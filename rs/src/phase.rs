//! The shared contract between the engine, the search and the network.
//!
//! A turn is not one decision. Flat, its widest measured node had ~1.9M edges,
//! which no policy head can emit a distribution over. Factored into the chain
//! `sample_legal_move` already walks, the widest node measured over 38,816
//! turns is **2,293**, with a 99th percentile of **164** — comfortably inside
//! `docs/SEARCH.md` §1.2's predicted 8,959 / 225, because the gear resize left
//! one mirror space per small gear instead of two and the mirrors were the
//! whole tail.
//!
//! `Phase` names where in that chain a position sits. It is `Copy` and carries
//! only what the edge set depends on, so `(GameState, Phase)` is a complete
//! search node.
//!
//! Definition follows `docs/SEARCH.md` §2.7. Both the tree and the encoder are
//! written against it, so it lives here rather than in either of them.

use crate::effect::Choice;
use crate::ids::*;
use crate::moves::Placement;
use crate::state::GameState;

/// Where in a turn a position sits.
///
/// `P` is the player to move. Every variant belongs to `P` except `ExtraDay`,
/// whose mover is the claimer named in it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    /// Start of P's turn: beg for corn, or don't. Width 1..=4.
    Beg,
    /// Place workers or retrieve them. Width 1..=2, or a single `Pity` edge
    /// when neither is possible.
    Mode,
    /// `n` workers already placed this turn. Width <= 7.
    Placing { n: u8 },
    /// Choosing which worker to pick up next. Width <= 7.
    PickWorker,
    /// A worker is picked and its action not yet chosen. Width <= K.
    Take { worker: WorkerId },
    /// Between rounds: the claimer may advance the calendar a second day.
    /// **The mover is `claimer`, not P.** Width 2.
    ExtraDay { claimer: PlayerId },
    /// "The gods take pity" — reachable only when nothing else is legal.
    PityPlace,
    /// Setup. Two sequential picks (4 edges, then 3) rather than one 6-way
    /// choice, so the pick reuses the same per-candidate head as `Take`.
    DraftTile { dealt: [u8; 4], kept: u8 },
}

impl Phase {
    /// Whose decision this is. Differs from the turn holder only at `ExtraDay`.
    pub fn mover(self, turn: PlayerId) -> PlayerId {
        match self {
            Phase::ExtraDay { claimer } => claimer,
            _ => turn,
        }
    }

    /// A dense index for the network's phase embedding. Exhaustive on purpose:
    /// adding a phase must break the build rather than mis-index a table.
    pub fn tag(self) -> u8 {
        match self {
            Phase::Beg => 0,
            Phase::Mode => 1,
            Phase::Placing { .. } => 2,
            Phase::PickWorker => 3,
            Phase::Take { .. } => 4,
            Phase::ExtraDay { .. } => 5,
            Phase::PityPlace => 6,
            Phase::DraftTile { .. } => 7,
        }
    }

    pub const COUNT: usize = 8;
}

/// One edge out of a `Phase` node.
///
/// `StopPlacing` and `StopRetrieving` are the commit edges; `Pity` commits too.
/// `StopPlacing` is illegal at `n == 0` and `StopRetrieving` before any worker
/// has been resolved, mirroring the rule that a move must involve a worker.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Step {
    Beg(Option<Temple>),
    Mode(ModeChoice),
    Place(Placement),
    StopPlacing,
    PickWorker(WorkerId),
    Take(Choice),
    StopRetrieving,
    Pity(Placement),
    ExtraDay(bool),
    DraftTile(u8),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ModeChoice {
    Place,
    Retrieve,
    Pity,
}

/// What a network (or a stand-in) returns for one node.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Prior over the edges at this node, in the order the tree enumerated
    /// them. Must sum to 1 over the legal set.
    pub priors: Vec<f32>,
    /// One value per player, in seat order, in `(-1, 1)`.
    pub value: [f32; N_PLAYERS],
}

/// One node awaiting evaluation.
///
/// Carries the *edges themselves* rather than a count. That matters twice over:
/// the evaluator no longer has to re-derive a candidate list the search already
/// built (~35 us per `Take` node, more than two forward passes), and it removes
/// an unwritten contract in which the network had to regenerate the edge list
/// in exactly the tree's order for its priors to line up, with nothing testing
/// that the two agreed.
pub struct Query<'a> {
    pub state: &'a GameState,
    pub phase: Phase,
    pub turn: PlayerId,
    pub edges: &'a [Step],
}

/// Anything that can score a search node.
///
/// The heuristic stand-in and the trained network implement the same trait, so
/// the search never learns which one it has and the arena can pit them against
/// each other.
///
/// Implement [`Evaluator::evaluate_many`] if the backend batches — for a neural
/// network that is worth about 20x on CPU alone, because a GEMM at batch 1 is
/// latency-bound. The single-node entry points default to it, so a batching
/// implementor gets the scalar cases for free and cannot accidentally leave a
/// per-position path behind.
pub trait Evaluator: Send + Sync {
    /// Evaluate a batch. **This is the method to implement.**
    ///
    /// The default fans out to `evaluate`, which is correct but forfeits the
    /// batching win.
    fn evaluate_many(&self, queries: &[Query<'_>]) -> Vec<Evaluation> {
        queries
            .iter()
            .map(|q| self.evaluate(q.state, q.phase, q.turn, q.edges.len()))
            .collect()
    }

    /// Evaluate one node, given its edges.
    ///
    /// Defaults to a one-element `evaluate_many`, so a batching implementor
    /// need not write it.
    fn evaluate_edges(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        edges: &[Step],
    ) -> Evaluation {
        let q = Query {
            state,
            phase,
            turn,
            edges,
        };
        self.evaluate_many(std::slice::from_ref(&q))
            .pop()
            .expect("evaluate_many must return one Evaluation per Query")
    }

    /// `turn` is the player whose turn it is; use `phase.mover(turn)` for whose
    /// decision this actually is.
    ///
    /// Prefer [`Evaluator::evaluate_edges`] where the edges are to hand, which
    /// in the search they always are.
    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation;

    /// For logging and checkpoint provenance.
    fn name(&self) -> String {
        "unnamed".into()
    }
}

/// A stand-in built on `eval::heuristic`: uniform priors, heuristic values,
/// squashed into the same range a value head will produce.
///
/// It exists so the search, the arena and the self-play driver can all be
/// exercised end to end before a network is trained, and so there is a baseline
/// to measure the trained net against.
pub struct HeuristicEvaluator;

impl Evaluator for HeuristicEvaluator {
    fn evaluate(
        &self,
        state: &GameState,
        _phase: Phase,
        _turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        let raw: Vec<f32> = PlayerId::ALL
            .iter()
            .map(|&q| crate::eval::heuristic(state, q))
            .collect();
        let mean = raw.iter().sum::<f32>() / N_PLAYERS as f32;
        // Same squash the value head is trained on, so the two are comparable.
        let value = std::array::from_fn(|i| ((raw[i] - mean) / 25.0).tanh());

        let p = if n_edges == 0 {
            0.0
        } else {
            1.0 / n_edges as f32
        };
        Evaluation {
            priors: vec![p; n_edges],
            value,
        }
    }

    fn name(&self) -> String {
        "heuristic".into()
    }
}
