//! MCTS over the factored tree.
//!
//! `docs/SEARCH.md` §3 and §4. The three things that make this not a textbook
//! AlphaZero search:
//!
//! * **Nodes hold their state.** `GameState` is a few hundred bytes of `Copy`
//!   with no heap, so a node is a memcpy. Replaying from the root would save
//!   that and cost nothing in `apply` — but the expensive half of a descent is
//!   *generating* a node's edges, and replay means regenerating
//!   `choices_for_worker` at every ancestor to know what edge index 3 meant.
//!   The state is also the transposition key, so it has to be there anyway.
//!
//! * **The value is a 4-vector and it is never negated.** Backup adds the whole
//!   vector at every level; the player identity enters only at selection, where
//!   a node maximises `to_move`'s component. That is max^n, and §4.3 argues the
//!   game genuinely is not zero-sum, so a scalar could not express it. Along a
//!   sub-decision chain the mover does not change for ~8 consecutive levels;
//!   `ExtraDay` is the one mid-round node where it does.
//!
//! * **`c_puct` is log-scaled.** Factoring makes visit counts span orders of
//!   magnitude *within one turn* — a `Mode` node sees every simulation entering
//!   the turn, a `Take` three levels down a cold branch sees a handful — and one
//!   constant cannot be right for both.
//!
//! # Against the alpha-beta, at matched work
//!
//! Worth knowing before spending another day tuning `search.rs`. On the same
//! positions and at the same cost per turn — `mcts:2048` at 21.9 ms against
//! `minimax:8:200::greedy:capw=25` at 19.8 ms — this search is **+9.32**
//! centred score head to head (95% CI +6.09..+12.55, p < 0.0001, 39 blocks /
//! 156 games, solo mode) and takes 40.4% of its games against three of them,
//! where an equally strong agent would take 25%.
//!
//! The match is on *work* rather than on the clock, deliberately. Both sides
//! are budget-invariant: this search runs its 2048 simulations whatever else
//! the machine is doing, and the alpha-beta was given a budget too large to
//! bind so that it stops at `max_depth` rather than on the clock. That matters
//! because the two budgets are different kinds of thing, and a loaded machine
//! silently favours the one counted in simulations — the same comparison with
//! a 200 ms alpha-beta budget reads +11.79 (n = 23) instead of +9.46, and the
//! difference is the contention, not the players. The 21.9-vs-19.8 ms figures
//! are what make the two budgets comparable at all.
//!
//! Not done yet: this is single-threaded. Virtual loss is applied and removed
//! for real, on the mover's component only, so the mechanism is exercised and
//! the shape is right, but the statistics are plain `u32`/`f32` rather than
//! §3.1's atomics and there is no batching. Swapping in `AtomicU32` and signed
//! fixed-point `AtomicI32` is mechanical; the descent logic does not change.

use crate::ids::*;
use crate::phase::{Evaluator, Phase, Step};
use crate::state::GameState;
use crate::tree;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;

/// The knobs of §6.3, with §2.6 and §3.3's recommended values.
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    /// §4.5: calibrated for a value in (-1, 1), not AlphaZero's win/loss scale.
    pub c_puct_init: f32,
    pub c_puct_base: f32,
    /// First-play urgency, subtracted from the parent's Q.
    pub fpu_reduction: f32,
    /// Carry the subtree over between the sub-decisions of a turn.
    ///
    /// **This is a search-quality knob, not a speed one.** A reused node keeps
    /// the visits and values it accumulated while it was an interior node of
    /// the previous sub-decision's search, and then receives the full `sims`
    /// again -- so the effective budget at reused nodes is larger and the
    /// search finds different moves. Measured at 512 simulations it is a wash
    /// on wall-clock (0.77 vs 0.79 games/s): retaining costs a sweep of the
    /// arena and a rebuild of the index, which roughly cancels the expansions
    /// it saves.
    pub tree_reuse: bool,
    pub dirichlet_eps: f32,
    /// Dirichlet `alpha` is `scale / n_edges`, set per node: widths here run
    /// from 2 to 32 and a fixed alpha would be negligible at one end and
    /// overwhelming at the other.
    pub dirichlet_scale: f32,
    /// K of §2.6: how many edges a node opens with.
    pub max_edges: usize,
    pub widen_c: f32,
    pub widen_alpha: f32,
    /// The absolute ceiling widening may reach.
    pub widen_cap: usize,
    /// `tau` in §3.8. Zero picks the most-visited edge.
    pub temperature: f32,
    pub virtual_loss: u32,
    /// Guard against a descent that never terminates. A whole game from day 0
    /// is ~900 sub-decisions.
    pub max_depth: u32,
    pub seed: u64,
    /// Where an edge's prior comes from. See [`Priors`].
    pub priors: Priors,
    /// Softmax temperature for [`Priors::OnePly`], **in points** — the same
    /// scale `eval::heuristic` returns, so 4.0 means "a four-point edge over a
    /// sibling is worth e times the prior".
    pub prior_temp: f32,
    /// Do not spend a one-ply pass on a node narrower than this.
    ///
    /// The pass costs an `apply_step` and an `eval::heuristic` per edge, so its
    /// cost is linear in width while the *benefit* is not: a two-edge node is
    /// resolved by three simulations whatever its prior says. Widths here run
    /// 2..2293 with a median searched width of 3 (`phase.rs`), so the threshold
    /// is where most of the saving is.
    pub prior_min_edges: usize,
}

/// Where an edge's prior comes from.
///
/// # Why this is a knob rather than a decision
///
/// `phase::HeuristicEvaluator` fills `Evaluation::priors` with `1.0 / n_edges`,
/// so with no network the search is told *nothing* about which sub-decision is
/// worth exploring and PUCT's exploration term does the whole job of a policy
/// head. A trained net's policy head is a real prior and must not be
/// overwritten, so the source has to be selectable rather than wired in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Priors {
    /// Whatever the evaluator returned.
    Evaluator,
    /// One ply per edge: apply the `Step` to a copy, score the result with
    /// `eval::heuristic` for the *mover* (not the turn holder — they differ at
    /// `ExtraDay`), and softmax at [`MctsConfig::prior_temp`].
    ///
    /// `eval::heuristic` is the same estimate the alpha-beta uses at its
    /// leaves, and it prices a placed worker (`board_position`, and
    /// `TEMPO_PER_ROUND` for riding a gear) rather than only what a player is
    /// holding — which is what makes a one-ply score of "put a worker on gear
    /// X" mean anything at all.
    OnePly,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct_init: 2.0,
            c_puct_base: 19652.0,
            fpu_reduction: 0.2,
            tree_reuse: true,
            dirichlet_eps: 0.25,
            dirichlet_scale: 10.0,
            max_edges: 32,
            widen_c: 2.0,
            widen_alpha: 0.5,
            widen_cap: 128,
            temperature: 1.0,
            virtual_loss: 1,
            max_depth: 2048,
            seed: 0,
            priors: Priors::Evaluator,
            prior_temp: 4.0,
            prior_min_edges: 3,
        }
    }
}

/// Something that can bias a node's priors without excluding anything.
///
/// This is the seam `src/plan.rs` is being written against. A game-long plan —
/// a target monument, a temple being raced for, a research track — knows things
/// a one-ply score cannot: that *this* building is on the way to *that*
/// monument, four rounds out. The way it says so is by raising a prior, never
/// by removing an edge, because PUCT still reaches a low-prior edge given
/// enough simulations. A wrong plan then costs simulations, not correctness,
/// and that is the whole reason to let a plan touch the search at all.
///
/// Weights are multiplicative and applied *before* renormalisation, so `1.0` is
/// "no opinion" and the identity bias is exactly the unbiased search.
pub trait PriorBias: Send + Sync {
    /// Fill `out` — already `1.0` and the same length as `steps` — with a
    /// non-negative weight per edge.
    fn bias(
        &self,
        state: &GameState,
        phase: Phase,
        mover: PlayerId,
        steps: &[Step],
        out: &mut [f32],
    );

    /// For the agent label, so a run biased by a plan cannot be mistaken for
    /// one that was not.
    fn name(&self) -> String {
        "bias".into()
    }
}

struct Edge {
    step: Step,
    prior: f32,
    child: Option<u32>,
    n: u32,
    w: [f32; N_PLAYERS],
    /// Virtual loss currently on this edge. Always zero between simulations in
    /// a single-threaded search; the bookkeeping is here for the threaded one.
    vloss: u32,
}

struct Node {
    state: GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
    /// `phase.mover(turn)`, cached: selection reads it on every visit.
    mover: usize,
    edges: Box<[Edge]>,
    /// How many edges progressive widening has opened. Beyond `max_edges` the
    /// edge list is sorted by prior, so the open set is always the best
    /// `active` of them.
    active: usize,
    /// Legal steps at this node before §2.6's cap dropped any. `edges.len()` is
    /// already the truncated list, so without this there is nothing left that
    /// remembers a wide node was ever wide.
    n_legal: u32,
    visits: u32,
    w: [f32; N_PLAYERS],
    /// The evaluation this node was created with. `None` for a node the descent
    /// walks straight through — §3.6's "a node with a single edge should be
    /// collapsed without any evaluation".
    value: Option<[f32; N_PLAYERS]>,
    terminal: bool,
    /// Part of the root player's current turn, and therefore due Dirichlet
    /// noise (§3.7).
    in_root_turn: bool,
}

/// One searched sub-decision.
pub struct SearchResult {
    /// The step to play here, sampled from the visit counts per §3.8.
    pub step: Step,
    /// Visit count per open edge. This is the policy training target; normalise
    /// with [`SearchResult::policy`].
    pub visits: Vec<(Step, u32)>,
    /// Backed-up value at the searched node, per player, in seat order.
    pub root_value: [f32; N_PLAYERS],
    /// Simulations that actually ran. Zero when the node had one edge and the
    /// search was skipped.
    pub sims: u32,
    /// Arena size when the search finished. Diagnostics.
    pub nodes: usize,
    /// Legal edges before the §2.6 cap, so callers can see how much of the
    /// candidate list the tree opened.
    pub legal_edges: usize,
}

impl SearchResult {
    /// Visit share over the open edges. Sums to 1, or is uniform if nothing was
    /// visited.
    pub fn policy(&self) -> Vec<f32> {
        let total: u32 = self.visits.iter().map(|(_, n)| n).sum();
        if total == 0 {
            let p = 1.0 / self.visits.len().max(1) as f32;
            return vec![p; self.visits.len()];
        }
        self.visits
            .iter()
            .map(|(_, n)| *n as f32 / total as f32)
            .collect()
    }
}

/// One complete turn the tree searched, as the path of `Step`s that spells it.
#[derive(Clone, Debug)]
pub struct TurnLine {
    pub steps: Vec<Step>,
    /// Simulations that ran the whole turn out along this path — the numerator
    /// of the search's policy over *turns*.
    pub visits: u32,
    /// Backed-up value for the searched player at the commit edge, on the
    /// `z_rel` scale of §4.5 rather than in points.
    pub value: f32,
}

/// What one root search knows about complete turns.
///
/// # What it is not
///
/// It is not a ranking of the legal move list. A turn is a chain of ~8
/// sub-decisions and the tree only ever holds the paths its simulations walked,
/// so `lines` is bounded by `sims` and is a *sample shaped by the search* —
/// which is the interesting thing about it, and also why nothing here may claim
/// to be exhaustive.
pub struct TurnRanking {
    /// Best first by `visits`, truncated to the caller's `keep`.
    pub lines: Vec<TurnLine>,
    /// Complete turns the tree reached, before `keep` truncated the list.
    pub found: usize,
    /// Simulations that reached a commit edge. Smaller than `sims`, because a
    /// simulation that stops at a fresh node inside the turn never finishes
    /// one; this is the honest denominator for a visit share.
    pub committed: u32,
    pub sims: u32,
    /// True when every node on a searched path had all of its legal edges open
    /// — nothing lost to §2.6's cap, nothing still waiting on progressive
    /// widening — so every legal turn was at least *reachable*. It says nothing
    /// about whether the search actually looked at them.
    pub all_edges_open: bool,
    pub nodes: usize,
}

/// One turn played out as a chain of searches.
pub struct PlayedTurn {
    /// One entry per sub-decision, in the order they were taken. Each is a
    /// training example (§3.6: a searched turn yields ~8 policy targets, not 1).
    pub steps: Vec<(Phase, PlayerId, u8, SearchResult)>,
    /// Where the game continues, or `None` once it is over.
    pub next: Option<(Phase, PlayerId, u8)>,
}

pub struct Mcts<E: Evaluator> {
    eval: E,
    cfg: MctsConfig,
    rng: StdRng,
    nodes: Vec<Node>,
    /// Transposition index. `FxHashMap` rather than the default: the key is a
    /// whole 320-byte `GameState`, and SipHash over that costs about as much as
    /// encoding the position for the network.
    index: FxHashMap<(GameState, Phase, PlayerId, u8), u32>,
    /// Nodes carried over from the previous sub-decision, and nodes thrown
    /// away, for the last `search_at`. Observability for the reuse: if `reused`
    /// stays at zero across a turn the retention is not firing.
    reused: usize,
    discarded: usize,
    /// From `MctsConfig::tree_reuse`, or `TZOLKIN_NO_TREE_REUSE` in the
    /// environment. Reuse is **not** behaviour-neutral -- see `MctsConfig` --
    /// so being able to switch it off is how the two are compared.
    reuse_disabled: bool,
    /// Optional plan-level steer on the priors. See [`PriorBias`].
    bias: Option<std::sync::Arc<dyn PriorBias>>,
    /// Scratch for the one-ply prior pass, so a node of width 2293 does not
    /// allocate on every expansion.
    scratch: Vec<f32>,
}

impl<E: Evaluator> Mcts<E> {
    pub fn new(eval: E, config: MctsConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let reuse_disabled =
            !config.tree_reuse || std::env::var_os("TZOLKIN_NO_TREE_REUSE").is_some();
        Mcts {
            eval,
            cfg: config,
            rng,
            nodes: Vec::new(),
            index: FxHashMap::default(),
            reused: 0,
            discarded: 0,
            reuse_disabled,
            bias: None,
            scratch: Vec::new(),
        }
    }

    /// Install a plan-level steer on the priors. See [`PriorBias`].
    pub fn set_bias(&mut self, bias: Option<std::sync::Arc<dyn PriorBias>>) {
        self.bias = bias;
    }

    pub fn bias_name(&self) -> Option<String> {
        self.bias.as_ref().map(|b| b.name())
    }

    pub fn evaluator(&self) -> &E {
        &self.eval
    }

    pub fn config(&self) -> &MctsConfig {
        &self.cfg
    }

    pub fn config_mut(&mut self) -> &mut MctsConfig {
        &mut self.cfg
    }

    /// Nodes the last `search_at` carried over from the previous sub-decision,
    /// and nodes it discarded. Zero reuse across a whole turn means the
    /// retention is not firing.
    pub fn reuse_stats(&self) -> (usize, usize) {
        (self.reused, self.discarded)
    }

    /// Run `sims` simulations from this node and return the chosen step and the
    /// visit distribution over its edges.
    ///
    /// Correct at the start of a turn and at every node except a `PickWorker`
    /// that has already resolved a worker; use [`Mcts::search_at`] there, or
    /// [`Mcts::play_turn`], which threads the counter itself.
    pub fn search(
        &mut self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        sims: u32,
    ) -> SearchResult {
        self.search_at(state, phase, turn, 0, sims)
    }

    /// As [`Mcts::search`], with the retrieval counter `tree::legal_steps`
    /// needs at `PickWorker`.
    pub fn search_at(
        &mut self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        done: u8,
        sims: u32,
    ) -> SearchResult {
        let legal = tree::legal_steps(state, phase, turn, done);
        assert!(
            !legal.is_empty(),
            "no legal step at {phase:?} for {turn:?}; the pity rule should have made this impossible"
        );

        // A forced move is not worth an evaluation, let alone a search.
        if legal.len() == 1 {
            let step = legal.into_iter().next().unwrap();
            return SearchResult {
                visits: vec![(step.clone(), 0)],
                step,
                root_value: [0.0; N_PLAYERS],
                sims: 0,
                nodes: 0,
                legal_edges: 1,
            };
        }

        // Reuse the subtree if this node is already in the arena from the
        // previous sub-decision of the same turn.
        //
        // A turn is 5-8 sub-decisions and each one used to start from an empty
        // arena, throwing away the subtree under the edge just played --
        // including the node now being searched, with all of its statistics.
        //
        // Only within the turn. §3.7 puts Dirichlet noise on every node of the
        // root player's turn, so a node carrying `in_root_turn` was noised
        // under the same turn this search belongs to and its priors are still
        // the right ones. A node from outside that turn was not, and reusing it
        // would search un-noised priors as if they were noised; those are
        // cleared instead.
        let key = (*state, phase, turn, done);
        let reusable = if self.reuse_disabled {
            None
        } else {
            self.index
                .get(&key)
                .copied()
                .filter(|&i| self.nodes[i as usize].in_root_turn)
        };

        let root = match reusable {
            Some(idx) => {
                let before = self.nodes.len();
                let r = self.retain_subtree(idx);
                self.reused = self.nodes.len();
                self.discarded = before - self.nodes.len();
                r
            }
            None => {
                self.nodes.clear();
                self.index.clear();
                self.reused = 0;
                self.discarded = 0;
                self.node_for(*state, phase, turn, done, true).0
            }
        };

        for _ in 0..sims {
            self.simulate(root);
        }

        let e = self.choose(root);
        let node = &self.nodes[root as usize];
        let visits: Vec<(Step, u32)> = node.edges[..node.active]
            .iter()
            .map(|edge| (edge.step.clone(), edge.n))
            .collect();
        let root_value = if node.visits > 0 {
            std::array::from_fn(|i| node.w[i] / node.visits as f32)
        } else {
            node.value.unwrap_or([0.0; N_PLAYERS])
        };
        SearchResult {
            step: node.edges[e].step.clone(),
            visits,
            root_value,
            sims,
            nodes: self.nodes.len(),
            legal_edges: node.n_legal as usize,
        }
    }

    /// Search and play every sub-decision of one turn, advancing `state`.
    ///
    /// This is the shape the self-play driver wants: it threads the retrieval
    /// counter, stops on the commit edge, and hands back one policy target per
    /// node on the played path.
    pub fn play_turn(
        &mut self,
        state: &mut GameState,
        phase: Phase,
        turn: PlayerId,
        sims: u32,
    ) -> PlayedTurn {
        let mut at = (phase, turn, 0u8);
        let mut steps = Vec::new();
        loop {
            let (phase, turn, done) = at;
            let result = self.search_at(state, phase, turn, done, sims);
            let step = result.step.clone();
            steps.push((phase, turn, done, result));
            let transition = tree::apply_step(state, phase, turn, done, &step);
            match transition.next() {
                None => return PlayedTurn { steps, next: None },
                Some(next) => {
                    if transition.committed() {
                        return PlayedTurn {
                            steps,
                            next: Some(next),
                        };
                    }
                    at = next;
                }
            }
        }
    }

    /// Search one whole turn and rank the complete turns the tree explored.
    ///
    /// # Why this is not `play_turn`
    ///
    /// `play_turn` searches each link of the chain separately, so it spends
    /// `sims` per sub-decision and never holds more than one turn's prefix.
    /// This runs a *single* search rooted at `Beg` and reads complete
    /// root-to-commit paths out of the resulting tree, which is the only place
    /// the search's opinion about a whole turn exists as one number.
    ///
    /// It is an explanation hook — `Agent::ranked_moves`, the TUI's `agent`
    /// view — and deliberately not what the agent plays, per that trait's note
    /// that the two may cost very different amounts.
    pub fn ranked_turns(
        &mut self,
        state: &GameState,
        turn: PlayerId,
        sims: u32,
        keep: usize,
    ) -> TurnRanking {
        // A fresh arena, not the reuse path: a viewer asking about the same
        // position twice must get the same answer, and a retained subtree makes
        // the second answer depend on the first.
        self.nodes.clear();
        self.index.clear();
        self.reused = 0;
        self.discarded = 0;
        let root = self.node_for(*state, Phase::Beg, turn, 0, true).0;
        for _ in 0..sims {
            self.simulate(root);
        }

        let seat = turn.idx();
        let mut lines: Vec<TurnLine> = Vec::new();
        let mut committed = 0u32;
        let mut all_edges_open = true;
        let mut stack: Vec<(u32, Vec<Step>)> = vec![(root, Vec::new())];
        while let Some((idx, path)) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if node.active < node.n_legal as usize {
                all_edges_open = false;
            }
            // Collected first so the borrow ends before the recursion pushes.
            let taken: Vec<(usize, u32, f32, bool)> = node.edges[..node.active]
                .iter()
                .filter(|e| e.n > 0 && e.child.is_some())
                .map(|e| {
                    let c = e.child.unwrap();
                    (
                        c as usize,
                        e.n,
                        e.w[seat] / e.n as f32,
                        // `in_root_turn` is cleared by `create_child` exactly at
                        // a commit edge, so a child without it is the start of
                        // somebody else's turn.
                        !self.nodes[c as usize].in_root_turn,
                    )
                })
                .collect();
            let steps: Vec<Step> = node.edges[..node.active]
                .iter()
                .filter(|e| e.n > 0 && e.child.is_some())
                .map(|e| e.step.clone())
                .collect();
            for ((child, n, q, commits), step) in taken.into_iter().zip(steps) {
                let mut next = path.clone();
                next.push(step);
                if commits {
                    committed += n;
                    lines.push(TurnLine {
                        steps: next,
                        visits: n,
                        value: q,
                    });
                } else {
                    stack.push((child as u32, next));
                }
            }
        }
        lines.sort_by(|a, b| b.visits.cmp(&a.visits).then(b.value.total_cmp(&a.value)));
        let found = lines.len();
        lines.truncate(keep.max(1));
        TurnRanking {
            lines,
            found,
            committed,
            sims,
            all_edges_open,
            nodes: self.nodes.len(),
        }
    }

    pub fn arena_len(&self) -> usize {
        self.nodes.len()
    }

    /// `(legal edges, edges open)` for every node in the arena, interior ones
    /// included.
    ///
    /// The roots of a turn's sub-decisions are narrow — p50 3 — so a
    /// measurement taken there says nothing about whether §2.6's cap is set
    /// sensibly. The wide nodes are `Take` nodes further down a descent, and
    /// this is the only place they are visible.
    pub fn node_widths(&self) -> Vec<(u32, u32)> {
        self.nodes
            .iter()
            .filter(|n| !n.terminal)
            .map(|n| (n.n_legal, n.active as u32))
            .collect()
    }

    /// Total virtual loss still resting on edges anywhere in the arena.
    ///
    /// Zero between simulations by construction: `simulate` applies it on the
    /// way down and `backup` walks the whole descent path and lifts it. It is
    /// exposed because that is the one property of virtual loss the
    /// single-threaded search cannot demonstrate for itself. With a single
    /// descent in flight nothing ever *observes* a virtual loss -- the knob is
    /// measurably inert, and setting it to 0, 1 or 10 gives bit-identical visit
    /// counts -- so a bug that failed to lift one would sit undetected until
    /// the day tree parallelism lands, which is exactly the day it would start
    /// silently poisoning `Q`. See the module note on parallelism.
    pub fn virtual_loss_residue(&self) -> u32 {
        self.nodes
            .iter()
            .flat_map(|n| n.edges.iter())
            .map(|e| e.vloss)
            .sum()
    }

    // ---- one simulation -------------------------------------------------

    fn simulate(&mut self, root: u32) {
        let mut path: Vec<(u32, usize)> = Vec::with_capacity(32);
        let mut value = [0.0f32; N_PLAYERS];
        let vl = self.cfg.virtual_loss;

        for depth in 0.. {
            let cur = self.cursor(&path, root);
            // A descent can run to the end of the game through transpositions
            // and collapsed nodes; the guard is against a graph cycle, which
            // the state advancing on every edge should already rule out.
            if depth >= self.cfg.max_depth || self.nodes[cur as usize].terminal {
                value = self.nodes[cur as usize].value.unwrap_or([0.0; N_PLAYERS]);
                break;
            }
            self.widen(cur);
            let e = self.select(cur);

            let mover = self.nodes[cur as usize].mover;
            {
                let edge = &mut self.nodes[cur as usize].edges[e];
                edge.n += vl;
                edge.w[mover] -= vl as f32;
                edge.vloss += vl;
            }
            path.push((cur, e));

            let (child, is_new) = match self.nodes[cur as usize].edges[e].child {
                Some(c) => (c, false),
                None => {
                    let (c, is_new) = self.create_child(cur, e);
                    self.nodes[cur as usize].edges[e].child = Some(c);
                    (c, is_new)
                }
            };
            // A fresh node carrying a value is where this simulation stops. A
            // fresh node without one is a single-edge node the descent walks
            // straight through; an already-known node is a transposition, and
            // graph search descends through those too.
            if is_new {
                if let Some(v) = self.nodes[child as usize].value {
                    value = v;
                    break;
                }
            }
        }

        let leaf = self.cursor(&path, root);
        self.backup(&path, leaf, value);
    }

    /// The node the descent currently sits on: the child of the last edge
    /// taken, or the root.
    fn cursor(&self, path: &[(u32, usize)], root: u32) -> u32 {
        match path.last() {
            None => root,
            Some(&(n, e)) => self.nodes[n as usize].edges[e]
                .child
                .expect("descent followed an edge with no child"),
        }
    }

    fn widen(&mut self, idx: u32) {
        let MctsConfig {
            widen_c,
            widen_alpha,
            widen_cap,
            ..
        } = self.cfg;
        let node = &mut self.nodes[idx as usize];
        let cap = node.edges.len().min(widen_cap);
        while node.active < cap {
            // §2.6: admit the m-th child once N(node) >= C * m^alpha.
            let m = node.active as f32 + 1.0;
            if (node.visits as f32) >= widen_c * m.powf(widen_alpha) {
                node.active += 1;
            } else {
                break;
            }
        }
    }

    fn select(&self, idx: u32) -> usize {
        let node = &self.nodes[idx as usize];
        let mover = node.mover;
        let open = &node.edges[..node.active];

        let total: u32 = open.iter().map(|e| e.n).sum();
        // `sqrt(0)` would zero the exploration term on a node's first visit and
        // leave the choice to FPU, which is identical across unvisited edges.
        // Flooring at one lets the prior break that tie.
        let sqrt_total = (total as f32).max(1.0).sqrt();
        let c = self.cfg.c_puct_init
            + ((1.0 + total as f32 + self.cfg.c_puct_base) / self.cfg.c_puct_base).ln();

        // FPU anchors to the parent rather than to zero: "no information" and
        // "an even position" only coincide if the value scale is centred, and
        // guessing that is exactly what FPU exists to avoid.
        let parent_q = if node.visits > 0 {
            node.w[mover] / node.visits as f32
        } else {
            node.value.map(|v| v[mover]).unwrap_or(0.0)
        };
        let expanded: f32 = open.iter().filter(|e| e.n > 0).map(|e| e.prior).sum();
        let fpu = parent_q - self.cfg.fpu_reduction * expanded.max(0.0).sqrt();

        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, edge) in open.iter().enumerate() {
            let q = if edge.n > 0 {
                edge.w[mover] / edge.n as f32
            } else {
                fpu
            };
            let u = c * edge.prior * sqrt_total / (1.0 + edge.n as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    fn backup(&mut self, path: &[(u32, usize)], leaf: u32, value: [f32; N_PLAYERS]) {
        {
            let node = &mut self.nodes[leaf as usize];
            node.visits += 1;
            add(&mut node.w, &value);
        }
        for &(idx, e) in path.iter().rev() {
            let node = &mut self.nodes[idx as usize];
            let mover = node.mover;
            if idx != leaf {
                node.visits += 1;
                add(&mut node.w, &value);
            }
            let edge = &mut node.edges[e];
            // Lift the virtual loss, then record the real result. The whole
            // 4-vector propagates unchanged; no negation, no player-relative
            // flip -- that is what makes max^n work.
            edge.n -= edge.vloss;
            edge.w[mover] += edge.vloss as f32;
            edge.vloss = 0;
            edge.n += 1;
            add(&mut edge.w, &value);
        }
    }

    fn choose(&mut self, idx: u32) -> usize {
        let tau = self.cfg.temperature;
        let counts: Vec<f32> = {
            let node = &self.nodes[idx as usize];
            node.edges[..node.active].iter().map(|e| e.n as f32).collect()
        };
        if tau <= 1e-3 {
            return counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        let weights: Vec<f32> = counts.iter().map(|&n| n.powf(1.0 / tau)).collect();
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return self.rng.gen_range(0..counts.len().max(1));
        }
        let mut draw = self.rng.gen::<f32>() * total;
        for (i, &w) in weights.iter().enumerate() {
            draw -= w;
            if draw <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    // ---- expansion ------------------------------------------------------

    fn create_child(&mut self, parent: u32, e: usize) -> (u32, bool) {
        let (state, phase, turn, done, in_root) = {
            let node = &self.nodes[parent as usize];
            (
                node.state,
                node.phase,
                node.turn,
                node.done,
                node.in_root_turn,
            )
        };
        let step = self.nodes[parent as usize].edges[e].step.clone();

        let mut next = state;
        let transition = tree::apply_step(&mut next, phase, turn, done, &step);
        // The noise of §3.7 covers the root player's *turn*, which ends at the
        // first commit edge.
        let in_root = in_root && !transition.committed();

        match transition.next() {
            None => (self.push_terminal(next), true),
            Some((phase, turn, done)) => self.node_for(next, phase, turn, done, in_root),
        }
    }

    /// Keep only the subtree reachable from `root`, renumbering it to start at
    /// zero and rebuilding the transposition index.
    ///
    /// Moves nodes rather than cloning them: an arena entry owns a
    /// `Box<[Edge]>`, and copying those would give back much of what the reuse
    /// saves.
    fn retain_subtree(&mut self, root: u32) -> u32 {
        // Breadth-first, so `order[i]` is the node that becomes index `i`.
        let mut mapping: Vec<Option<u32>> = vec![None; self.nodes.len()];
        let mut order: Vec<u32> = Vec::with_capacity(self.nodes.len());
        mapping[root as usize] = Some(0);
        order.push(root);

        let mut i = 0;
        while i < order.len() {
            let old = order[i] as usize;
            i += 1;
            for e in self.nodes[old].edges.iter() {
                if let Some(c) = e.child {
                    if mapping[c as usize].is_none() {
                        mapping[c as usize] = Some(order.len() as u32);
                        order.push(c);
                    }
                }
            }
        }

        let mut slots: Vec<Option<Node>> =
            std::mem::take(&mut self.nodes).into_iter().map(Some).collect();
        let mut kept = Vec::with_capacity(order.len());
        for &old in &order {
            let mut n = slots[old as usize].take().expect("node visited twice");
            for e in n.edges.iter_mut() {
                // A child outside the retained set cannot happen -- the sweep
                // above followed every edge -- but dropping the index rather
                // than trusting it keeps a stale one from being dereferenced.
                e.child = e.child.and_then(|c| mapping[c as usize]);
            }
            kept.push(n);
        }
        self.nodes = kept;

        self.index.clear();
        for (i, n) in self.nodes.iter().enumerate() {
            self.index
                .insert((n.state, n.phase, n.turn, n.done), i as u32);
        }
        0
    }

    fn node_for(
        &mut self,
        state: GameState,
        phase: Phase,
        turn: PlayerId,
        done: u8,
        in_root_turn: bool,
    ) -> (u32, bool) {
        if let Some(&idx) = self.index.get(&(state, phase, turn, done)) {
            return (idx, false);
        }
        if state.over {
            return (self.push_terminal(state), true);
        }

        // A dead node would back up a terminal value for a live position, which
        // is worse than crashing, so this is an assert rather than a fallback.
        let steps = tree::legal_steps(&state, phase, turn, done);
        assert!(!steps.is_empty(), "{phase:?} for {turn:?} has no legal step");

        let n_legal = steps.len();
        let (mut edges, value) = if n_legal == 1 {
            // Collapsed: one edge means no decision, so no evaluation. The
            // descent walks straight through and the network call is saved.
            let step = steps.into_iter().next().unwrap();
            (vec![new_edge(step, 1.0)], None)
        } else {
            // Hand over the edges, not just a count. Passing a count forced
            // the evaluator to re-derive the list from the engine and match it
            // by length alone -- which abstained to a uniform prior on 15% of
            // real nodes, and would have scrambled the policy silently on any
            // right-length list in the wrong order.
            let eval = self.eval.evaluate_edges(&state, phase, turn, &steps);
            let mut value = eval.value;
            // §4.5: assert the constant-sum invariant exactly rather than
            // approximately, so denying the leader really does raise your own
            // component.
            recentre(&mut value);
            debug_assert_eq!(
                eval.priors.len(),
                n_legal,
                "evaluator returned the wrong prior count"
            );
            let priors = self.priors_for(&state, phase, turn, done, &steps, eval.priors);
            let mut edges: Vec<Edge> = steps
                .into_iter()
                .enumerate()
                .map(|(i, step)| new_edge(step, priors.get(i).copied().unwrap_or(0.0)))
                .collect();

            // Only a node wide enough to be capped gets reordered; below the cap
            // the edges stay in the order the generator produced them, which is
            // the order the encoder will see.
            if edges.len() > self.cfg.max_edges {
                edges.sort_by(|a, b| b.prior.total_cmp(&a.prior));
                edges.truncate(self.cfg.widen_cap.max(self.cfg.max_edges));
            }
            (edges, Some(value))
        };

        // Priors are renormalised over the retained edges, so the discarded
        // tail does not quietly drain the exploration term.
        let sum: f32 = edges.iter().map(|e| e.prior).sum();
        if sum > 0.0 {
            for edge in edges.iter_mut() {
                edge.prior /= sum;
            }
        } else {
            let p = 1.0 / edges.len() as f32;
            for edge in edges.iter_mut() {
                edge.prior = p;
            }
        }

        if in_root_turn && self.cfg.dirichlet_eps > 0.0 && edges.len() > 1 {
            self.add_noise(&mut edges);
        }

        let active = edges.len().min(self.cfg.max_edges);
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            state,
            phase,
            turn,
            done,
            mover: phase.mover(turn).idx(),
            edges: edges.into_boxed_slice(),
            active,
            n_legal: n_legal as u32,
            visits: 0,
            w: [0.0; N_PLAYERS],
            value,
            terminal: false,
            in_root_turn,
        });
        self.index.insert((state, phase, turn, done), idx);
        (idx, true)
    }

    fn push_terminal(&mut self, state: GameState) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            state,
            phase: Phase::Beg,
            turn: state.current,
            done: 0,
            mover: state.current.idx(),
            edges: Vec::new().into_boxed_slice(),
            active: 0,
            n_legal: 0,
            visits: 0,
            w: [0.0; N_PLAYERS],
            value: Some(z_rel(state.scores())),
            terminal: true,
            in_root_turn: false,
        });
        idx
    }

    /// The prior over a node's edges, before the §2.6 cap and the §3.7 noise.
    ///
    /// The evaluator's own priors are the default because a trained policy head
    /// *is* this, only better. [`Priors::OnePly`] exists because there is no
    /// head yet and `HeuristicEvaluator` returns `1.0 / n_edges`, which tells
    /// the search nothing.
    fn priors_for(
        &mut self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        done: u8,
        steps: &[Step],
        from_eval: Vec<f32>,
    ) -> Vec<f32> {
        let mover = phase.mover(turn);
        let mut p = match self.cfg.priors {
            Priors::OnePly if steps.len() >= self.cfg.prior_min_edges => {
                self.one_ply(state, phase, turn, done, steps, mover)
            }
            _ => from_eval,
        };
        if let Some(bias) = self.bias.clone() {
            self.scratch.clear();
            self.scratch.resize(steps.len(), 1.0);
            bias.bias(state, phase, mover, steps, &mut self.scratch);
            for (x, w) in p.iter_mut().zip(self.scratch.iter()) {
                // Clamped at zero, not renormalised here: node_for renormalises
                // over the retained edges anyway, and a plan that zeroed every
                // edge would fall through to that function's uniform fallback
                // rather than to a NaN.
                *x *= w.max(0.0);
            }
        }
        p
    }

    /// One `apply_step` and one `eval::heuristic` per edge, softmaxed at
    /// `prior_temp`.
    ///
    /// The alpha-beta work measured 2.16 µs to walk the move generator against
    /// 2.29 µs to walk *and* apply *and* score, so scoring a candidate the
    /// generator has already produced is the cheap end of expansion. What is
    /// not cheap is doing it on a node three edges wide that three simulations
    /// would have resolved anyway — hence `prior_min_edges`.
    fn one_ply(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        done: u8,
        steps: &[Step],
        mover: PlayerId,
    ) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(steps.len());
        let mut best = f32::NEG_INFINITY;
        for step in steps {
            let mut next = *state;
            let _ = tree::apply_step(&mut next, phase, turn, done, step);
            // The mover's own estimate, not `eval::margin`. Siblings of one
            // node differ almost only in what the mover did, so the
            // best-opponent term is common to them and subtracting it would
            // cost three more `heuristic` calls an edge to change nothing.
            let s = crate::eval::heuristic(&next, mover);
            if s > best {
                best = s;
            }
            out.push(s);
        }
        // Shifted by the max before exponentiating: `heuristic` runs to ~200
        // points late in a game and `exp(200/4)` is not a number.
        let t = self.cfg.prior_temp.max(1e-3);
        let mut sum = 0.0;
        for x in out.iter_mut() {
            *x = ((*x - best) / t).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in out.iter_mut() {
                *x /= sum;
            }
        }
        out
    }

    /// §3.7: noise on every node of the root player's turn, not only the literal
    /// root. The `Beg` node's edges are "beg or don't"; noise there cannot
    /// diversify which building gets constructed.
    fn add_noise(&mut self, edges: &mut [Edge]) {
        let n = edges.len();
        let alpha = (self.cfg.dirichlet_scale / n as f32).clamp(0.03, 5.0);
        let mut draws: Vec<f32> = (0..n).map(|_| gamma(&mut self.rng, alpha)).collect();
        let total: f32 = draws.iter().sum();
        if total <= 0.0 {
            return;
        }
        for d in draws.iter_mut() {
            *d /= total;
        }
        let eps = self.cfg.dirichlet_eps;
        for (edge, d) in edges.iter_mut().zip(draws) {
            edge.prior = (1.0 - eps) * edge.prior + eps * d;
        }
    }
}

fn add(acc: &mut [f32; N_PLAYERS], v: &[f32; N_PLAYERS]) {
    for (a, b) in acc.iter_mut().zip(v) {
        *a += b;
    }
}

fn new_edge(step: Step, prior: f32) -> Edge {
    Edge {
        step,
        prior,
        child: None,
        n: 0,
        w: [0.0; N_PLAYERS],
        vloss: 0,
    }
}

/// §4.5's target: bounded, centred, and dense enough to carry signal from round
/// one. A one-hot winner label is two bits a game, which this project's ~10^5
/// self-play games cannot afford.
pub fn z_rel(scores: [i16; N_PLAYERS]) -> [f32; N_PLAYERS] {
    let mean = scores.iter().map(|&s| s as f32).sum::<f32>() / N_PLAYERS as f32;
    let mut v: [f32; N_PLAYERS] = std::array::from_fn(|i| ((scores[i] as f32 - mean) / 25.0).tanh());
    recentre(&mut v);
    v
}

/// 1 / |winners| each. Reporting and the auxiliary heads only; search does not
/// consume it.
pub fn win_share(state: &GameState) -> [f32; N_PLAYERS] {
    let winners = state.winners();
    let share = 1.0 / winners.len().max(1) as f32;
    let mut out = [0.0; N_PLAYERS];
    for p in winners {
        out[p.idx()] = share;
    }
    out
}

fn recentre(v: &mut [f32; N_PLAYERS]) {
    let mean = v.iter().sum::<f32>() / N_PLAYERS as f32;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

// ---- Dirichlet ---------------------------------------------------------

/// Gamma(shape, 1) by Marsaglia-Tsang, with the standard boost for shape < 1.
///
/// Hand-rolled because `rand_distr` is not a dependency and this is the only
/// distribution the search needs.
fn gamma(rng: &mut StdRng, shape: f32) -> f32 {
    if shape < 1.0 {
        let u: f32 = rng.gen::<f32>().max(1e-9);
        return gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f32 = rng.gen::<f32>().max(1e-9);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn standard_normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-9);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
