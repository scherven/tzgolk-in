//! The factored action tree: a turn as a chain of sub-decisions.
//!
//! `docs/SEARCH.md` §2.4-2.7. A turn is not one decision but the chain
//! `sample_legal_move` already walks — beg? place or retrieve? which worker?
//! which space? stop or continue? — and the flat `Move` is the *product* of that
//! chain flattened out. Building the tree over the chain turns a node of width
//! up to ~1.9M into a path of ~8 nodes of width <= 32.
//!
//! The rule this module lives by: **it states no rule of its own.** Every edge
//! list comes from an engine generator — `beg_options`, `lowest_free`,
//! `Placement::space_cost`, `on_board`, `choices_for_worker`, `pity_moves`,
//! `has_normal_move` — and every application is the mutation `apply_move` would
//! have made. That is what makes `reachable_after_turn` provably equal to
//! `legal_moves` + `apply_move` (§2.8, `tests/tree.rs`) rather than merely
//! believed equal.
//!
//! One thing `Phase` does not carry: how many workers a retrieval has already
//! resolved. `Placing { n }` carries its counter; `PickWorker` does not, and
//! `StopRetrieving` turns on exactly that fact — an empty `Move` is illegal
//! (`check_move`, "retrieval move with no workers"). `phase.rs` is a shared
//! contract with the encoder and is not mine to widen, so the counter travels
//! alongside as `done`. See the note on `legal_steps`.

use crate::effect::Choice;
use crate::ids::*;
use crate::moves::{self, MoveKind, Placement};
use crate::phase::{ModeChoice, Phase, Step};
use crate::state::GameState;
use std::collections::HashSet;

/// `Phase::DraftTile::kept` before either pick has been made.
pub const DRAFT_NONE: u8 = 0xFF;

/// Where a step leaves the search.
///
/// A commit edge (`StopPlacing`, `StopRetrieving`, `Pity`) runs the whole
/// end-of-turn sequence — `refill_buildings`, the seat handoff, and at the end
/// of a round `resolve_first_player`, the `ExtraDay` offer and the day advance
/// with its food day and scoring — so the caller never has to know where a
/// round boundary is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Transition {
    /// Same turn, next sub-decision.
    Step {
        phase: Phase,
        turn: PlayerId,
        done: u8,
    },
    /// The chain ended: the turn (or the between-round `ExtraDay` node) is over
    /// and the search has moved on to another player.
    Commit { phase: Phase, turn: PlayerId },
    /// The game finished on this edge. `state.scores()` is final.
    Over,
}

impl Transition {
    /// Whether this edge ended the turn.
    pub fn committed(self) -> bool {
        !matches!(self, Transition::Step { .. })
    }

    /// The node the step leads to, or `None` once the game is over.
    pub fn next(self) -> Option<(Phase, PlayerId, u8)> {
        match self {
            Transition::Step { phase, turn, done } => Some((phase, turn, done)),
            Transition::Commit { phase, turn } => Some((phase, turn, 0)),
            Transition::Over => None,
        }
    }
}

// ---- edges -------------------------------------------------------------

/// Every legal step out of one node.
///
/// `done` is the number of workers this turn has already resolved. It is read
/// only at `PickWorker`, where it decides whether `StopRetrieving` is offered;
/// pass 0 anywhere else and at the start of a turn. It exists because
/// `Phase::PickWorker` carries no counter of its own — see the module note.
///
/// The `Take` list is returned whole. The K = 32 cap of §2.6 is a *search*
/// decision made against the priors, so it lives in `mcts`, not here: this
/// function's answer has to stay the legal set or the equivalence test in §2.8
/// is testing something else.
pub fn legal_steps(state: &GameState, phase: Phase, turn: PlayerId, done: u8) -> Vec<Step> {
    match phase {
        Phase::Beg => beg_steps(state, turn),
        Phase::Mode => mode_steps(state, turn),
        Phase::Placing { n } => {
            let mut out = place_steps(state, turn, n);
            // An empty `Move` is not a move: `check_move` rejects one outright,
            // and `place_rec` only ever emits from depth 1.
            if n > 0 {
                out.push(Step::StopPlacing);
            }
            out
        }
        Phase::PickWorker => {
            let mut out: Vec<Step> = state.on_board(turn).map(Step::PickWorker).collect();
            if done > 0 {
                out.push(Step::StopRetrieving);
            }
            out
        }
        Phase::Take { worker } => match state.loc(worker).on_board() {
            Some((gear, pos)) => moves::choices_for_worker(state, turn, gear, pos)
                .into_iter()
                .map(Step::Take)
                .collect(),
            None => Vec::new(),
        },
        Phase::ExtraDay { .. } => vec![Step::ExtraDay(false), Step::ExtraDay(true)],
        Phase::PityPlace => moves::pity_moves(state, turn)
            .into_iter()
            .filter_map(|m| match m.kind {
                MoveKind::Pity { spot, .. } => Some(Step::Pity(spot)),
                _ => None,
            })
            .collect(),
        Phase::DraftTile { dealt, kept } => dealt
            .iter()
            .copied()
            .filter(|&t| t != kept)
            .map(Step::DraftTile)
            .collect(),
    }
}

/// Beg options that actually lead somewhere.
///
/// `beg_options` answers "may this player beg on that temple", which is not
/// quite the edge set: `visit_normal_moves` runs the whole placement and
/// retrieval walk *per beg variant*, and a variant that yields no move
/// contributes nothing to the flat list. With 0 corn and the cheapest space at
/// index 2, `beg: None` is such a variant — begging is the only way to move at
/// all — and an unfiltered `Beg` node would offer a dead branch for it.
///
/// Filtering also earns the pity gate. Nothing survives the filter exactly when
/// no ordinary move exists at all under any beg variant, which is precisely
/// `has_normal_move`'s question and precisely the condition `visit_legal_moves`
/// uses before falling through to `pity_moves`.
fn beg_steps(state: &GameState, turn: PlayerId) -> Vec<Step> {
    let mut out: Vec<Step> = moves::beg_options(state, turn)
        .into_iter()
        .filter(|&beg| {
            let mut probe = *state;
            apply_beg(&mut probe, turn, beg);
            can_place(&probe, turn) || can_retrieve(&probe, turn)
        })
        .map(Step::Beg)
        .collect();

    if out.is_empty() {
        debug_assert!(!moves::has_normal_move(state, turn));
        // `pity_moves` never begs, so the pitied player's single edge is the
        // empty one and the temple tracks are left alone.
        out.push(Step::Beg(None));
    }
    out
}

fn mode_steps(state: &GameState, turn: PlayerId) -> Vec<Step> {
    let mut out = Vec::with_capacity(2);
    if can_place(state, turn) {
        out.push(Step::Mode(ModeChoice::Place));
    }
    if can_retrieve(state, turn) {
        out.push(Step::Mode(ModeChoice::Retrieve));
    }
    if out.is_empty() {
        // Degenerate by construction: `beg_steps` only routes here when the
        // gods are the last option left.
        debug_assert!(!moves::pity_moves(state, turn).is_empty());
        out.push(Step::Mode(ModeChoice::Pity));
    }
    out
}

/// The `Placing { n }` edges, without the stop.
///
/// Applying a placement pays for it immediately, so the player's remaining corn
/// *is* `budget - cost_so_far` and the budget check `cost + n + pos <= budget`
/// collapses to `n + pos <= corn`. Nothing in placement reads corn, so paying
/// as you go lands on the same state as `apply_move`'s single deduction.
fn place_steps(state: &GameState, turn: PlayerId, n: u8) -> Vec<Step> {
    if state.available(turn).next().is_none() {
        return Vec::new();
    }
    let corn = state.players[turn.idx()].corn;
    let mut out = Vec::with_capacity(6);
    for gear in Gear::ALL {
        if let Some(pos) = state.lowest_free(gear) {
            let spot = Placement::Gear(gear, pos);
            if n + spot.space_cost() <= corn {
                out.push(Step::Place(spot));
            }
        }
    }
    if state.first_player_space.is_none() && n <= corn {
        out.push(Step::Place(Placement::FirstPlayer));
    }
    out
}

fn can_place(state: &GameState, turn: PlayerId) -> bool {
    !place_steps(state, turn, 0).is_empty()
}

fn can_retrieve(state: &GameState, turn: PlayerId) -> bool {
    state.on_board(turn).next().is_some()
}

// ---- application -------------------------------------------------------

fn apply_beg(state: &mut GameState, turn: PlayerId, beg: Option<Temple>) {
    if let Some(t) = beg {
        state.players[turn.idx()].corn = 3;
        state.temple_step(turn, t, -1);
    }
}

/// Apply a step and say where the search goes next.
///
/// `done` is the retrieval counter described on `legal_steps`.
pub fn apply_step(
    state: &mut GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
    step: &Step,
) -> Transition {
    match phase {
        Phase::ExtraDay { claimer } => {
            let Step::ExtraDay(take) = step else {
                panic!("{step:?} is not an ExtraDay edge");
            };
            // `spend_extra_day` only turns the tile over; the days themselves
            // move in one `advance_days`, so a food day passed over resolves on
            // the round landed on rather than the one left behind.
            let days = if *take {
                state.spend_extra_day(claimer);
                2
            } else {
                1
            };
            state.advance_days(days);
            start_of_round(state)
        }
        Phase::DraftTile { dealt, kept } => {
            let Step::DraftTile(id) = step else {
                panic!("{step:?} is not a DraftTile edge");
            };
            // The same application `Game::new` makes inline. If a tile ever
            // carries a symbol that is itself a decision ("advance any one
            // track"), it needs a resolver in `data::tiles` and this becomes
            // its second caller rather than growing a rule of its own.
            for e in crate::data::tiles::TILES[*id as usize] {
                e.apply(state, turn);
            }
            if kept == DRAFT_NONE {
                Transition::Step {
                    phase: Phase::DraftTile { dealt, kept: *id },
                    turn,
                    done: 0,
                }
            } else {
                // Setup deals to the seats in order, so seat 3's second pick
                // hands over to seat 0 -- who is the starting first player.
                // Nothing builds a `DraftTile` position today: `Game::new`
                // still resolves the draft with its own rng, so this is here
                // for when it stops doing that.
                Transition::Commit {
                    phase: Phase::Beg,
                    turn: turn.next(1),
                }
            }
        }
        _ => match advance_within_turn(state, phase, turn, done, step) {
            Some((phase, done)) => Transition::Step { phase, turn, done },
            None => {
                // `Game::play` refills after every move, never mid-choice, so a
                // double-build cannot reach a card its own first half dealt.
                state.refill_buildings();
                end_of_turn(state, turn)
            }
        },
    }
}

/// Apply the move half of a step without any of the end-of-turn machinery.
///
/// Returns where the turn continues, or `None` if the step committed. This is
/// the primitive the §2.8 walk is built on, and it is public so that tests can
/// traverse the turn's node graph — including the nodes a commit would hide.
pub fn step_within_turn(
    state: &mut GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
    step: &Step,
) -> Option<(Phase, u8)> {
    advance_within_turn(state, phase, turn, done, step)
}

/// Apply the *move* half of a step: exactly the mutation `apply_move` makes.
///
/// Returns where the turn continues, or `None` on a commit edge — at which
/// point `state` sits where `apply_move` would have left it, before the refill
/// and the handoff. That instant is what §2.8 pins the tree against, which is
/// why this is a separate function from `apply_step`.
fn advance_within_turn(
    state: &mut GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
    step: &Step,
) -> Option<(Phase, u8)> {
    match (phase, step) {
        (Phase::Beg, Step::Beg(beg)) => {
            apply_beg(state, turn, *beg);
            Some((Phase::Mode, 0))
        }
        (Phase::Mode, Step::Mode(mode)) => {
            let next = match mode {
                ModeChoice::Place => Phase::Placing { n: 0 },
                ModeChoice::Retrieve => Phase::PickWorker,
                ModeChoice::Pity => Phase::PityPlace,
            };
            Some((next, 0))
        }
        (Phase::Placing { n }, Step::Place(spot)) => {
            let w = state
                .available(turn)
                .next()
                .expect("Placing offered an edge with no worker to place");
            let cost = n + spot.space_cost();
            let corn = &mut state.players[turn.idx()].corn;
            debug_assert!(*corn >= cost, "placement not affordable");
            *corn = corn.saturating_sub(cost);
            put_worker(state, w, *spot);
            Some((Phase::Placing { n: n + 1 }, 0))
        }
        (Phase::Placing { .. }, Step::StopPlacing) => None,
        (Phase::PickWorker, Step::PickWorker(w)) => Some((Phase::Take { worker: *w }, done)),
        (Phase::PickWorker, Step::StopRetrieving) => None,
        (Phase::Take { worker }, Step::Take(choice)) => {
            choice.apply(state, turn);
            state.retrieve_worker(worker);
            Some((Phase::PickWorker, done + 1))
        }
        (Phase::PityPlace, Step::Pity(spot)) => {
            let w = state
                .available(turn)
                .next()
                .expect("PityPlace offered an edge with no worker to place");
            state.players[turn.idx()].corn = 0;
            put_worker(state, w, *spot);
            None
        }
        _ => panic!("{step:?} is not an edge of {phase:?}"),
    }
}

fn put_worker(state: &mut GameState, w: WorkerId, spot: Placement) {
    match spot {
        Placement::Gear(gear, pos) => state.place_worker(w, gear, pos),
        Placement::FirstPlayer => state.place_on_first_player(w),
    }
}

/// Hand off to the next seat, and run the round end when the seat comes back
/// round to the first player.
///
/// The `ExtraDay` decision sits before the day advance, matching both §2.7 and
/// `Game::end_round`: the privilege may not push a worker off a gear, and that
/// has to be judged against the board as it stands rather than after a rotation
/// has already carried the workers off.
fn end_of_turn(state: &mut GameState, turn: PlayerId) -> Transition {
    let next = turn.next(1);
    state.current = next;
    if next != state.first_player {
        return Transition::Commit {
            phase: Phase::Beg,
            turn: next,
        };
    }

    if let Some(claimer) = state.resolve_first_player() {
        if state.may_take_extra_day(claimer) {
            // The mover at that node is the claimer, not the player whose turn
            // just ended, so `turn` names them too and `(state, phase, turn)`
            // stays a canonical key.
            return Transition::Commit {
                phase: Phase::ExtraDay { claimer },
                turn: claimer,
            };
        }
    }
    state.advance_day();
    start_of_round(state)
}

fn start_of_round(state: &mut GameState) -> Transition {
    if state.over {
        return Transition::Over;
    }
    state.current = state.first_player;
    Transition::Commit {
        phase: Phase::Beg,
        turn: state.first_player,
    }
}

// ---- reconstruction ----------------------------------------------------

/// Reassemble the `Move` a path of steps describes, so the engine's own
/// `check_move` can be run against it.
///
/// The search never needs this to play — the child node already holds the
/// resulting state — but §2.8 asks for it under `debug_assertions`, and the UI
/// and the move log speak `Move`.
pub fn move_from_path(steps: &[Step]) -> Option<moves::Move> {
    let mut beg = None;
    let mut placements = moves::Placements::new();
    let mut retrievals = moves::Retrievals::new();
    let mut pity: Option<(WorkerId, Placement)> = None;
    let mut corn_cost = 0u8;
    let mut picked: Option<WorkerId> = None;

    for step in steps {
        match step {
            Step::Beg(t) => beg = *t,
            Step::Mode(_) => {}
            Step::Place(spot) => {
                corn_cost += placements.len() as u8 + spot.space_cost();
                // Worker identity is filled in by `retag`, which needs the
                // state; a bare path does not carry it.
                placements.push((WorkerId(u8::MAX), *spot));
            }
            Step::PickWorker(w) => picked = Some(*w),
            Step::Take(c) => {
                let w = picked.take()?;
                retrievals.push((w, c.clone()));
            }
            Step::Pity(spot) => pity = Some((WorkerId(u8::MAX), *spot)),
            Step::StopPlacing | Step::StopRetrieving => break,
            Step::ExtraDay(_) | Step::DraftTile(_) => return None,
        }
    }

    let kind = if let Some((worker, spot)) = pity {
        MoveKind::Pity { worker, spot }
    } else if !retrievals.is_empty() {
        MoveKind::Retrieve(retrievals)
    } else if !placements.is_empty() {
        MoveKind::Place(placements)
    } else {
        return None;
    };
    Some(moves::Move {
        kind,
        beg,
        corn_cost,
    })
}

/// Replay a path of steps, stopping at the commit edge.
///
/// Leaves `state` exactly where `apply_move` would — before the end-of-turn
/// refill and the handoff — which is what makes it comparable to the flat
/// generator. Returns false if the path ran out before committing.
pub fn apply_path(state: &mut GameState, turn: PlayerId, steps: &[Step]) -> bool {
    let mut at = (Phase::Beg, 0u8);
    for step in steps {
        let (phase, done) = at;
        match advance_within_turn(state, phase, turn, done, step) {
            Some(next) => at = next,
            None => return true,
        }
    }
    false
}

/// Fill in the worker identities a bare path cannot know, by replaying the
/// placement against `state`.
pub fn retag_workers(state: &GameState, turn: PlayerId, m: &mut moves::Move) {
    let mut probe = *state;
    // `pity_moves` records the corn given up, which a bare path cannot know.
    if matches!(m.kind, MoveKind::Pity { .. }) {
        m.corn_cost = state.players[turn.idx()].corn;
    }
    match &mut m.kind {
        MoveKind::Place(v) => {
            for (w, spot) in v.iter_mut() {
                let Some(next) = probe.available(turn).next() else {
                    return;
                };
                *w = next;
                put_worker(&mut probe, next, *spot);
            }
        }
        MoveKind::Pity { worker, .. } => {
            if let Some(next) = probe.available(turn).next() {
                *worker = next;
            }
        }
        MoveKind::Retrieve(_) => {}
    }
}

// ---- the §2.8 reference walk -------------------------------------------

/// Every distinct `GameState` a turn can reach, taken at the instant
/// `apply_move` would have left it.
///
/// The left-hand side of §2.8's equivalence. It is `pub` because the test that
/// makes this whole module trustworthy lives outside the crate.
///
/// The `(state, phase)` memo is the same collapse `retrieve_rec` performs with
/// its `reached` set: two commuting retrieval orders arrive at one state whose
/// subtree has already been walked. In the search proper this memo is the
/// transposition table, and the orderings share statistics instead of being
/// discarded.
pub fn reachable_after_turn(state: &GameState, turn: PlayerId) -> HashSet<GameState> {
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    walk(state, Phase::Beg, turn, 0, &mut seen, &mut out);
    out
}

/// The memo key. `done` is redundant with the state within a turn — reaching a
/// state after k retrievals means exactly k of the player's workers have left
/// the gears — but keying on it costs a byte and removes the need to trust that
/// argument.
type Seen = HashSet<(GameState, Phase, u8)>;

fn walk(
    state: &GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
    seen: &mut Seen,
    out: &mut HashSet<GameState>,
) {
    if !seen.insert((*state, phase, done)) {
        return;
    }
    for step in legal_steps(state, phase, turn, done) {
        let mut next = *state;
        match advance_within_turn(&mut next, phase, turn, done, &step) {
            Some((phase, done)) => walk(&next, phase, turn, done, seen, out),
            None => {
                out.insert(next);
            }
        }
    }
}

/// Every path of sub-decisions a turn can take, as the `Step` sequences that
/// make them up. Unmemoised, so it is exponential in the retrieval count — for
/// tests on small positions only.
pub fn enumerate_paths(state: &GameState, turn: PlayerId, limit: usize) -> Vec<Vec<Step>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    paths_rec(state, Phase::Beg, turn, 0, &mut cur, &mut out, limit);
    out
}

fn paths_rec(
    state: &GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
    cur: &mut Vec<Step>,
    out: &mut Vec<Vec<Step>>,
    limit: usize,
) {
    if out.len() >= limit {
        return;
    }
    for step in legal_steps(state, phase, turn, done) {
        let mut next = *state;
        let cont = advance_within_turn(&mut next, phase, turn, done, &step);
        cur.push(step);
        match cont {
            Some((phase, done)) => paths_rec(&next, phase, turn, done, cur, out, limit),
            None => out.push(cur.clone()),
        }
        cur.pop();
        if out.len() >= limit {
            return;
        }
    }
}

/// The `Take` node's candidate list, exposed so callers can size it without
/// building `Step`s. Used by the width diagnostics.
pub fn take_candidates(state: &GameState, turn: PlayerId, worker: WorkerId) -> Vec<Choice> {
    match state.loc(worker).on_board() {
        Some((gear, pos)) => moves::choices_for_worker(state, turn, gear, pos),
        None => Vec::new(),
    }
}
