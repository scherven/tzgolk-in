//! A hand-written position evaluator, and exhaustive move ranking on top of it.
//!
//! [`heuristic`] estimates a player's **final score**. The contract is the one
//! the value head will inherit — `(&GameState, PlayerId) -> f32`, in points —
//! so swapping a trained net in stays local.
//!
//! The shape it is built to is worth stating, because it is what the first
//! version got wrong. A Tzolk'in position is not a pile of resources; it is an
//! *engine* plus a set of *scheduled payouts*. So the estimate is
//!
//! ```text
//!   points already banked
//! + what liquidation pays today   (exact: the same arithmetic as `end_game`)
//! + the temple payouts still to come, projected from where everyone stands
//! + the engine, valued at what it can still convert before the calendar ends
//! - the food the engine will fail to pay for
//! ```
//!
//! Everything below the first line is an estimate of a game that has not
//! finished, so once `over` is set they all go away and `heuristic` is simply
//! `state.scores()[p]` — `end_game` has by then folded the liquidation into
//! `points` itself. A leaf evaluation and a real result are therefore the same
//! number, which is what lets the search compare a forced win against a guess.
//!
//! [`margin`] is the zero-sum reduction the search runs on: how far ahead of the
//! best opponent a player stands. It is what makes denying an opponent — a
//! contested temple top, the last monument, the skull bank — score as a gain
//! rather than as nothing at all.

use crate::data::monuments::def as mdef;
use crate::data::temples::TEMPLES;
use crate::ids::*;
use crate::moves::Move;
use crate::state::{GameState, LAST_DAY, POINT_DAYS, RESOURCE_DAYS};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

// ---- tuning ------------------------------------------------------------

/// The base game's liquidation rate: four corn to a point.
const CORN_PER_POINT: f32 = 4.0;

/// What a *spendable* corn is worth beyond its liquidation value. Corn buys
/// placement depth, feeds workers and mirrors actions, none of which the 4:1
/// conversion sees. Applies to the first [`LIQUID_CORN`] only — a hoard past
/// what a player can plausibly spend really is worth just 1/4 a point each.
const CORN_PREMIUM: f32 = 0.10;
const LIQUID_CORN: f32 = 16.0;

/// Per-block premium over liquidation value, for the same reason: blocks build,
/// and a building is worth more than the three corn its stone converts to.
const BLOCK_PREMIUM: f32 = 0.55;

/// A skull liquidates at 3 (`GameState::end_game`) but *spends* at Chichen for
/// 4 to 13 points plus a temple step. This is the premium for holding one while
/// a Chichen space and the time to reach it both remain.
const SKULL_PREMIUM: f32 = 2.2;

/// Points lost per worker that goes unfed on a food day.
const STARVE_POINTS: f32 = 3.0;

/// Corn a player is assumed to bring in per round before the next food day.
///
/// **Zero, and that is the point.** At 1.9 this forgave any shortfall more than
/// three rounds out — which is every shortfall at the start of a round block —
/// so `starvation_risk` only ever fired on a player already almost out of time.
/// Taking the assumption away is the largest single effect measured on this
/// file since the term re-pricing: **+3.96 greedy:64 [+3.65, +4.28] and +2.11
/// mcts:256 [+1.79, +2.44] at 800 blocks**, and +4.24 at `greedy:full`.
///
/// The mechanism is a double count rather than a calibration error. The corn
/// this projected is exactly the corn the *search* is separately planning to
/// gather, so crediting it here too pays for it twice; the term's job is to
/// score the position as it stands. 0.0 and 0.2 are indistinguishable under
/// MCTS (−0.11 [−0.35, +0.13] paired at 800 blocks) and greedy separates them
/// in favour of 0.0 (+2.15 against +1.55 at 0.4, monotone). The constant is
/// kept rather than deleted because it is the knob a future sweep needs.
/// `docs/FINDINGS-eval.md` F23, F23a.
const CORN_INCOME_PER_ROUND: f32 = 0.0;

/// Value of one worker-action, in points, *averaged over the actions a player
/// actually gets to take*. Well under the 2-3 the mid-range entries of the
/// space table pay, because most of a worker's actions are the cheap ones it
/// takes on the way to the good one, and because a worker that has to be fed
/// four times over the calendar has already spent 2 points of corn on itself.
///
/// It is this low because [`BOARD_SCALE`]'s term has already paid for the
/// action every *placed* worker is about to take; what is left here is the
/// throughput of workers that are not on a gear yet. Measured, not reasoned:
/// swept jointly with `BOARD_SCALE` (`evalab --ab 'board=..,av=..'`, 150 blocks
/// a cell) the score is flat over 0.2..0.4 and falls away by 0.6, and the old
/// 1.2 is off the end of the ridge — along the committed `TEMPO_PER_ROUND` the
/// score is monotone decreasing in this constant all the way down to 0.0.
///
/// 0.2 rather than 0.4 on a **600-block** paired re-run, +0.71 [+0.51, +0.91]
/// greedy:64 and +0.38 [+0.09, +0.68] mcts:256 — the one comparison of the two
/// that was run end to end on a single build. Anything inside 0.2..0.4 is
/// within noise of it once this file's own change is compiled into the MCTS
/// prior (`eval::heuristic` orders MCTS's edges, so landing here moves the
/// platform every later sweep is measured on — `docs/FINDINGS-eval.md` F14).
/// Treat the digit as "the low end of a plateau", not as resolved to 0.1.
/// `docs/FINDINGS-eval.md` F7, F10, F14.
const ACTION_VALUE: f32 = 0.2;

/// `board_position` sums, over every placed worker, the best space that worker
/// can still ride to. That sum is optimistic twice over, and the two compound:
/// the action it prices is *also* paid for by [`ACTION_VALUE`]'s per-worker
/// throughput in `engine_value`, and each worker is credited with the maximum
/// over its whole gear as if no other worker of its own and no opponent were
/// competing for those spaces.
///
/// Halving the term was the largest single effect ever measured on this file:
/// **+10.48 greedy:64 [+9.07, +11.89] and +9.04 mcts:256 [+7.86, +10.23]** at
/// 150 blocks, on a plateau over 0.35..0.50 with 0.25 and 0.70 clearly worse.
///
/// **That plateau moved when [`GEAR_SCALE`] landed**, and the direction is up
/// rather than down: on the new table 0.40 is −0.99 [−1.33, −0.65], 0.55 is
/// +0.56 [+0.29, +0.82], 0.65 is +1.36 [+0.88, +1.85] and 0.70 is +1.41
/// [+1.02, +1.83], all `mcts:256` against the 0.5 it replaces. This is not
/// level compensation — `GEAR_SCALE`'s n-weighted mean is 0.98, and a 2% cut in
/// the table is not answered by a 30% rise in its scale. What changed is the
/// *shape*: Chichen's entries came down 30% and Palenque's went up 50%, and a
/// term whose value is a max over a gear responds to the mix, not the mean.
/// **And it depends on how deep the search is.** On the two shallow agents 0.65
/// is the optimum and 0.80 is free: `greedy:64` reads −0.03 [−0.44, +0.38] and
/// `mcts:256` −0.03 [−0.39, +0.33] at 400 blocks each, with greedy falling to
/// −0.96 * by 1.0. On `mcts:1024:cp=0.05` — the agent that descends about six
/// turns rather than one — 0.80 is **+1.42 [+0.92, +1.93] * at 227 blocks**,
/// win rate 0.317, and 0.90 is +2.45 [+1.74, +3.17] * at 122.
///
/// So this is not one optimum measured badly, it is a different optimum per
/// depth, monotone in depth. `board_position` prices what is on the board
/// *now*; a search that plays the ride out verifies it instead of taking it on
/// trust, and can afford to hear it louder. 0.80 is taken because it costs
/// nothing distinguishable on any agent and gains a point and a half on the one
/// closest to the deliverable's `mcts:8192:heuristic:cp=0.02`; 0.90 looks better
/// still on that agent and is left to whoever has its 250-block number.
/// `docs/FINDINGS-eval.md` F5a, F7, F29c, F34, F35.
///
/// **0.90 now has that number, and it is 600 blocks rather than 250.** On
/// `evalab-p25` (rev `48c39bd`), against this file as it shipped at 0.80:
/// `greedy:64` −0.21 [−0.46, +0.04] at 800 blocks, `mcts:256` −0.09
/// [−0.43, +0.26] at 400, and **`mcts:1024:cp=0.05` +0.48 [+0.19, +0.76] \* at
/// 600**, win rate 0.273. Free on both shallow agents, distinguishable on the
/// one that searches. The deep curve peaks near 1.0-1.1 (+0.93 \*, +1.01 \*)
/// and turns over by 1.2, but 1.0 costs `greedy:64` −1.00 \* and 1.1 costs
/// −1.53 \*, so 0.90 is the last value that is free everywhere.
/// `docs/FINDINGS-eval.md` F41, F45b, F47.
///
/// Do **not** also subtract the placed worker from `engine_value`'s action
/// count. That corrects the same double count a second time and measures
/// −12.09 / −9.95 with a 0.028 win rate, the worst configuration in the log
/// after ungating the space table entirely (F5b).
///
/// The constants this one is supposed to be double-counting against were all
/// re-swept on top of `board = 0.8` and not one of them moved: `ACTION_VALUE`
/// is +0.01 ± 0.04 at zero on 800 blocks, `action_cap` is −0.00 ± 0.03,
/// `GEAR_SCALE[Chichen]` and `[Palenque]` are still unimodal at 0.7 and 1.5.
/// **`BOARD_SCALE` is not coupled to any of them** (F43).
const BOARD_SCALE: f32 = 0.9;

/// [`temple_outlook`] is *under*-priced, which is the one thing nobody
/// expected: scaling it alone is monotone improving from 0.5 to 1.4 on both
/// agents, and shrinking it to 0.5 costs 14 points and drops the MCTS win rate
/// to 0.065. On top of a corrected `BOARD_SCALE`/`ACTION_VALUE` the joint peak
/// is here, worth **+1.41 mcts:256 [+0.40, +2.41]** paired at 150 blocks; 1.6
/// and 1.8 fall away again.
///
/// This contradicts the plan workstream's refit, which fitted `temple_outlook`
/// at −3.86 and read it as harmful. That fit was taken against an evaluator
/// whose `board` and `engine` were 30 of a 43-point estimate; the three terms
/// are strongly collinear, so least squares drove the temple coefficient
/// negative to cancel the other two rather than because temples are worth
/// less. Correct board and engine first and the sign flips. `docs/FINDINGS-eval.md` F8.
///
/// **All of that was measured on agents that search one turn, and an agent
/// that searches six wants the opposite.** `temple_outlook` prices where
/// everyone's standings will be on a scoring day up to thirteen days away; a
/// one-ply agent has to take that on trust, and a search that plays the ride
/// out watches the standings actually move and does not need to be told twice.
/// Against this file as it shipped at 1.4, on `evalab-p25`:
///
/// | `temple` | `greedy:64` (800) | `mcts:256` (400) | `mcts:1024:cp=0.05` |
/// | --- | --- | --- | --- |
/// | 0.7 | **−4.02** * | — | +1.36 * (190) |
/// | 0.9 | −1.25 * | — | +1.74 * (250) |
/// | 1.0 | −0.59 * | +0.16 | +1.33 * (250) |
/// | 1.1 | −0.43 * | +0.27 | +1.40 * (400) |
/// | **1.2** | **−0.13 [−0.35, +0.09]** | **+0.19 [−0.11, +0.49]** | **+1.35 [+1.00, +1.71]** * (400) |
///
/// The deep tier is a **plateau** from 0.7 to 1.2 at about +1.4 and the shallow
/// one is a cliff, so 1.2 is taken as the point on that plateau which costs
/// `greedy:64` and `mcts:256` nothing distinguishable. This is not the temple
/// term being worth less than F8 measured — F8 is right about the two agents it
/// measured — it is the largest term in the evaluator being a *forecast*, and
/// the deliverable being `mcts:8192:heuristic:cp=0.02,pmin=2`.
///
/// The control that makes this shape rather than magnitude: moving both
/// constants the *other* way (`board = 0.7, temple = 1.7`) costs `greedy:64`
/// −0.25 [−0.51, +0.01] — nothing — and the deep search **−2.23 [−2.61, −1.85]
/// \***, win rate 0.185. Same magnitude change, opposite sign of result.
/// `docs/FINDINGS-eval.md` F41, F45, F46a, F47.
const TEMPLE_SCALE: f32 = 1.2;

/// Rounds a worker typically spends riding a gear between actions.
const ROUNDS_PER_ACTION: f32 = 2.6;

/// Residual value of a building already built. Its payoff is in the state
/// already; what is left is that three monuments count the pile.
const BUILDING_VALUE: f32 = 0.45;

/// Global scale on [`research_step_value`], which is priced per *use*.
///
/// The other half of the over-priced `engine_value` that [`ACTION_VALUE`] found
/// in its worker half: at 0.5 a maxed track was several points of pure
/// forecast. Located by `evalab --promise`, which measures what the estimate
/// actually *moves by* when a standing worker's action is taken — the first
/// research space is the widest disagreement on the board, priced at 2.00 by
/// `space_value` and delivering 0.60. Sweeping the side that is not the space
/// table settles which of the two is wrong: 0.0, 0.10 and 0.15 are a plateau at
/// +1.6 greedy:64, 0.35 is +0.96 and 0.75 is −3.40, so it is this one.
///
/// Worth **+1.11 mcts:256 [+0.75, +1.47]** on its own at 400 blocks, and it
/// adds to the [`CORN_INCOME_PER_ROUND`] change rather than overlapping it
/// (+1.73 and +1.11 alone, +2.25 together). Kept just off zero so the search
/// still has a reason to finish a track; the data cannot tell 0.0 from 0.15.
/// `docs/FINDINGS-eval.md` F19b, F20, F23.
const RESEARCH_SCALE: f32 = 0.05;

/// Fraction of a face-up monument's score credited to the player closest to
/// affording it. See [`monument_outlook`].
const MONUMENT_SHARE: f32 = 0.45;

/// Confidence in a projected temple payout: the next scoring day is nearly
/// locked in, a later one is a forecast that standings will move on.
const TEMPLE_NEAR: f32 = 0.85;
const TEMPLE_FAR: f32 = 0.55;

/// What one more step on a temple is worth between scoring days, as a fraction
/// of the jump it unlocks. Without it the search sees climbing as free.
///
/// The `climb` loop stops at the ceiling `GameState::temple_ceiling` imposes
/// rather than at the top of the track: the exclusive top step of a temple an
/// opponent already stands on cannot be climbed to, and crediting it paid for a
/// move `temple_step` clamps away. Small but measurable despite being ~0.1 of
/// one step's jump — **+0.08 greedy:64 [+0.01, +0.14] and +0.19 mcts:256
/// [+0.02, +0.35]** on its own — and it does not separate from zero once the
/// two constants above have landed (+0.05 [−0.02, +0.12] paired at 800 blocks).
/// It is kept because the rule says so, not because it pays.
///
/// The scale itself is at an optimum and flat around it: 0.0 and 0.25 both read
/// zero, 0.50 is −0.32. `docs/FINDINGS-eval.md` F18, F21a.
const TEMPLE_CLIMB: f32 = 0.10;

/// Per-block bonus for holding a *spread* of block types rather than a stack of
/// one, on top of [`BLOCK_PREMIUM`]. Mixed build costs are limited by the
/// scarcest type.
const BLOCK_BREADTH: f32 = 0.50;

/// A held Palenque tile, which monuments #7 and #8 pay 4 each for.
const TILE_PREMIUM: f32 = 0.25;

/// What one round of one worker's throughput costs, in points: the price of
/// leaving a worker standing on a gear for one more rotation instead of taking
/// the action under it.
///
/// This is what stops a worker being valued at whatever the top of its gear
/// pays. The old code discounted a later space by a flat 0.85 *regardless of
/// distance*, so a worker on Chichen space 1 was priced at the 13-point space
/// nine rotations away, and every worker on a gear scored the same. Charging
/// per space ridden is worth **+2.02 centred points** (95% CI +1.21..+2.83,
/// 500 rotation blocks / 2000 games, `heuristic:32` both sides).
///
/// Swept at 3000 rotation blocks (12000 games) per value, `heuristic:32`
/// against the frozen cbb61a2 evaluator, paired on seed. Centred score against
/// that baseline, and the paired difference from 0.52:
///
/// ```text
///   0.34  +5.42   -3.41 (-3.76..-3.06)
///   0.42  +7.07   -1.76 (-2.04..-1.47)
///   0.52  +8.83    --                   <- peak
///   0.62  +7.54   -1.29 (-1.63..-0.95)
///   0.74  +5.20   -3.63 (-4.05..-3.22)
/// ```
///
/// Unimodal, and every alternative loses by more than its interval. A quadratic
/// through the middle three puts the vertex at 0.512, so 0.52 is the value.
/// (An earlier 300-block sweep read 0.45 as the peak; at 600 blocks the same
/// seeds reversed it. 300 blocks is not enough to separate 0.45 from 0.52.)
const TEMPO_PER_ROUND: f32 = 0.52;

// ---- the evaluator -----------------------------------------------------

/// The estimate, term by term. Every field is in points and they sum to
/// [`heuristic`].
///
/// This exists so error can be *attributed*. A single number cannot say which
/// half of the estimate is wrong, and the calibration study in
/// `tests/rules.rs::evaluator_calibration` reads these to decide what to fix:
/// it regresses the realised final score on each term and reports which one
/// moves when the estimate misses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Components {
    /// Points already on the scorepad. Exact.
    pub banked: f32,
    /// What `end_game` would pay for held corn, skulls and monuments. Exact.
    pub liquidation: f32,
    pub held: f32,
    pub temple: f32,
    pub engine: f32,
    pub board: f32,
    pub monument: f32,
    /// Already signed as a *penalty*: the total subtracts it.
    pub starvation: f32,
}

impl Components {
    /// Field names in the order [`Components::terms`] returns them.
    pub const NAMES: [&'static str; 8] = [
        "banked",
        "liquidation",
        "held",
        "temple",
        "engine",
        "board",
        "monument",
        "starvation",
    ];

    /// The terms as a slice, signed the way they enter the sum.
    pub fn terms(&self) -> [f32; 8] {
        [
            self.banked,
            self.liquidation,
            self.held,
            self.temple,
            self.engine,
            self.board,
            self.monument,
            -self.starvation,
        ]
    }

    #[inline]
    pub fn total(&self) -> f32 {
        self.terms().iter().sum()
    }
}

/// Estimated final score for `p`, in points.
///
/// Exactly `g.scores()[p]` once `g.over`, so a finished line and a guess about
/// one are the same kind of number and the search can compare them.
#[inline]
pub fn heuristic(g: &GameState, p: PlayerId) -> f32 {
    components(g, p).total()
}

/// [`heuristic`], with the sum left unfolded. See [`Components`].
#[inline]
pub fn components(g: &GameState, p: PlayerId) -> Components {
    let pl = &g.players[p.idx()];
    let mut c = Components {
        banked: pl.points as f32,
        ..Components::default()
    };

    // A finished game needs no estimate. `end_game` has already folded corn,
    // skulls and monuments into `points`, and the resources it converted are
    // still sitting in the player — so adding `liquidation` here would pay for
    // them a second time.
    if g.over {
        return c;
    }

    // What liquidation would pay if the game ended now, by the same arithmetic
    // `end_game` uses. Exact, unlike everything below it.
    c.liquidation = liquidation(g, p);

    let rounds_left = (LAST_DAY.saturating_sub(g.day)) as f32;
    // 1.0 at the start of the game, 0.0 on the last day. Everything speculative
    // is scaled by some function of this.
    let horizon = rounds_left / LAST_DAY as f32;

    c.held = held_premium(g, p, rounds_left);
    // The two scales are applied here rather than inside the functions so that
    // "this term is mis-priced by k" stays one number, in the same place
    // `plan::Schedule` applies its weights and `bin/evalab` measured them.
    c.temple = temple_outlook(g, p) * TEMPLE_SCALE;
    c.engine = engine_value(g, p, rounds_left, horizon);
    c.board = board_position(g, p, rounds_left) * BOARD_SCALE;
    c.monument = monument_outlook(g, p, horizon);
    c.starvation = starvation_risk(g, p);

    c
}

/// How far ahead of the best opponent `p` stands.
///
/// This is the quantity the search maximises. Using it rather than the raw
/// estimate is what makes a move that costs an opponent more than it costs you
/// — taking the monument they were saving for, standing on a temple top,
/// emptying the skull bank — read as progress.
pub fn margin(g: &GameState, p: PlayerId) -> f32 {
    let mine = heuristic(g, p);
    let best_other = PlayerId::ALL
        .iter()
        .filter(|&&q| q != p)
        .map(|&q| heuristic(g, q))
        .fold(f32::NEG_INFINITY, f32::max);
    mine - best_other
}

// ---- components --------------------------------------------------------

/// Exactly what `GameState::end_game` would add to this player's points if the
/// game ended right now. Only meaningful before it has actually run.
fn liquidation(g: &GameState, p: PlayerId) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = (pl.total_corn() / 4) as f32;
    v += pl.get(Resource::Skull) as f32 * 3.0;
    for id in pl.monument_ids() {
        v += (mdef(id).score)(g, p) as f32;
    }
    v
}

/// What holdings are worth *over* their liquidation value, because they can
/// still be spent. Goes to zero as the calendar runs out.
fn held_premium(g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
    let pl = &g.players[p.idx()];
    // Spending takes a turn, so the premium is gone well before the last day.
    let spendable = (rounds_left / 6.0).min(1.0);
    if spendable <= 0.0 {
        return 0.0;
    }

    let mut v = (pl.corn as f32).min(LIQUID_CORN) * CORN_PREMIUM * spendable;

    // Blocks build, and buildings are the densest points in the game. A spread
    // of block types is worth more than the same count of one type, because
    // building costs are mixed — three of each beats nine wood.
    let counts = [
        pl.get(Resource::Wood) as f32,
        pl.get(Resource::Stone) as f32,
        pl.get(Resource::Gold) as f32,
    ];
    let total: f32 = counts.iter().sum();
    v += total.min(9.0) * BLOCK_PREMIUM * spendable;
    // Reward breadth: the minimum across types is what a mixed cost is limited
    // by. Worth about one extra block each.
    let breadth = counts.iter().copied().fold(f32::INFINITY, f32::min);
    v += breadth.min(3.0) * BLOCK_BREADTH * spendable;

    // Skulls are Chichen fuel. Only while there is a space left to spend them on
    // and a turn in which to do it.
    let chichen_left = 9u32.saturating_sub(g.chichen_filled.count_ones()) as f32;
    if chichen_left > 0.0 {
        let usable = (pl.get(Resource::Skull) as f32).min(chichen_left);
        v += usable * SKULL_PREMIUM * spendable;
    }

    // Palenque tiles feed monuments #7 and #8 at 4 points each. Even without
    // one in hand they are a live option while monuments remain.
    if g.face_up_monuments().next().is_some() {
        v += (pl.corn_tiles + pl.wood_tiles) as f32 * TILE_PREMIUM * spendable;
    }

    v
}

/// The temple payouts still on the calendar, projected from where everyone
/// stands right now.
///
/// Both kinds of scoring day are modelled: the point days pay
/// `temple_points` (which already handles the majority prize and its tie
/// split), the resource days hand out blocks and skulls. Each is discounted for
/// how far away it is, because standings move.
fn temple_outlook(g: &GameState, p: PlayerId) -> f32 {
    let mut v = 0.0;

    for (n, &d) in POINT_DAYS.iter().filter(|&&d| d > g.day).enumerate() {
        // `advance_days` pays out at the age current when the day resolves, and
        // bumps the age afterwards.
        let age = 1 + POINT_DAYS.iter().filter(|&&x| x < d).count() as u8;
        let pts = g.temple_points(p, age) as f32;
        // The nearest payout is close to locked in; a later one is a forecast.
        v += pts * if n == 0 { TEMPLE_NEAR } else { TEMPLE_FAR };
    }

    for (n, &_d) in RESOURCE_DAYS.iter().filter(|&&d| d > g.day).enumerate() {
        let mut haul = 0.0;
        for t in Temple::ALL {
            let step = g.temple_pos(p, t);
            for &(at, r) in TEMPLES[t.idx()].resources {
                if step >= at {
                    haul += match r {
                        // A skull is worth its liquidation plus its Chichen use.
                        Resource::Skull => 3.0 + SKULL_PREMIUM,
                        other => other.corn_value() as f32 / CORN_PER_POINT + BLOCK_PREMIUM,
                    };
                }
            }
        }
        v += haul * if n == 0 { TEMPLE_NEAR } else { TEMPLE_FAR };
    }

    // A step is worth more than its current payout when it is one short of a
    // jump, and the tracks are steep at the top. Value the *next* step so the
    // search sees climbing as progress even between scoring days.
    let mut climb = 0.0;
    for t in Temple::ALL {
        let d = &TEMPLES[t.idx()];
        let step = g.temple_pos(p, t) as usize;
        let top = (d.steps - 1) as usize;
        // `GameState::temple_ceiling` is private; this is its rule. A track
        // whose exclusive top an opponent holds stops one short, and a step
        // that would go nowhere is not progress worth paying for.
        let ceiling = if PlayerId::ALL
            .iter()
            .any(|&q| q != p && g.temple_pos(q, t) as usize == top)
        {
            top - 1
        } else {
            top
        };
        if step + 1 <= ceiling {
            climb += (d.points[step + 1] - d.points[step]) as f32;
        }
    }
    v += climb * TEMPLE_CLIMB;

    v
}

/// Workers, research and the permanent food discounts: everything that converts
/// remaining rounds into points.
fn engine_value(g: &GameState, p: PlayerId, rounds_left: f32, horizon: f32) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = 0.0;

    // Each worker takes roughly one action every `ROUNDS_PER_ACTION` rounds,
    // and each action is worth about `ACTION_VALUE`. That is the whole reason
    // to buy a fourth, fifth and sixth worker — and the reason they stop being
    // worth buying near the end.
    let actions_each = (rounds_left / ROUNDS_PER_ACTION).min(6.0);
    let workers = g.n_unlocked(p) as f32;
    v += workers * actions_each * ACTION_VALUE;

    // Feeding is a cost per food day, not per round; the permanent discounts
    // are worth the corn they save over every food day still to come.
    let food_days_left = RESOURCE_DAYS
        .iter()
        .chain(POINT_DAYS.iter())
        .filter(|&&d| d > g.day)
        .count() as f32;
    let saved = pl.free_workers as f32 * 2.0
        + (pl.worker_discount as f32).min(2.0) * workers;
    v += saved * food_days_left / CORN_PER_POINT;

    // Research. Each level is valued at what it actually pays, scaled by how
    // many actions remain to use it on.
    let uses = (rounds_left / 3.0).min(7.0);
    for s in Science::ALL {
        let lvl = g.level(p, s);
        for l in 1..=lvl {
            v += research_step_value(s, l) * uses * RESEARCH_SCALE;
        }
        // Monument #11 pays 9/20/33 for maxed tracks and #12 pays 3 a level, so
        // a track one short of the top is worth finishing.
        if lvl == 2 && horizon > 0.15 {
            v += 0.4;
        }
    }

    // Buildings are mostly one-shot and their payoff is already in the state.
    // What survives is that three monuments count them, so the pile itself has
    // a residual value.
    v += pl.n_buildings() as f32 * BUILDING_VALUE;

    v
}

/// Per-use value of one research level, in points.
///
/// These are the real effects from `research.rs`, priced: an extra corn is
/// 1/4 a point plus its spending premium, an extra block rather more.
fn research_step_value(s: Science, level: u8) -> f32 {
    match (s, level) {
        // +1 corn on green Palenque spaces.
        (Science::Agriculture, 1) => 0.35,
        // +1 on blue, and irrigation: corn from an exhausted jungle space.
        (Science::Agriculture, 2) => 0.45,
        // +3 corn on green — the biggest single corn step in the game.
        (Science::Agriculture, 3) => 0.75,
        // One extra block per gather, by type. Gold is worth the most and is
        // gated behind the two below it.
        (Science::Extraction, 1) => 0.55,
        (Science::Extraction, 2) => 0.65,
        (Science::Extraction, 3) => 0.75,
        // +1 corn, then +2 points, on every build. Then a block off the price.
        (Science::Architecture, 1) => 0.30,
        (Science::Architecture, 2) => 0.85,
        (Science::Architecture, 3) => 0.80,
        // Foresight: use the next Chichen space when yours is taken.
        (Science::Theology, 1) => 0.45,
        // Devout: a block buys one more temple step alongside a skull.
        (Science::Theology, 2) => 0.65,
        // A second skull from Yaxchilan 4.
        (Science::Theology, 3) => 0.90,
        _ => 0.0,
    }
}

/// What this player's workers are standing on, **over and above** the generic
/// action `engine_value` has already paid them for.
///
/// A worker's value is the action it will eventually take, not its distance
/// along a gear. A worker one space from a double build is worth far more than
/// one three spaces along Palenque, and the old `pos * 0.2` could not say so.
///
/// It is a *differential* rather than an outright price because `engine_value`
/// already pays every unlocked worker `ACTION_VALUE` per `ROUNDS_PER_ACTION`
/// rounds, whether it is in hand or on a gear. Paying the space value on top of
/// that counted a placed worker's next action twice, so placing always beat
/// holding by construction. The calibration study measured the damage: fitting
/// the realised final score on the components, centred across the four seats of
/// one position, gave this term a coefficient of **-0.44** — and negative in
/// every one of the nine day buckets, so it was steering the search the wrong
/// way for the whole game rather than only at one end of it.
/// (`tests/rules.rs::evaluator_calibration`.)
fn board_position(g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
    let mut v = 0.0;
    if rounds_left <= 0.0 {
        return 0.0;
    }
    // Once per position rather than once per space considered: the loop below
    // runs up to the width of a gear for every placed worker.
    let hungry = hungry(g, p);

    for w in g.on_board(p) {
        let Some((gear, pos)) = g.loc(w).on_board() else {
            continue;
        };
        let last = gear.size() - 1;
        // How far it can ride before the calendar ends or it falls off the top.
        let reach = last.min(pos.0.saturating_add(rounds_left as u8));

        // It can act now, or wait for something better up the gear. Waiting
        // costs a round of this worker's own throughput per space ridden --
        // which is the only thing that stops every worker on a gear being
        // valued at whatever the top of that gear pays. The old flat 0.85
        // multiplier did not depend on the distance, so a worker on Chichen
        // space 1 was priced at the 13-point space nine rotations away.
        let mut worth = space_value_scaled(g, p, gear, pos.0, hungry);
        for j in (pos.0 + 1)..=reach {
            let waited = (j - pos.0) as f32 * TEMPO_PER_ROUND;
            worth = worth.max(space_value_scaled(g, p, gear, j, hungry) - waited);
        }

        // A worker on the top space is picked up by the next rotation whether
        // its owner wants it or not: take it this turn or lose the action.
        if pos.0 == last {
            worth *= 0.5;
        }
        v += worth;
    }

    // The first player space: the corn pot, the marker, and the extra day.
    // Differential for the same reason -- that worker is spending its action
    // here rather than on a gear.
    if let Some(w) = g.first_player_space {
        if w.owner() == p {
            v += g.accumulated_corn as f32 / CORN_PER_POINT + 1.2;
        }
    }
    // The unused first-player tile is a free calendar day, which is a whole
    // extra round of gear movement for everyone but is claimed by its holder.
    if g.players[p.idx()].may_skip_day && rounds_left > 2.0 {
        v += 0.6;
    }

    v
}

/// Per-gear correction on the hand table, indexed by `Gear::ALL`:
/// Palenque, Yaxchilan, Tikal, Uxmal, Chichen.
///
/// The hand numbers are right *within* a gear and were wrong *between* gears,
/// and the reason is structural: [`space_value`] prices what a space hands over
/// and never what it charges. `evalab --promise`'s `dead` column measures the
/// consequence — the fraction of standing workers for which the evaluator,
/// offered the action, declines it and skips. Palenque and Yaxchilan, whose
/// actions are pure gathering, read **0.00 and 0.01**. Chichen reads **0.5 to
/// 0.7**, because a skull is held at `3.0 + SKULL_PREMIUM` = 5.2 through
/// `liquidation` and `held_premium` while the bottom of the gear pays 4 printed
/// points, so the evaluator will not cash a skull there and the table prices
/// those spaces as if it would. Tikal 1 reads **0.99**: a research advance
/// costs 1/2/3 blocks (`options::recurse`) and at `RESEARCH_SCALE = 0.05` is
/// worth a tenth of one.
///
/// **Only two of the five move.** The sizes are the arena's, not the
/// diagnostic's — `--promise` reads Uxmal as *under*-priced and shrinking it a
/// quarter measures **+1.80 on `greedy:64` and −0.46 on `mcts:256`**, the
/// largest greedy/MCTS sign split in `docs/FINDINGS-eval.md`. What survives the
/// search is Chichen, the gear whose payout is points a deeper search cannot
/// improve on: **+0.38 [+0.13, +0.64] mcts:256 at 400 blocks**, +0.36 greedy.
/// Palenque rides along at +0.11 [−0.09, +0.30] alone and +0.17 [−0.03, +0.38]
/// paired on top of Chichen; the pair is **+0.56 [+0.29, +0.83] mcts:256 and
/// +1.04 [+0.79, +1.30] greedy:64**. Palenque's own interval covers zero — it
/// is here on the strength of the pair and of a sharp greedy optimum (1.25 is
/// +0.23, 1.5 is +0.64, 2.0 is −0.59, 2.5 is −12.8), so drop it first if this
/// ever has to shrink. Yaxchilan loses in *both* directions and Tikal's greedy
/// gain did not get an MCTS tier.
/// `docs/FINDINGS-eval.md` F26b-F26e, F27.
const GEAR_SCALE: [f32; 5] = [1.5, 1.0, 1.0, 1.0, 0.7];

/// Extra multiplier on the Palenque table for a player who cannot pay the next
/// food bill out of the corn already in hand.
///
/// [`GEAR_SCALE`] says the corn gear is worth half again what the table pays.
/// This says it is not worth the same to everyone. Once
/// [`CORN_INCOME_PER_ROUND`] went to zero, corn's worth to a player who is
/// short is the three points a head [`starvation_risk`] is charging them, and
/// to a player who is not it is a quarter point plus [`CORN_PREMIUM`]; a flat
/// multiplier cannot say that and a gate can. The test is the one
/// `starvation_risk` fires on, without its arithmetic.
///
/// **The three agents rank this knob three different ways**, which is why it is
/// 0.25 and not larger. `greedy:64` at 800 blocks reads +0.19 / +1.73 / +1.06 /
/// +1.31 / **−1.90** / **−16.01** at 0.10 / 0.20 / 0.25 / 0.33 / 0.50 / 1.00 —
/// jagged, because a gate's effect is discontinuous in how often it flips a
/// max-over-the-gear comparison, and off a cliff by 0.5. `mcts:256` is monotone
/// *increasing* over the same range (+0.06 / +0.33 / +1.02 / +1.06), and
/// `mcts:1024:cp=0.05`, the deep search, reads **+0.70 [+0.21, +1.19] at 0.20**
/// where `mcts:256` reads +0.14. 0.25 is the largest value that is positive on
/// every agent measured; past 0.33 the one-ply agent that every screening
/// measurement in this file uses loses two points, and an evaluator nobody can
/// screen against is not worth the extra tenth.
///
/// An earlier attempt landed 1.0 and was retracted within two minutes: `pneed`
/// was swept against a base whose Palenque scale was 1.0, and carrying the same
/// constant onto [`GEAR_SCALE`]'s 1.5 made the total 3.0 — the configuration
/// that reads −16 on greedy. Two MCTS runs on disjoint seeds had both endorsed
/// it. `docs/FINDINGS-eval.md` F29a. Worth **+1.15 [+0.75, +1.55] * on `mcts:256`** at 298 blocks against the evaluator `GEAR_SCALE` landed, and **+1.29 [+0.59, +1.99] *** on a disjoint seed set — the larger of the two effects this run found, and the only one whose greedy reading (+1.48) and search reading agree in size rather than only in sign.
/// `docs/FINDINGS-eval.md` F27b, F28c, F28f.
const HUNGRY_CORN: f32 = 0.25;

/// Point-equivalent of the action at one board space.
///
/// Hand-priced against the space generators in `src/spaces`. Gated where the
/// action needs something the player may not have: Chichen without a skull is
/// worth nothing, and neither is a build with no blocks. [`GEAR_SCALE`] then
/// corrects the level of the gear as a whole.
fn space_value(g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
    space_value_scaled(g, p, gear, pos, hungry(g, p))
}

/// [`space_value`] with the hungry test already answered, so `board_position`
/// can hoist it out of its loop over the reachable gear.
fn space_value_scaled(g: &GameState, p: PlayerId, gear: Gear, pos: u8, hungry: bool) -> f32 {
    let mut scale = GEAR_SCALE[gear as usize];
    if hungry && gear == Gear::Palenque {
        scale *= 1.0 + HUNGRY_CORN;
    }
    scale * space_value_raw(g, p, gear, pos)
}

/// Whether `p` cannot pay the next food bill out of the corn in hand. The same
/// test [`starvation_risk`] fires on, without its arithmetic.
fn hungry(g: &GameState, p: PlayerId) -> bool {
    let pl = &g.players[p.idx()];
    let mouths = g.n_unlocked(p);
    let free = (pl.free_workers as usize).min(mouths);
    let each = 2u8.saturating_sub(pl.worker_discount);
    (mouths - free) as f32 * each as f32 > pl.corn as f32
}

/// [`space_value`] before [`GEAR_SCALE`]: the hand table as it was written.
fn space_value_raw(g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
    let pl = &g.players[p.idx()];
    match gear {
        // Corn and wood. The jungle spaces climb 5/7/9 corn or 2/3/4 wood.
        Gear::Palenque => match pos {
            0 => 0.0,
            1 => 0.9,
            2 => 1.6,
            3 => 1.6,
            4 => 2.1,
            5 => 2.6,
            _ => 2.6,
        },
        // Raw resources, and the game's only repeatable skull.
        Gear::Yaxchilan => match pos {
            0 => 0.0,
            1 => 0.8,
            2 => 1.3,
            3 => 1.8,
            // A skull is 3 points liquidated and much more at Chichen — but
            // only if the bank still has one.
            4 => {
                if g.skulls_remaining > 0 {
                    3.0 + SKULL_PREMIUM * 0.5
                } else {
                    0.2
                }
            }
            5 => 2.7,
            _ => 3.4,
        },
        // Research, construction and temples: the densest gear in the game.
        Gear::Tikal => {
            let can_build = pl.n_blocks() >= 2;
            match pos {
                0 => 0.0,
                1 => 2.0,
                2 => if can_build { 3.2 } else { 0.4 },
                3 => 3.4,
                4 => if can_build { 5.2 } else { 0.6 },
                // Two temple steps, for one block.
                5 => if pl.n_blocks() >= 1 { 4.0 } else { 0.3 },
                _ => 5.2,
            }
        }
        Gear::Uxmal => match pos {
            0 => 0.0,
            1 => if pl.corn >= 3 { 2.2 } else { 0.3 },
            2 => 1.1,
            // Buying a worker is the strongest early action on the board and
            // worthless once there is no time to use it.
            3 => {
                let locked = WORKERS_PER_PLAYER - g.n_unlocked(p);
                if locked > 0 {
                    3.4
                } else {
                    0.2
                }
            }
            4 => if pl.corn >= 4 { 3.0 } else { 0.3 },
            5 => if pl.corn >= 1 { 4.0 } else { 0.2 },
            _ => 4.0,
        },
        // Chichen pays in points directly, but only for a player holding a
        // skull and only on a space nobody has used.
        Gear::Chichen => {
            if pl.get(Resource::Skull) == 0 {
                return 0.1;
            }
            if pos >= 1 && pos <= 9 && g.chichen_is_full(Pos(pos)) {
                // Foresight steps up to the next space instead.
                return if g.foresight(p) { 3.0 } else { 0.1 };
            }
            // The printed rewards, 4..13 points, less the skull spent.
            match pos {
                0 => 0.0,
                1 => 3.0,
                2 => 3.8,
                3 => 4.6,
                4 => 5.4,
                5 => 6.2,
                6 => 6.8,
                7 => 8.0,
                8 => 9.0,
                9 => 10.5,
                _ => 10.5,
            }
        }
    }
}

/// A face-up monument the player is close to affording is worth a fraction of
/// what it would score.
///
/// Monuments are the largest single scores in the game and the row is shared,
/// so "I am two gold from a 20-point card" is a real feature of a position.
fn monument_outlook(g: &GameState, p: PlayerId, horizon: f32) -> f32 {
    if horizon <= 0.10 {
        return 0.0;
    }
    let pl = &g.players[p.idx()];
    let mut best: f32 = 0.0;

    for id in g.face_up_monuments() {
        let d = mdef(id);
        // Blocks still needed. Skulls in a monument cost are counted the same
        // way; #13 wants one.
        let short: i32 = Resource::ALL
            .iter()
            .map(|&r| (d.cost[r.idx()] as i32 - pl.get(r) as i32).max(0))
            .sum();
        if short > 4 {
            continue;
        }
        // What it would pay if taken now. Self-counting monuments (#2/#4/#5,
        // which count themselves) come out one card short, which is the right
        // direction to be wrong in.
        let pays = (d.score)(g, p) as f32;
        if pays <= 0.0 {
            continue;
        }
        // Full value when it is already affordable, falling away with distance.
        best = best.max(pays * MONUMENT_SHARE / (1.0 + short as f32));
    }

    best * horizon.min(1.0)
}

/// Points expected to be lost to unfed workers on the next food day.
///
/// This is the term whose absence let the old evaluator buy a sixth worker with
/// two corn in hand and call it progress. The rule costs 3 points a head, which
/// is more than most single actions are worth.
///
/// Everything in here except [`CORN_INCOME_PER_ROUND`] is already at an
/// optimum, checked in both directions on top of the corrected income: scaling
/// the whole term by 0.7 or 1.4 reads −0.30 and −0.24, moving the urgency
/// coefficient to 0.5 reads −0.26 and its floor to 0.1 reads −0.07, none of
/// them distinguishable from zero at 400 blocks. `docs/FINDINGS-eval.md` F23a.
fn starvation_risk(g: &GameState, p: PlayerId) -> f32 {
    let Some(next) = RESOURCE_DAYS
        .iter()
        .chain(POINT_DAYS.iter())
        .copied()
        .filter(|&d| d > g.day)
        .min()
    else {
        return 0.0;
    };

    let pl = &g.players[p.idx()];
    let mouths = g.n_unlocked(p);
    let free = (pl.free_workers as usize).min(mouths);
    let each = 2u8.saturating_sub(pl.worker_discount);
    if each == 0 {
        return 0.0;
    }
    let owed = (mouths - free) as f32 * each as f32;

    let rounds = (next - g.day) as f32;
    let expected = pl.corn as f32 + rounds * CORN_INCOME_PER_ROUND;
    let short = owed - expected;
    if short <= 0.0 {
        return 0.0;
    }
    // Every `each` corn short is one worker unfed.
    let unfed = short / each as f32;
    // Discount a shortfall that is still several rounds out — there is time to
    // fix it, but the search should still see it coming.
    let urgency = (1.0 / (1.0 + rounds * 0.25)).max(0.3);
    unfed * STARVE_POINTS * urgency
}

// ---- ranking -----------------------------------------------------------

/// A scored, ordered slice of a position's move list.
#[derive(Clone, Debug, Default)]
pub struct Ranking {
    /// Best first. Scores are [`margin`]-scale points unless a search says
    /// otherwise.
    pub moves: Vec<(Move, f32)>,
    /// Legal moves at this position, counting every spelling.
    pub total: usize,
    /// Legal moves that do something different from each other, as far as the
    /// scan could tell — `total` less the restatements it dropped. Equal to
    /// `total` when nothing was deduplicated.
    pub distinct: usize,
    /// Whether `total` is the whole move list rather than a capped walk.
    pub exhaustive: bool,
    /// What produced the ranking, for the status line.
    pub note: String,
}

impl Ranking {
    pub fn best(&self) -> Option<&Move> {
        self.moves.first().map(|(m, _)| m)
    }
}

/// Keeps the best `k` distinct moves of a stream without holding the rest.
///
/// The point of this type: the widest node measured is ~1.9M moves at 264 bytes
/// each, so the exhaustive pass has to *score* every move without ever *owning*
/// more than `k` of them. Amortised O(n): the buffer grows to `2k` and is
/// pruned by a single sort.
///
/// Distinctness is [`Move::same_effect`], not `Eq`. Without it a shortlist of
/// ten is routinely five moves written out twice — placement enumerates every
/// assignment of interchangeable workers to the same set of spaces — and the
/// search wastes most of its root beam re-deepening one position.
struct TopK {
    k: usize,
    buf: Vec<(Move, f32)>,
    /// The lowest score currently kept, once the buffer has been pruned once.
    floor: f32,
    /// Moves dropped as restatements of one already held.
    dupes: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            k: k.max(1),
            buf: Vec::with_capacity(k.max(1) * 2),
            floor: f32::NEG_INFINITY,
            dupes: 0,
        }
    }

    fn offer(&mut self, m: &Move, score: f32) {
        // The score test comes first: it rejects the overwhelming majority for
        // the price of one comparison, so the linear distinctness scan below
        // only runs on moves that were going to be kept anyway.
        if score <= self.floor {
            return;
        }
        if self.buf.iter().any(|(x, _)| x.same_effect(m)) {
            self.dupes += 1;
            return;
        }
        self.buf.push((m.clone(), score));
        if self.buf.len() >= self.k * 2 {
            self.prune();
        }
    }

    fn prune(&mut self) {
        self.buf.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.buf.truncate(self.k);
        if self.buf.len() == self.k {
            self.floor = self.buf[self.k - 1].1;
        }
    }

    fn finish(mut self) -> Vec<(Move, f32)> {
        self.buf.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.buf.truncate(self.k);
        self.buf
    }
}

/// The state a move leads to, as the next player will actually see it.
///
/// `refill_buildings` is part of `Game::play`, so leaving it out evaluates a
/// building row no player is ever shown.
#[inline]
pub fn successor(g: &GameState, p: PlayerId, m: &Move) -> GameState {
    let mut probe = *g;
    crate::moves::apply_move(&mut probe, p, m);
    probe.refill_buildings();
    probe
}

/// Rank an already-materialised move list by one-ply [`margin`].
pub fn rank(g: &GameState, p: PlayerId, moves: &[Move]) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = moves
        .iter()
        .enumerate()
        .map(|(i, m)| (i, margin(&successor(g, p, m), p)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// How long an exhaustive walk may run before it gives up and says so.
///
/// Exhaustive is the contract, and on all but a handful of positions it costs
/// under a millisecond. But the move space is not bounded by anything a caller
/// can see in advance — a strong player holding six workers and a pile of corn
/// reaches turns whose retrieval space runs into the millions — and an agent
/// that can stall for minutes on one turn is not an agent. So the walk carries
/// a deadline, and a `Ranking` that hit it says `exhaustive: false` rather than
/// quietly implying it saw everything.
pub const FULL_BUDGET: Duration = Duration::from_millis(2_500);

/// Moves scored between deadline checks. `Instant::now` is ~20 ns, which is a
/// real fraction of the ~1 us it costs to score a move, so it is not worth
/// asking on every one.
const DEADLINE_STRIDE: usize = 1024;

/// Sampled draws mixed in when the walk gave up, to counter the fact that a
/// prefix of traversal order is all placements and first-worker retrievals.
const GIVE_UP_TOP_UP: usize = 64;

/// Score **every** legal move with `score` and keep the best `keep`.
///
/// `score` is handed the successor state, already refilled. This is the
/// primitive `heuristic:full` and the search root are both built on: no cap, no
/// sampling, no traversal-order bias. It is affordable because scoring is
/// streamed — `visit_legal_moves` hands over one move at a time, it is applied
/// to a copy of the state, scored, and dropped unless it is good enough to keep
/// — so a six-figure node costs time but not memory.
///
/// `budget` bounds that time. `None` means genuinely unbounded, which is right
/// where a human is waiting on one position and wrong anywhere a turn has to
/// come back. Cost is one `score` call per legal move, so an evaluator that is
/// not cheap has no business here either way.
pub fn rank_all_within<F>(
    g: &GameState,
    p: PlayerId,
    keep: usize,
    budget: Option<Duration>,
    mut score: F,
) -> Ranking
where
    F: FnMut(&GameState) -> f32,
{
    let started = Instant::now();
    let deadline = budget.map(|b| started + b);
    let mut top = TopK::new(keep);
    let mut total = 0usize;

    let flow = crate::moves::visit_legal_moves(g, p, |m| {
        total += 1;
        top.offer(m, score(&successor(g, p, m)));
        if let Some(d) = deadline {
            if total % DEADLINE_STRIDE == 0 && Instant::now() >= d {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    });

    let gave_up = flow.is_break();
    if gave_up {
        // What was walked is a prefix of traversal order, which is placements
        // and then retrievals worker by worker — not a slice of the move space.
        // These draws come from the rollout policy, which is shaped like real
        // play, so the shortlist is at least not blind to a whole move kind.
        // Seeded from the position, so the same turn ranks the same way twice.
        let mut rng = StdRng::seed_from_u64(
            (g.day as u64) << 32 | (p.0 as u64) << 8 | g.current.0 as u64,
        );
        for _ in 0..GIVE_UP_TOP_UP {
            let Some(m) = crate::moves::sample_legal_move(g, p, &mut rng) else {
                break;
            };
            top.offer(&m, score(&successor(g, p, &m)));
        }
    }

    let ms = started.elapsed().as_secs_f64() * 1e3;
    let dupes = top.dupes;
    Ranking {
        moves: top.finish(),
        total,
        distinct: total.saturating_sub(dupes),
        exhaustive: !gave_up,
        note: if gave_up {
            format!("gave up at {total} moves after {ms:.0} ms — the list is wider than that")
        } else {
            format!("all {total} moves scored in {ms:.0} ms")
        },
    }
}

/// [`rank_all_within`] with no deadline. Only for a caller that can wait.
pub fn rank_all_by<F>(g: &GameState, p: PlayerId, keep: usize, score: F) -> Ranking
where
    F: FnMut(&GameState) -> f32,
{
    rank_all_within(g, p, keep, None, score)
}

/// Every legal move, scored by [`margin`], with no deadline.
pub fn rank_all(g: &GameState, p: PlayerId, keep: usize) -> Ranking {
    rank_all_by(g, p, keep, |s| margin(s, p))
}

/// The same under a deadline, which is what anything driving a game wants.
pub fn rank_all_capped(g: &GameState, p: PlayerId, keep: usize, budget: Duration) -> Ranking {
    rank_all_within(g, p, keep, Some(budget), |s| margin(s, p))
}

/// The same, stopping after `cap` moves.
///
/// Not a sample: it takes the *first* `cap` in traversal order, which is
/// systematically "place one worker low on Palenque". Use it where a bounded
/// prefix is genuinely wanted, not where a representative slice is.
pub fn rank_capped(g: &GameState, p: PlayerId, keep: usize, cap: usize) -> Ranking {
    let mut top = TopK::new(keep);
    let mut total = 0usize;

    let flow = crate::moves::visit_legal_moves(g, p, |m| {
        total += 1;
        top.offer(m, margin(&successor(g, p, m), p));
        if total >= cap {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });

    let capped = flow.is_break();
    let dupes = top.dupes;
    Ranking {
        moves: top.finish(),
        total,
        distinct: total.saturating_sub(dupes),
        exhaustive: !capped,
        note: if capped {
            format!("first {total} moves in traversal order")
        } else {
            format!("all {total} moves scored")
        },
    }
}
