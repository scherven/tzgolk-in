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

/// Corn a player can expect to bring in per round, used only to decide whether
/// a food day is survivable. Deliberately pessimistic: the penalty should fire
/// on positions that are genuinely short, not on every position that is not
/// already holding the whole bill.
const CORN_INCOME_PER_ROUND: f32 = 1.9;

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
/// Halving the term is the largest single effect measured on this file:
/// **+10.48 greedy:64 [+9.07, +11.89] and +9.04 mcts:256 [+7.86, +10.23]** at
/// 150 blocks. The optimum is a plateau, not a digit — 0.35, 0.40 and 0.50 are
/// mutually indistinguishable on both agents while 0.25 and 0.70 are clearly
/// worse — so read this as "about a half". `docs/FINDINGS-eval.md` F5a, F7.
///
/// Do **not** also subtract the placed worker from `engine_value`'s action
/// count. That corrects the same double count a second time and measures
/// −12.09 / −9.95 with a 0.028 win rate, the worst configuration in the log
/// after ungating the space table entirely (F5b).
const BOARD_SCALE: f32 = 0.5;

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
const TEMPLE_SCALE: f32 = 1.4;

/// Rounds a worker typically spends riding a gear between actions.
const ROUNDS_PER_ACTION: f32 = 2.6;

/// Residual value of a building already built. Its payoff is in the state
/// already; what is left is that three monuments count the pile.
const BUILDING_VALUE: f32 = 0.45;

/// Global scale on [`research_step_value`], which is priced per *use*.
const RESEARCH_SCALE: f32 = 0.5;

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
/// Known wart, left alone deliberately: the `climb` loop tests only
/// `step + 1 < steps`, so it credits the move on to the **exclusive top step
/// even while an opponent stands there** and `GameState::temple_ceiling` will
/// refuse it. Worth ~0.1 of one step's jump, far below anything measurable
/// here, and fixing it means reimplementing a private helper of `state.rs`.
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
        if (step + 1) < d.steps as usize {
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
        let mut worth = space_value(g, p, gear, pos.0);
        for j in (pos.0 + 1)..=reach {
            let waited = (j - pos.0) as f32 * TEMPO_PER_ROUND;
            worth = worth.max(space_value(g, p, gear, j) - waited);
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

/// Point-equivalent of the action at one board space.
///
/// Hand-priced against the space generators in `src/spaces`. Gated where the
/// action needs something the player may not have: Chichen without a skull is
/// worth nothing, and neither is a build with no blocks.
fn space_value(g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
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
