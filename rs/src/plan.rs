//! Phase-dependent evaluation, layered *over* `eval::components` rather than
//! forked from it.
//!
//! # The problem
//!
//! `eval::heuristic` is one function applied identically on day 1 and day 26.
//! It is not calendar-blind — `rounds_left` scales `engine`, `held` and
//! `board`, and `temple_outlook` walks the remaining scoring days at the right
//! age — but every term enters the sum with a weight of exactly 1, all game.
//! The *relative* price of an engine against a temple against solvency is the
//! same in the opening as on the last day, and that is wrong in a way no
//! per-term decay can fix.
//!
//! # What the data says
//!
//! `tests/rules.rs::evaluator_calibration` regresses each seat's realised final
//! score on the eight components, centred across the four seats of the same
//! position, so what is left is exactly the quantity a search is ranking. A
//! coefficient of 1.0 means the term is already priced right. Over 300 games of
//! `heuristic:32` (123,296 seat-positions):
//!
//! ```text
//!    days   banked  liquid.    held  temple  engine   board  monum.  starv.
//!     0-2     1.15     1.13   -9.70   -0.10   -5.33   -0.33   17.84   -0.94
//!     3-5     0.92     0.48   -6.61    0.19   -3.35   -0.87   16.86    2.08
//!     6-8     1.01    -2.21   -0.41    0.87   -1.66   -0.47    9.16    2.56
//!    9-11     1.01    -2.00    1.33    1.25   -1.00   -0.22    8.39    4.00
//!   12-14     0.92    -0.45    1.33    1.28   -0.77   -0.37    4.55    1.95
//!   15-17     1.21     1.43    0.25    0.97   -0.38   -0.45   -0.16    3.21
//!   18-20     1.17     1.43    0.43    1.08   -0.12   -0.31   -2.28    1.45
//!   21-23     1.14     1.42    0.31    1.06   -0.07   -0.14   -0.42    1.95
//!   24-26     1.07     1.18    0.23    1.10    0.44   -0.10    0.27    1.22
//! ```
//!
//! Three shapes are visible and none of them is flat:
//!
//! * **`temple` is worthless before day 6 and correctly priced after day 9.**
//!   Where everyone stands on the tracks on day 3 says nothing about the day-14
//!   payout, because every seat still climbs four or five steps. `eval` pays it
//!   at `TEMPLE_NEAR = 0.85` from day 0.
//! * **`engine` points the wrong way, and worst early.** A seat carrying more
//!   `engine` than its table-mates at day 1 finishes *behind* them. The raw fit
//!   in the same test prices one extra unlocked worker at **-10.2 realised
//!   points** (-14.4 before day 9), which is very close to the -12 a worker
//!   costs if it goes unfed at all four food days — so this is not "workers are
//!   bad", it is "this agent buys workers it cannot feed".
//! * **`starvation` is under-weighted by 1.2x to 4x, everywhere.** Same story
//!   from the other side.
//!
//! # The shape chosen
//!
//! A **weight schedule keyed on the calendar's own landmarks**: five knots at
//! days 0, 8, 14, 21 and 27 — the two resource days, the two point days and
//! the start — with linear interpolation between them. Six free terms times
//! five knots is 30 numbers, which is small enough to fit at the sample sizes
//! this project can afford and expressive enough to bend where the table above
//! bends.
//!
//! Linear interpolation rather than piecewise-constant phases because the
//! agent compares positions *across* a day boundary: `GreedyAgent::extra_day`
//! scores `advance_days(1)` against `advance_days(2)`, and a weight that jumps
//! at day 14 would make that comparison read a discontinuity in the *weights*
//! as a difference in the *positions*.
//!
//! `banked` and `liquidation` are **pinned at 1.0** and are not fitted. Both
//! are exact — points on the pad and what `end_game` would pay today — so
//! there is no defensible reading of "a banked point is worth 0.9 of a banked
//! point". Pinning them also fixes the gauge: with the two exact terms at 1,
//! every other weight is readable as "worth this many points per point of
//! `eval`'s estimate", and the whole sum stays in points. It also removes the
//! collinearity the calibration test flags between `held` and `liquidation`
//! (both are near-linear in the same block count, so only their sum is
//! identified) by giving that sum a fixed anchor.
//!
//! Weights are **clamped to `[0, MAX_W]`**. The fitted coefficients are
//! associations under a policy that was maximising the *un*-reweighted score,
//! not causal prices. "This term is noise, ignore it" survives being optimised
//! against; "the negative of this term is good" does not — an agent told that
//! `engine` is worth -5 will refuse to buy a fourth worker and lose to one that
//! has six.
//!
//! # What is rejected, and why
//!
//! * **Named discrete phases** (`Draft`/`Opening`/`PreFood`/...) with a weight
//!   vector each. Discontinuous, for the `extra_day` reason above, and the cut
//!   points would be guesses where the knots are the calendar's own structure.
//! * **Distance to the next scoring day as the axis** instead of absolute day.
//!   It is the right axis for `starvation` and `temple` and the wrong one for
//!   everything else: `engine` and `monument` care about how much game is
//!   *left*, not about when the next payout is, and day 9 and day 22 are both
//!   "5 days out" while being opposite ends of the game. Absolute day already
//!   determines distance-to-next-payout (the calendar is fixed), so the day
//!   axis is strictly more expressive; the fit can discover the sawtooth if it
//!   is there. `starvation` keeps its own distance term because `eval` already
//!   computes one.
//! * **A learned per-day table** (28 x 6). More parameters than the measurement
//!   can separate, and the fitted table is visibly smooth.
//!
//! # Beyond the day: weights that depend on the position
//!
//! A "phase" is not only a date. `held` prices corn, blocks and skulls above
//! liquidation *because they can still be spent* — `eval` scales that by
//! `min(rounds_left / 6, 1)`, a global clock that knows nothing about whether
//! **this** player can actually reach a space that converts **this** asset.
//! Because the gears turn exactly one space a day, that is arithmetic rather
//! than search: see [`conversion_reach`]. The reach factor multiplies the
//! schedule's `held` weight, so it modulates `eval`'s pricing without
//! duplicating any of it.
//!
//! # The plan
//!
//! The agent instance is shared across seats and across every game in a
//! rotation block (`arena.rs::play_block` hands the same `&dyn Agent` to four
//! seats), and `Evaluator::evaluate` takes `&self`. A stored, mutable plan is
//! therefore not available, and would be wrong even if it were: a search
//! evaluates hypothetical successors, and a plan committed in one branch would
//! leak into its siblings.
//!
//! So commitment is bought a different way. What a plan actually does to an
//! evaluation is make it **convex in progress**: two half-finished lines must
//! score worse than one finished one, or a per-turn argmax has no reason to
//! concentrate. `max` over the lines is convex and is a pure function of the
//! position, so [`focus`] adds a bonus proportional to the *best* line's
//! progress rather than to the sum. With total progress fixed, that is maximal
//! when it all sits on one line — which is the commitment, obtained without
//! storing anything. Abandonment is automatic and continuous: a line whose
//! calendar feasibility has run out contributes zero, and the max moves to
//! whatever is still reachable.

use crate::effect::Effect;
use crate::eval;
use crate::ids::*;
use crate::moves::Placement;
use crate::phase::{Evaluation, Evaluator, Phase, Step};
use crate::state::{GameState, LAST_DAY, POINT_DAYS, RESOURCE_DAYS};

// =======================================================================
// The weight schedule
// =======================================================================

/// Where the schedule's knots sit: the start, the two resource days and the
/// two point days. Every interesting discontinuity in the game is one of
/// these, so the schedule is free to bend at each and is linear in between.
pub const KNOTS: [u8; 5] = [0, 8, 14, 21, 27];

/// Index into `Components::terms()`. Named because a transposed weight vector
/// is silent and catastrophic.
pub const BANKED: usize = 0;
pub const LIQUIDATION: usize = 1;
pub const HELD: usize = 2;
pub const TEMPLE: usize = 3;
pub const ENGINE: usize = 4;
pub const BOARD: usize = 5;
pub const MONUMENT: usize = 6;
pub const STARVATION: usize = 7;

/// Upper clamp on a fitted weight.
///
/// `monument` fits at 17.8 in the first three days, on a term whose sd is 0.22
/// — a 17x correction on a small term, which is real signal but is also the
/// one place a single noisy bucket could dominate the whole estimate. 4.0 lets
/// it be the strongest term in the sum without letting it be the *only* one.
pub const MAX_W: f32 = 4.0;

/// Per-term weights at the five [`KNOTS`], linearly interpolated in between.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    /// `w[knot][term]`, term order as `eval::Components::NAMES`.
    pub w: [[f32; 8]; KNOTS.len()],
}

impl Schedule {
    /// All ones: reproduces `eval::heuristic` exactly, term for term.
    ///
    /// This is the null the arena measures against, and it is a *strong* null:
    /// with this schedule and [`PlanWeights::OFF`] the layer must play the
    /// identical game to `phase::HeuristicEvaluator`, so any centred score
    /// away from zero is a bug in the plumbing rather than an effect.
    pub const IDENTITY: Schedule = Schedule {
        w: [[1.0; 8]; KNOTS.len()],
    };

    /// The weights at `day`, linear between knots and flat outside them.
    pub fn at(&self, day: u8) -> [f32; 8] {
        let d = day.min(LAST_DAY);
        let mut k = 0;
        while k + 2 < KNOTS.len() && d >= KNOTS[k + 1] {
            k += 1;
        }
        let (a, b) = (KNOTS[k], KNOTS[k + 1]);
        let t = ((d.saturating_sub(a)) as f32 / (b - a) as f32).clamp(0.0, 1.0);
        std::array::from_fn(|i| self.w[k][i] * (1.0 - t) + self.w[k + 1][i] * t)
    }

    /// Clamp every free weight into `[0, MAX_W]` and pin the two exact terms.
    ///
    /// Applied to anything that comes out of a fit, because a fit can return a
    /// negative coefficient and an evaluator must not be told that destroying
    /// its own engine is progress.
    pub fn sanitise(mut self) -> Schedule {
        for k in 0..KNOTS.len() {
            self.w[k][BANKED] = 1.0;
            self.w[k][LIQUIDATION] = 1.0;
            for t in [HELD, TEMPLE, ENGINE, BOARD, MONUMENT, STARVATION] {
                self.w[k][t] = self.w[k][t].clamp(0.0, MAX_W);
            }
        }
        self
    }

    /// `IDENTITY` blended toward `self` by `t`.
    ///
    /// The shrinkage knob. The fit is an association measured under a policy
    /// that was *not* using its own output, so the honest reading of a fitted
    /// weight is a direction, not a destination: the further the schedule
    /// moves, the further off-distribution the data that chose it. `t` is swept
    /// in the arena rather than assumed — see `planlab --shrink`.
    pub fn shrunk(&self, t: f32) -> Schedule {
        let mut out = Schedule::IDENTITY;
        for k in 0..KNOTS.len() {
            for i in 0..8 {
                out.w[k][i] = 1.0 + (self.w[k][i] - 1.0) * t;
            }
        }
        out
    }

    /// The same average level, with all the calendar shape removed: every knot
    /// set to that term's mean weight over days 0..=27.
    ///
    /// The control the whole module rests on. If a flattened schedule measures
    /// as well as the shaped one, the gain was a constant rescale of `eval`'s
    /// terms and *not* phase-dependence, and saying so is worth more than the
    /// schedule is.
    pub fn flattened(&self) -> Schedule {
        let mut acc = [0.0f32; 8];
        for d in 0..=LAST_DAY {
            let w = self.at(d);
            for i in 0..8 {
                acc[i] += w[i] / (LAST_DAY as f32 + 1.0);
            }
        }
        Schedule { w: [acc; KNOTS.len()] }
    }

    /// One term forced to a flat `v` at every knot.
    pub fn with(mut self, term: usize, v: f32) -> Schedule {
        for k in 0..KNOTS.len() {
            self.w[k][term] = v;
        }
        self
    }

    /// Rust source for this table, so a `planlab fit` run can be pasted back
    /// here rather than loaded from a file the arena would have to ship.
    pub fn to_source(&self) -> String {
        let mut s = String::from("Schedule { w: [\n");
        for (k, d) in KNOTS.iter().enumerate() {
            s.push_str(&format!("    /* day {d:>2} */ ["));
            for t in 0..8 {
                s.push_str(&format!("{:.2}, ", self.w[k][t]));
            }
            s.push_str("],\n");
        }
        s.push_str("] }");
        s
    }
}

/// The fitted schedule.
///
/// Produced by `planlab fit`: within-position centred ridge of realised final
/// score on the six free components, one fit per knot-centred day window, then
/// clamped by [`Schedule::sanitise`]. See the module docs for the raw
/// coefficients it came from.
///
/// Reading it as strategy, which is the point of the exercise:
///
/// * **Day 0-8 is not about the temples.** `temple` starts at 0 and only
///   reaches 1 around day 11; nothing about a day-2 temple position survives to
///   day 14.
/// * **`engine` is never worth its face value** and is worth least at the
///   start, where `eval` pays a worker six actions' worth of credit it will
///   spend the whole game failing to feed.
/// * **Solvency is worth about double what `eval` charges for it**, all game.
/// * **`monument` is the one term worth *more* than face**, heavily so before
///   day 14 — being three blocks from a 20-point card on day 5 is a real
///   feature of a position and `MONUMENT_SHARE = 0.45` divided by `1 + short`
///   nearly erases it.
pub const FITTED: Schedule = Schedule {
    w: [
        //           banked  liquid   held  temple  engine  board  monum  starv
        /* day  0 */ [1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 3.49, 1.20],
        /* day  8 */ [1.00, 1.00, 0.00, 1.08, 0.00, 0.00, 4.00, 1.96],
        /* day 14 */ [1.00, 1.00, 0.73, 1.32, 0.00, 0.00, 2.91, 1.72],
        /* day 21 */ [1.00, 1.00, 0.76, 1.15, 0.00, 0.00, 1.18, 1.82],
        /* day 27 */ [1.00, 1.00, 0.85, 1.20, 0.80, 0.00, 1.02, 1.11],
    ],
};

/// [`FITTED`] pulled halfway back toward [`Schedule::IDENTITY`].
///
/// The fit is an association under a policy that was *not* using these weights,
/// so the further the schedule moves the further off-distribution the estimate
/// it came from is. Shrinking is the cheap insurance against that, and the
/// arena says which of the two to keep.
pub const HALF: Schedule = Schedule {
    w: [
        /* day  0 */ [1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 2.25, 1.10],
        /* day  8 */ [1.00, 1.00, 0.50, 1.04, 0.50, 0.50, 2.50, 1.48],
        /* day 14 */ [1.00, 1.00, 0.87, 1.16, 0.50, 0.50, 1.96, 1.36],
        /* day 21 */ [1.00, 1.00, 0.88, 1.08, 0.50, 0.50, 1.09, 1.41],
        /* day 27 */ [1.00, 1.00, 0.93, 1.10, 0.90, 0.50, 1.01, 1.06],
    ],
};

/// The second iteration: [`FITTED`] refit on the policy that [`FITTED`]
/// produced.
///
/// The first fit is measured on positions the *unmodified* evaluator reaches,
/// so the further its weights move the less the data that chose them describes
/// the agent that will use them. Refitting closes that loop once. The result is
/// much closer to identity — most of the first fit's correction was real and
/// has been taken — and what is left is where the *calendar shape* lives:
///
/// * `held` climbs 0.00 -> 0.84. Corn and blocks held on day 2 predict nothing;
///   the same holdings on day 24 are about to become points.
/// * `engine` climbs 0.00 -> 0.84 the same way, which is the opposite of the
///   sign `eval` gives it: `engine_value` fades *out* as the calendar runs down
///   (`actions_each = rounds_left / ROUNDS_PER_ACTION`), and the fit says the
///   fade is far too slow early and roughly right at the end.
/// * `starvation` is worth ~1.9x face before day 14 and ~1.0x after, which is
///   the shape a *penalty for a bill you still have time to fail to pay* should
///   have.
/// * `temple` is 0.23 on day 0 and ~1.1 from day 8 on: nothing about a day-2
///   temple standing survives to the day-14 payout.
///
/// Measured against the flat-rescale control in `planlab`; see the report.
pub const REFIT: Schedule = Schedule {
    w: [
        //           banked  liquid   held  temple  engine  board  monum  starv
        /* day  0 */ [1.00, 1.00, 0.00, 0.23, 0.00, 0.32, 1.34, 1.44],
        /* day  8 */ [1.00, 1.00, 0.00, 1.05, 0.34, 0.90, 1.41, 1.91],
        /* day 14 */ [1.00, 1.00, 0.31, 1.02, 0.51, 0.37, 1.15, 0.93],
        /* day 21 */ [1.00, 1.00, 0.54, 1.04, 0.78, 0.44, 1.28, 0.97],
        /* day 27 */ [1.00, 1.00, 0.84, 1.18, 0.84, 0.23, 1.08, 0.98],
    ],
};

// =======================================================================
// Calendar reach: what this player can still convert
// =======================================================================

/// The last day a worker of `p` could still take an action on `gear` at a
/// space of index >= `min_pos`, given that gears turn exactly one space a day.
///
/// Returns `None` if no worker can get there before the calendar ends. This is
/// the "calendar scheduling" the deterministic rotation makes available:
/// a worker at `(gear, pos)` is at `pos + k` after `k` more days and rides off
/// past `gear.size() - 1`, and a worker in hand enters at the lowest free
/// space and needs one turn to be placed first. No search — arithmetic.
pub fn days_to_reach(g: &GameState, p: PlayerId, gear: Gear, min_pos: u8) -> Option<u8> {
    let left = LAST_DAY.saturating_sub(g.day);
    if left == 0 {
        return None;
    }
    let last = gear.size() - 1;
    let mut best: Option<u8> = None;
    for w in g.on_board(p) {
        let Some((gr, pos)) = g.loc(w).on_board() else {
            continue;
        };
        if gr != gear || pos.0 > last {
            continue;
        }
        // Already there, or `min_pos - pos` rotations away — but only if it has
        // not ridden off the end by then.
        let need = min_pos.saturating_sub(pos.0);
        if pos.0 + need <= last && need <= left {
            best = Some(best.map_or(need, |b: u8| b.min(need)));
        }
    }
    // A worker in hand can be placed next turn and then ride. It enters at the
    // lowest free space, which this cannot see, so assume the entry space --
    // pessimistic by however far up the gear is already occupied.
    if best.is_none() && g.n_unlocked(p) > g.on_board(p).count() {
        let need = 1 + min_pos;
        if need <= left && min_pos <= last {
            best = Some(need);
        }
    }
    best
}

/// How much of `eval`'s `held` premium this player can actually realise: 1.0
/// when every asset class held has a converting space still in reach, falling
/// toward 0 when it does not.
///
/// `eval::held_premium` scales the whole term by `min(rounds_left / 6, 1)` — a
/// clock, not a plan. Three skulls on day 22 with no worker inside nine
/// rotations of a free Chichen space are worth 3 apiece and not a point more,
/// and the global clock still says `spendable = 0.83`.
///
/// Weighted by what each class contributes to that premium, so a player holding
/// only corn is not penalised for being unable to reach Chichen.
pub fn conversion_reach(g: &GameState, p: PlayerId) -> f32 {
    let pl = &g.players[p.idx()];
    let mut weight = 0.0;
    let mut got = 0.0;

    // Blocks convert at a Tikal build (spaces 2 and 4) or by climbing a temple
    // (Tikal 5). Weight ~ what `BLOCK_PREMIUM` is paid on.
    let blocks = pl.n_blocks() as f32;
    if blocks > 0.0 {
        weight += blocks.min(9.0);
        if days_to_reach(g, p, Gear::Tikal, 2).is_some() {
            got += blocks.min(9.0);
        }
    }

    // Skulls convert at Chichen, and only onto a space nobody has used.
    let skulls = pl.get(Resource::Skull) as f32;
    if skulls > 0.0 {
        let lowest_free = (1..=9u8).find(|&i| !g.chichen_is_full(Pos(i)));
        weight += skulls * 2.0; // SKULL_PREMIUM is the largest of the three
        if let Some(pos) = lowest_free {
            if days_to_reach(g, p, Gear::Chichen, pos).is_some() {
                got += skulls * 2.0;
            }
        }
    }

    // Corn is spendable anywhere a worker is placed, including on placement
    // depth itself, so it needs no specific space -- only a turn.
    let corn = (pl.corn as f32).min(16.0) * 0.25;
    weight += corn;
    if LAST_DAY.saturating_sub(g.day) >= 2 {
        got += corn;
    }

    if weight <= 0.0 {
        1.0
    } else {
        got / weight
    }
}

// =======================================================================
// The plan: convexity over lines
// =======================================================================

/// The scoring lines a game can commit to — a **disjoint partition**, which is
/// the only kind a dispersion statistic means anything over.
///
/// Also the vocabulary for asking what an action is *for*: `bin/movestats.rs`
/// tags effects with it.
///
/// The first version of this had six variants, `Monuments` and `Buildings`
/// among them. That was wrong and measurably so: both keyed on the block count,
/// so gathering a block advanced both, and a statistic over them was not a
/// measure of commitment at all but a second, differently scaled copy of `held`
/// — which the schedule had just finished *down*-weighting. Over 300 rotation
/// blocks it cost **-1.15 centred points** (-1.95..-0.34) at weight 1.0 and got
/// steadily worse as the weight rose.
///
/// These three compete for the same worker-turns on different gears: skulls
/// come off Yaxchilan and Chichen, construction off Tikal 2 and 4, temple steps
/// off Tikal 5 and Chichen. Nothing advances two of them at once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Line {
    /// Skulls into Chichen Itza: 4-13 points a head plus a temple step.
    Skulls,
    /// Blocks into buildings and monuments, on the back of Architecture.
    Construction,
    /// Climb a temple track and hold the top on a scoring day.
    Temples,
}

impl Line {
    pub const ALL: [Line; 3] = [Line::Skulls, Line::Construction, Line::Temples];
}

/// The age the *next* point day will pay out at, or `None` past the last one.
///
/// The whole reason the temples are phase-dependent: `advance_days` pays at the
/// age current when the day resolves, and the prizes invert between the two —
/// brown 6 then 2, yellow 2 then 6 (`data/temples.rs`). Which temple is worth
/// climbing depends on which scoring day you are aiming at, and that is a
/// strategic fact sitting in the data rather than in anyone's judgement.
pub fn next_scoring_age(g: &GameState) -> Option<u8> {
    let next = *POINT_DAYS.iter().find(|&&d| d > g.day)?;
    Some(1 + POINT_DAYS.iter().filter(|&&x| x < next).count() as u8)
}

/// How far up one track this player is, scaled by how much the prize on offer
/// at the next scoring day is worth relative to the best in the game.
///
/// Topping yellow in age 1 pays 2 and topping brown pays 6, so the same height
/// is not the same plan.
pub fn temple_share(g: &GameState, p: PlayerId, t: Temple, age: u8) -> f32 {
    let d = &crate::data::temples::TEMPLES[t.idx()];
    let base = crate::data::temples::STARTING_STEP as f32;
    let top = (d.steps - 1) as f32;
    let prize = if age == 1 { d.age1_prize } else { d.age2_prize } as f32;
    let climbed = (g.temple_pos(p, t) as f32 - base).max(0.0);
    ((climbed / (top - base)) * (prize / 6.0)).min(1.0)
}

/// **How far along** this line the player is, as a fraction in `[0, 1]` of what
/// a game spent on it would look like.
///
/// A fraction, not a point total: [`focus`] measures dispersion across the
/// lines, and a dispersion over quantities on different scales is just a vote
/// for whichever one has the biggest numbers.
///
/// A line the calendar has closed reads 0 — which is the abandonment rule, and
/// it needs no trigger because it is continuous in the position.
pub fn line_progress(g: &GameState, p: PlayerId, l: Line) -> f32 {
    let pl = &g.players[p.idx()];
    match l {
        Line::Skulls => {
            let free = (1..=9u8).find(|&i| !g.chichen_is_full(Pos(i)));
            let open = free
                .and_then(|pos| days_to_reach(g, p, Gear::Chichen, pos))
                .is_some();
            if !open {
                return 0.0;
            }
            // Skulls in hand plus the theology that keeps producing them.
            // Three Chichen visits is a game's worth of this line.
            let v =
                pl.get(Resource::Skull) as f32 + g.level(p, Science::Theology) as f32 * 0.5;
            (v / 3.0).min(1.0)
        }
        Line::Construction => {
            if days_to_reach(g, p, Gear::Tikal, 2).is_none() {
                return 0.0;
            }
            // Monuments count double: they are the expensive end of the same
            // line, and #2, #4 and #5 pay per monument of a colour.
            let v = g.level(p, Science::Architecture) as f32
                + pl.n_buildings() as f32
                + pl.monument_ids().count() as f32 * 2.0;
            (v / 6.0).min(1.0)
        }
        Line::Temples => {
            let Some(age) = next_scoring_age(g) else {
                return 0.0;
            };
            Temple::ALL
                .iter()
                .map(|&t| temple_share(g, p, t, age))
                .fold(0.0f32, f32::max)
        }
    }
}

/// How hard the layer leans on concentration, in points per point of the
/// leading line's progress. Zero switches the plan off entirely.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanWeights {
    /// Multiplier on the best line's progress.
    pub focus: f32,
    /// Whether the schedule's `held` weight is modulated by
    /// [`conversion_reach`].
    pub reach: bool,
}

impl PlanWeights {
    pub const OFF: PlanWeights = PlanWeights {
        focus: 0.0,
        reach: false,
    };
}

/// The commitment term: the **dispersion** of progress across the [`Line`]s.
///
/// `max - mean` is zero when every line is equally advanced and largest when
/// one is finished and the rest are untouched. That is the property a plan is
/// supposed to add and the only one — it deliberately carries no "has more
/// stuff" component, because every point of *having* is already priced by
/// `eval`'s own terms and paying for it twice is exactly what the first version
/// of this did.
///
/// Abandonment needs no rule and no stored commitment. A line the calendar has
/// closed reads zero progress, so the dispersion re-forms around whatever is
/// still reachable.
///
/// Faded out over the last two days: nothing started then finishes, and by then
/// the only correct plan is liquidation.
pub fn focus(g: &GameState, p: PlayerId, k: f32) -> f32 {
    if k == 0.0 {
        return 0.0;
    }
    let left = LAST_DAY.saturating_sub(g.day) as f32;
    if left <= 2.0 {
        return 0.0;
    }
    let f: [f32; 3] = std::array::from_fn(|i| line_progress(g, p, Line::ALL[i]));
    let max = f.iter().copied().fold(0.0f32, f32::max);
    let mean = f.iter().sum::<f32>() / f.len() as f32;
    k * (max - mean) * (left / LAST_DAY as f32).min(1.0)
}

// =======================================================================
// The evaluator
// =======================================================================

/// `eval::components`, reweighted by the calendar and by what the position can
/// still convert, plus the plan's convexity term.
///
/// Deliberately owns no pricing of its own: every point in the sum came out of
/// `eval::components`, so a change to `eval.rs`'s constants moves this
/// evaluator with it instead of leaving the two to drift apart.
/// How the starting-tile draft is decided.
///
/// A named choice rather than a bool because the interesting control is the
/// third one: measuring `Random` against `Board` is how you find out whether
/// the draft is worth having an opinion about at all, which has to be settled
/// before any effort goes into having a better one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DraftMode {
    /// `record::best_pair` over the general board evaluator, as `GreedyAgent`
    /// does today: score the post-draft position with a function built for
    /// positions that have a board.
    Board,
    /// [`draft_value`]: price the holdings directly, for a full calendar.
    Prior,
    /// A uniform pick, as `RandomAgent` does. The control.
    Random,
    /// `record::best_pair` over the **unweighted** evaluator: the schedule
    /// drives play, but the draft is scored with `eval`'s own day-0 pricing.
    ///
    /// Worth a mode of its own because the schedule's day-0 weights are fitted
    /// on mid-game *discrimination between seats* and are the wrong prior for a
    /// deal: with `engine` at 0 on day 0 they price tile 4 (`UnlockWorker`) at
    /// **0.00**, where `eval` prices it at 7.20 — see `planlab tiles`.
    Unweighted,
}

pub struct PlanEvaluator {
    pub sched: Schedule,
    pub plan: PlanWeights,
    /// How the starting-tile pair is chosen. See [`DraftMode`].
    pub draft: DraftMode,
    pub label: &'static str,
}

impl PlanEvaluator {
    /// The unmodified evaluator, as a control.
    pub fn identity() -> Self {
        PlanEvaluator {
            sched: Schedule::IDENTITY,
            plan: PlanWeights::OFF,
            draft: DraftMode::Board,
            label: "identity",
        }
    }

    /// Phase weights only.
    pub fn phase() -> Self {
        PlanEvaluator {
            sched: FITTED,
            plan: PlanWeights::OFF,
            draft: DraftMode::Board,
            label: "phase",
        }
    }

    /// Estimated final score for `p`, in points — the same contract as
    /// `eval::heuristic`.
    pub fn raw(&self, g: &GameState, p: PlayerId) -> f32 {
        let c = eval::components(g, p);
        // A finished game is not an estimate. `eval::components` has already
        // reduced to the scorepad, and reweighting a scorepad would break the
        // one property that lets a search compare a forced line against a
        // guess: a leaf evaluation and a real result are the same number.
        if g.over {
            return c.banked;
        }
        let t = c.terms();
        let mut w = self.sched.at(g.day);
        if self.plan.reach {
            w[HELD] *= conversion_reach(g, p);
        }
        let mut v = 0.0;
        for k in 0..8 {
            v += w[k] * t[k];
        }
        v + focus(g, p, self.plan.focus)
    }
}

impl Evaluator for PlanEvaluator {
    fn evaluate(
        &self,
        state: &GameState,
        _phase: Phase,
        _turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        // Identical shaping to `phase::HeuristicEvaluator`, so the two are
        // interchangeable in `GreedyAgent` and a duel between them measures the
        // weights and nothing else.
        let raw: [f32; N_PLAYERS] = std::array::from_fn(|i| self.raw(state, PlayerId(i as u8)));
        let mean = raw.iter().sum::<f32>() / N_PLAYERS as f32;
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
        self.label.into()
    }
}

// =======================================================================
// The draft
// =======================================================================

/// What a kept starting-tile pair is worth, scored **without** a board
/// evaluator.
///
/// `record::best_pair` currently scores the post-draft position with the
/// general evaluator, which is the wrong tool: there is no board. Every term
/// that makes `eval::heuristic` work — `board`, `monument`, `temple_outlook`'s
/// projected standings, `starvation`'s corn-income model — is being asked about
/// a position with no worker placed, no gear turned and 27 days of noise ahead
/// of it. What tiles are actually worth is what they *hand you*, priced for a
/// full calendar's use.
///
/// So this prices the holdings directly, at the day-0 end of the schedule:
///
/// * A worker unlocked on day 0 is the largest single item in the deal — it
///   acts for the whole game — but it also eats at all four food days, which is
///   the correction the calibration's -14.4 points per early worker is pointing
///   at. Net, not gross.
/// * A research level bought before any action is worth more than one bought on
///   day 20, because everything downstream of it is cheaper.
/// * Corn is the one thing that is *always* convertible, and 27 days is long
///   enough to spend all of it, so it is priced above its 4:1 liquidation.
/// * Temple steps are priced against the **age-1** prizes, because the day-14
///   payout is the one a starting step is still standing in front of. Brown
///   pays 6 then, yellow 2.
pub fn draft_value(g: &GameState, p: PlayerId) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = 0.0;

    // Corn: liquidation plus a full calendar of spending premium.
    v += pl.corn as f32 * 0.45;

    // Blocks: a block placed into a building or a monument is worth far more
    // than its 3-corn liquidation, and a spread beats a stack because build
    // costs are mixed.
    let counts = [
        pl.get(Resource::Wood) as f32,
        pl.get(Resource::Stone) as f32,
        pl.get(Resource::Gold) as f32,
    ];
    // Gold is scarcest and gates the expensive monuments; wood is the one
    // Palenque hands out for free.
    v += counts[0] * 0.9 + counts[1] * 1.1 + counts[2] * 1.4;
    v += counts.iter().copied().fold(f32::INFINITY, f32::min) * 0.6;

    // A skull on day 0 is 27 days of Chichen options, the densest points in the
    // game per action.
    v += pl.get(Resource::Skull) as f32 * 4.0;

    // A fourth worker, net of the eight corn it eats over four food days.
    let extra = g.n_unlocked(p) as f32 - 3.0;
    v += extra * 4.5;
    // Free-feeding removes exactly that cost, all game.
    v += pl.free_workers as f32 * 2.0;

    // Research before the first placement compounds: `RESEARCH_SCALE` in `eval`
    // is priced per use and day 0 is the maximum number of uses there will
    // ever be.
    for s in Science::ALL {
        v += g.level(p, s) as f32 * 1.6;
    }

    // Temple steps, against the age-1 prizes -- the payout a starting step is
    // actually standing in front of.
    for t in Temple::ALL {
        let d = &crate::data::temples::TEMPLES[t.idx()];
        let step = g.temple_pos(p, t) as usize;
        let base = crate::data::temples::STARTING_STEP as usize;
        if step > base {
            let climbed = (d.points[step] - d.points[base]) as f32;
            // Height plus a share of the age-1 prize this step is competing
            // for: brown 6, green 4, yellow 2.
            v += climbed + (step - base) as f32 * d.age1_prize as f32 * 0.25;
        }
    }

    // Palenque tiles: monuments #7 and #8 pay 4 each for them.
    v += (pl.corn_tiles + pl.wood_tiles) as f32 * 0.4;

    v
}

// =======================================================================
// The agent
// =======================================================================

/// A `record::GreedyAgent` over [`PlanEvaluator`], with the draft optionally
/// taken by [`draft_value`] instead of by the board evaluator.
///
/// This wrapper exists only because `GreedyAgent::draft` is not overridable
/// from outside `record.rs`; everything else delegates.
pub struct PlanAgent {
    pub inner: crate::record::GreedyAgent<PlanEvaluator>,
}

impl PlanAgent {
    pub fn new(ev: PlanEvaluator, k: usize) -> Self {
        PlanAgent {
            inner: crate::record::GreedyAgent {
                ev,
                cands: crate::record::Candidates::Sampled(k),
                record: false,
            },
        }
    }
}

impl crate::record::Agent for PlanAgent {
    fn play_turn(
        &self,
        g: &GameState,
        p: PlayerId,
        temp: f32,
        rng: &mut rand::rngs::StdRng,
    ) -> Option<crate::record::TurnOutcome> {
        self.inner.play_turn(g, p, temp, rng)
    }

    fn extra_day(
        &self,
        g: &GameState,
        p: PlayerId,
        rng: &mut rand::rngs::StdRng,
    ) -> (bool, Option<crate::record::Node>) {
        self.inner.extra_day(g, p, rng)
    }

    fn draft(
        &self,
        g: &GameState,
        p: PlayerId,
        dealt: [u8; 4],
        rng: &mut rand::rngs::StdRng,
    ) -> [u8; 2] {
        match self.inner.ev.draft {
            DraftMode::Prior => crate::record::best_pair(g, p, dealt, |s| draft_value(s, p)),
            DraftMode::Board => self.inner.draft(g, p, dealt, rng),
            DraftMode::Unweighted => {
                let flat = PlanEvaluator::identity();
                crate::record::best_pair(g, p, dealt, |s| flat.raw(s, p))
            }
            DraftMode::Random => {
                use rand::Rng;
                let mut idx = [0usize, 1, 2, 3];
                for i in (1..4).rev() {
                    idx.swap(i, rng.gen_range(0..=i));
                }
                [dealt[idx[0]], dealt[idx[1]]]
            }
        }
    }

    fn name(&self) -> String {
        self.inner.name()
    }
}

// =======================================================================
// The plan as a prior
// =======================================================================

/// Which [`Line`]s a step advances, as a 3-bit mask, decided from the `Step`
/// alone.
///
/// **Read off the effect vocabulary, never by applying the step.** That is the
/// whole reason a plan can afford to touch a prior at all: a `Take` node on
/// Tikal 6 offers ~124 choices, and the alternative — apply, then re-derive
/// `line_progress` — is 124 `apply_step`s and 372 `days_to_reach` walks for
/// one expansion. This is a scan of at most eight `Effect`s with no state
/// touched and nothing copied.
///
/// The classification is deliberately *exactly* the inputs [`line_progress`]
/// reads, term for term. A prior that pointed somewhere the value function does
/// not follow would spend simulations proving itself wrong — the search visits
/// the edge, gets no credit, and has to unlearn the steer. So raw blocks are
/// **not** Construction here, for the same reason `lines_do_not_share_inputs`
/// asserts they are not Construction there: they are already priced by `held`.
///
/// A mask rather than one line, because a Chichen action genuinely is both a
/// skull spent and a temple step, and forcing a choice between them would be a
/// worse reading than reporting both.
fn lines_advanced(step: &Step) -> u8 {
    let Step::Take(choice) = step else {
        return 0;
    };
    let mut mask = 0u8;
    for e in choice.0.iter() {
        let l = match *e {
            // What `Line::Skulls` counts is skulls in hand plus the theology
            // that keeps producing them; a Chichen space is where they become
            // points and it is the only effect that marks one unambiguously.
            Effect::Res(Resource::Skull, n) if n > 0 => Line::Skulls,
            Effect::FillChichen(_) => Line::Skulls,
            Effect::AdvanceResearch(Science::Theology) => Line::Skulls,
            // Architecture, buildings, monuments: the three things
            // `Line::Construction` reads. Blocks are not among them.
            Effect::Build(_) | Effect::TakeMonument(_) => Line::Construction,
            Effect::AdvanceResearch(Science::Architecture) => Line::Construction,
            Effect::TempleStep(_, n) if n > 0 => Line::Temples,
            _ => continue,
        };
        mask |= 1 << (l as u8);
    }
    mask
}

/// Which lines a *placement* can serve, by gear.
///
/// Coarser than [`lines_advanced`] and deliberately so: a placement does not
/// commit to an action, it buys the option of one several rotations later.
/// Uxmal is blank rather than marked as everything — its mirror space reaches
/// any gear, so "Uxmal serves all three lines" and "no opinion" are the same
/// statement and the second one is free.
fn lines_placed(step: &Step) -> u8 {
    let Step::Place(Placement::Gear(gear, _)) = step else {
        return 0;
    };
    match gear {
        // Skulls come off Yaxchilan and are cashed at Chichen, which pays a
        // temple step for every skull it takes.
        Gear::Yaxchilan => 1 << (Line::Skulls as u8),
        Gear::Chichen => (1 << (Line::Skulls as u8)) | (1 << (Line::Temples as u8)),
        // Tikal 2 and 4 build; Tikal 5 climbs.
        Gear::Tikal => (1 << (Line::Construction as u8)) | (1 << (Line::Temples as u8)),
        // Palenque is corn and wood: it feeds every line and commits to none.
        Gear::Palenque | Gear::Uxmal => 0,
    }
}

/// How hard the plan steers PUCT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiasWeights {
    /// Strength, in nats of prior per unit of the leading line's lead. Zero is
    /// the identity bias, and must reproduce the unbiased search exactly.
    pub k: f32,
    /// Also steer `Step::Place`, by gear. A separate knob because a placement
    /// commits to an option rather than to an action, and the two claims are
    /// worth measuring apart.
    pub place: bool,
    /// Below this lead the plan abstains — and does no per-edge work at all.
    ///
    /// On day 0 every line reads zero, `max - mean` is exactly zero, and "which
    /// line am I on" has no answer; any steer there is noise on a coin flip.
    /// This is also where most of the saving is, because the opening is where
    /// the tree is widest.
    pub min_lead: f32,
    /// **The control, not a mode to ship.** Raise every line-advancing edge by
    /// the leader's multiplier and demote nothing, so the bias no longer knows
    /// or cares which line is ahead.
    ///
    /// It exists because "steer toward the committed line" and "steer toward
    /// any edge that scores at all" boost overlapping sets of edges, and a
    /// number measured against an unbiased search cannot tell them apart. This
    /// arm holds the gate, the strength schedule and the edge set fixed and
    /// varies only the *direction*, so the difference between it and
    /// [`PlanBias`] is exactly the plan.
    pub blind: bool,
}

impl BiasWeights {
    pub const OFF: BiasWeights = BiasWeights {
        k: 0.0,
        place: false,
        min_lead: 0.02,
        blind: false,
    };
}

/// [`crate::mcts::PriorBias`] over [`Line`]: raise the prior on edges that
/// advance the line this player is already committed to, lower it on the ones
/// that split the commitment, and never remove anything.
///
/// # Why the multiplier has the shape it does
///
/// The plan's whole content is that progress should be **convex** — two
/// half-finished lines are worth less than one finished one — and [`focus`]
/// spends that as `max - mean` over the three lines. This prior is the same
/// statistic differentiated. Adding `d` to line `l` moves `max - mean` by
///
/// * `d - d/3 = +2d/3` when `l` is already the leader, and
/// * `-d/3` when it is not,
///
/// so **+2/3 and -1/3 are not tuned numbers**: they are the derivative of the
/// commitment term over three lines. The one free parameter is `k`, which
/// converts a lead into nats of prior.
///
/// The whole thing scales with the current lead, because a position with no
/// lead has nothing to be loyal to and steering on a tie is steering on noise.
///
/// # Why this cannot exclude a move
///
/// The weight is `exp` of a bounded quantity, so it is strictly positive, and
/// PUCT's exploration term still reaches every edge given enough simulations —
/// a wrong plan costs simulations, not correctness. The one exception is a node
/// wider than `MctsConfig::max_edges`, where `Mcts::node_for` sorts by prior
/// before truncating to `widen_cap`; there a prior really is an exclusion, and
/// that is why `k` is swept rather than assumed.
pub struct PlanBias {
    pub w: BiasWeights,
}

impl PlanBias {
    pub fn new(k: f32) -> Self {
        PlanBias {
            w: BiasWeights {
                k,
                ..BiasWeights::OFF
            },
        }
    }
}

/// Marginal effect on `max - mean` of advancing the leading line, and of
/// advancing one of the other two. Derived, not fitted: see [`PlanBias`].
const LEAD_SLOPE: f32 = 2.0 / 3.0;
const FOLLOW_SLOPE: f32 = -1.0 / 3.0;

impl crate::mcts::PriorBias for PlanBias {
    fn bias(
        &self,
        state: &GameState,
        _phase: Phase,
        mover: PlayerId,
        steps: &[Step],
        out: &mut [f32],
    ) {
        if self.w.k == 0.0 {
            return;
        }
        // The same fade `focus` uses: in the last two days nothing that is
        // started finishes, and the only correct plan is liquidation.
        let left = LAST_DAY.saturating_sub(state.day) as f32;
        if left <= 2.0 {
            return;
        }

        // Three `line_progress` calls per *node*, not per edge. That is the
        // whole cost model: a node-level plan and an edge-level lookup.
        let f: [f32; 3] = std::array::from_fn(|i| line_progress(state, mover, Line::ALL[i]));
        let max = f.iter().copied().fold(0.0f32, f32::max);
        let mean = f.iter().sum::<f32>() / 3.0;
        let lead = max - mean;
        if lead < self.w.min_lead {
            return;
        }
        let leader = Line::ALL[f.iter().position(|&x| x == max).unwrap_or(0)] as u8;

        let s = self.w.k * lead * (left / LAST_DAY as f32).min(1.0);
        let up = (s * LEAD_SLOPE).exp();
        let down = (s * FOLLOW_SLOPE).exp();
        for (step, o) in steps.iter().zip(out.iter_mut()) {
            let mut mask = lines_advanced(step);
            if self.w.place {
                mask |= lines_placed(step);
            }
            if mask == 0 {
                continue;
            }
            // An edge that touches the leading line counts as advancing it even
            // when it also touches another. A Chichen skull *is* a temple step,
            // and refusing to credit it for the line it is on would be a worse
            // reading than crediting it twice.
            *o = if self.w.blind || mask & (1 << leader) != 0 {
                up
            } else {
                down
            };
        }
    }

    fn name(&self) -> String {
        format!(
            "plan-bias:k={}{}{}",
            self.w.k,
            if self.w.place { ":place" } else { "" },
            if self.w.blind { ":blind" } else { "" }
        )
    }
}

/// The cheap, honest baseline the plan has to beat: apply each step, score the
/// result with [`PlanEvaluator::raw`], softmax.
///
/// This is `mcts::Priors::OnePly` with the plan-weighted evaluator underneath
/// instead of `eval::heuristic`, written as a bias so the two arms sit on the
/// same seam — a bias multiplies whatever prior it is handed, so installing
/// this over a uniform prior *is* a one-ply softmax.
///
/// It exists to answer one question and is not meant to be shipped. The costs
/// are not comparable: `O(width)` state copies and full evaluations here
/// against [`PlanBias`]'s `O(width)` eight-element scans, so a tie is a loss
/// for this one.
///
/// `done` is passed as 0 because [`crate::mcts::PriorBias::bias`] is not given
/// it. That is safe rather than approximate: `tree::advance_within_turn` uses
/// `done` only to label the phase it returns, never to mutate, and the returned
/// transition is discarded here.
pub struct OnePlyBias {
    pub ev: PlanEvaluator,
    /// Softmax temperature, in points — `eval::heuristic`'s own scale, so 4.0
    /// means "a four-point edge over a sibling is worth e times the prior".
    pub temp: f32,
}

impl crate::mcts::PriorBias for OnePlyBias {
    fn bias(
        &self,
        state: &GameState,
        phase: Phase,
        mover: PlayerId,
        steps: &[Step],
        out: &mut [f32],
    ) {
        let mut best = f32::NEG_INFINITY;
        for (step, o) in steps.iter().zip(out.iter_mut()) {
            let mut next = *state;
            // `mover` doubles as the turn holder: the two differ only at
            // `ExtraDay`, and every caller searches that node with the claimer
            // in both roles.
            let _ = crate::tree::apply_step(&mut next, phase, mover, 0, step);
            let s = self.ev.raw(&next, mover);
            best = best.max(s);
            *o = s;
        }
        // Shifted by the max before exponentiating: `raw` runs to ~200 points
        // late in a game and `exp(50)` is not a number.
        let t = self.temp.max(1e-3);
        for o in out.iter_mut() {
            *o = ((*o - best) / t).exp();
        }
    }

    fn name(&self) -> String {
        format!("one-ply-bias:t={}", self.temp)
    }
}

/// Days until the next day on which anyone is fed. `None` past the last one.
///
/// Exposed because it is the axis the `starvation` and `temple` terms actually
/// live on, and `planlab` reports along it.
pub fn days_to_food(g: &GameState) -> Option<u8> {
    RESOURCE_DAYS
        .iter()
        .chain(POINT_DAYS.iter())
        .copied()
        .filter(|&d| d > g.day)
        .min()
        .map(|d| d - g.day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;

    /// The identity schedule must reproduce `eval::heuristic` to the bit, or
    /// every duel against it is measuring plumbing rather than weights.
    #[test]
    fn identity_schedule_is_the_evaluator() {
        let ev = PlanEvaluator::identity();
        for seed in 0..40u64 {
            let mut g = Game::new(seed);
            for _ in 0..30 {
                if g.state.over {
                    break;
                }
                for p in PlayerId::ALL {
                    let a = ev.raw(&g.state, p);
                    let b = crate::eval::heuristic(&g.state, p);
                    assert!(
                        (a - b).abs() < 1e-3,
                        "seed {seed} day {} seat {p:?}: {a} vs {b}",
                        g.state.day
                    );
                }
                g.play_round();
            }
        }
    }

    /// Weights must be continuous in the day: `extra_day` compares a one-day
    /// advance against a two-day one, and a jump in the schedule would read as
    /// a difference in the positions.
    #[test]
    fn schedule_is_continuous() {
        let s = FITTED;
        for d in 0..LAST_DAY {
            let a = s.at(d);
            let b = s.at(d + 1);
            for k in 0..8 {
                assert!(
                    (a[k] - b[k]).abs() <= MAX_W / 4.0,
                    "term {k} jumps {} -> {} between day {d} and {}",
                    a[k],
                    b[k],
                    d + 1
                );
            }
        }
        // And it must hit the knots exactly.
        for (k, &d) in KNOTS.iter().enumerate() {
            assert_eq!(s.at(d), s.w[k], "knot at day {d}");
        }
    }

    /// A finished game evaluates to the scorepad, whatever the schedule says —
    /// the property that lets a search compare a forced result against a guess.
    #[test]
    fn finished_games_are_exact() {
        let ev = PlanEvaluator {
            sched: FITTED,
            plan: PlanWeights {
                focus: 1.0,
                reach: true,
            },
            draft: DraftMode::Prior,
            label: "t",
        };
        for seed in 0..10u64 {
            let mut g = Game::new(seed);
            let mut guard = 0;
            while !g.state.over && guard < 200 {
                g.play_round();
                guard += 1;
            }
            assert!(g.state.over);
            let scores = g.state.scores();
            for p in PlayerId::ALL {
                assert_eq!(ev.raw(&g.state, p), scores[p.idx()] as f32);
            }
        }
    }

    /// `focus` must read zero when the lines are level and rise with the
    /// spread between them, at equal total progress. That is the whole
    /// commitment mechanism and the property the first version got wrong, so it
    /// is asserted on the statistic rather than assumed.
    #[test]
    fn focus_measures_spread_not_size() {
        let spread = |f: [f32; 3]| {
            let max = f.iter().copied().fold(0.0f32, f32::max);
            max - f.iter().sum::<f32>() / 3.0
        };
        assert_eq!(spread([0.5, 0.5, 0.5]), 0.0, "level lines are no plan");
        assert_eq!(spread([0.1, 0.1, 0.1]), 0.0, "and neither is having little");
        // Same total, all on one line: the commitment.
        assert!(spread([0.9, 0.0, 0.0]) > spread([0.3, 0.3, 0.3]));
        // More of everything must not read as more commitment.
        assert_eq!(spread([0.9, 0.0, 0.0]), spread([0.9, 0.0, 0.0]));
        assert!(spread([1.0, 1.0, 1.0]) < spread([0.6, 0.0, 0.0]));
    }

    /// The lines must be disjoint in what they read, or the dispersion is not
    /// a plan signal. Gaining a block advances construction and nothing else.
    #[test]
    fn lines_do_not_share_inputs() {
        let mut g = Game::new(5).state;
        g.day = 4;
        let p = PlayerId(0);
        let before: [f32; 3] =
            std::array::from_fn(|i| line_progress(&g, p, Line::ALL[i]));
        g.players[p.idx()].res[Resource::Stone.idx()] += 4;
        let after: [f32; 3] = std::array::from_fn(|i| line_progress(&g, p, Line::ALL[i]));
        assert_eq!(
            before, after,
            "raw blocks must not move any line: they are already priced by `held`"
        );
    }

    /// The age inversion is the phase-dependent strategic fact the temples
    /// carry: the same brown step is a 6-point plan before day 14 and a
    /// 2-point one after.
    #[test]
    fn temple_lines_invert_between_the_ages() {
        let mut g = Game::new(11).state;
        let p = PlayerId(0);
        g.temples[Temple::Brown.idx()][p.idx()] = 4;
        g.temples[Temple::Yellow.idx()][p.idx()] = 4;
        g.day = 5;
        assert_eq!(next_scoring_age(&g), Some(1));
        let (b1, y1) = (
            temple_share(&g, p, Temple::Brown, 1),
            temple_share(&g, p, Temple::Yellow, 1),
        );
        g.day = 18;
        assert_eq!(next_scoring_age(&g), Some(2));
        let (b2, y2) = (
            temple_share(&g, p, Temple::Brown, 2),
            temple_share(&g, p, Temple::Yellow, 2),
        );
        assert!(b1 > y1, "brown pays 6 and yellow 2 at the day-14 payout");
        assert!(y2 > b2, "and the other way round at day 27");
    }

    /// Reachability is arithmetic, not search: a worker one space below a
    /// target is one day away, and the last day of the calendar reaches
    /// nothing.
    #[test]
    fn reach_respects_the_calendar() {
        let g = Game::new(3).state;
        // Nothing is placed at setup, so every reach comes from a worker in
        // hand: one turn to place, then `min_pos` rotations.
        assert_eq!(days_to_reach(&g, PlayerId(0), Gear::Tikal, 3), Some(4));
        let mut end = g;
        end.day = LAST_DAY;
        assert_eq!(days_to_reach(&end, PlayerId(0), Gear::Tikal, 3), None);
    }
}
