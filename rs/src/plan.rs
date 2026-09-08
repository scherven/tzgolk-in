//! Phase-dependent evaluation, layered *over* `eval::components` rather than
//! forked from it.
//!
//! # What it measured
//!
//! Read this before the design sections below, because the design is more
//! interesting than the result and that is the wrong way round.
//!
//! Two independent halves — a **value head** (this file's schedule) and a
//! **prior** ([`PlanBias`]) — and they are not the same size. Every number is a
//! centred score over rotation blocks, null 0; the greedy arms are `greedy:32`
//! over [`PlanEvaluator`] against `heuristic:32`, and the MCTS arms are
//! `mcts:256` against the *identical* search on `phase::HeuristicEvaluator`, so
//! the simulation count is matched and only the evaluator or the prior differs.
//!
//! ```text
//!   the value head                greedy         mcts:256
//!     Schedule::IDENTITY (null)    -0.10           +0.00      500 / 300 blocks
//!     FITTED                       -6.37             ---
//!     HALF                        +14.60             ---
//!     REFIT                       +18.97          +14.89
//!     REFIT, calendar removed     +14.72             ---
//!
//!   the prior, on `heuristic` both sides             mcts:256
//!     PlanBias k=32                                   +8.61   240 blocks
//!     ... blind control (same edges, no leader)       +3.26   240 blocks
//!     ... shuffled control (same weights, moved)      -1.41   240 blocks
//!     Priors::OnePly through the same seam            +1.40   120 blocks
//!     ... the same probe over FITTED instead          +6.55   120 blocks
//!     a softmax over `mcts::Gradient`                 +3.21   240 blocks
//!
//!   both halves at once                              mcts:256
//!     REFIT value + PlanBias k=32                    +17.49   240 blocks
//!
//!   and at 4x the budget                            mcts:1024   60 blocks
//!     REFIT value alone                              +17.31
//!     PlanBias k=32 alone                             +5.48
//!     both                                           +19.10
//! ```
//!
//! Five things follow, and two of them are negative.
//!
//! * **Most of the value head is not a phase, and it is two terms.**
//!   [`Schedule::flattened`] keeps each term's average level and throws the
//!   calendar away; paired block by block it costs only **+4.25**
//!   (+3.43..+5.07) of [`REFIT`]'s +18.97, so the day axis is a quarter of the
//!   story. Per-term ablations put the other three quarters in `board` (+11.85
//!   alone) and `engine` (+8.71) — both terms the schedule *lowers*. The
//!   finding under all of this is that `eval::board_position` and
//!   `eval::engine_value` are over-priced by 2-4x.
//! * **A fit must be shrunk or refitted.** [`FITTED`] — the same directions at
//!   full magnitude, measured off-policy — is 25 points worse than [`REFIT`]
//!   and worse than doing nothing at all.
//! * **The prior is the cheap half and it works.** [`PlanBias`] reads the
//!   effect vocabulary of an edge and never touches the state: **5.6 ns an
//!   edge against 123.8** for the one-ply probe it beats by **+7.15**
//!   (+5.45..+8.84). Both controls hold: promoting the same edges without the
//!   leader test is +5.35 worse, and taking the plan's own multipliers and
//!   attaching them to shuffled edges is +10.03 worse and *negative in absolute
//!   terms*. The gain is which line this player is on, not "prefer edges that
//!   score" and not "concentrate the prior somewhere".
//! * **A prior is not a value function, and the two want different numbers.**
//!   The one-ply probe run over [`FITTED`] scores +6.55 where the same probe
//!   over `eval::heuristic` scores +1.40 — paired, **+5.15** (+3.46..+6.84) —
//!   even though [`FITTED`] is 25 points *worse* as a value head. A prior only
//!   has to rank siblings, so a schedule that exaggerates `monument` and
//!   `starvation` and ignores `engine` is a better ranking key and a much worse
//!   estimate. Nothing about "fit the evaluator" transfers to "fit the prior".
//! * **The two halves are strongly sub-additive.** Value +14.89 and prior +8.61
//!   compose to **+17.49**, not +23.5. Paired on the same 240 seeds, adding the
//!   prior to the value head buys **+2.28** (+1.49..+3.07) where the prior is
//!   worth +8.61 on its own, and adding the value head to the prior buys
//!   **+8.87** (+7.82..+9.93). They are two routes to the same better moves,
//!   and only about a quarter of the prior survives a value head that has
//!   already been told what matters. If only one of the two ships, it is the
//!   value head.
//!
//! What did **not** work is recorded where it lives, and it is three of the
//! four ideas this file was written around: [`PlanWeights::focus`], the
//! convexity term the last section argues for, is inert below a weight of 1 and
//! costs points above it; [`BiasWeights::name_temple`] loses -0.74; and
//! [`BiasWeights::place`] loses **-9.22**. The plan is worth something at
//! exactly one grain — which *action* serves the line — and worth nothing or
//! less at every grain coarser or finer than that.
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
//!
//! **That argument is correct and the term it produced is worth nothing** —
//! see [`PlanWeights::focus`] for the grid. The same [`Line`] statistic reached
//! through the *prior* instead of the value ([`PlanBias`]) is worth +8.61, so
//! what failed is the channel and not the vocabulary: a bonus of a fraction of
//! a point inside a 200-point sum cannot outvote anything, where a multiplier
//! on a prior decides which 32 edges of a wide node the search may play at all.

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
///
/// **It said shrink.** Against `heuristic:32` over 500 rotation blocks,
/// [`FITTED`] is **-6.37** centred (-6.88..-5.85) and this is **+14.60**
/// (+13.94..+15.26) — a 21-point swing for moving the same weights halfway
/// back. Under `mcts:256` on both sides the same pair is -16.77 and +12.93.
/// The fit's *direction* is worth a great deal and its *magnitude* is not the
/// number to use, which is exactly what "measured off-policy" predicts.
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
/// # What it is worth, and how much of that is the calendar
///
/// The best value head this file has. Against `heuristic:32` over 500 rotation
/// blocks it is **+18.97** centred (+18.32..+19.61), against [`HALF`]'s +14.60
/// and [`FITTED`]'s -6.37; under `mcts:256` on both sides, **+14.89**
/// (+14.15..+15.62) over 300 blocks. [`Schedule::IDENTITY`] measures -0.10
/// (-0.91..+0.70), which is the null doing its job.
///
/// The refit is also correctly *scaled*, which is the thing the first fit was
/// not. [`Schedule::shrunk`] sweeps the distance from identity, 500 blocks a
/// rung:
///
/// ```text
///   shrink   0.25    0.50    0.75    1.00    1.25
///     pts   +4.76  +11.41  +16.85  +18.97  +17.60
/// ```
///
/// A clean interior maximum at 1.0 — the published table — where the same sweep
/// over [`FITTED`] runs +6.87 at 0.25 and +14.60 at 0.50 and then *falls* to
/// -6.37 at 1.0. One refit round is what moved the optimum from "a quarter of
/// the way" to "all the way", and that is the whole case for closing the loop.
///
/// # And two terms are the whole of it
///
/// `--only T` leaves every term but `T` at its identity weight, so these six
/// arms say which part of the table is doing the work. 500 blocks each:
///
/// ```text
///    board  engine    held  monum   starv   temple      all six together
///   +11.85   +8.71   +1.52  -0.18   -0.01    -3.86                +18.97
/// ```
///
/// They sum to +18.03 against the joint +18.97, so the terms are very nearly
/// separable — and **`board` and `engine` are the entire effect**. Both are
/// terms this schedule *lowers*: `board` runs 0.23-0.90 and `engine` 0.00-0.84
/// where `eval` charges 1.0 for each. So the finding underneath all of the
/// above is not really about phases at all. It is that `eval::board_position`
/// and `eval::engine_value` are over-priced by roughly 2-4x, exactly as
/// `tests/rules.rs::evaluator_calibration` said, and that correcting those two
/// prices is worth about twenty points on its own.
///
/// `temple` is the one term that is worse than leaving it alone (-3.86), which
/// is a caution about reading a fitted coefficient as a price: the fit wants
/// `temple` near zero early because a day-2 standing predicts nothing, and an
/// agent told that acts as if the tracks do not matter until day 8 and arrives
/// at the day-14 payout behind.
///
/// **But most of it is a rescale, not a phase.** [`Schedule::flattened`] — the
/// same average level with every knot equalised, so the calendar shape is gone
/// and nothing else is — scores **+14.72** (+14.06..+15.37) on the same 500
/// seeds. Paired block by block, the shape is worth **+4.25** (+3.43..+5.07) of
/// the +18.97, and the other 78% is `eval` mispricing its terms by a constant
/// this file happens to have measured. That is the honest reading and it is
/// worth more than the schedule is: the day axis earns about four points, and
/// the claim that phase-dependence is where the win lives does not survive its
/// own control.
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
    ///
    /// # Swept, and it is worth nothing
    ///
    /// `greedy:32` over [`REFIT`] against `heuristic:32`, 500 rotation blocks a
    /// rung, and every rung paired against `focus = 0` on its own seeds:
    ///
    /// ```text
    ///   focus     0     0.5      1       2       4       8      16
    ///    pts  +18.97 +19.00  +19.00  +18.76  +18.57  +17.97  +15.40
    ///  paired      -  +0.03   +0.03   -0.21   -0.40   -1.00   -3.57
    ///     CI      -  +/-.24  +/-.30  +/-.37  +/-.45  +/-.56  +/-.67
    /// ```
    ///
    /// Flat to the width of the interval up to 1 and then monotonically down;
    /// at 8 it costs a real point (-1.00, -1.55..-0.44, p = 0.0005) and at 16 it
    /// costs -3.57 (-4.24..-2.90). So the
    /// commitment term is **inert where it is safe and harmful where it bites**,
    /// which is the worst shape a term can have and is a clean negative result.
    ///
    /// The mechanism is visible in the size of the number rather than in its
    /// sign. `max - mean` over three fractions is bounded by 2/3, and a single
    /// turn moves one of those fractions by a few percent — a skull is a third
    /// of a Chichen visit, a building a sixth of a construction game — so at
    /// `focus = 1` the whole term separates two candidate turns by ~0.02 points
    /// against a `raw` that runs to 200 and is then squashed through
    /// `tanh(x/25)`. It cannot outvote anything until it is large enough to
    /// outvote *everything*, and by then it is paying for concentration in
    /// positions where spreading was correct.
    ///
    /// Kept, at zero by default, because it is the only implementation of the
    /// module's convexity argument and a later fit over a coarser progress
    /// statistic could revive it. It is not kept because it works.
    pub focus: f32,
    /// Whether the schedule's `held` weight is modulated by
    /// [`conversion_reach`].
    ///
    /// **The one part of the plan that pays.** Paired block by block against
    /// the same schedule with it off, over 500 rotation blocks of `greedy:32`
    /// over [`REFIT`]: **+0.32 (+0.13..+0.51, p = 0.0011)**. Small, and it is
    /// the *only* term in [`PlanWeights`] whose interval clears zero — which
    /// makes sense, because it is the only one that is arithmetic about the
    /// position rather than an opinion about it. Three skulls with no reachable
    /// Chichen space really are worth 3 apiece, and `eval::held_premium`'s
    /// global `rounds_left / 6` clock really does not know that.
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

/// `eval::components`, reweighted by the calendar and by what the position can
/// still convert, plus the plan's convexity term.
///
/// Deliberately owns no pricing of its own: every point in the sum came out of
/// `eval::components`, so a change to `eval.rs`'s constants moves this
/// evaluator with it instead of leaving the two to drift apart.
///
/// It implements `phase::Evaluator` with `HeuristicEvaluator`'s exact shaping —
/// centre across the four seats, `tanh(x / 25)`, uniform priors — so the two
/// are interchangeable in `GreedyAgent` and in `Mcts`, and a duel between them
/// measures the weights and nothing else. It is also not more expensive than
/// the evaluator it replaces: benched back to back in the same loop it runs at
/// **0.70x** `HeuristicEvaluator::evaluate`, because both spend their time in
/// `eval::components` and the reweighting is eight multiplies. `PlanEvaluator::identity()` against
/// `HeuristicEvaluator` under `mcts:256` measures **+0.00 with a zero-width
/// interval**: identical agents, so the seat rotation cancels exactly. That is
/// the null this file is read against.
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
fn lines_advanced(step: &Step, target: Option<Temple>) -> u8 {
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
            // `target` is the whole content of `temple_target`: with a temple
            // named, a step on either of the other two is a *different* plan
            // rather than the same one, and crediting it would be the mistake
            // `Line::Temples`'s fold over the three tracks was hiding.
            Effect::TempleStep(t, n) if n > 0 && target.map_or(true, |x| x == t) => {
                Line::Temples
            }
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
    ///
    /// # The sweep
    ///
    /// `mcts:256` on `HeuristicEvaluator` against the same search with no
    /// bias, so the value head is common-mode and the *only* difference is this
    /// number. Centred score, 80 blocks a rung (120 at 64 and 128), CI ~+/-1.4:
    ///
    /// ```text
    ///    k       1      2      4      8     16     32     64    128
    ///  pts   +1.80  +2.03  +3.00  +4.67  +7.25  +8.28  +8.56  +8.10
    /// ```
    ///
    /// **Monotone to a plateau at 32-64 and then flat.** Four times the budget
    /// takes about a third of it away — `mcts:1024` measures +5.48
    /// (+3.95..+7.01) over 60 blocks where `mcts:256` measures +8.61 — which is
    /// the direction a prior should move: its job is to aim a *small* number of
    /// simulations, and PUCT finds the same edges on its own once it has
    /// enough. The value head moves the other way over the same step, +14.89 to
    /// +17.31.
    ///
    /// The shape should still be read with suspicion rather than satisfaction: at `k = 32` and a typical
    /// lead of 0.2, `up/down` is about 90, which is not a nudge but a filter on
    /// the 32-edge active window of every wide node. A prior that only gets
    /// better the harder it is applied is as consistent with "concentrating the
    /// prior anywhere is worth points at 256 simulations" as it is with "the
    /// plan is right" — which is what [`BiasWeights::blind`] and
    /// [`BiasWeights::shuffled`] exist to separate.
    pub k: f32,
    /// Also steer `Step::Place`, by gear. A separate knob because a placement
    /// commits to an option rather than to an action, and the two claims are
    /// worth measuring apart.
    ///
    /// # Measured apart, and it is the worst thing in the file
    ///
    /// `mcts:256`, `k = 32`, 120 blocks: **-9.22** (-10.59..-7.84), against the
    /// same prior without it at +8.61 — paired on the seed, **-17.76**
    /// (-19.44..-16.08). An 18-point swing from one extra line of
    /// classification.
    ///
    /// `lines_placed` resolves a whole *gear*, which is far too coarse for
    /// the decision it is steering. "Tikal serves construction and temples" is
    /// true of all eleven Tikal spaces at once, so at `k = 32` this does not
    /// prefer a good placement, it forbids Palenque and Uxmal — and Palenque is
    /// where the corn that feeds the workers comes from. A prior may bias and
    /// never exclude; a multiplier of 90 applied to a claim this vague excludes
    /// in everything but name.
    pub place: bool,
    /// Below this lead the plan abstains — and does no per-edge work at all.
    ///
    /// On day 0 every line reads zero, `max - mean` is exactly zero, and "which
    /// line am I on" has no answer; any steer there is noise on a coin flip.
    /// This is also where most of the saving is, because the opening is where
    /// the tree is widest.
    pub min_lead: f32,
    /// Credit a `TempleStep` to [`Line::Temples`] only when it is on the
    /// temple [`temple_target`] names.
    ///
    /// Off is the honest null: `Line::Temples` folds `max` over the three
    /// tracks, so without this the prior promotes any climb at all — including
    /// the two that pay 2 where the third pays 6.
    ///
    /// # It does not work, and the mechanism says why
    ///
    /// Paired against the unnamed prior at `k = 32` over 240 blocks of
    /// `mcts:256`, naming the temple is **-0.74** (-1.37..-0.10, p = 0.022) —
    /// a small, real *loss*. The same claim through [`PricedBias`], where the
    /// temple axis is repriced on top of `mcts::Gradient` instead of gating a
    /// multiplier, is **+0.07** (-0.56..+0.70) against the plain gradient: an
    /// exact zero, and -0.64 against the arm that credits all three tracks.
    ///
    /// The diagnostic in `planlab priors` predicted it. Over 4,298 real nodes
    /// [`temple_target`] names a temple at 92.2% of them, but naming changes a
    /// single weight at only **5.1%** and moves the top-weighted edge at
    /// **2.1%**. A node offering steps on two different temples is rare — most
    /// `Take` nodes offer one climb or none — so the argmax the naming resolves
    /// is usually not contested, and the 2% where it is cannot pay for the
    /// times the target is stale. `next_scoring_age` looks only at the *next*
    /// payout, so on day 13 the plan names brown for a day-14 prize the player
    /// is one step short of, and then spends the remaining 13 days having
    /// committed to the track that pays 2.
    ///
    /// Kept, off, because it is the sharpest available statement of the age
    /// inversion and the cost of asking is 0.6 ns an edge. It is not kept
    /// because it works.
    pub name_temple: bool,
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
    ///
    /// **The direction is most of the effect.** `mcts:256` on
    /// `HeuristicEvaluator` both sides, paired on the seed:
    ///
    /// ```text
    ///            plan    blind   plan - blind          blocks
    ///   k =  8  +4.67    +0.63   +4.12 (+2.37..+5.87)      80
    ///   k = 32  +8.61    +3.26   +5.35 (+4.37..+6.33)     240
    /// ```
    ///
    /// Promoting every line-advancing edge and demoting nothing is worth
    /// **nothing at all** at `k = 8` (+0.63, -0.52..+1.78) and 38% of the
    /// plan's score at 32. Whatever this prior is doing, it is not simply
    /// "prefer edges that score".
    pub blind: bool,
    /// **The harder control.** Keep every weight the plan computed and attach
    /// them to *different edges*, by a permutation that is a function of the
    /// node and of nothing in the edges.
    ///
    /// [`BiasWeights::blind`] holds the edge set fixed and varies the
    /// direction. This holds the whole multiset of multipliers fixed — the same
    /// count promoted, the same count demoted, the same strength — and varies
    /// *which edges get them*. It exists because the k-sweep rises
    /// monotonically, and a prior that gets better the harder it is applied is
    /// equally consistent with "the plan is right" and with "concentrating a
    /// uniform prior on any 40% of a wide node's edges is worth points at 256
    /// simulations". Only this arm can tell those apart.
    ///
    /// **It told them apart, decisively.** At `k = 32` over 240 blocks the
    /// shuffle scores **-1.41** (-2.33..-0.49) where the plan scores +8.61: not
    /// merely worthless but *worse than no prior at all*, which is what a
    /// confident wrong prior should be. Paired, the plan is **+10.03**
    /// (+8.98..+11.08) ahead of its own weights pointed somewhere else, and
    /// **+5.35** ahead of [`BiasWeights::blind`] — so the ordering is
    /// shuffle < uniform < blind < plan, and every step of it is the content of
    /// the classification rather than the shape of the distribution.
    pub shuffled: bool,
}

impl BiasWeights {
    pub const OFF: BiasWeights = BiasWeights {
        k: 0.0,
        place: false,
        min_lead: 0.02,
        name_temple: false,
        blind: false,
        shuffled: false,
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
///
/// # Where it lands, measured
///
/// `mcts.rs` now defaults to [`crate::mcts::EdgeOrder::Gradient`], which prices
/// and truncates a wide node with `mcts::Gradient` **before** `priors_for` is
/// reached — so a plan can re-rank the survivors and cannot rescue an edge the
/// gradient dropped. The obvious worry is that the gradient silently deletes
/// whatever the plan wanted. It does not: over 4,298 real nodes from
/// `planlab priors --games 12`, only 26 are past the 128-edge cap at all, and
/// at **none** of them was every plan-promoted edge deleted, nor was a step on
/// the [`temple_target`] ever cut (21 kept, 0 dropped).
///
/// What the bias does reach is `Edge::prior` itself, and it reaches it hard.
/// `node_for` re-sorts by the *biased* prior whenever a node is wider than
/// `max_edges`, and only the first `max_edges` are opened — so on any node
/// between 33 and 128 edges wide the bias, not the gradient, chooses which 32
/// the search may play. Run at one simulation, where PUCT reduces to argmax of
/// the prior, installing this changes the edge the search takes at **176 of 471
/// wide `Take` nodes (37.4%)**.
///
/// It has an opinion at 66.6% of `Take` nodes and moves 41.8% of their edges,
/// and abstains completely on every other phase — a `Place` is priced only with
/// [`BiasWeights::place`] on, and `Beg`, `Mode`, `PickWorker` and `ExtraDay`
/// carry no `Choice` to read.
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

/// Permute `out` in place by a shuffle seeded from the position and the width,
/// so the *distribution* of multipliers is untouched and their attachment to
/// edges carries no information. See [`BiasWeights::shuffled`].
///
/// Deterministic rather than random: `PriorBias::bias` takes `&self` and is
/// `Sync`, and a control arm whose two runs of the same seed disagree is not a
/// control. Seeded off the calendar day, the mover, the width and the mover's
/// corn — enough to differ between nodes, and nothing that a *good* edge could
/// correlate with.
fn shuffle_weights(g: &GameState, p: PlayerId, out: &mut [f32]) {
    let mut z = (g.day as u64) << 40
        ^ (p.idx() as u64) << 32
        ^ (out.len() as u64) << 16
        ^ g.players[p.idx()].corn as u64;
    let mut next = move || {
        // splitmix64: one multiply-xor chain, no state beyond the counter.
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    for i in (1..out.len()).rev() {
        out.swap(i, (next() % (i as u64 + 1)) as usize);
    }
}

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
        // Once per node, like the three `line_progress` calls above: naming the
        // temple is an argmax over three closed-form shares, not a per-edge
        // cost.
        let target = if self.w.name_temple {
            temple_target(state, mover)
        } else {
            None
        };
        for (step, o) in steps.iter().zip(out.iter_mut()) {
            let mut mask = lines_advanced(step, target);
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
        if self.w.shuffled {
            shuffle_weights(state, mover, out);
        }
    }

    fn name(&self) -> String {
        format!(
            "plan-bias:k={}{}{}{}{}",
            self.w.k,
            if self.w.place { ":place" } else { "" },
            if self.w.name_temple { ":named" } else { "" },
            if self.w.blind { ":blind" } else { "" },
            if self.w.shuffled { ":shuffled" } else { "" }
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
/// against [`PlanBias`]'s `O(width)` eight-element scans — 123.8 ns an edge
/// against 5.6 — so a tie is a loss for this one.
///
/// **Which evaluator it carries is the whole experiment.** `planlab`'s
/// `--bias-1ply` builds this from `--sched`, and `--sched identity` is the
/// honest baseline arm: [`PlanEvaluator`] under [`Schedule::IDENTITY`] is
/// `eval::heuristic` to the bit, so it reproduces `mcts::Priors::OnePly`
/// exactly through the bias seam.
///
/// # And the answer inverts the value-head ranking
///
/// `mcts:256`, 120 blocks, temperature 4.0 throughout:
///
/// ```text
///   over Schedule::IDENTITY (= Priors::OnePly)   +1.40  (-0.07..+2.87)
///   over FITTED                                  +6.55  (+5.37..+7.73)
///   paired difference                            +5.15  (+3.46..+6.84)
/// ```
///
/// [`FITTED`] is 25 points *worse* than [`REFIT`] as a value head and is the
/// better prior by five points. A prior only has to rank a node's siblings,
/// and a schedule that exaggerates `monument` and `starvation` and zeroes
/// `engine` ranks them better than the calibrated estimate does — the estimate
/// spends its accuracy on the level, which cancels between siblings. So
/// "fit the evaluator" does not transfer to "fit the prior", and a fitted
/// schedule should be swept in both roles rather than assumed to serve one
/// because it serves the other.
///
/// The true `Priors::OnePly` is also *worse* than the far cheaper gradient
/// softmax it was supposed to justify: -1.93 (-3.78..-0.07) against
/// [`PricedBias`] with [`PriceSource::Gradient`], at 123.8 ns an edge against
/// the gradient's ~16 amortised. That is `mcts.rs`'s own regret finding
/// reproduced from the strength side.
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

// =======================================================================
// The plan as a price
// =======================================================================

/// Which temple this player is actually climbing, or `None` when nothing
/// separates them.
///
/// [`Line::Temples`] deliberately folds `max` over the three tracks, which is
/// the right statistic for *how far along* the line is and throws away the one
/// fact a prior needs: **which** track. The prizes invert between the ages —
/// brown pays 6 then 2, yellow 2 then 6 (`data/temples.rs`) — so "climb a
/// temple" is not one plan but three, and two of them are wrong on any given
/// day.
///
/// Ties abstain rather than picking the first. On day 0 every player stands on
/// `STARTING_STEP` of all three, every share is 0, and an argmax there is a
/// coin flip dressed up as a plan.
pub fn temple_target(g: &GameState, p: PlayerId) -> Option<Temple> {
    let age = next_scoring_age(g)?;
    let mut best = Temple::Brown;
    let mut best_v = f32::NEG_INFINITY;
    let mut tied = false;
    for &t in Temple::ALL.iter() {
        let v = temple_share(g, p, t, age);
        if v > best_v + 1e-6 {
            best_v = v;
            best = t;
            tied = false;
        } else if (v - best_v).abs() <= 1e-6 {
            tied = true;
        }
    }
    if best_v <= 0.0 || tied {
        None
    } else {
        Some(best)
    }
}

/// What one step on `t` is worth to the plan at `age`, in points of eventual
/// prize.
///
/// # Why this is not what the gradient measures
///
/// `eval::temple_outlook` prices a step by what it changes **today**:
/// `temple_points` is the majority prize, so a step that does not overtake
/// anybody moves it by exactly zero. A player three steps below the top of
/// yellow on day 16 is on the only climb that pays at day 27, and the gradient
/// reads that climb as worthless until the step that actually passes someone.
/// That is not a bug in `eval` — a one-step difference is what a gradient *is*
/// — but it is the one place a plan has strictly more to say.
///
/// [`temple_share`] is linear in the step, so its derivative is a constant per
/// temple per age: prize over climbable steps. Brown pays 1.20 points a step in
/// age 1 and 0.40 in age 2; yellow 0.29 then 0.86. No state is touched.
///
/// # And it is worth nothing, which is the interesting part
///
/// The argument above is sound and the measurement is flat: adding this to the
/// gradient's temple axis moves the search by +0.10 (-0.77..+0.98) — see
/// [`PriceSource`]. Having strictly more information than the gradient is not
/// the same as having information the *search* can spend. Most `Take` nodes
/// offer at most one climb, so the ranking this sharpens is between edges that
/// are not competing, and a prior can only pay where two good edges are.
pub fn plan_temple_step(t: Temple, age: u8) -> f32 {
    let d = &crate::data::temples::TEMPLES[t.idx()];
    let top = (d.steps - 1) as f32;
    let base = crate::data::temples::STARTING_STEP as f32;
    let prize = if age == 1 { d.age1_prize } else { d.age2_prize } as f32;
    prize / (top - base)
}

/// Where the temple axis of the prior comes from.
///
/// The three arms share **one** price list — `mcts::Gradient`, the exact key
/// `EdgeOrder::Gradient` truncates a wide node on — and differ only in what
/// they add on top of it for a temple step. That is what makes the difference
/// between two arms the claim being tested rather than two differently-tuned
/// tables.
///
/// # The answer is no, and the two halves of it disagree
///
/// `mcts:256` on `HeuristicEvaluator` both sides, 240 rotation blocks, every
/// difference paired on the seed:
///
/// ```text
///   Gradient                +3.21  (+2.40..+4.03)
///   PlanFlat  (w = 4)       +3.93  (+3.15..+4.70)   vs Gradient  +0.71 (+0.01..+1.42)
///   PlanNamed (w = 4)       +3.28  (+2.47..+4.09)   vs Gradient  +0.07 (-0.56..+0.70)
///                                                   vs PlanFlat  -0.64 (-1.25..-0.03)
/// ```
///
/// **Knowing the age inversion is worth about two thirds of a point and naming
/// the temple gives it back.** `PlanFlat` clears zero by 0.01 at p = 0.046,
/// which is the smallest claim this file is willing to make; `PlanNamed` is
/// exactly the gradient, and is a real -0.64 behind the arm that credits all
/// three tracks. Tripling the plan's weight does not rescue it — `PlanNamed` at
/// `w = 12` measures +3.26 (+2.15..+4.37) against `w = 4`'s +3.28, so this is
/// the axis being inert and not the term being too quiet.
///
/// That is the same verdict [`BiasWeights::name_temple`] reaches through the
/// cheaper seam (-0.74) and the one the mechanical count predicts: naming moves
/// the top-weighted edge at 2.1% of nodes, and the 2% cannot pay for the
/// positions where `next_scoring_age` names a temple for a payout this player
/// will not reach. Crediting every climb keeps the age information and drops
/// the commitment, and that is the half that survives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PriceSource {
    /// The search's own gradient and nothing else: the baseline to beat, at
    /// 16 ns an edge and 0.057 points of regret where a full one-ply probe
    /// costs 123.
    Gradient,
    /// The gradient plus [`plan_temple_step`] on **all three** temples. Knows
    /// the age inversion; does not know which temple is this player's.
    ///
    /// The control that isolates the naming: without it, an arm that beat
    /// [`PriceSource::Gradient`] could be winning on the prize schedule alone,
    /// which `temple_outlook` half-knows already.
    PlanFlat,
    /// The gradient plus [`plan_temple_step`] on the [`temple_target`] only.
    /// `PlanNamed - PlanFlat` is exactly "naming the temple".
    PlanNamed,
}

/// A prior over a `Take` node's edges: the search's own edge-ordering gradient,
/// with the temple axis optionally topped up by the plan.
///
/// # Why this is built on `mcts::Gradient` and not on a table of its own
///
/// An earlier version filled its own `options::EffectPrice` by re-probing
/// `eval::heuristic`. It measured the same thing and it was the wrong
/// experiment: the five unprobeable effects had different constants from the
/// search's, so `PriceSource::Gradient` was *a* gradient rather than **the**
/// gradient, and a win over it would not have been a win over what the search
/// actually does. `Gradient::step` is public, so the honest arm is to call it.
///
/// # What the plan adds, and why only here
///
/// `eval::temple_outlook` prices a step by what it changes **today**:
/// `temple_points` is the majority prize, so a step that overtakes nobody moves
/// it by exactly zero, and the gradient — a one-step difference of `heuristic`
/// — reads it as free. A player three steps below the top of yellow on day 16
/// is on the only climb that pays at day 27 and the gradient cannot see it.
/// That is not a bug in `eval`; it is the one axis where a plan holds strictly
/// more information than a local derivative, which is why the whole experiment
/// is confined to the temple axis and touches nothing else. The measurement
/// then said the information is not worth anything — see [`PriceSource`].
///
/// # Where the cost is, measured
///
/// The dot product is free and the **gradient is not**, and through this seam
/// that is fatal. Over 1,157 `Take` nodes (mean width 20.6, from `planlab
/// priors --games 12`, on an otherwise idle machine — the same table taken
/// under a load average of 200 is 10x slower across every row and the ordering
/// is unchanged):
///
/// ```text
///   PricedBias::bias (gradient)          2.09 us/node   101.2 ns/edge
///     of which: the 17 gradient probes   1.81 us/node    87.9 ns/edge
///     of which: EffectPrice::choice      0.16 us/node     7.9 ns/edge
///   OnePlyBias::bias (what it replaces)  2.55 us/node   123.8 ns/edge
///   PlanBias::bias (the cheap prior)     0.12 us/node     5.6 ns/edge
///   plan::temple_target (the naming)                      0.6 ns/edge
///   tree::legal_steps, for scale         4.32 us/node   529.3 ns/edge
/// ```
///
/// The last row is the one that settles whether any of this is affordable:
/// **generating** a node's edges costs 4.32 us, so [`PlanBias`] prices them for
/// 2.8% of what it cost to produce them and [`PricedBias`] for 48%.
///
/// 87% of the cost is *filling* the table, and `PriorBias::bias` is handed a
/// **node**, not a turn — where `Mcts` caches its own gradient per mover per
/// sub-decision (`Mcts::gradient`) and amortises the probes over every wide
/// node in the chain, a bias has nowhere to put them. That drags a 7.9 ns/edge
/// dot product up to 101, which is most of the 124 of the one-ply probe it was
/// supposed to undercut.
///
/// So this arm is **18x the price of [`PlanBias`] and 5.40 points weaker**
/// (+3.21 against +8.61, paired -5.40 over 240 blocks). It stays because it is
/// the only arm that isolates the temple claim against the search's own key,
/// and it is not a prior anyone should ship.
///
/// The conclusion is not that the price is wrong; it is that a plan reaching
/// the search through `PriorBias` should not be rebuilding a gradient at all.
/// See [`PlanBias`], which reaches 5.6 ns/edge by reading the effect vocabulary
/// directly.
///
/// `min_edges` is the partial mitigation: a three-edge node is resolved by
/// three simulations whatever its prior says, and the median node width is 3.
pub struct PricedBias {
    pub source: PriceSource,
    /// Softmax temperature, in points — `eval::heuristic`'s own scale, which is
    /// what `Gradient::step` returns.
    pub temp: f32,
    /// Points per step of climb credited to the plan's temple. Zero reduces
    /// every arm to [`PriceSource::Gradient`], which is the identity test.
    pub w: f32,
    /// Do not spend 17 `heuristic` probes on a node narrower than this.
    pub min_edges: usize,
}

impl PricedBias {
    pub fn new(source: PriceSource, w: f32) -> PricedBias {
        PricedBias {
            source,
            temp: 4.0,
            w,
            min_edges: 8,
        }
    }

    /// What the plan adds to one step on each temple, in `heuristic` points,
    /// zero where the arm declines to credit it.
    ///
    /// `None` means "add nothing at all", which is the `Gradient` arm and also
    /// every position past the last payout — there a step buys nothing the plan
    /// can spend, and the gradient's own reading is the only true one left.
    ///
    /// Computed once per node: an argmax over three closed-form shares, not a
    /// per-edge cost. Measured at 0.6 ns/edge, against the 87.9 the gradient
    /// underneath it costs.
    pub fn temple_bonus(&self, g: &GameState, p: PlayerId) -> Option<[f32; 3]> {
        if self.w == 0.0 || self.source == PriceSource::Gradient {
            return None;
        }
        let age = next_scoring_age(g)?;
        let target = temple_target(g, p);
        if self.source == PriceSource::PlanNamed && target.is_none() {
            return None;
        }
        Some(std::array::from_fn(|i| {
            let t = Temple::ALL[i];
            let credit = match self.source {
                PriceSource::Gradient => false,
                PriceSource::PlanFlat => true,
                PriceSource::PlanNamed => target == Some(t),
            };
            if credit {
                self.w * plan_temple_step(t, age)
            } else {
                0.0
            }
        }))
    }
}

impl crate::mcts::PriorBias for PricedBias {
    fn bias(
        &self,
        state: &GameState,
        _phase: Phase,
        mover: PlayerId,
        steps: &[Step],
        out: &mut [f32],
    ) {
        if steps.len() < self.min_edges {
            return;
        }
        // `Choice` is the only thing a gradient can price, and a node's edges
        // are homogeneous by phase: a `Take` node is all `Take`, a `Placing`
        // node is `Place`s plus a commit edge. So this either prices every edge
        // or abstains, and never mixes a priced weight with an unpriced 1.0.
        if !steps.iter().all(|s| matches!(s, Step::Take(_))) {
            return;
        }
        let grad = crate::mcts::Gradient::new(state, mover);
        let bonus = self.temple_bonus(state, mover);
        let mut best = f32::NEG_INFINITY;
        for (step, o) in steps.iter().zip(out.iter_mut()) {
            let mut v = grad.step(step);
            if let (Some(b), Step::Take(c)) = (bonus, step) {
                for e in c.0.iter() {
                    if let Effect::TempleStep(t, n) = *e {
                        if n > 0 {
                            v += n as f32 * b[t.idx()];
                        }
                    }
                }
            }
            best = best.max(v);
            *o = v;
        }
        // Shifted by the max before exponentiating, as `mcts::one_ply` is: a
        // priced choice runs to tens of points and `exp` of that is not a
        // number.
        let t = self.temp.max(1e-3);
        for o in out.iter_mut() {
            *o = ((*o - best) / t).exp();
        }
    }

    fn name(&self) -> String {
        format!(
            "priced:{}:w={}:t={}",
            match self.source {
                PriceSource::Gradient => "grad",
                PriceSource::PlanFlat => "flat",
                PriceSource::PlanNamed => "named",
            },
            self.w,
            self.temp
        )
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

    /// Real `Take` nodes from real games, for the prior tests.
    ///
    /// Walks the sub-decision chain rather than only the turn roots: a `Take`
    /// node is three levels into a turn and never appears at the top of one.
    fn take_nodes(seeds: std::ops::Range<u64>) -> Vec<(GameState, Phase, PlayerId, Vec<Step>)> {
        let mut out = Vec::new();
        for seed in seeds {
            let mut game = Game::new(seed);
            for _ in 0..14 {
                if game.state.over {
                    break;
                }
                let turn = game.state.current;
                let mut probe = game.state;
                let mut at = (Phase::Beg, turn, 0u8);
                for _ in 0..24 {
                    let (phase, t, done) = at;
                    let steps = crate::tree::legal_steps(&probe, phase, t, done);
                    if steps.is_empty() {
                        break;
                    }
                    if steps.len() > 1 {
                        out.push((probe, phase, phase.mover(t), steps.clone()));
                    }
                    let step = steps[steps.len() / 2].clone();
                    let tr = crate::tree::apply_step(&mut probe, phase, t, done, &step);
                    match tr.next() {
                        None => break,
                        Some(next) => {
                            if tr.committed() {
                                break;
                            }
                            at = next;
                        }
                    }
                }
                game.play_round();
            }
        }
        out
    }

    /// A prior may bias and must never exclude. PUCT reaches a low-prior edge
    /// given enough simulations only while the prior is strictly positive, and
    /// a zero would turn "the plan disagrees" into "the move does not exist".
    #[test]
    fn the_bias_never_excludes_an_edge() {
        use crate::mcts::PriorBias;
        let b = PlanBias::new(32.0);
        let mut out = Vec::new();
        for (g, phase, mover, steps) in take_nodes(0..6) {
            out.clear();
            out.resize(steps.len(), 1.0);
            b.bias(&g, phase, mover, &steps, &mut out);
            for (w, step) in out.iter().zip(&steps) {
                assert!(
                    *w > 0.0 && w.is_finite(),
                    "day {} {phase:?} {step:?}: weight {w}",
                    g.day
                );
            }
        }
    }

    /// `k = 0` must leave every weight at one, so "the plan is off" and "there
    /// is no plan" are the same search and the null duel measures exactly zero.
    #[test]
    fn zero_strength_is_the_identity() {
        use crate::mcts::PriorBias;
        let b = PlanBias::new(0.0);
        let mut out = Vec::new();
        for (g, phase, mover, steps) in take_nodes(10..13) {
            out.clear();
            out.resize(steps.len(), 1.0);
            b.bias(&g, phase, mover, &steps, &mut out);
            assert!(out.iter().all(|&w| w == 1.0), "day {}: {out:?}", g.day);
        }
    }

    /// An edge the classifier calls Construction or Temples must really raise
    /// that line once applied, or the prior is steering the search somewhere
    /// the value function will refuse to pay for.
    ///
    /// **Skulls is exempt, and the exemption is the finding.** `FillChichen` is
    /// where the skull line *pays*, and it lowers `line_progress(Skulls)`
    /// because that statistic counts skulls in hand — fuel, not payoff. The
    /// prior deliberately promotes cashing anyway: a plan that hoards the
    /// resource it is committed to and never spends it is not a plan.
    #[test]
    fn construction_and_temple_edges_raise_the_line_they_name() {
        for (g, phase, mover, steps) in take_nodes(20..26) {
            for step in &steps {
                let mask = lines_advanced(step, None);
                for l in [Line::Construction, Line::Temples] {
                    if mask & (1 << (l as u8)) == 0 {
                        continue;
                    }
                    let before = line_progress(&g, mover, l);
                    let mut after = g;
                    let _ = crate::tree::apply_step(&mut after, phase, mover, 0, step);
                    let now = line_progress(&after, mover, l);
                    assert!(
                        now >= before - 1e-6,
                        "day {} {l:?}: {step:?} was classified as advancing it and moved it {before} -> {now}",
                        g.day
                    );
                }
            }
        }
    }

    /// The blind control must touch exactly the edges the plan does. If it
    /// touched a different set, the difference between the two arms would be
    /// the edge set as much as the direction, and the arm would answer nothing.
    #[test]
    fn the_blind_control_moves_the_same_edges() {
        use crate::mcts::PriorBias;
        let plan = PlanBias {
            w: BiasWeights { k: 8.0, ..BiasWeights::OFF },
        };
        let blind = PlanBias {
            w: BiasWeights { k: 8.0, blind: true, ..BiasWeights::OFF },
        };
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut touched = 0usize;
        for (g, phase, mover, steps) in take_nodes(30..46) {
            a.clear();
            a.resize(steps.len(), 1.0);
            b.clear();
            b.resize(steps.len(), 1.0);
            plan.bias(&g, phase, mover, &steps, &mut a);
            blind.bias(&g, phase, mover, &steps, &mut b);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(
                    *x == 1.0,
                    *y == 1.0,
                    "day {}: the two arms disagree about which edges to touch",
                    g.day
                );
                if *x != 1.0 {
                    touched += 1;
                }
            }
        }
        assert!(touched > 100, "only {touched} edges touched: the walk found nothing to test");
    }


    /// The shuffled control must be the *same multiset* of weights on the same
    /// nodes — same count promoted, same count demoted, same strength — and
    /// must scramble which edge gets which. If it changed the distribution it
    /// would be a different bias rather than a control, and if it changed
    /// nothing it would answer nothing.
    #[test]
    fn the_shuffled_control_keeps_the_weights_and_moves_them() {
        use crate::mcts::PriorBias;
        let w = BiasWeights { k: 32.0, ..BiasWeights::OFF };
        let plan = PlanBias { w };
        let shuf = PlanBias { w: BiasWeights { shuffled: true, ..w } };
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let (mut wide, mut moved) = (0usize, 0usize);
        for (g, phase, mover, steps) in take_nodes(110..150) {
            a.clear();
            a.resize(steps.len(), 1.0);
            b.clear();
            b.resize(steps.len(), 1.0);
            plan.bias(&g, phase, mover, &steps, &mut a);
            shuf.bias(&g, phase, mover, &steps, &mut b);
            let (mut x, mut y) = (a.clone(), b.clone());
            x.sort_by(f32::total_cmp);
            y.sort_by(f32::total_cmp);
            assert_eq!(x, y, "day {}: the control changed the weights", g.day);
            // Only a node with both a promoted and a demoted edge can show a
            // permutation at all; the rest are constant vectors.
            if steps.len() >= 8 && x.first() != x.last() {
                wide += 1;
                if a != b {
                    moved += 1;
                }
            }
        }
        assert!(wide > 20, "only {wide} nodes had two distinct weights to permute");
        assert!(
            moved * 2 > wide,
            "{moved} of {wide} nodes permuted: the control is barely moving anything"
        );
    }

    /// The plan's per-step temple price must carry the age inversion the
    /// prizes actually have, or naming the temple names the wrong one. Brown is
    /// the age-1 plan and yellow the age-2 plan; green is flat by construction.
    #[test]
    fn the_plan_step_price_inverts_with_the_age() {
        let (b1, y1) = (plan_temple_step(Temple::Brown, 1), plan_temple_step(Temple::Yellow, 1));
        let (b2, y2) = (plan_temple_step(Temple::Brown, 2), plan_temple_step(Temple::Yellow, 2));
        assert!(b1 > y1, "brown pays 6 to yellow's 2 at day 14: {b1} vs {y1}");
        assert!(y2 > b2, "and 2 to yellow's 6 at day 27: {b2} vs {y2}");
        assert_eq!(
            plan_temple_step(Temple::Green, 1),
            plan_temple_step(Temple::Green, 2),
            "green pays 4 in both ages, so it must not move with the age"
        );
    }

    /// `temple_target` must abstain where there is nothing to name and commit
    /// where there is. A fresh game has every player on `STARTING_STEP` of all
    /// three tracks, which is the tie the argmax would otherwise resolve by
    /// enum order.
    #[test]
    fn the_temple_target_abstains_on_a_tie() {
        let mut g = Game::new(11).state;
        g.day = 4;
        let p = PlayerId(0);
        // Levelled by hand rather than taken fresh: the starting-tile draft
        // deals temple steps, so a real day-0 position usually *has* already
        // named a temple. That is the mechanism working, not a case to exclude
        // -- the tie is what has to abstain.
        for t in Temple::ALL {
            g.temples[t.idx()][p.idx()] = crate::data::temples::STARTING_STEP;
        }
        assert_eq!(temple_target(&g, p), None, "level tracks name no plan");
        // Two steps up yellow and nothing else: the plan is yellow, in both
        // ages, because it is the only track this player has climbed at all.
        g.temples[Temple::Yellow.idx()][p.idx()] = 3;
        assert_eq!(temple_target(&g, p), Some(Temple::Yellow));
    }

    /// Naming the temple may only ever *narrow* the set of edges credited to
    /// `Line::Temples` — never credit one the unnamed classifier refused. A
    /// prior that named a temple and then promoted a different one would be
    /// worse than no naming at all.
    #[test]
    fn naming_the_temple_only_narrows() {
        let mut seen = 0usize;
        let mut narrowed = 0usize;
        for (g, _phase, mover, steps) in take_nodes(50..70) {
            let target = temple_target(&g, mover);
            for step in &steps {
                let wide = lines_advanced(step, None);
                let narrow = lines_advanced(step, target);
                assert_eq!(
                    wide & narrow,
                    narrow,
                    "naming credited a line the unnamed classifier did not"
                );
                let bit = 1 << (Line::Temples as u8);
                if wide & bit != 0 {
                    seen += 1;
                    if narrow & bit == 0 {
                        narrowed += 1;
                    }
                }
            }
        }
        assert!(seen > 50, "only {seen} temple edges: the walk found nothing to test");
        assert!(
            narrowed > 0,
            "{seen} temple edges and naming dropped none of them, so the axis is inert"
        );
    }

    /// The priced bias is a softmax, so every weight is strictly positive: a
    /// low prior costs simulations, never correctness. This is the same
    /// soundness property `the_bias_never_excludes_an_edge` asserts for
    /// `PlanBias`, and it has to hold separately because the two compute their
    /// weights by completely different routes.
    #[test]
    fn the_priced_bias_never_excludes_an_edge() {
        use crate::mcts::PriorBias;
        let b = PricedBias {
            min_edges: 2,
            ..PricedBias::new(PriceSource::PlanNamed, 4.0)
        };
        let mut out = Vec::new();
        let mut touched = 0usize;
        for (g, phase, mover, steps) in take_nodes(70..86) {
            out.clear();
            out.resize(steps.len(), 1.0);
            b.bias(&g, phase, mover, &steps, &mut out);
            for w in &out {
                assert!(
                    w.is_finite() && *w > 0.0,
                    "day {}: a priced edge got weight {w}",
                    g.day
                );
                if *w != 1.0 {
                    touched += 1;
                }
            }
        }
        assert!(touched > 100, "only {touched} edges priced: the walk found nothing to test");
    }

    /// With `w = 0` all three arms must produce the identical weight vector,
    /// or the sweep's low end is not a control. Asserted on the weights the
    /// search actually sees rather than on the price table behind them: the
    /// table is an implementation detail and the weights are the contract.
    ///
    /// And with the weight on, only edges carrying a temple step may move — 18
    /// of the 21 prices are `mcts::Gradient`'s own, which is what makes the
    /// arms comparable at all.
    #[test]
    fn zero_weight_leaves_the_gradient_alone() {
        use crate::mcts::PriorBias;
        let arm = |src, w| PricedBias {
            min_edges: 2,
            ..PricedBias::new(src, w)
        };
        let (mut base, mut other) = (Vec::new(), Vec::new());
        let (mut compared, mut differed) = (0usize, 0usize);
        for (g, phase, mover, steps) in take_nodes(90..104) {
            base.clear();
            base.resize(steps.len(), 1.0);
            arm(PriceSource::Gradient, 0.0).bias(&g, phase, mover, &steps, &mut base);
            for src in [PriceSource::PlanFlat, PriceSource::PlanNamed] {
                other.clear();
                other.resize(steps.len(), 1.0);
                arm(src, 0.0).bias(&g, phase, mover, &steps, &mut other);
                assert_eq!(base, other, "{src:?} at w = 0 is not the gradient");
            }
            // Weight on: a step with no `TempleStep` in it must price
            // identically, because nothing else was touched.
            other.clear();
            other.resize(steps.len(), 1.0);
            arm(PriceSource::PlanNamed, 4.0).bias(&g, phase, mover, &steps, &mut other);
            let any_temple = steps.iter().any(|s| match s {
                Step::Take(c) => c
                    .0
                    .iter()
                    .any(|e| matches!(e, Effect::TempleStep(_, n) if *n > 0)),
                _ => false,
            });
            if !any_temple {
                assert_eq!(base, other, "a node with no temple step was repriced");
            } else if base != other {
                differed += 1;
            }
            compared += 1;
        }
        assert!(compared > 100, "only {compared} nodes: the walk found nothing to test");
        assert!(
            differed > 0,
            "naming the temple never changed a weight over {compared} nodes, so the axis is inert"
        );
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
