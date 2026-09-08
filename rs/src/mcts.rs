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
//! # Against the alpha-beta
//!
//! Worth knowing before spending another day tuning `search.rs`. On the same
//! positions, `mcts:2048` is **+6.37** centred score head to head against
//! `minimax:8:600000::greedy:capw=25` (95% CI +4.71..+8.02, p = 4.9e-14, **127
//! blocks / 508 games**, solo mode) and takes 35.2% of its games against three
//! of them (CI 30.8..39.7), where an equally strong agent would take 25%.
//!
//! **And it does that on less CPU, not the same.** The two budgets are
//! different kinds of thing — simulations against a depth — so they are matched
//! by measuring: `/usr/bin/time` on `sprobe mcts`, one spec per process, four
//! alternating runs each, puts `mcts:2048` at 2.398 s of user CPU and the
//! alpha-beta at 2.955 s. The alpha-beta gets **1.23x the work** and still
//! loses by six points. Use CPU seconds and not the clock for this: the same
//! two specs measured by wall clock on a loaded machine read 1.52x and 1.03x on
//! two consecutive runs, while the CPU figures repeat to within 3%.
//!
//! With the `quality` preset ([`Priors::OnePly`] at `prior_temp = 1`) the gap
//! is **+7.73** (CI +6.09..+9.36, 142 blocks / 568 games) and it takes **40.8%
//! of its games** against three alpha-betas. That is the ceiling this search
//! currently has against `search.rs`, and it is a prior away rather than a
//! budget away.
//!
//! The number this comment used to carry was **+9.32** from 39 blocks, then
//! **+3.27** from 150. Neither survived. The first was measured on a machine
//! running four other agents; the second on a search whose wide nodes were
//! ordered by `Choice`'s derived `Ord` (see [`EdgeOrder`]) and against an
//! `eval.rs` that has been tuned repeatedly since — which the alpha-beta
//! consumes at *every* leaf and this search only at expansions, so both sides
//! move when it does. The direction has survived three measurements and the
//! size has not: quote the interval, quote the block count, and re-measure
//! before quoting the number.
//!
//! # The simulation budget does nothing — until the prior is real
//!
//! Six head-to-heads, solo mode, `mcts:S` against `mcts:S/2` and one across
//! the whole span, all at temperature zero so nothing is sampled:
//!
//! | candidate | baseline | centred | 95% CI | blocks |
//! |---|---|---|---|---|
//! | `mcts:256`  | `mcts:128`  | -0.68 | -2.18..+0.82 | 150 |
//! | `mcts:512`  | `mcts:256`  | -0.74 | -2.19..+0.71 | 150 |
//! | `mcts:1024` | `mcts:512`  | +0.22 | -1.36..+1.80 | 150 |
//! | `mcts:2048` | `mcts:1024` | +0.22 | -1.46..+1.90 | 145 |
//! | `mcts:4096` | `mcts:2048` | +0.80 | -0.92..+2.52 | 100 |
//! | `mcts:2048` | `mcts:128`  | -1.08 | -2.79..+0.64 | 150 |
//!
//! **Sixteen times the budget is worth nothing.** Not "a gain too small to
//! resolve" — the wide-span interval excludes anything above +0.64. Every
//! `sims` number this project has chosen, `run.sh`'s included, was chosen on an
//! axis that has been measured flat.
//!
//! It is flat because of what a simulation carries, not because search does not
//! help. `HeuristicEvaluator` fills `Evaluation::priors` with `1.0 / n_edges`,
//! so PUCT's exploration term does the entire job of a policy head and the
//! thousandth simulation is spread as thinly as the first. Give the edges a
//! real prior — [`Priors::OnePly`] at `prior_temp = 1`, the `quality` preset —
//! and the same axis comes back to life:
//!
//! | candidate | baseline | centred | 95% CI | blocks |
//! |---|---|---|---|---|
//! | `mcts:2048` | `mcts:128` | -1.08 | -2.79..+0.64 | 150 |
//! | `mcts:2048:quality` | `mcts:256:quality` | +3.04 | +1.46..+4.62 | 150 |
//!
//! Same search, same budget ratio to within a factor of two, opposite answer.
//! The prior is also worth a great deal on its own: at equal simulations
//! `quality` is **+8.84** centred against `mcts:2048` (CI +7.24..+10.44, 202
//! blocks), and at 1360 simulations — **0.83x the CPU** of the 2048-simulation
//! default — still **+7.68** (CI +5.90..+9.46, 119 blocks, p = 2.4e-17).
//!
//! So the order to tune things in is: the prior first, then the budget, and
//! `max_edges` / `widen_cap` / `widen_c` never — all three are measured inert
//! at their own fields below.
//!
//! # Threading
//!
//! Still single-threaded. Virtual loss is applied and removed for real, on the
//! mover's component only, so the mechanism is exercised and the shape is
//! right, but the statistics are plain `u32`/`f32` rather than §3.1's atomics
//! and there is no batching. Swapping in `AtomicU32` and signed fixed-point
//! `AtomicI32` is mechanical; the descent logic does not change. It would buy
//! latency rather than throughput, though: self-play and the arena already fill
//! every core by running whole games in parallel.
//!
//! Where a single descent's time goes, from `sample` over a 20-second window of
//! a running arena (91,387 running samples, 28% of threads parked):
//!
//! | | share | owner |
//! |---|---|---|
//! | `Choice` sort / compare / eq | 24.7% | `options.rs` |
//! | `simulate` (descent + backup) | 16.9% | here |
//! | allocator | 14.6% | mostly `options.rs` |
//! | `eval::heuristic` | 7.3% | `eval.rs` |
//! | memmove / memset | 5.7% | mixed |
//! | `options::` generation | 5.6% | `options.rs` |
//! | `options::dominated_dedup` | 4.9% | `options.rs` |
//! | `search_at` (the prune sweep) | 4.4% | here |
//! | Vec/SmallVec build | 3.9% | mixed |
//! | transposition hash + eq | 3.3% | here |
//!
//! **Three tenths of the search is `options::dominated_dedup` and the
//! lexicographic `Choice` comparison underneath it**, reached from
//! `spaces::choices_at`. Nothing in this file can avoid it — a node generates
//! its edges exactly once, and the transposition index already stops the same
//! node being generated twice. It is the single biggest lever this search has
//! and it is in someone else's file.
//!
use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::options::EffectPrice;
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
    ///
    /// **Measured inert at 32.** Over 2,713,782 node expansions from
    /// `mcts:2048` on real positions (`bin/sprobe nodes`), legal edges per node
    /// run p50 2, p90 6, p99 27, max 441 — so this window is short of the legal
    /// list on **0.78%** of nodes, and progressive widening opens the rest of a
    /// 441-edge node by its 42nd visit. `cap_per_width` was the
    /// alpha-beta's biggest single win because its nodes were 184-644 moves
    /// wide; the analogous cap here has almost nothing to bite on. The wide
    /// nodes are `Take` nodes deep in a descent, not the roots of a turn's
    /// sub-decisions, which are p50 3.
    ///
    /// It is still the window that [`EdgeOrder`] is really about: 0.78% of
    /// nodes is where all of the ordering's effect lives, and it is why that
    /// effect is small.
    pub max_edges: usize,
    /// §2.6's progressive widening, `N >= widen_c * m^widen_alpha` for the
    /// m-th child. **Also measured inert**, and doubly so: a node opens with
    /// `min(edges, max_edges)` already active, so widening only ever governs
    /// the 33rd edge and up — 0.0% of nodes in the same 2.7M-expansion sample
    /// were sitting below the cap waiting on it — and at `2.0 * m^0.5` the
    /// 33rd edge needs 12 visits and the 441st needs 42. It is a soft delay
    /// measured in tens of visits, not a restriction.
    pub widen_c: f32,
    pub widen_alpha: f32,
    /// The absolute ceiling widening may reach, and the width past which edges
    /// are **deleted**.
    ///
    /// # It is not needed, and 128 is not why
    ///
    /// This is the one knob here that loses information: widening can reopen an
    /// edge it has not reached yet and cannot reopen one that is gone. So it
    /// was raced against `nocap` (`usize::MAX`), with [`EdgeOrder::Gradient`]
    /// deciding the order on both sides: **+0.15 centred, CI -1.47..+1.77, 138
    /// blocks / 552 games** — nothing, in either direction. It costs nothing
    /// either: 2.390 s of user CPU against 2.398 for the capped default, which
    /// is the same number.
    ///
    /// The reason is how little it cuts. Over **2,713,782 node expansions** by
    /// `mcts:2048` on 120 positions strided across 40 games (`bin/sprobe
    /// nodes`), legal edges per node run p50 2, p90 6, p99 27, max 441 — and
    /// the cap fires on **1,058 nodes, 0.039%**, deleting 27,412 of 8,870,785
    /// edges: **0.309% of all edge mass**. Memory says the same. `Edge` is 88
    /// bytes and `Node` 400, so the widest node in that sample is 39 kB
    /// uncapped against 11.7 kB capped, and the whole arena for a sub-decision
    /// is ~3,000 nodes.
    ///
    /// Keep it as a ceiling: `phase.rs` measured the widest factored node over
    /// 38,816 turns at **2,293** edges with a p99 of 164, so nodes past 128 do
    /// exist even though a 2048-simulation descent has not been caught
    /// expanding one, and 2,293 edges is 0.2 MB in a single node. But do not
    /// attribute anything to it, and see [`EdgeOrder`] before lowering it: a
    /// truncation is only as good as the key it truncates on.
    pub widen_cap: usize,
    /// How a node wider than `max_edges` decides which edges survive. See
    /// [`EdgeOrder`].
    pub ordering: EdgeOrder,
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
    ///
    /// **Sweep it before using it.** Against `mcts:2048` at equal simulations,
    /// centred score by temperature: 0.5 → +6.42 (22 blocks), **1.0 → +8.84
    /// (CI +7.24..+10.44, 202 blocks)**, 2.0 → +5.56 (28), 4.0 → +1.01 (27),
    /// 8.0 → +1.36 (20). The default of 4.0 throws away seven of the eight
    /// points a one-ply prior is worth: at four points per e-fold the prior is
    /// nearly flat again, which is the state [`Priors::Evaluator`] is already
    /// in. `quality` is the name for 1.0.
    pub prior_temp: f32,
    /// Do not spend a one-ply pass on a node narrower than this.
    ///
    /// The pass costs an `apply_step` and an `eval::heuristic` per edge, so its
    /// cost is linear in width while the *benefit* is not: a two-edge node is
    /// resolved by three simulations whatever its prior says. Widths here run
    /// 2..2293 with a median searched width of 3 (`phase.rs`), so the threshold
    /// is where most of the saving is.
    ///
    /// Set it low. At 3 the prior is worth +8.84 against `mcts:2048`; at 8 it
    /// is worth **+3.46** (CI +2.06..+4.85, 200 blocks). Skipping the probe on
    /// 3-to-7-edge nodes skips it on most of the tree, and most of the tree is
    /// where the strength was.
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
    ///
    /// **This is the largest measured effect in the file**, and the only knob
    /// that moved the search more than a point. At `prior_temp = 1` and
    /// `prior_min_edges = 3` — spelled `quality` — it is +8.84 centred against
    /// `mcts:2048` at equal simulations (CI +7.24..+10.44, 202 blocks / 808
    /// games) and +7.68 at 1360 simulations, which is 0.83x the default's CPU
    /// (CI +5.90..+9.46, 119 blocks). Against `minimax:8:600000::greedy:capw=25`
    /// it is +7.78 (CI +6.11..+9.45, 137 blocks) and takes **40.7% of its
    /// games** against three of them, where an equal agent takes 25%.
    ///
    /// It costs 1.34x the CPU per simulation at 2048, which the flat
    /// budget-vs-strength curve in the module header makes free: buy it by
    /// lowering `sims`, an axis that has been measured worth nothing.
    OnePly,
}

/// How a node too wide for `max_edges` decides which edges are opened first,
/// and — past `widen_cap` — which survive at all.
///
/// # Why this is not just "sort by prior"
///
/// It was, and it was wrong. `sort_by` is stable and `HeuristicEvaluator`
/// returns `1.0 / n_edges` for every edge, so sorting a uniform prior is a
/// no-op: the window opened in **generation order**, which is `Choice`'s
/// derived lexicographic `Ord` — a fact about the declaration order of the
/// `Effect` variants and about nothing whatsoever in the game.
///
/// How much that cost, priced against a one-ply score of every edge
/// (`bin/sprobe trunc`, three position samples of 210 / 1,320 / 2,880):
///
/// | | best outside the opening 32 | mean regret |
/// |---|---|---|
/// | generation order | 65.6% / 38.0% / 34.5% of wide nodes | 0.63 / 0.37 / 0.28 pts |
/// | gradient order | 12.5% / 12.5% / 6.0% | 0.02 / 0.06 / 0.01 pts |
///
/// **Read that as a delay, not a loss.** An earlier version of this comment
/// said the truncation "deleted the one-ply-best edge at 55.6% of wide nodes"
/// and that every number this search had produced was measured with the best
/// move missing. That conflated the two halves of §2.6. The column above is
/// `max_edges`, the *opening* window, and progressive widening reaches the
/// 33rd edge after 12 visits. Actual deletion needs a node past `widen_cap`,
/// and in a real descent that is 0.039% of expanded nodes and 0.309% of edge
/// mass — see [`MctsConfig::widen_cap`].
///
/// Which is why the head-to-head is small. `ord=grad` against `ord=prior`,
/// everything else equal: **+0.40 centred (CI -0.77..+1.57, 252 blocks / 1,008
/// games)**, and an earlier 200-block run of the same pair read +1.03 (CI
/// -0.33..+2.40). Inverse-variance pooled over both, **+0.67 (CI -0.22..+1.56,
/// 452 blocks / 1,808 games)** — under a point and not distinguishable from
/// zero. It is the right default anyway: it costs 1.8% of CPU (2.398 s against
/// 2.355 of user time), it is the only one of the two orders that is a fact
/// about the game rather than about `Effect`'s declaration order, and it is
/// what makes `widen_cap` safe to leave alone. It is just not where the
/// strength is — see [`Priors::OnePly`], which is worth twenty times as much.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeOrder {
    /// Sort on the prior as it stands. Right when the prior is real -- a
    /// trained policy head -- and a no-op when the evaluator abstains, which is
    /// what makes it the wrong default while there is no net.
    Prior,
    /// Price every edge against `eval`'s own local gradient ([`Gradient`]),
    /// keep the best `widen_cap`, and spend the one-ply probe only on those.
    Gradient,
}

/// `eval::heuristic`'s local gradient: a price per unit of everything an
/// `Effect` can hand out, in the points `heuristic` itself returns.
///
/// # Why a gradient and not a table of constants
///
/// A hand-written price list is a second opinion about the value function, and
/// it goes stale every time someone tunes `eval.rs`. This asks `eval` instead
/// -- probe `+1` of each axis against the position once, then price a `Choice`
/// as the dot product of its effects with the result. Linear, so it misses
/// every interaction between the effects of one choice; that is exactly why it
/// is allowed to *order* edges and never to value them.
///
/// Sixteen `heuristic` calls to build, then a handful of nanoseconds per edge
/// to apply, against roughly 8x that for the `apply_step`-and-score one-ply
/// probe — 16.0 against 123.0 ns on a quiet machine, 41.4 against 121.3 on a
/// loaded one, so quote the ratio and not the figures. That is what makes a
/// several-hundred-edge node orderable at all, and it cuts the mean regret
/// below the best edge in the opening window from 0.28-0.63 points to
/// 0.01-0.06 (see [`EdgeOrder`] for the samples).
///
/// It is an approximation and it says so: at the widest node in
/// `tests/search.rs::the_edge_cap_deletes_nothing_the_search_can_reach` — 298
/// edges — the gradient's top 128 misses the one-ply best just as generation
/// order does. Ordering, never valuing.
#[derive(Clone, Copy)]
pub struct Gradient {
    price: EffectPrice,
}

impl Gradient {
    /// Probe the position for one player. Sixteen `+1` perturbations plus the
    /// base, which is ~2 us -- paid once per mover per sub-decision, against
    /// the ~2.6 ms that sub-decision costs.
    pub fn new(g: &GameState, p: PlayerId) -> Gradient {
        let base = crate::eval::heuristic(g, p);
        let pr = |e| {
            let mut probe = *g;
            Choice::one(e).apply(&mut probe, p);
            crate::eval::heuristic(&probe, p) - base
        };
        let points = pr(Effect::Points(1));
        // The card- and space-naming effects cannot be probed without naming a
        // card, and probing each distinct one moves the regret inside
        // `max_edges` by 0.005 points for 26% more per edge. So: constants, on
        // the scale `search.rs` already prices them in, converted into
        // `heuristic` points by the probed price of a point so the two halves
        // of the sum are commensurate.
        let k = |pts: f32| pts * points / 4.0;
        Gradient {
            price: EffectPrice {
                corn: pr(Effect::Corn(1)),
                res: std::array::from_fn(|i| pr(Effect::Res(Resource::ALL[i], 1))),
                points,
                temple: std::array::from_fn(|i| pr(Effect::TempleStep(Temple::ALL[i], 1))),
                science: std::array::from_fn(|i| pr(Effect::AdvanceResearch(Science::ALL[i]))),
                unlock_worker: pr(Effect::UnlockWorker),
                free_worker: pr(Effect::FreeWorker(1)),
                worker_discount: pr(Effect::WorkerDiscount(1)),
                palenque_tile: k(2.0),
                burn_wood: k(-2.0),
                fill_chichen: k(0.0),
                build: k(8.0),
                monument: k(20.0),
            },
        }
    }

    /// One `Step`, priced.
    ///
    /// Only `Take` carries a `Choice`, and only `Take` nodes are ever wide: the
    /// 2,293-edge worst case (`phase.rs`) is one worker's options on a full gear, where
    /// `Placing` runs to a few hundred and every other phase is p50 3. A step
    /// this cannot price returns 0 and the stable tie-break leaves it in
    /// generation order, which is what those phases had anyway.
    #[inline]
    pub fn step(&self, s: &Step) -> f32 {
        match s {
            Step::Take(c) => self.price.choice(c),
            _ => 0.0,
        }
    }
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
            ordering: EdgeOrder::Gradient,
            temperature: 1.0,
            virtual_loss: 1,
            max_depth: 2048,
            seed: 0,
            // `Priors::OnePly` at `prior_temp = 1` -- what the `quality` preset
            // spelled -- is the default because it is worth +8.84 centred
            // against this same search with the old defaults (95% CI
            // +7.24..+10.44, 202 blocks), and because the simulation budget is
            // a dead axis without it: 16x the budget measures -1.08 under a
            // flat prior and 8x is worth +3.04 once the prior is real. The
            // `prior_temp` sweep against `mcts:2048` reads 0.5 -> +6.42,
            // 1.0 -> +8.84, 2.0 -> +5.56, 4.0 -> +1.01, so the 4.0 this
            // shipped with was throwing away seven of the eight points.
            priors: Priors::OnePly,
            prior_temp: 1.0,
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
    /// Transposition index, `digest -> the nodes carrying it`.
    ///
    /// **The key is eight bytes, not the position.** A map keyed by `NodeKey`
    /// stores a 320-byte `GameState` per entry, so the index for one
    /// sub-decision's 3,184 nodes was 1.3 MB of table that every probe walked
    /// and every insert memcpy'd -- while the arena already holds each of those
    /// states anyway. Keyed by the digest it is 76 kB, the insert copies eight
    /// bytes, and `retain` renumbers without touching a state at all. Measured
    /// **5.8% less CPU per turn** (+/-2.6%, six paired runs of `sprobe mcts
    /// mcts:2048`), output bit-identical.
    ///
    /// Collisions are resolved against the arena, not by the hash: a bucket is
    /// a list of node ids and [`Mcts::find_hashed`] compares the whole
    /// `NodeKey` against each. That is the same comparison `HashMap` was doing
    /// through `Eq`, so nothing about correctness moved -- see [`NodeKey`] for
    /// what the digest is allowed to leave out because of it.
    index: FxHashMap<u64, smallvec::SmallVec<[u32; 2]>>,
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
    /// The descent path, hoisted out of `simulate`. It is `Vec::with_capacity`
    /// per simulation otherwise, which at 2048 simulations a search is 2048
    /// mallocs -- and a `sample` profile put the malloc family at 14% of stack
    /// tops. Taken and put back around each descent, because the descent needs
    /// `&mut self` for everything else it touches.
    path: Vec<(u32, usize)>,
    /// One [`Gradient`] per mover, built on first use and thrown away at the
    /// next `search_at`. Lazy because most sub-decisions never reach a node
    /// wide enough to need an ordering at all.
    grad: [Option<Gradient>; N_PLAYERS],
    /// The position the gradients are anchored at -- the state `search_at` was
    /// given, not the node being expanded. See [`Mcts::gradient`].
    grad_root: Option<GameState>,
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
            path: Vec::with_capacity(32),
            grad: [None; N_PLAYERS],
            grad_root: None,
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

        // Re-anchor the ordering gradients on this sub-decision's own root. A
        // turn moves a handful of resources, so re-probing per sub-decision
        // rather than per turn is strictly fresher, and costs 16 `heuristic`
        // calls only for a mover that actually meets a wide node.
        self.grad = [None; N_PLAYERS];
        self.grad_root = Some(*state);

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
        let key = NodeKey { state: *state, phase, turn, done };
        let reusable = if self.reuse_disabled {
            None
        } else {
            self.find(&key)
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

    /// What one edge and one node cost in bytes.
    ///
    /// The whole argument for `widen_cap` is that an uncapped node is
    /// unaffordable, and that claim is arithmetic that nothing outside this
    /// module can do: `Edge` holds a `Step`, and a `Step::Take` holds a
    /// `Choice`, which is a `SmallVec<[Effect; 8]>` — inline, so an edge is far
    /// wider than the `u32` an edge index suggests. A node's array is
    /// `n_legal * edge_bytes()`.
    pub const fn edge_bytes() -> usize {
        std::mem::size_of::<Edge>()
    }

    /// A `Node` without its edge array. Dominated by the `GameState` it holds,
    /// which is the transposition key and so has to be there anyway (§3).
    pub const fn node_bytes() -> usize {
        std::mem::size_of::<Node>()
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
        let mut path = std::mem::take(&mut self.path);
        path.clear();
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
        self.path = path;
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

        // Renumber the index in place rather than rebuilding it. Reinserting
        // every survivor would hash each of them again -- 2,310 nodes retained
        // per sub-decision -- where `retain` hashes nothing. With the digest
        // key this is now pure integer work: no state is read, compared or
        // copied by the sweep.
        self.index.retain(|_, b| {
            b.retain(|i| mapping[*i as usize].is_some());
            for i in b.iter_mut() {
                *i = mapping[*i as usize].expect("the survivors were just filtered");
            }
            !b.is_empty()
        });
        0
    }

    /// The node holding this key, if the arena already has one.
    ///
    /// Two positions in one bucket is a digest collision, which at 64 bits over
    /// the ~3,200 nodes of a sub-decision is not something that happens; the
    /// list is there so that when it does, the answer is still right.
    fn find(&self, key: &NodeKey) -> Option<u32> {
        self.find_hashed(key.digest(), key)
    }

    fn find_hashed(&self, digest: u64, key: &NodeKey) -> Option<u32> {
        let bucket = self.index.get(&digest)?;
        bucket.iter().copied().find(|&i| {
            let n = &self.nodes[i as usize];
            n.phase == key.phase && n.turn == key.turn && n.done == key.done && n.state == key.state
        })
    }

    fn node_for(
        &mut self,
        state: GameState,
        phase: Phase,
        turn: PlayerId,
        done: u8,
        in_root_turn: bool,
    ) -> (u32, bool) {
        // Terminal first. `push_terminal` never inserts into the index, so the
        // lookup below could not have hit for an over position -- it was a
        // whole state hash spent to be told nothing, on every simulation that
        // reaches the end of the game.
        if state.over {
            return (self.push_terminal(state), true);
        }
        let key = NodeKey { state, phase, turn, done };
        // Digest once and carry it to the insert below. The miss path is the
        // common one -- an edge reaches here only the first time it is
        // followed, after which `Edge::child` caches the answer -- so the probe
        // that finds nothing is what this hash is mostly paying for.
        let digest = key.digest();
        if let Some(i) = self.find_hashed(digest, &key) {
            return (i, false);
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
            // Select first, price second. The one-ply prior costs an
            // `apply_step` and a `heuristic` per edge (123.0 ns) where the
            // gradient costs 16.0, so on a node past the cap the cheap key
            // picks the survivors and the expensive one is spent only on them
            // -- 34.1 us against 53.3 at the wide nodes, for the same top 32.
            let (steps, from_eval) =
                self.select_edges(&state, phase, turn, steps, eval.priors);
            let priors = self.priors_for(&state, phase, turn, done, &steps, from_eval);
            let mut edges: Vec<Edge> = steps
                .into_iter()
                .enumerate()
                .map(|(i, step)| new_edge(step, priors.get(i).copied().unwrap_or(0.0)))
                .collect();

            // Only a node wide enough to have been selected gets reordered;
            // below the cap the edges stay in the order the generator produced
            // them, which is the order the encoder will see. Above it,
            // `select_edges` has already put them in gradient order and dropped
            // the tail, so under `EdgeOrder::Gradient` this re-sort only matters
            // for `Priors::OnePly`, where it promotes the true one-ply best of
            // the survivors to the front of the `active` window, and the
            // `truncate` is a no-op. Under `EdgeOrder::Prior` the two lines are
            // the whole cap -- and are exactly the code that deleted the best
            // edge at 55.6% of wide nodes, kept so the two can be raced.
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
        // Indexed after the push, not before it: the id is `nodes.len()`, and
        // everything between the probe above and here -- `legal_steps`, the
        // evaluator, `select_edges` -- is free to touch the arena.
        let idx = self.nodes.len() as u32;
        self.index.entry(digest).or_default().push(idx);
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

    /// The turn-root gradient for one player, built on first use.
    ///
    /// Anchored at the state `search_at` was given rather than at the node
    /// being expanded: within a turn the two differ by a handful of resources,
    /// and re-probing per node would spend 16 `heuristic` calls at every wide
    /// node to move an ordering the linear approximation has already blurred.
    /// Keyed by mover because a descent crosses turns -- `max_depth` is 2048
    /// sub-decisions -- and a price list is a claim about one player's
    /// position, not about the position.
    fn gradient(&mut self, at: &GameState, mover: PlayerId) -> Gradient {
        if let Some(g) = self.grad[mover.idx()] {
            return g;
        }
        // `search_at` sets the anchor. A caller that reached `node_for` by
        // another route -- the tests do -- prices at the node itself, which is
        // stricter rather than cheaper.
        let anchor = self.grad_root.unwrap_or(*at);
        let g = Gradient::new(&anchor, mover);
        self.grad[mover.idx()] = Some(g);
        g
    }

    /// Cut a node wider than `max_edges` down to `widen_cap`, best first.
    ///
    /// Returns the surviving steps in the order the search should open them,
    /// with the evaluator's priors permuted and sliced to match. Below the cap
    /// this is the identity, so the generator's order -- the encoder's order --
    /// survives everywhere it can.
    fn select_edges(
        &mut self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        steps: Vec<Step>,
        from_eval: Vec<f32>,
    ) -> (Vec<Step>, Vec<f32>) {
        if steps.len() <= self.cfg.max_edges || self.cfg.ordering == EdgeOrder::Prior {
            return (steps, from_eval);
        }
        let g = self.gradient(state, phase.mover(turn));
        let key: Vec<f32> = steps.iter().map(|s| g.step(s)).collect();
        let mut order: Vec<u32> = (0..steps.len() as u32).collect();
        // A full sort of the widest node `phase.rs` has measured (2,293 edges)
        // is well under the pricing it already cost, so `select_nth_unstable` would
        // be optimising the smaller half. Ties break on generation order rather
        // than on `sort_unstable`'s arbitrary choice, so that a node's surviving
        // edge set is a function of the position and two runs of the same seed
        // agree.
        order.sort_unstable_by(|&a, &b| {
            key[b as usize]
                .total_cmp(&key[a as usize])
                .then(a.cmp(&b))
        });
        order.truncate(self.cfg.widen_cap.max(self.cfg.max_edges));

        // Moved out of their slots rather than cloned: a `Choice` owns a
        // `SmallVec`, and this runs on the widest nodes in the tree.
        let mut slots: Vec<Option<Step>> = steps.into_iter().map(Some).collect();
        let mut out_steps = Vec::with_capacity(order.len());
        let mut out_priors = Vec::with_capacity(order.len());
        for &i in &order {
            out_steps.push(slots[i as usize].take().expect("each edge selected once"));
            out_priors.push(from_eval.get(i as usize).copied().unwrap_or(0.0));
        }
        (out_steps, out_priors)
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

/// The transposition key, with a hash that does not walk 320 bytes one at a
/// time.
///
/// # Why this hash is hand-written
///
/// `GameState` derives `Hash`, and a derived `Hash` over nested `u8` arrays
/// calls `Hasher::write_u8` once per byte -- about 270 rounds of `FxHasher` for
/// one lookup. A `sample` profile of `mcts:2048` put `GameState::hash` at
/// **11.2% of all stack tops**, second only to `simulate` itself, and the
/// `FxHashMap` in `Mcts::index` is the only thing that ever hashes a state.
///
/// **Correctness does not rest on this hash.** `HashMap` resolves every bucket
/// with `Eq`, which is still the derived whole-state comparison, so the digest
/// only has to spread. That frees it to skip what cannot vary inside one
/// search, and to pack what remains eight bytes at a time:
///
/// * `gears` is redundant with `workers` -- `gears[g].occ[p] == w` and
///   `workers[w] == OnGear { gear: g, pos: p }` are the same fact written
///   twice, and `state.rs` moves them together. 55 bytes.
/// * the three `Deck::ids` arrays are shuffled once at setup and never
///   permuted again; only `next` advances. 45 bytes.
/// * `Player::color` is fixed for the whole game. 4 bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
struct NodeKey {
    state: GameState,
    phase: Phase,
    turn: PlayerId,
    done: u8,
}

impl NodeKey {
    /// The bucket this key lands in. One `write_u64`'s worth of work: the
    /// map's key *is* this number, so `FxHasher` sees eight bytes rather than
    /// the ~270 rounds a derived `Hash` over nested `u8` arrays would cost. A
    /// `sample` profile of `mcts:2048` had `GameState::hash` at **11.2% of all
    /// stack tops** before this existed; it is 3.3% now, and the paired
    /// measurement of the change alone is **8.5% less CPU per turn** (+/-3.2%,
    /// six runs).
    #[inline]
    fn digest(&self) -> u64 {
        mix(state_digest(&self.state), self.phase_word())
    }
}

impl NodeKey {
    /// `Phase` with its payload, not just its tag. The siblings of a `Take`
    /// node differ only in `worker`, and dropping it would put every worker of
    /// a position in one bucket -- correct, because `Eq` decides, and slow for
    /// exactly the nodes this search spends its time on.
    #[inline]
    fn phase_word(&self) -> u64 {
        let payload = match self.phase {
            Phase::Beg | Phase::Mode | Phase::PickWorker | Phase::PityPlace => 0,
            Phase::Placing { n } => n as u64,
            Phase::Take { worker } => worker.0 as u64,
            Phase::ExtraDay { claimer } => claimer.idx() as u64,
            Phase::DraftTile { dealt, kept } => {
                u32::from_le_bytes(dealt) as u64 | (kept as u64) << 32
            }
        };
        self.phase.tag() as u64
            | (self.turn.idx() as u64) << 8
            | (self.done as u64) << 16
            | payload << 24
    }
}

/// FNV-1a's step, over a whole word instead of a byte.
#[inline]
fn mix(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(0x0100_0000_01b3)
}

#[inline]
fn mix_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    let mut it = bytes.chunks_exact(8);
    for c in &mut it {
        h = mix(h, u64::from_le_bytes(c.try_into().expect("chunks_exact(8)")));
    }
    let rem = it.remainder();
    if !rem.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rem.len()].copy_from_slice(rem);
        h = mix(h, u64::from_le_bytes(buf));
    }
    h
}

/// A worker's whole location in one byte, so 24 of them are three words.
///
/// `3 + gear * MAX_GEAR_SPACES + pos` tops out at `3 + 4*11 + 10 = 57`, which
/// is why this fits at all.
#[inline]
fn worker_byte(w: crate::state::WorkerLoc) -> u8 {
    use crate::state::WorkerLoc;
    match w {
        WorkerLoc::Locked => 0,
        WorkerLoc::Available => 1,
        WorkerLoc::FirstPlayerSpace => 2,
        WorkerLoc::OnGear { gear, pos } => {
            3 + gear as u8 * crate::state::MAX_GEAR_SPACES as u8 + pos.0
        }
    }
}

/// Everything about a position that can change inside one search, folded to a
/// word. See [`NodeKey`] for what is left out and why.
fn state_digest(g: &GameState) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;

    let mut wb = [0u8; N_WORKERS];
    for (b, w) in wb.iter_mut().zip(g.workers.iter()) {
        *b = worker_byte(*w);
    }
    h = mix_bytes(h, &wb);

    for p in &g.players {
        h = mix(
            h,
            p.corn as u64
                | (p.res[0] as u64) << 8
                | (p.res[1] as u64) << 16
                | (p.res[2] as u64) << 24
                | (p.res[3] as u64) << 32
                | ((p.points as u16) as u64) << 40
                | (p.corn_tiles as u64) << 56,
        );
        h = mix(
            h,
            p.buildings as u64
                | (p.monuments as u64) << 32
                | (p.wood_tiles as u64) << 48
                | (p.free_workers as u64) << 56,
        );
        h = mix(
            h,
            p.worker_discount as u64 | (p.may_skip_day as u64) << 8,
        );
    }
    for row in &g.temples {
        h = mix_bytes(h, row);
    }
    for row in &g.research {
        h = mix_bytes(h, row);
    }
    for t in &g.palenque {
        h = mix(h, t.corn as u64 | (t.wood as u64) << 8);
    }
    let opt = |o: Option<u8>| o.map_or(0u64, |v| v as u64 + 1);
    let mut up = 0u64;
    for (i, b) in g.buildings_up.iter().enumerate() {
        up ^= opt(b.map(|x| x.0)) << (i * 9);
    }
    h = mix(h, up);
    let mut mu = 0u64;
    for (i, m) in g.monuments_up.iter().enumerate() {
        mu ^= opt(m.map(|x| x.0)) << (i * 9);
    }
    h = mix(h, mu);
    mix(
        h,
        g.chichen_filled as u64
            | (g.accumulated_corn as u64) << 16
            | (g.skulls_remaining as u64) << 24
            | (opt(g.first_player_space.map(|w| w.0))) << 32
            | (g.current.idx() as u64) << 40
            | (g.first_player.idx() as u64) << 43
            | (g.age as u64) << 46
            | (g.day as u64) << 48
            | (g.over as u64) << 56
            // The decks are only ever drawn from, so `next` is the whole of
            // their state that a search can move.
            | (g.age1.next as u64) << 57,
    ) ^ mix(0, g.age2.next as u64 | (g.monument_deck.next as u64) << 8)
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
