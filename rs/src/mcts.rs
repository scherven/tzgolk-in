//! MCTS over the factored tree.
//!
//! # The spec to run, if you read nothing else
//!
//! ```text
//! mcts:8192:heuristic:deeper        # == mcts:8192:heuristic:cp=0.02,pmin=2
//! ```
//!
//! Two constants away from the shipped defaults, and both of them were shipped
//! one to two orders of magnitude wrong for a search factored into
//! sub-decisions. Against `mcts:8192:heuristic:cp=0.02`, `pmin=2` is **+2.78**
//! (CI +2.39..+3.16, **600 paired blocks**) and reproduces on three separate
//! evaluators (+4.73 / 50 blk, +2.78 / 600 blk, +3.12 / 117 blk); against the
//! shipped `c_puct` at the same budget, `cp=0.02` alone is **+9.85** (CI
//! +8.29..+11.40, 44 paired blocks). Cost on the current build is 182 ms of
//! user CPU per turn against `mcts:2048:heuristic:quality`'s 18.8 — and
//! **`pmin=2` itself is free**: 0.98x `cp=0.02` measured in the same sample
//! (182.0 against 186.5 ms), so it buys three points for nothing.
//!
//! **More budget is worth more, and costs a lot**: `mcts:32768:heuristic:deeper`
//! is the strongest thing measured — the budget rung is **+4.30** (CI
//! +3.40..+5.19, 110 paired blocks) — at **823 ms/turn, 44x**. The ladder stops
//! there: 65,536 is **-0.79** over 32,768 (CI -3.46..+1.88, 9 blk), agreeing in
//! sign and size with an independent +0.70 (14 blk) on an older platform.
//!
//! **The three constants are additive, not multiplicative.** `pmin=2` is worth
//! the same three points at 32,768 as at 8,192 — difference-in-differences
//! **+0.20** (CI -1.12..+1.53, 109 paired blocks, p = 0.76) — so `c_puct` sets
//! the descent length, the budget supports the tree that length digs, and
//! `prior_min_edges` fixes the narrow nodes it passes through, each
//! independently. The axis looked dead for years because all three were wrong
//! at once.
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
//! of its games** against three alpha-betas.
//!
//! This comment used to call that "the ceiling this search currently has
//! against `search.rs`, and it is a prior away rather than a budget away".
//! **It was neither: it was a `c_puct` away.** `mcts:8192:heuristic:cp=0.02`
//! against the same alpha-beta is **+8.50** (CI +6.62..+10.39, 27 blocks) and
//! takes 47.7% of its games. There was no ceiling; there was a constant set
//! two orders of magnitude too high, which meant no amount of extra budget
//! could turn into extra lookahead. See [`MctsConfig::c_puct_init`].
//!
//! **And with `pmin=2` on top, on the post-`d7b4e42` evaluator, there is not
//! much of a contest.** `mcts:8192:heuristic:deeper` against the same
//! alpha-beta is **+13.65** (CI +11.90..+15.40, 26 blocks) and takes **77.4%
//! of its games** against three of them. Against a stateless `heuristic:full`
//! — whose null control returns **exactly +0.00 with a win rate of exactly
//! 0.250 over 90 blocks**, so no null correction is needed at all — it is
//! **+13.63** (CI +12.11..+15.15, 44 blocks, **77.8%** of games) where
//! `mcts:2048:heuristic:quality` is **+3.41** (183 blocks, 35.8%). Paired on
//! seed against that common opponent the difference is **+10.34** (CI
//! +8.50..+12.17, 44 paired blocks, p = 2e-29).
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
//! # The simulation budget, and why it looked dead
//!
//! Six head-to-heads under a **uniform** prior once put every rung from 128 to
//! 2048 within +/-1.1 points of every other, with the widest span (`mcts:2048`
//! against `mcts:128`) at -1.08, CI -2.79..+0.64. A note here then claimed a
//! real prior revived the axis: `mcts:2048:quality` over `mcts:256:quality` at
//! +3.04 (CI +1.46..+4.62, 150 blocks).
//!
//! **That claim does not reproduce.** Re-run on 2026-09-08 against the champion
//! at `--seed 3000000`, with `null-self` (candidate spec == baseline spec) at
//! **-0.10, CI -0.61..+0.40 over 374 blocks** as the zero:
//!
//! | candidate | centred vs `mcts:2048:quality` | 95% CI | blocks |
//! |---|---|---|---|
//! | `mcts:1` | **-32.10** | -33.10..-31.10 | 84 |
//! | `mcts:32` | -2.28 | -3.57..-0.99 | 77 |
//! | `mcts:128` | -0.69 | -1.71..+0.33 | 98 |
//! | `mcts:256` | -0.24 | -1.24..+0.76 | 110 |
//! | `mcts:512` | +0.34 | -0.71..+1.39 | 110 |
//! | `mcts:1024` | -0.26 | -1.29..+0.77 | 102 |
//! | `mcts:4096` | +0.50 | -0.83..+1.82 | 62 |
//! | `mcts:8192` | +0.62 | -0.13..+1.36 | 158 |
//! | `mcts:16384` | +0.56 | -0.94..+2.05 | 47 |
//! | `mcts:32768` | +0.14 | -1.75..+2.03 | 24 |
//!
//! So the axis is not dead — it is **saturated**. `mcts:1` is the prior's argmax
//! with no search at all and it is not a player: -32 points and a **0% win
//! rate**. The knee is between 32 and 256 simulations, and above 256 a
//! 128-fold span is worth nothing anybody can measure.
//!
//! # It saturates because the search cannot afford to look at a reply
//!
//! `Mcts::depth_stats` counts descent depth in **sub-decisions**, and a turn is
//! ~8 of them. At the shipped `c_puct` (`bin/sprobe budget`):
//!
//! | sims | mean descent | that is | arena nodes |
//! |---|---|---|---|
//! | 256 | 5.6 | less than its own turn | 342 |
//! | 2048 | 8.1 | its own turn, exactly | 2756 |
//! | 16384 | 10.8 | its turn plus a third of the next | 21715 |
//!
//! Sixty-four times the budget buys 1.9x the depth, and the tree grows at ~1.4
//! nodes per simulation at every rung — the budget is buying **width inside the
//! current turn**, where a one-ply prior has already decided nearly everything.
//! That is the whole of it: the simulations were not wasted on a bad prior, they
//! were spent re-deciding what the prior had decided, because they could not
//! reach anything new.
//!
//! # Buy depth instead, and then the budget pays
//!
//! [`MctsConfig::c_puct_init`] is the knob that converts budget into depth, and
//! at 2.0 it was set one to two orders of magnitude too high.
//!
//! Two arena runs sharing a baseline and a seed sequence are a matched pair, and
//! the matched form is the number to quote. Holding the budget at **8,192
//! simulations** and changing nothing but `c_puct`:
//!
//! **`cp=0.02` - `cp=2.0` = +5.93 centred (95% CI +4.27..+7.59, 47 paired
//! blocks, p = 9e-13).**
//!
//! Neither half is worth much alone — the same `c_puct` cut at 2,048
//! simulations is +0.82 (127 paired blocks) and 4x the budget at `cp=2.0` is
//! +0.62 — which is why every 1-D sweep this project ran found "about a point":
//!
//! | | `cp = 2.0` (shipped) | `cp = 0.06` | `cp = 0.02` |
//! |---|---|---|---|
//! | 2,048 sims | 0 by construction | **+1.04** (+0.41..+1.66, 264 blk) | +0.45 (75 blk) |
//! | 8,192 sims | +0.62 (158 blk) | +1.94 (69 blk) | **+6.27** (+5.28..+7.26, 88 blk) |
//! | 16,384 sims | +0.56 (47 blk) | — | +7.94 (+6.35..+9.53, 45 blk) |
//! | 32,768 sims | +0.14 (24 blk) | — | +9.36 (+7.99..+10.73, 33 blk) |
//!
//! `mcts:8192:heuristic:cp=0.02` takes **42.9% of its games** against three
//! copies of the old champion, where an equal agent takes 25%, and scores 38.9
//! against their 30.4. Its descent runs a mean of **~50 sub-decisions — about
//! six turns** — against the old champion's 8.5.
//!
//! **And it is cheaper than the width it beats.** User CPU per turn, one spec
//! per process: the old champion 14.77 ms, `mcts:8192:cp=0.02` 146.04 ms —
//! **10x** — and `mcts:16384:quality` **187.12 ms**. The CPU-matched comparison
//! is tilted 1.28x *against* the deep spec and it still wins by six points.
//! Memory is the other price: 412 MB max RSS against 186 MB.
//!
//! **Four opponents, four designs, one direction.** Against a stateless
//! `heuristic:full` the old champion is +3.53 (108 blk) and this is **+11.28**
//! (+9.75..+12.80, 38 blk); against `minimax:8:600000::greedy:capw=25` it is
//! **+8.50** (+6.62..+10.39, 27 blk) where the old champion is +7.73; and in
//! `--mode pairs`, where both sides share an agent instance across two seats and
//! `bin/arena.rs`'s one-instance-for-three-seats asymmetry cannot apply, it is
//! **+3.46** (+2.38..+4.55, 18 blk) against a pairs null of +0.27. It is not an
//! artefact of racing a shallow search of its own family.
//!
//! See [`MctsConfig::c_puct_init`] for the full sweep and
//! `docs/FINDINGS-mcts.md` for the run files and block counts.
//!
//! # Lowering `c_puct` re-opened every knob that had been measured inert
//!
//! A knob swept at `cp = 2.0` was swept on a search that could not look ahead,
//! and "inert" was a statement about that search, not about this one. Paired on
//! seed at 8,192 simulations, changing one flag against `cp = 0.02`:
//!
//! | flag | at `cp = 2.0` | at `cp = 0.02` | paired blk |
//! |---|---|---|---|
//! | **`pmin=2`** | -1.35 (settled *negative*) | **+4.73** (+3.22..+6.24) | 50 |
//! | `fpu=0.6` | +0.21 (inert) | **-4.75** (-5.58..-3.92) | 195 |
//! | `fpu=0.0` | +0.31 (inert) | **-2.90** (-4.64..-1.15) | 61 |
//! | `q0=1,qn=16` | +0.42 (inert) | **-4.18** (-6.00..-2.35) | 38 |
//! | `noreuse` | -0.03 (inert) | **-2.00** (-3.26..-0.73) | 76 |
//! | `priors=mixed` | +0.38 (inert) | **-4.26** (-7.31..-1.21) | 13 |
//! | `pt=0.5` | +0.00 (inert) | **-1.19** (-1.82..-0.55) | 261 |
//! | `nocap` | +0.72 (inert) | +0.19 (-0.51..+0.89) | 44 |
//! | `k=8` | +0.74 (inert) | +0.06 (-1.53..+1.64) | 19 |
//! | `pick=q` | -2.41 | -1.60 (-3.27..+0.08) | 30 |
//!
//! **One mechanism explains the whole column.** A descent is ~56 sub-decisions
//! at `cp = 0.02` against ~8 at `cp = 2.0`, so anything that prices a *narrow
//! or barely-visited node* went from touching a handful of nodes per descent to
//! touching most of them: `pmin` (median searched width is **2**, so
//! `prior_min_edges = 3` left about 28 nodes per descent with a uniform prior),
//! `fpu` and `q_pseudo` (the frontier is no longer a transient), `tree_reuse`
//! (the previous sub-decision built far more of the tree the next one needs),
//! and the prior source (consulted at every node of a much longer descent).
//! What did **not** move is the width family — `max_edges`, `widen_cap`,
//! `widen_c` — because those act on the few wide nodes, and the wide nodes are
//! not where a deep descent lives.
//!
//! `pmin=2` reproduces on the post-`d7b4e42` evaluator at **+3.06** (+1.43..+4.68,
//! 46 paired blocks), so it is not an artefact of the platform it was found on.
//!
//! **The order to tune things in** is therefore: the prior first, then
//! `c_puct` and `sims` **jointly**, then re-sweep everything that touches a
//! narrow node (`pmin`, `fpu`, `q_pseudo`, `tree_reuse`) **at the `c_puct` you
//! settled on** — and `max_edges` / `widen_cap` / `widen_c` never.
//!
//! # Threading
//!
//! Still single-threaded. Virtual loss is applied and removed for real, on the
//! mover's component only, so the mechanism is exercised and the shape is
//! right, but the statistics are plain `u32`/`f32` rather than §3.1's atomics
//! and there is no batching.
//!
//! **The exchange rate is now measured**: +2.45 centred per 4x simulations
//! (`32768:cp=0.02` - `8192:cp=0.02`, 99 paired blocks, p = 5e-6) at 4.85x the
//! user CPU, so **+1.22 per doubling of the budget for 2.2x the CPU**.
//!
//! **That makes threading worth zero, and the sentence this comment used to
//! carry — "threading buys simulations, so it is worth exactly what
//! simulations are worth" — is a wall-clock claim wearing a strength claim's
//! clothes.** K threads do not create simulations; they spend K cores to
//! finish the same ones sooner. Every number in this file is strength *per
//! CPU*, the constraint the user lifted was the clock rather than the cycles,
//! and virtual-loss tree parallelism is a slightly *worse* use of N
//! simulations than one thread's N. No arena measurement here could show a
//! threading win. What it buys is **latency for the shipped agent** — a TUI
//! turn uses one of fourteen cores, so at a fixed 0.8 s/turn eight threads
//! would be worth about `1.22 * log2(8) = +3.7`. That is a
//! deliverable-comfort case, and it is the honest one.
//!
//! **And the swap is not mechanical.** `Edge::w` and `Node::w` are
//! `[f32; N_PLAYERS]`, and both encodings cost something: §3.4's signed
//! fixed-point `AtomicI32` changes the arithmetic, so the search plays
//! differently *at one thread* and every row in `docs/FINDINGS-mcts.md` would
//! need re-measuring; an `AtomicU32` of `f32::to_bits` with a compare-exchange
//! add is bit-identical at one thread but backup does `4 * depth` of them and
//! depth is now ~56, which is ~7M CAS per sub-decision at 32,768 simulations —
//! buying threads by making the single-threaded search slower.
//!
//! `docs/SEARCH.md` §3.5 also calls batching "the real reason for
//! parallelism", sized against a **network** call of 0.1-1 ms. There is no
//! network: the leaf is `eval::heuristic`, and a `sample` profile of a live
//! `mcts:32768:cp=0.02` puts the whole `eval::*` family at ~1,850 samples
//! against `simulate`'s 11,367. **The search is descent-bound, not
//! evaluation-bound** — the opposite of the regime §3.5 was written for — so
//! there is nothing to batch. The hot loop is `select`, and `logf` for the
//! `c_puct_base` term is on its own ~6% of the non-idle profile.
//!
//! Two caveats stand whatever the curve does. Root- or leaf-parallel MCTS is
//! *weaker* than the same total simulations in one tree, so the sims curve is an
//! upper bound and not an estimate. And it buys latency rather than measurement
//! throughput: self-play and the arena already fill every core by running whole
//! games in parallel, so a threaded search makes one turn faster without making
//! an experiment faster. At 8,192 simulations and `cp=0.02` a turn is ~180 ms of
//! user CPU, which no interface needs help with.
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
    /// §4.5 says this is calibrated for a value in (-1, 1) rather than
    /// AlphaZero's win/loss scale. **It is not calibrated for anything, and 2.0
    /// is one to two orders of magnitude too high.**
    ///
    /// # What it actually buys, which is depth
    ///
    /// `c` enters selection as `u = c * prior * sqrt(N) / (1 + n)`, so lowering
    /// it makes the descent commit to the best-looking child sooner and go
    /// further before it comes back to explore. In a search factored into
    /// sub-decisions that is the *only* way to buy lookahead, because a turn is
    /// ~8 sub-decisions and simulations alone do not deepen a tree fast enough
    /// to reach one. At 2,048 simulations (`bin/sprobe budget`, 147
    /// sub-decisions, mean descent depth in sub-decisions):
    ///
    /// | `c_puct_init` | mean descent | top edge's visit share | user CPU |
    /// |---|---|---|---|
    /// | 2.0 (shipped) | 8.0 | 83.5% | 14.9 ms/turn |
    /// | 0.25 | 11.3 | 90.6% | — |
    /// | 0.06 | **21.8** | 93.3% | 19.3 ms (1.29x) |
    /// | 0.02 | 48.3 | 93.8% | 31.8 ms (2.13x) |
    /// | 0.005 | 83.0 | 93.8% | 50.7 ms (3.39x) |
    ///
    /// For comparison, sixty-four times the *budget* at `c_puct = 2.0` moves the
    /// mean descent from 5.6 to 10.8 — so dividing this constant by 33 buys more
    /// lookahead than a 64x budget does, at 1.3x the CPU instead of 64x.
    ///
    /// # And the best value depends on the budget
    ///
    /// A deep descent needs enough simulations to support the tree it is
    /// digging, so the two knobs are one knob. Centred against
    /// `mcts:2048:heuristic:quality`:
    ///
    /// | | `cp = 2.0` | `cp = 0.06` | `cp = 0.02` | `cp = 0.01` |
    /// |---|---|---|---|---|
    /// | 2,048 sims | 0 by construction | **+1.04** (264 blk) | +0.70 (114 blk) | -0.13 (39) |
    /// | 4,096 sims | — | — | **+4.64** (72 blk) | — |
    /// | 8,192 sims | +0.42 (352 blk) | +1.94 (69 blk) | **+5.99** (+5.53..+6.45, 353 blk) | +3.74 (24) |
    /// | 16,384 sims | +0.56 (47 blk) | — | +7.94 (45 blk) | — |
    /// | 32,768 sims | +0.14 (24 blk) | — | **+9.29** (+8.50..+10.08, 120 blk) | +9.02 (24) |
    /// | 65,536 sims | — | — | +11.75 (14 blk) | — |
    ///
    /// **The optimum falls with the budget and then stops falling.** 0.06 at
    /// 2,048, 0.02 at 8,192, and at 32,768 the 0.01 and 0.02 cells are
    /// indistinguishable (paired +0.25, CI -3.51..+4.01, 15 blocks). `cp` sets
    /// the descent length — 56 sub-decisions at 0.02, 78 at 0.01, 95 at 0.005,
    /// and *within 9% independent of the budget* — so once the length is right
    /// more simulations only support it. **Depth is the axis; the budget is the
    /// support.** Paired against `8192:cp=0.02`: 2,048 is -4.79, 4,096 is
    /// -1.90, and 32,768 is **+4.03** (CI +3.14..+4.92, 127 paired blk, p=6e-19
    /// — re-measured 2026-09-09 on a frozen binary; the earlier +2.45 at 99
    /// blocks was on an older evaluator). At `pmin=2` the same rung is **+4.30**
    /// (110 paired blk). **65,536 is -0.79** over 32,768 (CI -3.46..+1.88, 9
    /// blk), which agrees in sign and size with an independent +0.70 (14 blk)
    /// on the older platform: two looks at the top of the ridge, neither able
    /// to find a gain. **The ridge stops paying at 32,768.**
    ///
    /// The matched form, two runs sharing a baseline and a seed sequence with
    /// only `c_puct` between them: at 8,192 simulations **+5.93 (CI
    /// +4.27..+7.59, 47 paired blocks, p = 9e-13)**.
    ///
    /// At 2,048 simulations `cp = 0.02` descends a mean of 48 sub-decisions on a
    /// tree with 2,048 simulations to spread over them, which is
    /// over-commitment; at 8,192 the same depth is four times better supported.
    /// **Sweep the two together or you will find "about a point" whichever one
    /// you move**, which is what the 1-D sweeps before 2026-09-08 found.
    ///
    /// The full 1-D curve at 2,048 simulations is single-peaked at 0.06-0.12
    /// with a trough at 0.25-1.0 and the shipped 2.0 on the far shoulder:
    /// -2.71, -1.34, -0.13, +0.45, **+1.07**, +1.08, -0.69, -0.91, -0.58, 0,
    /// +0.23 at cp = 0.002, 0.005, 0.01, 0.02, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0,
    /// 4.0.
    pub c_puct_init: f32,
    /// `c = c_puct_init + ln((1 + N + base) / base)`, so this is how much *more*
    /// a well-visited node explores.
    ///
    /// At 19652 it does almost nothing here — `ln(21701/19652) = 0.099` at
    /// N = 2048 — which is why [`MctsConfig::c_puct_init`] is effectively the
    /// whole constant. It is the natural place to put "breadth at the root,
    /// exploitation below": the root of a turn carries every simulation and a
    /// node eight levels down carries a handful. `cp=0.005, cpb=1000` gives
    /// `c(N=20) = 0.026` and `c(N=2048) = 1.12`, which reaches `cp=0.06`'s mean
    /// descent of 21.8 while keeping the root's top-edge share at 86% instead of
    /// 93%. Measured only as `cpb=1000` alone (+0.56, -0.50..+1.61, 99 blocks).
    ///
    /// # "It does almost nothing" is only true at 2,048 simulations
    ///
    /// `ln((1 + N + 19652) / 19652)` grows with the budget, and at `cp = 0.02`
    /// it is not a correction, it is the constant:
    ///
    /// | sims at the root | 2048 | 8192 | 32768 | 65536 |
    /// |---|---|---|---|---|
    /// | `c` with `cp = 0.02` | 0.119 | 0.368 | **1.001** | **1.487** |
    ///
    /// So `mcts:8192:cp=0.02` and `mcts:32768:cp=0.02` differ in *two* things:
    /// the budget, and a root that explores four times as hard. **Raising the
    /// budget at a fixed `(init, base)` raises root breadth as a side effect**,
    /// which is a second reason the budget axis came alive when `init` came
    /// down.
    ///
    /// Separating them does not pay. Giving 8,192 simulations 32,768's root
    /// breadth (`cp=0.02,cpb=4913`, so `c(N=8192) = 1.00`) is **-0.83** (CI
    /// -2.44..+0.79, 54 paired blocks); the reverse pairing `cp=0.005,cpb=3000`
    /// matched `cp=0.02` to within its interval. Read that as good news about
    /// the shape of the optimum: anything reaching a mean descent of 35-60
    /// sub-decisions scores the same, so the result is about **depth**, not
    /// about which pair of constants buys it. `cp` alone is the simpler
    /// spelling.
    pub c_puct_base: f32,
    /// First-play urgency, subtracted from the parent's Q.
    ///
    /// # Measured inert at `c_puct = 2.0`, and worth five points at 0.02
    ///
    /// The 1-D sweep against the shipped `cp = 2.0` put 0.0, 0.6 and 1.2 at
    /// +0.31, +0.21 and +0.22, every interval straddling zero, and
    /// [`MctsConfig::q_init`] explains why: an unvisited edge is a state that
    /// lasts two descents when nodes are p50 3 edges wide and the budget is
    /// thousands.
    ///
    /// **At `cp = 0.02` the same flag is worth five points across its range**,
    /// paired on seed at 8,192 simulations against the `fpu = 0.2` default:
    ///
    /// | `fpu` | centred | 95% CI | paired blk | p |
    /// |---|---|---|---|---|
    /// | 0.0 | **-2.90** | -4.64..-1.15 | 61 | 9e-4 |
    /// | 0.2 (default) | 0 by construction | — | — | — |
    /// | 0.6 | **-4.75** | -5.58..-3.92 | 195 | 3e-29 |
    ///
    /// A 56-sub-decision descent creates fresh nodes for most of its depth, so
    /// "unvisited edge" is not a transient at the frontier — it *is* the
    /// frontier, and what FPU charges it decides where the search goes. The
    /// shipped 0.2 sits at an interior optimum: the knob went from inert to
    /// decisive **without its best value moving**, so nothing needs changing —
    /// but do not re-derive "inert" from the old sweep and delete it.
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
    ///
    /// **Re-measured on the real prior, 2026-09-08: it is a wash on strength
    /// too.** `noreuse` against the default is **+0.07** (CI -0.89..+1.03, 129
    /// blocks). So the paragraph above is right that this is a search-quality
    /// knob and wrong to imply the quality moves: the larger effective budget at
    /// reused nodes buys nothing, which is what the flat budget curve in the
    /// module header would predict.
    ///
    /// **And that is a `c_puct = 2.0` fact.** At `cp = 0.02` and 8,192
    /// simulations, `noreuse` is **-2.00** (CI -3.26..-0.73, 76 paired blocks,
    /// p = 0.002) — reuse is worth two points, not zero. The sentence above is
    /// still the right explanation and now has the opposite sign: the effective
    /// budget at reused nodes *does* buy something once a descent is 56
    /// sub-decisions instead of 8, because the previous sub-decision built far
    /// more of the tree the next one needs.
    ///
    /// Keep it on for the strength *and* the memory: `record::SearchAgent` shares
    /// one `Mcts` across every seat it is asked to play (`bin/selfplay.rs:449`
    /// hands one instance all four), so the arena it retains is shared too.
    pub tree_reuse: bool,
    /// §3.7's root exploration noise: `prior = (1-eps) * prior + eps * Dir`,
    /// applied at every `in_root_turn` node rather than only the literal root.
    ///
    /// **It has no `mcts:` flag on purpose, and adding one is a mistake.**
    /// `record::SearchAgent::with_config` zeroes this field whenever
    /// `Exploration::Off`, which is every caller that is not generating
    /// training data — the arena included — so a flag would parse, be
    /// overwritten, and print `:eps=` into a label that did not describe the
    /// search. This was tried on 2026-09-08 and reverted within the hour.
    ///
    /// The thing worth checking is the gate, not the value: noise at tau = 0 is
    /// a search playing something other than its best move, and the only
    /// defence against that is `Exploration`.
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
    /// `tau` in §3.8. Zero picks by [`MctsConfig::root_pick`]; above zero the
    /// step is always sampled from the visit counts, because that is the
    /// distribution §3.8's training target is defined on.
    pub temperature: f32,
    /// What a temperature-zero search plays: the most-visited edge, or the
    /// best-valued one.
    ///
    /// # Visits are nearly the prior here, and Q is what the search added
    ///
    /// Most-visited is the standard answer and it is the robust one when visits
    /// are earned — a single lucky rollout cannot make an edge the most
    /// visited. That argument assumes the visit distribution is *about* the
    /// search. `bin/sprobe budget` says it is mostly about the prior: 85% of a
    /// 2,048-simulation root's visits sit on one edge, and `mcts:1` — the
    /// prior's argmax with no search whatever — plays the same move as
    /// `mcts:2048` on 81% of sub-decisions. Whatever the simulations learned is
    /// in the *values* they backed up, not in a count that was already decided.
    ///
    /// See [`RootPick`] for how the value variant guards against the fluke the
    /// visit rule exists to prevent.
    pub root_pick: RootPick,
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
    /// Score the one-ply probe with `eval::margin` rather than `eval::heuristic`.
    ///
    /// # The sibling-common argument is wrong for exactly the moves that matter
    ///
    /// [`Priors::OnePly`] scores a child with the mover's own estimate and
    /// [`one_ply`](Mcts::one_ply) justifies that by saying the best-opponent
    /// term is common to a node's siblings. It is common to siblings that
    /// differ only in what the mover *gains*. It is not common to siblings that
    /// differ in what an opponent is *denied* — taking the gear space they
    /// needed, the monument they were saving for, the last skull — and denial
    /// is a large part of how this game is won. `eval::margin`'s own doc comment
    /// calls it "the quantity the search maximises".
    ///
    /// It costs four `heuristic` calls an edge instead of one. That is
    /// affordable here and nowhere else in this project: the probe is ~12% of a
    /// search at `sims = 2048`, and the simulations it would be competing for
    /// are worth +0.35 per eightfold (see the module header), so trading
    /// simulations for a better prior is the trade this search wants.
    ///
    /// The spread of a margin is wider than the spread of a raw estimate, so
    /// `prior_temp` does not carry over — sweep the two together.
    pub prior_margin: bool,
    /// How much of the one-ply score to spend as an unvisited edge's Q, on top
    /// of first-play urgency.
    ///
    /// # The prior was throwing away half of what it computed
    ///
    /// [`Priors::OnePly`] applies `eval::heuristic` to every child and then
    /// keeps only the *softmax* of the result. A softmax is a statement about
    /// which edge to try; the scores it was built from are also a statement
    /// about what each edge is worth, and PUCT was being told the first and not
    /// the second. Every unvisited sibling therefore entered `select` with the
    /// identical Q — `parent_q - fpu_reduction * sqrt(expanded)` — and the
    /// prior had to carry the whole difference through `u`, where it decays as
    /// `sqrt(N)/(1+n)` and is gone by the time a node is resolved.
    ///
    /// This adds `q_init * tanh((h_i - mean_h) / 25)` to that edge's FPU. The
    /// squash and the 25 are `phase::HeuristicEvaluator`'s own points-to-value
    /// map, so `q_init = 1` means "trust the one-ply score exactly as far as
    /// the evaluator's value head is trusted"; linearising the four-player
    /// centring puts the self-consistent value nearer 0.75.
    ///
    /// **Default 0, which is the behaviour this shipped with**, so a spec that
    /// does not name it searches identically to one from before the field
    /// existed — `tests/search.rs` checks that byte for byte. `Edge::q0` fits
    /// in the padding `Edge` already had, so `edge_bytes()` is still 88 and the
    /// knob costs nothing at all when it is off.
    ///
    /// # On its own it is inert, and the reason is the shape of the tree
    ///
    /// `q0=1` and `q0=4` both change **0.0%** of played turns against the
    /// default (`bin/sprobe mcts`, 405 turns) — even at `cp=0.06`, where the
    /// exploration term is small enough that the offset should dominate. An
    /// unvisited edge is a transient state here: nodes are p50 3 edges wide and
    /// the budget is thousands of simulations, so every edge is visited within
    /// the first few descents and an FPU term is never consulted again. FPU is
    /// a knob for searches whose nodes are wider than their budgets.
    ///
    /// It becomes live through [`MctsConfig::q_pseudo`], which keeps the same
    /// offset in Q *after* the edge has been visited.
    pub q_init: f32,
    /// Pseudo-visits carrying [`MctsConfig::q_init`]'s estimate, so that it
    /// decays with real evidence instead of vanishing at the first visit.
    ///
    /// `Q = (w + k * (parent_q + q0)) / (n + k)`. At `k = 0` the edge is scored
    /// exactly as before. At `k = 4` an edge's first four simulations are half
    /// discounted toward what the one-ply probe said about it, and by fifty
    /// visits the probe is worth 8% of the estimate.
    ///
    /// This is the form in which "the prior computed a value and the search
    /// threw it away" is actually testable: `q_init` alone is consulted once
    /// per edge and then never again.
    ///
    /// **Testable, and the answer is no.** At `cp = 2.0`, `q0=1,qn=16` changed
    /// 2.2% of turns and measured +0.42, inert. At `cp = 0.02` and 8,192
    /// simulations it is **-4.18 (CI -6.00..-2.35, 38 paired blocks, p = 4e-6)**.
    ///
    /// Same shape as [`MctsConfig::fpu_reduction`]: anything that prices an
    /// *unvisited or barely visited* edge is inert when the descent is 8
    /// sub-decisions deep and decisive when it is 56, because at 56 the
    /// barely-visited edges are the descent. Here the sign is bad — pulling a
    /// young edge's Q toward the one-ply probe overrides the values the deeper
    /// search is actually returning, and those are worth more than the probe
    /// precisely because the search can now see further than one ply.
    pub q_pseudo: f32,
    /// Do not spend a one-ply pass on a node narrower than this.
    ///
    /// The pass costs an `apply_step` and an `eval::heuristic` per edge, so its
    /// cost is linear in width while the *benefit* is not: a two-edge node is
    /// resolved by three simulations whatever its prior says. Widths here run
    /// 2..2293 with a median searched width of 3 (`phase.rs`), so the threshold
    /// is where most of the saving is.
    ///
    /// Set it low, but not to 2. At 3 the prior is worth +8.84 against
    /// `mcts:2048`; at 8 it is worth **+3.46** (CI +2.06..+4.85, 200 blocks).
    /// Skipping the probe on 3-to-7-edge nodes skips it on most of the tree,
    /// and most of the tree is where the strength was.
    ///
    /// Downward it stops paying and starts costing — **at `c_puct = 2.0`.**
    /// `pmin=2` measures -1.35 against the `pmin=3` default (CI -2.37..-0.34,
    /// 96 blocks, 2026-09-08), and the explanation was that a two-edge node is
    /// resolved by three simulations whatever its prior says, so the probe buys
    /// nothing there while softmaxing two scores at `pt=1` produces a *sharper*
    /// prior than the uniform it replaces.
    ///
    /// # That verdict inverts in the deep regime, and this is the second-largest
    /// constant in the file
    ///
    /// At `cp = 0.02` and 8,192 simulations, `pmin=2` reproduces on **three
    /// separate evaluators** — +4.73 (CI +3.22..+6.24, 50 paired blk),
    /// **+2.78** (CI +2.39..+3.16, **600 paired blk**, p = 5e-46), and +3.12
    /// (CI +2.23..+4.00, 117 paired blk) — so it is not an artefact of the
    /// platform it was found on. Call it **three points**.
    ///
    /// **It is free.** Measured against `cp=0.02` in one sample, `pmin=2` is
    /// **0.98x** the user CPU (182.0 against 186.5 ms/turn, two alternated
    /// reps). An earlier 1.15x came from comparing two separately-taken
    /// samples. A more accurate prior makes the descent more decisive, and the
    /// descent it saves pays for the extra one-ply probes at two-edge nodes.
    ///
    /// **It does not stack with the budget.** The increment is +3.20 (CI
    /// +2.23..+4.17, 109 paired blk) at 32,768 simulations against +3.00 on the
    /// same seeds at 8,192 — a difference-in-differences of **+0.20 (CI
    /// -1.12..+1.53, p = 0.76)**. `c_puct`, the budget and this knob are three
    /// independent contributions, which is why no 1-D sweep ever found any of
    /// them. (An eight-block reading once suggested +7.8 of stacking here; at
    /// 109 blocks it is +0.20. Low-block readings drift.)
    ///
    /// The mechanism is the descent length. A node's *median* searched width is
    /// **2**, so `prior_min_edges = 3` means roughly half of all nodes get a
    /// uniform prior. At `cp = 2.0` a descent is ~8 sub-decisions and passes
    /// through a handful of them; at `cp = 0.02` it is ~56 and passes through
    /// about **28 blind nodes per descent**. "A two-edge node is resolved by
    /// three simulations" is true of a node the search will return to; it is
    /// false of a node the search visits once on its way sixty levels down,
    /// where the choice of edge *is* the descent.
    ///
    /// 2 is the floor (`record.rs` clamps with `.max(2)`), so this knob is now
    /// pinned at its most aggressive setting and there is nothing further to
    /// win on this axis.
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
    /// [`Gradient`]'s linear price list, softmaxed at the same `prior_temp`.
    ///
    /// The units line up with [`Priors::OnePly`] by construction — both are
    /// `eval::heuristic` points — so the temperature means the same thing, but
    /// the *spread* does not, and `prior_temp` has to be swept again rather
    /// than carried over.
    ///
    /// **It is blind wherever `Gradient::step` is.** Only `Step::Take` carries
    /// a `Choice`, so `Placing`, `PickWorker`, `Mode` and `Beg` all price to
    /// zero and softmax back to uniform. `Placing` is not a narrow phase — it
    /// runs to a few hundred edges — so this is not a small blind spot, and it
    /// is why [`Priors::Mixed`] exists.
    ///
    /// # Is `OnePly` worth 7.7x per edge? At equal simulations, by 0.77 points
    ///
    /// Against `mcts:2048:heuristic:quality` at the same 2,048 simulations this
    /// is **-0.77 (CI -1.49..-0.04, 191 blocks)** — a real loss, but a small
    /// one for a prior that is blind on every phase but `Take`. It costs
    /// **0.79x** the user CPU of the one-ply probe (11.55 ms/turn against
    /// 14.62) and only **1.05x** the uniform prior's 11.02, so as a prior it is
    /// very nearly free.
    ///
    /// Spend the saving and the loss goes away: `mcts:2450:heuristic:priors=grad`
    /// is measured at **0.98x** the champion's user CPU and scores **+0.47 (CI
    /// -0.48..+1.42, 96 blocks)**. **At matched CPU the two priors are
    /// indistinguishable.** Which of them to prefer is therefore a question
    /// about what a simulation is worth — see the module header, where the
    /// answer turns out to depend entirely on `c_puct`.
    ///
    /// # And in the deep regime it is not close: -4.04
    ///
    /// Everything above is measured at `c_puct = 2.0`, where a descent is ~8
    /// sub-decisions. At `cp = 0.02` and 8,192 simulations this prior is
    /// **-4.04 (CI -4.72..-3.36, 321 paired blocks, p = 1e-31)** — five times
    /// the loss, and no longer recoverable by spending the saving.
    ///
    /// The direction follows from where the blind spot is. **The prior is
    /// consulted at every node of a descent**, and a 56-sub-decision descent
    /// consults it seven times as often as an 8-sub-decision one, so a prior
    /// that softmaxes to uniform on `Placing`, `PickWorker`, `Mode` and `Beg`
    /// commits a blind step seven times as often. `OnePly`'s 7.7x per-edge
    /// cost is a wash at `cp = 2.0` and clearly worth paying at `cp = 0.02` —
    /// which is the answer to "is `OnePly` worth it", and it is
    /// budget-dependent in the same way everything else in this file turned
    /// out to be.
    Gradient,
    /// [`Priors::Gradient`] on `Take` nodes, [`Priors::OnePly`] everywhere else.
    ///
    /// The two costs and the two blind spots line up the right way round: the
    /// wide phase is the one the gradient can price at 16 ns an edge, and the
    /// phases it cannot price are the ones where a 123 ns one-ply probe is
    /// affordable because there are few edges to spend it on.
    ///
    /// It behaves like the compromise it is: **+0.48 (CI -0.25..+1.20, 190
    /// blocks)** against the one-ply default at equal simulations, for 0.97x the
    /// user CPU, and it changes only 0.7% of played turns (`bin/sprobe mcts`).
    /// Nothing to choose between it and `OnePly` on the evidence; it is here
    /// because it is the shape a trained policy head would want if the head
    /// covered only some phases.
    ///
    /// **In the deep regime the compromise is not one: -4.26** (CI
    /// -7.31..-1.21, 13 paired blocks) at `cp = 0.02` and 8,192 simulations,
    /// which is [`Priors::Gradient`]'s -4.04 essentially undiminished. The
    /// reason is that the phase it hands to the gradient — `Take` — is exactly
    /// where a long descent spends its length, so half a blind prior is most of
    /// a blind prior. Do not read the +0.48 above as "free"; it is a
    /// `c_puct = 2.0` number.
    Mixed,
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

/// What a temperature-zero search plays. See [`MctsConfig::root_pick`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RootPick {
    /// §3.8's rule: the most-visited edge.
    Visits,
    /// The best mean value for the mover, among edges carrying at least a
    /// hundredth of the node's visits.
    ///
    /// The floor is the whole difference between this and a rule that would
    /// hand the turn to any edge a single fortunate simulation ran through. At
    /// 2,048 simulations it asks for 20 visits before an edge may be played,
    /// which is enough that its mean is a mean; at 8 it degenerates to
    /// "anything visited", which is the right answer when there is nothing to
    /// be robust about.
    Value,
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
            root_pick: RootPick::Visits,
            virtual_loss: 1,
            max_depth: 2048,
            seed: 0,
            // `Priors::OnePly` at `prior_temp = 1` -- what the `quality` preset
            // spelled -- is the default because it is worth +8.84 centred
            // against this same search with the old defaults (95% CI
            // +7.24..+10.44, 202 blocks). The `prior_temp` sweep against
            // `mcts:2048` reads 0.5 -> +6.42, 1.0 -> +8.84, 2.0 -> +5.56,
            // 4.0 -> +1.01, so the 4.0 this shipped with was throwing away
            // seven of the eight points; re-swept at 1.0 on 2026-09-08,
            // 0.5 -> +0.00 (98 blk) and 1.5 -> -0.18 (88 blk), so 1.0 is a
            // genuine optimum and a flat one.
            //
            // The reason this comment used to give -- "the simulation budget is
            // a dead axis without it, and 8x is worth +3.04 once the prior is
            // real" -- **did not replicate**: that exact pair, corrected by its
            // own null, is +0.03 (CI -1.84..+1.90, 43 paired blocks). The
            // budget is unlocked by `c_puct_init`, not by the prior. See the
            // module header.
            priors: Priors::OnePly,
            prior_temp: 1.0,
            prior_margin: false,
            q_init: 0.0,
            q_pseudo: 0.0,
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
    /// What an *unvisited* edge's Q is worth relative to its siblings, from the
    /// same one-ply scores the prior was softmaxed from. Zero unless
    /// [`MctsConfig::q_init`] is set. See [`MctsConfig::q_init`].
    q0: f32,
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
    /// Descent depth in sub-decisions, for the last `search_at`. Counters, read
    /// by [`Mcts::depth_stats`]; nothing in selection or backup looks at them.
    deepest: u32,
    depth_sum: u64,
    depth_n: u64,
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
            deepest: 0,
            depth_sum: 0,
            depth_n: 0,
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
        // Per sub-decision, so `depth_stats` describes this search and not the
        // whole turn.
        self.deepest = 0;
        self.depth_sum = 0;
        self.depth_n = 0;

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

    /// How deep the deepest descent of the last `search_at` went, and the mean
    /// descent depth, both counted in **sub-decisions**.
    ///
    /// The unit is the point. A turn is ~8 sub-decisions, so a tree 7 deep has
    /// not finished looking at its own turn and has certainly not seen a reply.
    /// That is the number that says whether a budget increase bought lookahead
    /// or only width. Counters only; they do not enter selection.
    pub fn depth_stats(&self) -> (u32, f64) {
        (
            self.deepest,
            self.depth_sum as f64 / self.depth_n.max(1) as f64,
        )
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
        let mut reached = 0u32;
        let mut value = [0.0f32; N_PLAYERS];
        let vl = self.cfg.virtual_loss;

        for depth in 0.. {
            let cur = self.cursor(&path, root);
            // A descent can run to the end of the game through transpositions
            // and collapsed nodes; the guard is against a graph cycle, which
            // the state advancing on every edge should already rule out.
            reached = depth;
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
        self.deepest = self.deepest.max(reached);
        self.depth_sum += reached as u64;
        self.depth_n += 1;
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
        let k_pseudo = self.cfg.q_pseudo;

        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, edge) in open.iter().enumerate() {
            // An unvisited edge is worth the parent, less urgency, plus what
            // the one-ply probe already said about *this* edge. With `q_init`
            // and `q_pseudo` both at 0 neither term is present and every
            // unvisited sibling shares one Q, which is the behaviour every
            // measurement before 2026-09-08 was made on.
            let q = if k_pseudo > 0.0 {
                (edge.w[mover] + k_pseudo * (parent_q + edge.q0))
                    / (edge.n as f32 + k_pseudo)
            } else if edge.n > 0 {
                edge.w[mover] / edge.n as f32
            } else {
                fpu + edge.q0
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
            if self.cfg.root_pick == RootPick::Value {
                if let Some(i) = self.best_by_value(idx) {
                    return i;
                }
            }
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

    /// The best-valued open edge, or `None` if nothing cleared the visit floor
    /// and the caller should fall back to visits.
    fn best_by_value(&self, idx: u32) -> Option<usize> {
        let node = &self.nodes[idx as usize];
        let mover = node.mover;
        let open = &node.edges[..node.active];
        let total: u32 = open.iter().map(|e| e.n).sum();
        let floor = (total / 100).max(1);
        open.iter()
            .enumerate()
            .filter(|(_, e)| e.n >= floor)
            .max_by(|a, b| {
                let q = |e: &Edge| e.w[mover] / e.n as f32;
                q(a.1).total_cmp(&q(b.1))
            })
            .map(|(i, _)| i)
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
            (vec![new_edge(step, 1.0, 0.0)], None)
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
            let (priors, q0) = self.priors_for(&state, phase, turn, done, &steps, from_eval);
            let mut edges: Vec<Edge> = steps
                .into_iter()
                .enumerate()
                .map(|(i, step)| {
                    new_edge(
                        step,
                        priors.get(i).copied().unwrap_or(0.0),
                        q0.get(i).copied().unwrap_or(0.0),
                    )
                })
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
    ) -> (Vec<f32>, Vec<f32>) {
        let mover = phase.mover(turn);
        let wide = steps.len() >= self.cfg.prior_min_edges;
        let is_take = matches!(phase, Phase::Take { .. });
        // The raw one-ply scores, in `heuristic` points, before the softmax
        // collapses them. Empty means "no scores here": the evaluator's own
        // prior, or a node too narrow to probe.
        let raw: Vec<f32> = match self.cfg.priors {
            _ if !wide => Vec::new(),
            Priors::OnePly => self.one_ply(state, phase, turn, done, steps, mover),
            Priors::Gradient => self.grad_scores(state, steps, mover),
            // `Gradient::step` prices a `Choice` and nothing else, so on any
            // other phase the gradient softmax is uniform and the one-ply probe
            // is the only one of the two that says anything.
            Priors::Mixed if is_take => self.grad_scores(state, steps, mover),
            Priors::Mixed => self.one_ply(state, phase, turn, done, steps, mover),
            Priors::Evaluator => Vec::new(),
        };
        let q0 = self.q_init_from(&raw);
        let mut p = if raw.is_empty() {
            from_eval
        } else {
            let mut p = raw;
            let best = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            self.softmax(&mut p, best);
            p
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
        (p, q0)
    }

    /// The unvisited-edge Q offset for each edge, from the raw one-ply scores.
    ///
    /// Centred on the sibling mean rather than the max: FPU already decides how
    /// pessimistic an unvisited edge is in absolute terms, and this only has to
    /// say which of them is better than which. Squashed through
    /// `HeuristicEvaluator`'s own `tanh(pts / 25)` so the offset is on the same
    /// scale as the Q it is added to, and bounded by `q_init` however wild the
    /// point spread gets.
    fn q_init_from(&self, raw: &[f32]) -> Vec<f32> {
        let k = self.cfg.q_init;
        if k == 0.0 || raw.is_empty() {
            return Vec::new();
        }
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        raw.iter().map(|h| k * ((h - mean) / 25.0).tanh()).collect()
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
        for step in steps {
            let mut next = *state;
            let _ = tree::apply_step(&mut next, phase, turn, done, step);
            // The mover's own estimate, not `eval::margin`. Siblings of one
            // node differ almost only in what the mover did, so the
            // best-opponent term is common to them and subtracting it would
            // cost three more `heuristic` calls an edge to change nothing.
            out.push(if self.cfg.prior_margin {
                crate::eval::margin(&next, mover)
            } else {
                crate::eval::heuristic(&next, mover)
            });
        }
        out
    }

    /// [`Gradient`]'s price list as a prior: one dot product per edge instead of
    /// an `apply_step` and a `heuristic` call.
    ///
    /// 16 ns an edge against the one-ply probe's 123 (see [`Gradient`]), and the
    /// probe itself is amortised — `gradient` builds one price list per mover
    /// per sub-decision and caches it. The scores are in the same `heuristic`
    /// points [`one_ply`](Mcts::one_ply) returns, so `prior_temp` carries over
    /// unchanged as a *unit*; the spread does not, so it does not carry over as
    /// a *value*.
    fn grad_scores(&mut self, state: &GameState, steps: &[Step], mover: PlayerId) -> Vec<f32> {
        let g = self.gradient(state, mover);
        steps.iter().map(|s| g.step(s)).collect()
    }

    /// Softmax in place at `prior_temp`, shifted by `best` before exponentiating:
    /// `heuristic` runs to ~200 points late in a game and `exp(200/4)` is not a
    /// number.
    fn softmax(&self, out: &mut [f32], best: f32) {
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

fn new_edge(step: Step, prior: f32, q0: f32) -> Edge {
    Edge {
        step,
        prior,
        q0,
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
