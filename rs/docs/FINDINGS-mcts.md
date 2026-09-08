# FINDINGS — `src/mcts.rs`

Every measurement this workstream lands, appended the moment it lands. Three
agent runs were killed by usage limits on 2026-09-07 and only what was on disk
survived; assume the same.

## The answer, if you read nothing else

**`mcts:8192:heuristic:deeper`** — sugar for `mcts:8192:heuristic:cp=0.02,pmin=2`.
**Two constants**, both shipped one to two orders of magnitude wrong for a
search factored into sub-decisions, and neither of them the simulation budget.

| | `c_puct_init` | `prior_min_edges` |
|---|---|---|
| shipped | 2.0 | 3 |
| wants to be | **0.02** | **2** |
| worth, at 8,192 sims, paired on seed | **+10.39** (+7.68..+13.10, 13 blk, `v8`) | **+3.06** (+1.43..+4.68, 46 blk, `v8`) / +4.73 (+3.22..+6.24, 50 blk, `v6`) |

Against a stateless `heuristic:full`, whose null control is **exactly +0.00 with
a win rate of exactly 0.250**, the champion takes **80% of its games** where the
champion this run started with takes 34%.

**Why, in one sentence each.**

* `c_puct` is the only thing that converts a simulation budget into *lookahead*
  in a search factored into sub-decisions; at 2.0 the descent stopped after ~8
  sub-decisions, which is one turn, so every extra simulation re-decided what
  the one-ply prior had already decided. At 0.02 the mean descent is ~56.
* `prior_min_edges = 3` gave a **uniform** prior to every node narrower than
  three edges, and the *median searched node is two edges wide* — so about half
  of every descent was blind. At 8 sub-decisions that was a handful of nodes and
  cost nothing; at 56 it is ~28 blind nodes per descent, and fixing it is worth
  three to five points for **1.15x the CPU**.

**They are one finding, not two.** Lowering `c_puct` did not merely add points —
it re-opened every knob that had been measured inert, because "inert" was a
statement about a search that could not look ahead. `fpu`, `q_pseudo`,
`tree_reuse`, `prior_temp` and the prior source all flipped in the same
direction and for the same reason; `max_edges` / `widen_cap` / `widen_c` did
not, because they act on wide nodes and a deep descent lives in narrow ones.

**More budget is worth more and costs a lot.** The ladder at `cp = 0.02`, paired
against 8,192: 2,048 is -4.79, 4,096 is -1.90, 32,768 is **+2.45** (99 paired
blocks), 65,536 is +3.15 (14 blocks — i.e. +0.70 over 32,768, interval covering
zero). **The ridge stops paying at about 32,768 simulations**, and that rung
costs 59x the old champion's user CPU per turn against the shipped rung's 12.6x.

## How to run it

    # step through a game against the champion, and watch its search panel
    cargo run --release --bin tui -- --agent mcts:8192:heuristic:deeper

    # `deeper` is sugar for cp=0.02,pmin=2; these two are the same agent and
    # `examples/nametest.rs` and `tests/search.rs` both check that they print
    # the same name. `deep` is the older sugar and means cp=0.02 alone.
    cargo run --release --bin tui -- --agent mcts:8192:heuristic:cp=0.02,pmin=2

    # the strongest thing measured, at 59x the old champion's CPU per turn
    cargo run --release --bin tui -- --agent mcts:32768:heuristic:deeper

    # race it, from a PINNED binary, never target/release
    <mine>/arena --candidate mcts:8192:heuristic:deeper \
      --baseline heuristic:full --games 800 --seed 3000000 --resume \
      --out <mine>/champ.jsonl

    # and the control that says the harness is fair: this one returns
    # exactly +0.00 with a win rate of exactly 0.250
    <mine>/arena --candidate heuristic:full --baseline heuristic:full \
      --games 400 --seed 3000000 --resume --out <mine>/null.jsonl

**Two caveats for the viewer.**

1. A deep search is a *more* concentrated one: the root's top edge holds 93.5%
   of visits at `cp=0.02` against 88.4% at the default, so a panel that ranks
   turns by visit count will look more one-sided than before.
   `mcts::TurnLine` carries `value` next to `visits`; that is the field to show
   beside it if the ranking is to stay informative.
2. **It is slow on purpose.** 181 ms of user CPU per turn at 8,192 simulations
   and 853 ms at 32,768, single-threaded, against the old champion's 14.4 ms.
   The TUI must show progress rather than appear frozen. Threading would fix
   the latency and buy no strength per CPU — see the `# Threading` section of
   `src/mcts.rs`.

## How to read a row

* **Strength** is arena `--mode solo` mean **centred score**, candidate minus
  the four-seat mean. Null is 0; null win rate is 25%. **The block is the
  independent unit**, not the game — `n` is always blocks, and 4 games make a
  block. An effect smaller than its interval is not an improvement.
* **Cost** is `/usr/bin/time` **user CPU**, never wall clock. The same pair
  measured 1.52x and 1.03x on consecutive wall-clock runs while the CPU figures
  repeated to 3%.
* Every race in a wave runs in **one pinned binary** and on **one seed
  sequence**, so any two runs in a wave are also a matched (paired-on-seed)
  comparison of their candidates. `mc/pair.py A B` does that.

## Design of the 2026-09-08 waves

All runs share `--baseline mcts:2048:heuristic:quality` — the reigning champion
at the start of the run — and `--seed 3000000`. That makes every row directly
readable as "against the champion", and makes any two rows a paired comparison
against a common opponent on identical decks.

## Binaries

Every one of these was checked against its predecessor on 24 games at seed 99000
and is **identical at the default config** — each addition defaults to off. That
is what makes rows from different waves poolable. Re-run the check before
trusting a new build: another workstream was transiently `git show`-ing older
`options.rs` / `moves.rs` into the working tree during this session.

| tag | adds | default changed? |
| --- | --- | --- |
| `arena_v1` | baseline, git `3fe213e` | — |
| `arena_v2` | `Priors::Gradient`, `Priors::Mixed` | no |
| `arena_v3` | `q_init` | no (0.0) |
| `arena_v4` | `prior_margin` | no (false) |
| `arena_v5` | `RootPick` | no (Visits) |
| `arena_v6` | `q_pseudo`, `Mcts::depth_stats` | no (0.0; counters only) |
| `arena_v7` | the `deep` preset | no (sugar only) |
| **`arena_v8`** | **nothing of mine — it is `arena_v6`'s source plus the evaluator of `d7b4e42` and the generation workstream's in-flight `options.rs`** | **YES, and that is the point** |

Almost every run below used `arena_v6`; the `sims-*` ladder used `arena_v1`.

**`arena_v8` is not poolable with anything above it.** `arena_v6` was built at
00:43, after `94d85f3` and *before* `d7b4e42` (02:50), so every `v6` row is on
the older evaluator. The same 24-game check at seed 99000 that certified
`v1..v7` shows **0 of 6 blocks identical** between v6 and v8. Rows are named
`v8-*` for that reason and **must never be pooled or paired with a `v6` row**;
each platform has its own null (`null-self` -0.10 over 374 blk on v6,
`v8-null` -1.41 over 145 blk on v8). Fingerprint in `mc/bin/arena_v8.rev`.

## Filename → meaning

Everything lives under
`<scratch>/mc/`: `runs/` the block files, `bin/` the pinned binaries, `logs/`
per-job stdout, `jobs/` the lane job lists, `dirty/` the pre-cleanup copies of
the files a broken summariser wrote into.

Unless a row says otherwise the baseline is `mcts:2048:heuristic:quality`, the
seed is 3000000, the mode is solo, and the binary is `arena_v6`.

| file | candidate | note |
| --- | --- | --- |
| `null-self` | `mcts:2048:heuristic:quality` | **the control.** Candidate == baseline; the zero for every row below |
| `null-greedy` | `heuristic:full` vs `heuristic:full` | stateless control; returns -0.01, win 0.250 |
| `null-noreuse`, `null-256`, `null-256b` | as named | further controls |
| `nullp-self` | as `null-self`, `--mode pairs` | symmetric-sharing control |
| `sims-N` | `mcts:N:heuristic:quality` | the budget ladder at the shipped `c_puct` |
| `low-N` | `mcts:N:heuristic:quality` | the bottom of the same ladder, N in 1..128 |
| `c-cpX` | `cp=X` | `c_puct_init` sweep at 2048 sims |
| `c-fpuX`, `c-cpb1e3` | `fpu=X`, `cpb=1000` | the other PUCT knobs; all inert |
| `p-grad`, `p-mixed` | `priors=grad`, `priors=mixed` | prior source at equal sims |
| `p-grad-cpu` | `mcts:2450:heuristic:priors=grad` | the gradient prior at **matched CPU** (0.98x) |
| `p-pmin2`, `p-ptX` | `pmin=2`, `pt=X` | prior sharpness |
| `q-qX`, `q-qnX` | `q0=X`, `q0=1,qn=X` | one-ply value as FPU / as pseudo-visits |
| `m-pmarg` | `pmarg` | margin-scored one-ply probe |
| `r-pickq` | `pick=q` | root selection by value |
| `s-noreuse`, `s-k8`, `s-nocap`, `s-widen` | as named | the "measured inert" stale list |
| `x-SIMScpC` | `mcts:SIMS:heuristic:cp=0.0C` | **the deep ladder** — where the strength is |
| `d-grad`, `d-pt2`, `d-mixed` | prior variants at `8192:cp=0.02` | does the prior retune in the deep regime |
| `g-*-vs-greedy` | vs `heuristic:full` | independent confirmation against a stateless opponent |
| `pr-8192cp002` | `--mode pairs` | symmetric-sharing confirmation |
| `rep-2048v256` | `mcts:2048:quality` vs `mcts:256:quality` | replication of the module header's +3.04 |
| `d-pmin2`, `d-fpu00`, `d-noreuse`, `d-nocap`, `d-k8`, `d-qn16`, `d-pickq`, `d-mixed2`, `d-pmarg`, `d-cpb4913` | one flag on top of `8192:cp=0.02` | **the re-sweep of the `cp=2.0`-era verdicts** |
| `x-4096cp002`, `x-4096cp004`, `x-2048cp002`, `x-16384cp002`, `x-32768cp001`, `x-65536cp002` | the ridge | `cp` x budget |
| `x-32768pmin2` | `32768:cp=0.02,pmin=2` | the champion rung; **8 blocks, killed by memory** |
| **`v8-*`** | **`arena_v8`, the shipping evaluator** | `v8-null` its null, `v8-null-greedy` its stateless control, `v8-8192cp002`/`v8-8192pmin2` the champion pair, `v8-champ-vs-greedy`/`v8-oldchamp-vs-greedy`/`v8-champ-vs-ab`/`v8-pairs-champ` the independent opponents |

## Tools

| | |
| --- | --- |
| `mc/sum.py RUN...` | read-only summary. **Never point the arena at a live run file** — see the retraction |
| `mc/pair.py A B` | paired-on-seed difference, ~4x tighter than two unpaired intervals |
| `mc/pool.py A1,A2 -- B1,B2` | the same for groups |
| `mc/cost.sh REPS SPEC...` | per-turn **user CPU**, one spec per process, alternated |
| `mc/lane.sh JOBS PAR` | run a job list at a concurrency limit; refuses a second writer on one file |
| `bin/sprobe budget SIMS...` | agreement, visit concentration and **descent depth** by budget; `SPROBE_CP`/`SPROBE_CPB` override the constants |
| `bin/sprobe mcts SPEC...` | ms/turn and move agreement with the first spec |
| `bin/sprobe values` | is the points-to-value map saturating |

## Measurements

<!-- append below, newest last -->

---

## RETRACTION — every arena number logged before 00:53 was contaminated

**What happened.** `mc/sum.sh` summarised a run by handing the live `.jsonl` to
`arena --resume`. It passed no `--seed`, so the arena used its default seed
sequence, decided that none of the blocks on file were the ones it wanted, and
**played a fresh `random` vs `random` run into the file it was being asked to
read**. A second version passed the right seed but still let the arena fill the
gaps a killed run had left. Between 00:15 and 00:53 this appended 96-199 junk
blocks to every file it touched.

**Why it was not obvious.** The junk blocks pulled the means toward zero, which
is exactly the answer the experiment was testing for. Worse, two files summarised
together got junk at the *same* seeds from the *same* deterministic
`random`-vs-`random` games, so paired differences at those seeds were **exactly
0** — deflating the mean and the interval at once. `mc/pool.py` reported the
budget bound as "+0.01, CI -0.43..+0.45, 104 paired blocks" when it had ~25 real
blocks and no such precision.

**Caught by** an impossible arithmetic: block counts rose by 99 in two minutes of
wall clock on a machine that produces one block every eight CPU-seconds.

**Fixed.** `mc/sum.py` computes the statistic in Python and never runs the arena
— the block is the unit and the statistic is a mean over blocks, so there was
never anything for the arena to compute. Verified identical to the arena's own
summary on a clean file (both -32.069, CI -33.084..-31.053, 83 blocks). Junk
blocks were stripped (originals kept in `mc/dirty/`), and `pair.py`/`pool.py`
now drop any seed below 2e6 as well.

**Unaffected:** everything from `bin/sprobe`, which never touches a run file —
the budget/agreement diagnostics and the value-scale measurement below stand.
The three entries above them are superseded by the clean numbers that follow.

### 2026-09-08 00:12 — `dirichlet_eps` has no spec flag, and must not get one

Tried adding `eps=`. `record::SearchAgent::with_config` already zeroes
`dirichlet_eps` under `Exploration::Off`, which is every arena caller, so §3.7's
root noise was **already off** in evaluation. The flag parsed, was overwritten,
and printed `:eps=0` into the label of every default spec — caught by
`examples/nametest.rs` inside ten minutes. Reverted; the reason is now a comment
on the field so the next person does not spend the same hour.

### 2026-09-08 00:26 — binaries `v1`..`v6` are behaviourally identical at default

24 games, seed 99000, `mcts:512:heuristic:quality` vs `heuristic:8`: the block
files are identical up to completion order. `v2` adds `Priors::Gradient`/`Mixed`,
`v3` `q_init`, `v4` `prior_margin`, `v5` `RootPick`, `v6` `q_pseudo` — every one
defaulted off. So all runs below are one matched experiment sharing the baseline
`mcts:2048:heuristic:quality` and the seed sequence from 3000000, and any two of
them may be paired on seed (`mc/pair.py`, `mc/pool.py`).

Re-run the check before trusting a new binary: another workstream was
transiently `git show`-ing old `options.rs`/`moves.rs` into the working tree
during this session, so a rebuild can silently catch a different engine.

### 2026-09-08 00:31 — how often more simulations change the move (`bin/sprobe budget`)

`SPROBE_GAMES=8 SPROBE_EVERY=5 SPROBE_NODES=40`, 196 sub-decisions, every rung
choosing at the *same* nodes (all driven along the top rung's path).

| sims | agrees with 16384 | top edge's visit share | arena nodes | ms/decision |
|---|---|---|---|---|
| 256 | 95.4% | 80.8% | 366 | 0.82 |
| 1024 | 96.4% | 83.7% | 1456 | 3.14 |
| 4096 | 97.4% | 86.9% | 5745 | 12.42 |
| 16384 | 100% | 88.8% | 23364 | 67.23 |

and at the bottom of the ladder, 191 sub-decisions driven along `mcts:2048`:

| sims | agrees with 2048 | top edge's visit share | arena | ms/dec |
|---|---|---|---|---|
| 1 | 81.2% | 100% | 2 | 0.00 |
| 8 | 90.6% | 79.2% | 12 | 0.01 |
| 32 | 94.2% | 78.9% | 46 | 0.06 |
| 128 | 95.8% | 80.0% | 183 | 0.17 |
| 512 | 96.9% | 82.2% | 731 | 0.93 |

**Read these per sub-decision, and a turn is about eight of them.** 81.2% per
sub-decision is `0.812^8 = 19%` of *turns* played the same way, not 81% — an
earlier version of this entry drew the opposite conclusion from the same table
and called the search "a 19% correction on a policy player". The arena says the
reverse: `mcts:1` is **-32.10** centred and wins **0%** of its games. The prior
alone is not a player. What the table actually shows is where the *marginal*
simulation stops mattering: 95.4% per sub-decision at 256 against 16,384 is
`0.954^8 = 69%` of turns identical.

The arena grows at ~1.4 nodes per simulation at every rung, so extra budget buys
a wider tree and not a deeper one — and at ~8 sub-decisions per turn, even
16,384 simulations see barely one opponent cycle ahead.

### 2026-09-08 00:45 — which knobs change a move at all (`bin/sprobe mcts`)

Before spending an arena hour on a knob, ask whether it moves anything. 405
turns, 135 positions, each spec played against the same position and compared to
the default's move. Agreement here is a **lower bound on divergence** — it is one
sample of positions, and in a real game differences compound.

| spec | ms/turn | x default | agrees with default |
|---|---|---|---|
| `quality` (default) | 23.4 | 1.00 | 100% |
| `cp=0.06` | 34.4 | 1.48 | **65.9%** |
| `cp=0.25` | 27.8 | 1.19 | 88.1% |
| `cp=0.5` | 26.6 | 1.10 | 91.1% |
| `cp=1.0` | 24.9 | 1.07 | 97.0% |
| `priors=grad` | 19.6 | **0.81** | 92.6% |
| `priors=mixed` | 21.4 | 0.88 | 99.3% |
| `q0=1.0,qn=16` | 23.8 | 1.02 | 97.8% |
| `pick=q` | 23.5 | 0.97 | 98.5% |
| `pmarg` | 30.8 | 1.27 | 99.3% |
| `q0=1.0` / `q0=4.0` | 23.6 | 1.01 | **100.0% / 100.0%** |

1. **`c_puct` has more leverage on what this search plays than anything else in
   the file**, and lowering it costs CPU (1.48x at 0.06) because a more
   exploitative descent goes deeper and transposes less.
2. **`q_init` alone is exactly inert** — 0 of 405 turns at `q0=4`, even at
   `cp=0.06`. Nodes are p50 3 edges wide against thousands of simulations, so
   "unvisited edge" is a state that lasts two descents. FPU is a knob for
   searches whose nodes are wider than their budgets, and this is not one.
   `q_pseudo` is the form that survives the first visit; it reaches 2.2% at
   `qn=16`.
3. **`pmarg` costs 27% and changes 0.7% of turns.** `eval::heuristic` does not
   price an opponent's *lost options*, so taking the space they needed does not
   move their estimate and a margin-scored probe is very nearly a
   heuristic-scored one. Making denial visible is an `eval.rs` change.

### 2026-09-08 00:47 — the value map is not the problem (`bin/sprobe values`)

A saturated value function would explain a flat budget curve by construction.
`phase::HeuristicEvaluator` maps points to value with `tanh((h - mean) / 25)`;
over 3,840 (position, player) leaves from 960 positions:

| | p05 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| `h - mean`, points | -12.0 | -6.2 | +0.1 | +5.0 | +13.6 |
| `tanh(d/25)` | -0.445 | -0.245 | +0.004 | +0.198 | +0.497 |

**0.0% of leaves exceed \|0.9\|.** The map is well scaled; the hypothesis is dead.

It leaves one number for the `c_puct` sweep: the value function's *working* range
is about +/-0.5, and two siblings differ by a few points, which is a Q difference
near 0.1. `c_puct_init = 2.0` was "calibrated for a value in (-1, 1)". At
`N = 2048` an edge holding a tenth of the prior and fifty visits carries
`u = 0.18` — nearly twice the Q difference it is supposed to be arbitrating.

### 2026-09-08 00:53 — first clean arena sweep

All against `mcts:2048:heuristic:quality`, `--seed 3000000`, solo, `arena_v1..v6`
(identical at default). Junk stripped; block counts are what survived, so these
are screens, not verdicts. `+` is better than the champion.

| run | spec | blk | centred | 95% CI |
|---|---|---|---|---|
| `low-1` | `mcts:1` | 84 | **-32.10** | -33.10..-31.10 |
| `low-32` | `mcts:32` | 77 | **-2.28** | -3.57..-0.99 |
| `sims-256` | `mcts:256` | 110 | -0.24 | -1.24..+0.76 |
| `sims-512` | `mcts:512` | 110 | +0.34 | -0.71..+1.39 |
| `sims-1024` | `mcts:1024` | 102 | -0.26 | -1.29..+0.77 |
| `sims-4096` | `mcts:4096` | 62 | +0.50 | -0.83..+1.82 |
| `sims-8192` | `mcts:8192` | 91 | **+1.17** | +0.19..+2.15 |
| `sims-16384` | `mcts:16384` | 47 | +0.56 | -0.94..+2.05 |
| `sims-32768` | `mcts:32768` | 24 | +0.14 | -1.75..+2.03 |
| `p-mixed` | `priors=mixed` | 101 | +0.72 | -0.33..+1.76 |
| `p-grad` | `priors=grad` | 100 | -0.31 | -1.19..+0.58 |
| `p-pmin2` | `pmin=2` | 96 | **-1.35** | -2.37..-0.34 |
| `p-pt0.5` | `pt=0.5` | 98 | +0.00 | -1.11..+1.12 |
| `p-pt1.5` | `pt=1.5` | 88 | -0.18 | -1.32..+0.96 |
| `r-pickq` | `pick=q` | 47 | **-2.34** | -3.85..-0.82 |
| `s-noreuse` | `noreuse` | 56 | +1.26 | -0.22..+2.75 |
| `m-pmarg` | `pmarg` | 49 | +0.47 | -1.04..+1.98 |
| `q-q2.0` | `q0=2.0` | 61 | +1.01 | -0.18..+2.21 |
| `c-fpu0.6` | `fpu=0.6` | 13 | +2.55 | -0.10..+5.20 |
| `c-fpu0.0` | `fpu=0.0` | 14 | +2.21 | -0.18..+4.61 |
| `c-cp0.06` | `cp=0.06` | 26 | +1.68 | -0.10..+3.46 |

**The budget axis is real at the bottom and saturates by 256.** `mcts:1` is not
a player at all (-32.1, 0% win rate); `mcts:32` is 2.3 points down and the
interval excludes zero; from 256 upward nothing is distinguishable from the
2,048 default except a marginal `mcts:8192` at +1.17 (+0.19..+2.15). **The knee
is between 32 and 256 simulations, and above it the curve is flat to 32,768.**

Two settled negatives at 95%: `pmin=2` (-1.35) and `pick=q` (-2.34). Most-visited
really is the right root rule even though visits are prior-heavy — see
`RootPick`.

Live leads, all under-powered: `fpu`, `noreuse`, `cp=0.06`, `priors=mixed`,
`q0=2.0`. CPU redirected to them.

### 2026-09-08 01:00 — the arena's null control, and a +1.6 that was not there

Run the arena with the candidate spec *identical to the baseline spec*. The
answer must be zero. Four controls at 20 blocks each read:

| control | candidate = baseline | centred @20 blk | 95% CI |
|---|---|---|---|
| `null-greedy` | `heuristic:full` | **-0.01** | -0.03..+0.01 |
| `null-self` | `mcts:2048:heuristic:quality` | +1.63 | -0.13..+3.40 |
| `null-noreuse` | `...:noreuse` | +0.59 | -1.18..+2.36 |
| `null-256` | `mcts:256:heuristic:quality` | +0.38 | -2.04..+2.80 |

**I wrote +1.63 up as a harness bias. It was not one.** At 162 blocks
`null-self` reads **+0.21 (CI -0.56..+0.98)**, win rate 0.265 against a null of
0.250 — consistent with zero, and the interval at 20 blocks (+/-1.77) had said
so all along. This entry is kept in its corrected form because the error is the
one this project's own standing rule names: *never call an effect smaller than
its interval an improvement*, and I called one on 20 blocks because it had a
mechanism ready to explain it.

**What survives is worth keeping.**

* **The seating scheme is sound.** A stateless `GreedyAgent` returns -0.01 with
  a win rate of exactly 0.250 — the rotation cancels seat advantage to two
  decimal places, which is the strongest validation of the arena in this file.
* **The mechanism is real code even if the effect is not measurable.**
  `bin/arena.rs:229` builds the seats as
  `if seats[s] { cand } else { base }` from *one* baseline instance, so three
  baseline seats share one `Mutex<Mcts>` — one tree arena, one transposition
  index, one RNG. `bin/selfplay.rs:449` is stronger still:
  `[agent.as_ref(); N_PLAYERS]`, one instance for **all four** seats, which is
  how every training game is generated. Whether that costs anything is now an
  open question with a 162-block bound of at most about a point on it, not an
  established bug. `null-self` keeps running.
* **`null-self` is the zero for every solo number below**, whatever its true
  value, and `mc/pair.py RUN null-self` is how an effect should be quoted.
  Paired contrasts between two candidate runs never needed it: they share the
  baseline and the seed, so any such offset cancels exactly.

### 2026-09-08 01:15 — cost per turn, user CPU (`mc/cost.sh`)

`/usr/bin/time` on `bin/sprobe mcts`, **one spec per process**, four runs each,
alternated so anything else on the machine is common-mode; `positions()` setup
(0.01 s) subtracted; 264 turns per run. Repeatability across separate invocations
was 2.5% on the same spec, which is the 3% the standing rule claims.

| spec | ms/turn user CPU | x default |
|---|---|---|
| `mcts:256:heuristic:quality` | **1.35** | **0.09** |
| `mcts:2048:heuristic:priors=eval` (uniform prior) | 11.02 | 0.75 |
| `mcts:2048:heuristic:priors=grad` | 11.55 | 0.79 |
| `mcts:2048:heuristic:priors=mixed` | 14.17 | 0.97 |
| `mcts:2048:heuristic:quality` (default) | **14.62** | 1.00 |
| `mcts:2450:heuristic:priors=grad` | 14.66 | **0.98** |
| `mcts:8192:heuristic:quality` | 83.54 | 5.71 |

The one-ply prior costs **1.33x** the uniform one (14.62 / 11.02), reproducing
the 1.34x in the module header. The gradient prior costs **1.05x** uniform — it
is very nearly free — and **0.79x** the one-ply probe.

`mcts:2450:heuristic:priors=grad` is the CPU-matched partner for
`mcts:2048:heuristic:quality`, measured at 0.98x. That pair is racing as
`p-grad-cpu` and is the honest form of "is `OnePly` worth 7.7x per edge".

**For the deliverable:** the champion turn costs **14.6 ms of user CPU**. Even
`mcts:8192` is 84 ms. Nothing in this file is anywhere near a per-turn time cap,
and the constraint the user lifted was never binding.

### 2026-09-08 01:35 — **the search never sees a reply, and `c_puct` is why** (`bin/sprobe budget`)

`Mcts` now counts descent depth in **sub-decisions** (`Mcts::depth_stats`,
counters only, nothing in selection reads them). A turn is ~8 sub-decisions, so
depth 8 is "my own turn" and depth ~32 is "one full round".

Depth against budget, default `c_puct = 2.0`, 147 sub-decisions:

| sims | deepest descent | mean descent | arena | ms/dec |
|---|---|---|---|---|
| 256 | 9.1 | **5.60** | 342 | 1.48 |
| 2048 | 13.0 | **8.09** | 2756 | 12.96 |
| 16384 | 17.3 | **10.84** | 21715 | 116.70 |

**Sixty-four times the budget buys 1.9x the mean depth and still never reaches
an opponent's reply.** 256 simulations do not finish looking at their own turn;
2,048 just barely do; 16,384 get a third of the way into the next player's. The
extra budget goes into *width inside the current turn*, where the prior has
already decided almost everything — which is the whole explanation of the flat
curve above 256.

Depth against `c_puct`, all at **2,048 simulations** (`SPROBE_CP`):

| `c_puct_init` | deepest | mean descent | top-visit share | ms/dec |
|---|---|---|---|---|
| 2.0 (default) | 12.9 | 8.02 | 83.5% | 8.53 |
| 0.25 | 19.1 | 11.33 | 90.6% | 8.96 |
| 0.06 | 54.4 | **21.8** | 93.3% | 11.29 |
| 0.02 | 140.6 | **48.3** | 93.8% | 20.27 |
| 0.01 | 196.8 | 68.8 | 92.9% | 26.39 |
| 0.005 | 230.4 | 83.0 | 93.8% | 31.15 |
| 0.002 | 265.8 | **95.3** | 94.2% | 34.37 |

**`c_puct = 0.06` at 2,048 simulations reaches mean depth 21.8 — twice what
16,384 simulations reach at the default, for 1.3x the CPU rather than 64x.** At
0.002 the mean descent is 95 sub-decisions, about twelve turns.

The cost is breadth: the root's top edge goes from 83.5% of visits to 94%. This
is the exploration/exploitation dial doing exactly what it says, and the reason
it was mis-set is in the value-scale entry above — `c_puct_init = 2.0` was
calibrated for values in (-1, 1) and this evaluator works in about (-0.5, +0.5),
with sibling Q differences near 0.1.

Arena confirmation is running across `cp` in {0.06, 0.02, 0.01, 0.005, 0.002}
and at 8192 simulations. First reads: `cp=0.06` **+1.22 (+0.33..+2.12, 107
blocks)**, `cp=0.02` +4.68 (+1.30..+8.07, **13 blocks only**), `cp=0.25` -0.69,
`cp=0.5` -0.91. The curve is not monotone in the middle; the floor is where the
signal is.

### 2026-09-08 01:45 — the `c_puct` curve at 2048 sims, and the null settles at zero

`null-self` at **310 blocks: -0.10 (CI -0.65..+0.45), win 0.251**. The arena is
unbiased for this baseline; solo centred scores below can be read directly.

`c_puct_init` against the `cp=2.0` default, everything else at the default,
2,048 simulations:

| `cp` | centred | 95% CI | blk |
|---|---|---|---|
| 0.002 | **-2.71** | -4.06..-1.37 | 38 |
| 0.005 | -1.34 | -2.83..+0.16 | 36 |
| 0.01 | -0.13 | -1.67..+1.41 | 39 |
| 0.02 | +0.45 | -0.75..+1.66 | 75 |
| **0.06** | **+1.06** | **+0.35..+1.76** | **176** |
| 0.12 | +1.08 | -1.54..+3.70 | 15 |
| 0.25 | -0.69 | -1.64..+0.25 | 112 |
| 0.5 | -0.91 | -2.00..+0.19 | 90 |
| 1.0 | -0.58 | -1.61..+0.44 | 81 |
| 2.0 | 0 by construction | — | — |
| 4.0 | +0.23 | -0.94..+1.41 | 80 |

Single-peaked around **0.06-0.12**, worth about a point, at 1.29x the user CPU
(19.27 ms/turn against 14.94). Note the shape: the default 2.0 is on a shoulder,
0.25-1.0 is a *trough*, and the peak is more than an order of magnitude below
where this shipped.

**And it interacts with the budget, which is the point.** `mcts:8192` alone is
+0.75 (-0.03..+1.53, 142 blk) — the flat curve. With `cp` lowered it is not
flat at all:

| spec | centred | 95% CI | blk |
|---|---|---|---|
| `mcts:8192:quality` | +0.75 | -0.03..+1.53 | 142 |
| `mcts:8192:cp=0.06` | +2.08 | +0.67..+3.48 | 41 |
| `mcts:8192:cp=0.02` | **+7.79** | +5.14..+10.44 | **18 only** |

and at 2,048 simulations `cp=0.02` is worth only +0.45 where `cp=0.06` is +1.06 —
**the best `cp` falls as the budget rises**, which is what a depth explanation
predicts. `cp=0.02` at 2,048 descends a mean of 48 sub-decisions on a tree with
2,048 simulations to spread over them; at 8,192 the same depth is four times
better supported. Depth without budget is over-commitment; budget without depth
is width the prior had already decided.

**18 blocks is not a result** — `cp=0.02` itself read +4.68 at 13 blocks and
settled to +0.45 at 75. Confirmation running at 8192/16384/32768 x
cp {0.02, 0.01, 0.005}.

### 2026-09-08 02:05 — **depth and budget compound: `mcts:8192:cp=0.02` is +6.4**

Against `mcts:2048:heuristic:quality`, solo, `--seed 3000000`, `arena_v6`.
`null-self` at 374 blocks is **-0.10 (-0.61..+0.40)**, so these read directly.

| spec | centred | 95% CI | blk | win |
|---|---|---|---|---|
| `mcts:8192:quality` | +0.62 | -0.13..+1.36 | 158 | 0.265 |
| `mcts:2048:cp=0.06` | +1.07 | +0.44..+1.70 | 261 | 0.277 |
| `mcts:8192:cp=0.06` | +1.82 | +0.76..+2.87 | 68 | 0.292 |
| `mcts:8192:cp=0.005` | +4.88 | +2.05..+7.70 | 15 | 0.392 |
| `mcts:8192:cp=0.01` | +3.51 | +1.22..+5.79 | 18 | 0.243 |
| **`mcts:8192:cp=0.02`** | **+6.37** | **+4.87..+7.88** | **42** | **0.438** |
| `mcts:16384:cp=0.02` | +7.51 | +4.43..+10.59 | 15 | 0.425 |
| `mcts:32768:cp=0.02` | +9.78 | +6.75..+12.81 | 8 | 0.562 |

Neither ingredient is worth much alone — 4x the budget is +0.62, `cp=0.06` at the
default budget is +1.07 — and together they are **+6.37 with a 0.438 win rate
against a null of 0.25**. The games are clean: 27 days, zero aborts, candidate
mean score 38.9 against the baseline's 30.4.

**The mechanism is the previous entry.** Lowering `c_puct` spends the budget on
depth instead of width; depth is only worth having if there are enough
simulations to support the deeper tree; and width was worth nothing because the
prior had already decided it. That is also why the best `cp` moves with the
budget — 0.06 at 2,048 simulations, 0.02 at 8,192 — and why the two 1-D sweeps
that preceded this both found "about a point".

**Cost.** `mcts:8192:cp=0.02` is not cheap: 8192 simulations cost 5.6x the
default and `cp=0.02` costs a further 2.1x, so roughly **12x the user CPU per
turn** — about 180 ms against 14.9. The CPU-matched comparison is against
`mcts:16384:quality`, which measures **+0.56** (-0.94..+2.05, 47 blocks). The
depth is buying it, not the CPU.

Confirmation in flight: `x-8192cp002` to 200+ blocks, the same spec against
`heuristic:full` (a *stateless* opponent, to rule out exploiting a shallow MCTS
baseline specifically), and `--mode pairs`.

### 2026-09-08 02:15 — the deep ladder firms up

Same design. `null-self` 374 blocks at -0.10 (-0.61..+0.40).

`c_puct` at **8,192** simulations (the axis moves with the budget — at 2,048 the
peak was 0.06):

| `cp` | centred | 95% CI | blk |
|---|---|---|---|
| 2.0 | +0.62 | -0.13..+1.36 | 158 |
| 0.06 | +1.94 | +0.88..+3.01 | 69 |
| **0.02** | **+6.65** | **+5.35..+7.96** | **54** |
| 0.01 | +3.74 | +1.92..+5.57 | 24 |
| 0.005 | +4.12 | +1.96..+6.29 | 21 |

Budget at **`cp = 0.02`**:

| sims | centred | 95% CI | blk | win |
|---|---|---|---|---|
| 2048 | +0.45 | -0.75..+1.66 | 75 | 0.230 |
| 8192 | +6.65 | +5.35..+7.96 | 54 | 0.442 |
| 16384 | +8.27 | +5.92..+10.63 | 22 | 0.460 |
| 32768 | +8.80 | +6.17..+11.44 | 12 | 0.500 |

Rising and flattening. 65,536 is running, as is `cp=0.01` at 32,768 — if the
peak `cp` keeps falling with the budget, that is where it should be.

**This is the same axis the module header called dead.** It was dead at
`c_puct = 2.0` and it is worth six points at `c_puct = 0.02`, because a
simulation is only worth having if the descent it feeds is long enough to reach
something the prior had not already decided.

### 2026-09-08 02:20 — the CPU-matched statement, and it is not close

`mc/cost.sh`, `/usr/bin/time` user CPU, one spec per process, alternated:

| spec | ms/turn user CPU | x old champion | centred vs old champion |
|---|---|---|---|
| `mcts:2048:heuristic:quality` | 14.77 | 1.00 | 0 by construction |
| **`mcts:8192:heuristic:cp=0.02`** | **146.04** | **9.89** | **+6.59** (60 blk) |
| `mcts:16384:heuristic:quality` | 187.12 | 12.67 | +0.56 (47 blk) |
| `mcts:32768:heuristic:quality` | 421.93 | 28.56 | +0.14 (24 blk) |

**`mcts:16384:quality` costs 28% *more* CPU than `mcts:8192:cp=0.02` and is six
points worse.** The comparison is not merely matched, it is tilted against the
new spec, and it still is not close. Spending a simulation budget on depth beats
spending it on width by about six points at the same CPU.

Descent depth of the two, `bin/sprobe budget`:

| spec | mean descent | deepest | arena nodes | top-visit |
|---|---|---|---|---|
| `mcts:2048:quality` | 8.5 (~1 turn) | 13.8 | 2,665 | 88.4% |
| `mcts:8192:cp=0.02` | **52.3 (~6.5 turns)** | 158.4 | 12,806 | 93.5% |

### 2026-09-08 02:20 — independent confirmation against a stateless opponent

The worry with a six-point jump measured only against `mcts:2048:quality` is
that a deep search might be exploiting a *shallow search of its own family*
rather than playing better. `heuristic:full` is a stateless `GreedyAgent` with no
tree at all — a different opponent in every sense, and the one whose null control
returns exactly 0.250:

| candidate | vs `heuristic:full` | 95% CI | blk | win |
|---|---|---|---|---|
| `mcts:2048:heuristic:quality` (old champion) | +4.25 | +2.76..+5.73 | 50 | 0.385 |
| `mcts:8192:heuristic:cp=0.02` | **+10.65** | +9.15..+12.15 | 8 | 0.562 |

The old champion's +4.25 reproduces `3fe213e`'s +4.15 against the same opponent,
which is a good sign for both. The deep spec is at 8 blocks and needs more, but
it is not the same number. `--mode pairs` is also running, where both sides share
an agent instance symmetrically.

### 2026-09-08 02:25 — the deep spec's other cost is memory

`/usr/bin/time -l`, one agent, single-threaded, same 135 turns:

| spec | max RSS |
|---|---|
| `mcts:2048:heuristic:quality` | 186 MB |
| `mcts:8192:heuristic:cp=0.02` | **412 MB** |

A low `c_puct` grows the arena in two ways at once — four times the simulations,
and descents that are six times longer, so `retain_subtree` carries more forward
between the sub-decisions of a turn. 412 MB is nothing for one agent in the TUI
and it is a real constraint for the arena: **two `mcts:65536:cp=0.02` runs at
`--concurrency 4` were killed with no output** while ten arena processes held
7 GB between them, and both were the largest-budget jobs. Anything past 32,768
simulations wants `--concurrency 2` or `noreuse`.

(The `ms/turn` figures printed alongside were 45.8 and 463 — three times the
`/usr/bin/time` user-CPU numbers of 14.8 and 146, on a machine running sixteen
other search threads. That gap is the entire reason the standing rule says user
CPU.)

### 2026-09-08 02:30 — what did not work, with final block counts

All against `mcts:2048:heuristic:quality`, solo, seed 3000000; `null-self` is
-0.10 (-0.61..+0.40, 374 blk), so read these against zero.

**Settled negatives** (interval excludes zero):

| knob | centred | 95% CI | blk | why |
|---|---|---|---|---|
| `pick=q` | **-2.41** | -3.88..-0.94 | 51 | most-visited really is the right root rule, even though 85% of visits are the prior's doing |
| `pmin=2` | **-1.35** | -2.37..-0.34 | 96 | a 2-edge node is resolved by three simulations, and softmaxing two scores is *sharper* than the uniform it replaces |
| `priors=grad` | **-0.77** | -1.49..-0.05 | 194 | blind on every phase but `Take`; paired against `priors=mixed` it is **-1.16 (-1.99..-0.32, 193 paired blk)** |

**Settled inert** (the stale "measured inert" list, re-measured on the real
prior as the brief asked, plus this run's own dead ends):

| knob | centred | 95% CI | blk |
|---|---|---|---|
| `noreuse` | -0.03 | -0.98..+0.93 | 131 |
| `fpu=0.0` / `0.6` / `1.2` | +0.31 / +0.21 / +0.22 | all straddle 0 | 73 / 83 / 65 |
| `k=8` (`max_edges`) | +0.74 | -0.45..+1.92 | 65 |
| `nocap` (`widen_cap`) | +0.72 | -0.46..+1.91 | 65 |
| `wc=1.0,wa=0.75` | +1.55 | -0.30..+3.40 | 19 |
| `cpb=1000` | +0.52 | -0.51..+1.55 | 102 |
| `pt=0.5` / `pt=1.5` | +0.00 / -0.18 | | 98 / 88 |
| `priors=mixed` | +0.38 | -0.34..+1.10 | 193 |
| `pmarg` | +0.21 | -0.93..+1.34 | 86 |
| `q0=0.5` / `1.0` / `2.0` | +0.87 / +0.42 / +1.01 | all straddle 0 | 65 / 65 / 61 |

`max_edges`, `widen_cap` and `widen_c` are confirmed inert **on the real prior**,
so the module header's "never" survives re-measurement. `fpu` is inert for the
structural reason in `MctsConfig::q_init`: an unvisited edge is a state that
lasts two descents here.

### 2026-09-08 02:30 — the +3.04 does not replicate, and a null is agent-specific

The module header's "8x the budget is worth +3.04 once the prior is real" was
`mcts:2048:quality` against a **`mcts:256:quality` baseline**, a different design
from everything else here, so it got its own null.

| run | centred | 95% CI | blk |
|---|---|---|---|
| `rep-2048v256` (`mcts:2048:quality` vs `mcts:256:quality`) | -0.34 | -1.56..+0.87 | 86 |
| `null-256b` (`mcts:256:quality` vs itself) | **-1.09** | -1.96..-0.21 | 141 |
| paired difference | **+0.03** | -1.84..+1.90 | 43 |

**+0.03, not +3.04.** And note the second row: the null for a `mcts:256`
baseline is -1.09 and *excludes zero*, where the null for a `mcts:2048` baseline
is -0.10 over 374 blocks. **Run a matched null for every baseline you use.** The
arena's seating is exact for a stateless agent (`null-greedy` = -0.01, win
0.250); with a stateful one, one candidate instance against one instance shared
by three seats is not perfectly symmetric, and how much that is worth depends on
the agent.

### 2026-09-08 02:30 — the bottom of the budget ladder, final

| spec | centred | 95% CI | blk | win |
|---|---|---|---|---|
| `mcts:1` | **-32.18** | -33.12..-31.24 | 94 | **0.000** |
| `mcts:8` | -14.48 | -15.53..-13.44 | 150 | 0.052 |
| `mcts:32` | -2.25 | -3.45..-1.04 | 86 | 0.225 |
| `mcts:64` | -1.21 | -2.89..+0.48 | 55 | 0.225 |
| `mcts:128` | -0.75 | -1.75..+0.26 | 100 | 0.245 |
| `mcts:256` | -0.24 | -1.24..+0.76 | 110 | 0.251 |

Paired, `mcts:128 - mcts:32` = +1.44 (+0.29..+2.59, 86 paired blk). **The knee is
between 32 and 128 simulations** at the shipped `c_puct`, and the prior's argmax
alone (`mcts:1`) wins none of 376 games. The search is doing nearly all of the
work; it just stops being able to use more of it once it has resolved its own
turn.

### 2026-09-08 02:35 — three independent confirmations of the deep spec

`mcts:8192:heuristic:cp=0.02` measured against opponents that are not the
baseline it was tuned against, and in a mode where instance sharing is
symmetric:

| check | opponent / mode | centred | 95% CI | blk | win (null) |
|---|---|---|---|---|---|
| solo, tuned baseline | `mcts:2048:quality` | +6.42 | +5.38..+7.47 | 78 | 0.438 (0.25) |
| **stateless opponent** | `heuristic:full` | **+11.28** | +9.75..+12.80 | 38 | 0.592 (0.25) |
| **the alpha-beta** | `minimax:8:600000::greedy:capw=25` | **+8.50** | +6.62..+10.39 | 27 | 0.477 (0.25) |
| **`--mode pairs`** | `mcts:2048:quality`, 2v2 | **+3.46** | +2.38..+4.55 | 18 | 0.611 (0.50) |

For scale, the old champion is +3.53 against `heuristic:full` (108 blk) and the
module header records +7.73 against the alpha-beta. The pairs-mode figure is
against a pairs null of +0.27 (30 blk) — the mode where both sides share an agent
instance across two seats, so the asymmetry in `bin/arena.rs:229` cannot explain
it. Four opponents, four designs, same direction.

Also: a `deep` preset now spells `cp=0.02`, following `quality`'s convention —
sugar in, flags out in the label, pinned by `tests/search.rs`. `MctsConfig`'s
default is deliberately **not** changed: the best `c_puct` falls as the budget
rises, so 0.02 is only right from ~8k simulations up, and other workstreams have
binaries pinned against the current default.

### 2026-09-08 02:40 — the effect isolated, paired at fixed budget

Two runs sharing a baseline and a seed sequence are a matched comparison, and
the matched form of the result is much tighter than either against the baseline.
Holding the budget at **8,192 simulations** and changing nothing but `c_puct`:

| contrast | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| **`8192:cp=0.02` - `8192:quality`** | **+5.93** | **+4.27..+7.59** | 47 | **9e-13** |
| `8192:cp=0.02` - `2048:cp=0.06` | +6.01 | +4.55..+7.47 | 43 | 2e-16 |
| `2048:cp=0.06` - `8192:quality` | +0.82 | -0.28..+1.92 | 127 | 0.14 |

**One constant, at a fixed simulation budget, is worth +5.93 points.** Neither
half is worth much alone — a quarter-order-of-magnitude `c_puct` cut at the small
budget is +0.82, and 4x the budget at the shipped `c_puct` is +0.62 — which is
exactly why two years of 1-D sweeps found nothing.

Above 8,192 at `cp=0.02` the budget keeps paying, but slowly and not resolvably:

| contrast | centred | 95% CI | paired blk |
|---|---|---|---|
| 16384 - 8192 | +1.19 | -1.05..+3.43 | 34 |
| 32768 - 16384 | +1.71 | -0.73..+4.15 | 23 |
| {16384, 32768} - 8192 | +0.32 | -2.35..+2.98 | 16 |

so about a point per doubling, none of it individually distinguishable from
zero, against 4x the CPU and 2x the memory each time. **`mcts:8192` is the rung
to ship**; 32,768 is nominally +1.7 better for six times the CPU and is not
proven.

### 2026-09-08 02:45 — the deeper the search, the more the prior matters

`priors=grad` costs 0.79x the one-ply probe and, at the shipped `c_puct`, is a
wash once the saving is spent on simulations (`p-grad-cpu`: +0.34, -0.60..+1.29,
99 blk at 0.98x CPU). The obvious guess was that in the deep regime — where
simulations are worth something again — the cheap prior would be *better*.

**It is much worse.** `mcts:8192:heuristic:cp=0.02,priors=grad` reads +1.08
(-1.87..+4.04, 18 blk) where the same spec with the one-ply prior reads +6.42.

That direction makes sense once stated: the prior is consulted at every node of a
descent, and lowering `c_puct` makes descents six times longer. A prior that is
blind on `Placing` (`Gradient::step` prices a `Choice` and nothing else) commits
a blind step six times as often. **`Priors::OnePly`'s 7.7x per-edge cost is a
wash at `cp=2.0` and clearly worth paying at `cp=0.02`** — which is the answer to
"is `OnePly` worth it", and it is budget-dependent in the same way everything
else here turned out to be.

### 2026-09-08 02:47 — the joint `(init, base)` cell is a different route to the same place

`c = c_puct_init + ln((1 + N + base) / base)`, so a small `init` with a small
`base` is exploitation deep in the tree and breadth at the root, where N is
largest. That is a better-shaped answer than lowering `init` alone, which flattens
exploration everywhere. At 8,192 simulations (`bin/sprobe budget`):

| `init` | `base` | c(N=20) | c(N=2048) | mean descent | root top-visit |
|---|---|---|---|---|---|
| 0.02 | 19652 | 0.021 | 0.119 | 46.6 | 95.7% |
| 0.005 | 3000 | 0.007 | 0.51 | 35.0 | 92.0% |
| 0.005 | 1000 | 0.026 | 1.12 | 21.6 | 91.1% |

It does what it says — 3.7 points of root breadth back for a quarter of the
depth — and **it is worth nothing**: `cp=0.005,cpb=3000` measures **+6.50
(+5.15..+7.86, 32 blk)** against `cp=0.02`'s **+6.33 (+5.31..+7.36, 82 blk)**.

Read that as good news about the shape of the optimum rather than a failure:
anything that reaches a mean descent of 35-47 sub-decisions scores the same, so
the result is about **depth**, not about the particular constants that buy it.
`cp=0.02` is the simpler spelling and is the one to ship.

### 2026-09-08 02:50 — cost of the deep ladder, user CPU

`mc/cost.sh`, one spec per process, alternated, two reps agreeing to 2%. This
sample is `SPROBE_EVERY=17` (135 turns) and is dearer per turn than the
`SPROBE_EVERY=11` sample used earlier (264 turns) — **quote the ratio, not the
milliseconds**, and note which sample a figure came from.

| spec | ms/turn (135-turn sample) | x old champion | ms/turn (264-turn sample) |
|---|---|---|---|
| `mcts:2048:heuristic:quality` | 16.37 | 1.00 | 14.77 |
| `mcts:8192:heuristic:cp=0.02` | 167.85 | **10.25** | 146.04 |
| `mcts:16384:heuristic:cp=0.02` | 368.74 | 22.52 | — |
| `mcts:32768:heuristic:cp=0.02` | 781.89 | 47.76 | — |

The champion costs **about ten times the old one per turn** — a sixth of a
second — and buys +6.3 centred. The rung above costs 2.2x that for a nominal
+1.2 that no interval resolves. **Ship 8,192.**

For scale on the other axis: `mcts:16384:heuristic:quality` — the same CPU spent
on width instead of depth — is 187.12 ms/turn on the 264-turn sample, *more* than
`mcts:8192:cp=0.02`'s 146.04, and scores +0.56 against its +6.3.

### 2026-09-08 04:40 — the runs that were in flight when the session was killed, at final block counts

Everything below kept running while the session was gone. `null-self` is
**-0.10 (-0.61..+0.40, 374 blk)**, so solo centred scores read directly.
All against `mcts:2048:heuristic:quality`, seed 3000000, `arena_v6`.

| run | spec | blk | centred | 95% CI | win |
|---|---|---|---|---|---|
| `x-8192cp002` | `mcts:8192:cp=0.02` | **353** | **+5.99** | +5.53..+6.45 | 0.400 |
| `sims-8192` | `mcts:8192:quality` | 352 | +0.42 | -0.11..+0.94 | 0.252 |
| `x-16384cp002` | `mcts:16384:cp=0.02` | 45 | +7.94 | +6.35..+9.53 | 0.481 |
| **`x-32768cp002`** | `mcts:32768:cp=0.02` | **110** | **+9.29** | **+8.50..+10.08** | **0.536** |
| `d-pt2` | `...,pt=2` | 268 | +6.24 | +5.70..+6.78 | 0.396 |
| `d-pt05` | `...,pt=0.5` | 270 | +4.91 | +4.35..+5.47 | 0.373 |
| `d-grad` | `...,priors=grad` | 330 | +2.12 | +1.62..+2.63 | 0.274 |
| `d-fpu06` | `...,fpu=0.6` | 205 | +1.19 | +0.54..+1.84 | 0.248 |
| `pr-8192cp002` | `--mode pairs` | 136 | +3.83 | +3.46..+4.20 | 0.668 (null 0.50) |
| `g-8192cp002-vs-greedy` | vs `heuristic:full` | 40 | +11.46 | +9.91..+13.00 | 0.588 |
| `g-champ-vs-greedy` | old champ vs `heuristic:full` | 142 | +3.35 | +2.49..+4.20 | 0.343 |
| `ab-8192cp002` | vs `minimax:8:600000::greedy:capw=25` | 45 | +8.70 | +7.38..+10.03 | 0.531 |
| `j-8192-cp0005-cpb3000` | `cp=0.005,cpb=3000` | 48 | +6.19 | +5.13..+7.25 | 0.401 |

**`mcts:8192:cp=0.02` is confirmed at 353 blocks: +5.99, interval +/-0.46.** The
02:35 figure of +6.42 at 78 blocks held. The pairs-mode and stateless-opponent
confirmations both firmed up rather than regressing.

#### Paired, and this is where the news is

| contrast | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| **`32768:cp=0.02` - `8192:cp=0.02`** | **+2.45** | **+1.39..+3.50** | 99 | **4.6e-06** |
| `16384:cp=0.02` - `8192:cp=0.02` | +1.11 | -0.65..+2.87 | 45 | 0.21 |
| `...,fpu=0.6` - `cp=0.02` | **-4.75** | -5.58..-3.92 | 195 | 3.4e-29 |
| `...,priors=grad` - `cp=0.02` | **-4.04** | -4.72..-3.36 | 321 | 1.2e-31 |
| `...,pt=0.5` - `cp=0.02` | **-1.19** | -1.82..-0.55 | 261 | 0.00024 |
| `...,pt=2` - `cp=0.02` | +0.21 | -0.45..+0.86 | 258 | 0.54 |

**1. "Ship 8,192" was wrong.** At 02:40 the 32768-8192 contrast was
+1.71 (-0.73..+4.15, 23 blk) and I wrote "not proven". At 99 paired blocks it is
**+2.45 and p = 5e-6**. The budget axis is still climbing at 32,768 — it was
under-powered, not flat. The ridge has to be mapped further out, and the best
`c_puct` at 32,768 may be below 0.02.

**2. The first flipped `cp=2.0`-era verdict: `fpu_reduction` is not inert, it is
a disaster.** At `cp=2.0` it measured +0.21/+0.31/+0.22 across 0.0/0.6/1.2, all
straddling zero, and the doc comment explains why: an unvisited edge is a state
that lasts two descents. At `cp=0.02` the same flag is **-4.75 with p=3e-29**.
The explanation inverts with the descent length: a search that descends 52
sub-decisions creates fresh nodes for most of its depth, so "unvisited edge" is
no longer a transient at the frontier — it is the frontier, and charging it
0.6 makes the deep search refuse to look at the moves it just reached.
**Everything measured inert at `cp=2.0` has to be re-measured.**

**3. `priors=grad` in the deep regime is settled at -4.04** (321 paired blocks),
against -0.77 at `cp=2.0`. Same mechanism as 02:45, now with an interval: the
prior is consulted at every node of a descent and a longer descent consults it
more.

**4. `prior_temp` is flat upward and negative downward** — `pt=2` is +0.21
(inert), `pt=0.5` is -1.19. A sharper prior hurts a deep search; a blunter one
does nothing. Leave `pt` at 1.0.

### 2026-09-08 04:50 — `c_puct_base` makes the root's exploration constant a function of the budget

`bin/sprobe budget` at three `c_puct_init` values, 90 positions, every rung
driven along the 65,536 rung's path so all three see the same nodes:

| `cp` | sims | agrees w/ 65536 | top-visit | arena nodes | deepest | **mean descent** | ms/dec |
|---|---|---|---|---|---|---|---|
| 0.02 | 8192 | 91.8% | 94.0% | 13,419 | 167 | **56.1** | 37.5 |
| 0.02 | 32768 | 96.7% | 94.0% | 55,414 | 187 | **59.6** | 164.5 |
| 0.02 | 65536 | 100% | 90.7% | 113,110 | 195 | **60.9** | 362.8 |
| 0.01 | 8192 | 87.1% | 93.2% | 11,894 | 239 | 78.2 | 49.3 |
| 0.01 | 32768 | 95.2% | 92.9% | 53,570 | 261 | 81.5 | 227.7 |
| 0.005 | 8192 | 87.1% | 93.4% | 12,443 | 282 | 94.5 | 51.6 |
| 0.005 | 32768 | 88.7% | 92.6% | 54,609 | 309 | 96.9 | 238.6 |

**Depth is set by `c_puct`, not by the budget — again.** Eight times the
simulations at a fixed `cp` moves the mean descent by 9% (56.1 -> 60.9). So
whatever the budget is buying above 8,192 (a measured +2.45 at 32,768), it is
*not* more lookahead; it is support for the tree the `c_puct` already dug.

**And there is a confound in the spec I have been calling one knob.**
`c = c_puct_init + ln((1 + N + base) / base)` with `base = 19652`:

| base | c at N=20 | N=2048 | N=8192 | N=32768 | N=65536 | N=131072 |
|---|---|---|---|---|---|---|
| **19652 (default)** | 0.021 | 0.119 | 0.368 | **1.001** | **1.487** | 2.057 |
| 78608 | 0.020 | 0.046 | 0.119 | 0.368 | 0.626 | 1.001 |
| 262144 | 0.020 | 0.028 | 0.051 | 0.138 | 0.243 | 0.425 |

At the root of a 32,768-simulation search the constant is **1.00, not 0.02** —
the `ln` term is fifty times `c_puct_init` — and at 65,536 it is 1.49. So
`mcts:8192:cp=0.02` and `mcts:32768:cp=0.02` do not differ only in budget:
the second one also explores the root four times as hard. **Raising the budget
at a fixed `(init, base)` raises root breadth as a side effect**, which is a
second reason the budget axis came alive when `init` came down, and it means
"the ridge" is a surface in three constants, not two.

Racing the 2x2 that separates them: `8192:cp=0.02,cpb=4913` (root c = 1.00 at
8,192 sims — 32,768's breadth on 8,192's budget) and
`32768:cp=0.02,cpb=78608` (root c = 0.368 — 8,192's breadth on 32,768's budget).

### 2026-09-08 04:52 — the deep search's profile is not the shallow one's

`sample` for 15 s on a live `mcts:32768:cp=0.02` arena, leaves by sample count:

| frame | samples |
|---|---|
| `Mcts::simulate` (with `select`/`cursor`/`backup` inlined) | **11,367** |
| `logf` + its stub | 1,486 |
| `Mcts::search_at` | 713 |
| `eval::*` (space_value, board_position, monument, temple, starvation, engine, held) | ~1,850 total |
| malloc/free family | ~1,300 |
| `options::dominated_dedup` + `options::recurse` | 373 |
| slice `partial_compare` (the `Choice` sort) | 359 |

**The module header's "`Choice` sort/compare/eq 24.7% + `dominated_dedup` 4.9%
is the biggest throughput lever" is a `cp=2.0` fact and is now false.** At
`cp=0.02` a descent is 56 sub-decisions instead of 8, so the search runs
`select` an order of magnitude more times per expansion, and the descent — not
move generation and not the evaluator — is where the CPU goes. `logf`, called
once per `select` for the `c_puct_base` term, is on its own about 6% of the
non-idle profile.

### 2026-09-08 04:55 — every arena number in this file is on the **pre-`d7b4e42`** evaluator

`mc/bin/arena_v6` was built at **00:43**. `94d85f3` (evaluator re-price) landed
at 00:19, so v6 has it; **`d7b4e42` ("stop assuming corn income, stop paying for
research stock") landed at 02:50, so v6 does not.** `arena_v7` was built at
02:34 and is also pre-`d7b4e42`, which is why the v6/v7 equivalence check passed.

Pinned `arena_v8` from the current tree (`mc/bin/arena_v8`, fingerprint in
`mc/bin/arena_v8.rev`: HEAD 5318276, `eval.rs` clean at HEAD, Apple M4 Pro,
Darwin 24.3.0 arm64) and ran the same 24-game equivalence check at seed 99000
that certified v1..v7:

| seed | v6 centred | v8 centred |
|---|---|---|
| 99000 | +18.44 | +16.62 |
| 99001 | +25.62 | +22.06 |
| 99002 | +29.56 | +30.69 |
| 99003 | +17.56 | +12.19 |
| 99004 | +18.75 | +9.94 |
| 99005 | +14.75 | +12.75 |

**0 of 6 blocks identical.** v8 is a different agent, as expected: `mcts.rs:1573`
calls `eval::heuristic` for edge ordering, so an `eval.rs` change moves the prior
*and* the leaf value, and the working tree also carries the generation
workstream's in-flight `options.rs`/`moves.rs`/`spaces/*`.

**What this does and does not invalidate.** Every v6 row is internally
comparable — one frozen platform, one seed sequence — so the *structure* stands:
`c_puct` converts budget into depth, the budget ladder rises to 32,768, `fpu`
and `priors=grad` flip in the deep regime. What is **not** transferable is any
absolute margin, because 94d85f3 already moved the same spec from +11.12 to
+3.5 against `heuristic:full`. A `v8-` wave is now running with its own null and
its own baseline arm so the shipping decision is made on the code that will
ship:

| run | candidate | why |
|---|---|---|
| `v8-null` | `mcts:2048:heuristic:quality` vs itself | the zero for the v8 rows |
| `v8-8192q` | `mcts:8192:heuristic:quality` | the budget alone, new evaluator |
| `v8-8192cp002` | `mcts:8192:heuristic:cp=0.02` | the champion, new evaluator |
| `v8-32768cp002` | `mcts:32768:heuristic:cp=0.02` | the rung above |

**Never pool a `v8-` row with a `v6` row.**

### 2026-09-08 05:00 — threading: not attempted, and the module header's case for it is a wall-clock case

The header says "threading buys simulations, so it is worth exactly what
simulations are worth", and at `cp=0.02` that is now a real number. The
exchange rate, measured: **+2.45 per 4x simulations** (`32768:cp=0.02` -
`8192:cp=0.02`, 99 paired blocks, p=5e-6) at **4.85x the user CPU**
(772.9 ms/turn against 159.3, one pinned binary, alternated) — so **+1.22 per
doubling of the budget, at 2.2x the CPU per doubling**.

**That sentence is true at fixed wall clock and false at fixed CPU**, and the
constraint the user actually lifted was the clock. K threads do not create
simulations; they spend K cores to finish the same simulations sooner. At fixed
CPU a K-thread search is worth **zero** points and slightly less than zero,
because virtual-loss tree parallelism is a worse use of N simulations than one
thread's N. Every number in this file is a strength-per-CPU number, and no
arena measurement I can run would show a threading win. What threading buys is
**latency for the shipped agent**: a TUI turn uses one of fourteen cores, so at
a fixed 0.8 s/turn, eight threads would be worth about `+1.22 * log2(8) = +3.7`.
That is the whole case, and it is a deliverable-comfort case, not a strength one.

**And the swap is not mechanical, for a reason the header does not mention.**
`Edge::w` and `Node::w` are `[f32; N_PLAYERS]`. There are two encodings and
each costs something:

* **§3.4's signed fixed-point `AtomicI32` (value x 2^16)** changes the
  arithmetic, so the search plays differently *at one thread*. Every row in
  this file would have to be re-measured on the new binary — the same
  poolability rule that made `v1..v7` worth certifying.
* **`AtomicU32` holding `f32::to_bits` with a compare-exchange add loop** is
  bit-identical at one thread, but backup does `4 * depth` of them and depth is
  now ~56, which is ~7M CAS per sub-decision at 32,768 simulations. It buys
  threads by making the single-threaded search slower — directly against the
  budget curve it is supposed to serve.

`SEARCH.md` §3.5 also says "batching is the real reason for parallelism",
sizing it against a **network** call of 0.1-1 ms. There is no network here; the
leaf is `eval::heuristic`, and the 04:52 profile puts the whole `eval::*` family
at ~1,850 samples against `simulate`'s 11,367. **The search is descent-bound,
not evaluation-bound** — the opposite of the regime §3.5 was written for — so
there is nothing to batch.

**Not attempted, deliberately.** The CPU it would have cost went to the ridge
and to the `v8` re-confirmation instead.

### 2026-09-08 05:00 — the `cp=2.0`-era verdicts, re-measured at `cp=0.02` (v6 platform)

All at 8,192 simulations, `cp=0.02`, changing one flag, **paired on seed**
against `x-8192cp002` (353 blocks). Positive is better than the deep champion.

| flag | at `cp=2.0` (old) | **at `cp=0.02`** | 95% CI | paired blk | p | verdict |
|---|---|---|---|---|---|---|
| `fpu=0.6` | +0.21 (inert) | **-4.75** | -5.58..-3.92 | 195 | 3e-29 | **flipped** |
| `fpu=0.0` | +0.31 (inert) | **-2.90** | -4.64..-1.15 | 61 | 9e-4 | **flipped** |
| `noreuse` | -0.03 (inert, 131 blk) | **-1.94** | -3.55..-0.32 | 51 | 0.016 | **flipped** |
| `priors=grad` | -0.77 | **-4.04** | -4.72..-3.36 | 321 | 1e-31 | same sign, 5x |
| `pt=0.5` | +0.00 (inert) | **-1.19** | -1.82..-0.55 | 261 | 2e-4 | **flipped** |
| `pt=2` | -0.18 (inert) | +0.21 | -0.45..+0.86 | 258 | 0.54 | still inert |
| `nocap` (`widen_cap`) | +0.72 (inert) | **+0.19** | -0.51..+0.89 | 44 | 0.59 | **still inert** |
| `cpb=4913` (root breadth) | — | -1.13 | -3.28..+1.01 | 30 | 0.28 | no |

**1. `fpu_reduction` is the sharpest flip, and it is a flip to "the default was
already right".** The shipped 0.2 sits at an interior optimum: 0.0 is -2.90 and
0.6 is -4.75. At `cp=2.0` all three were within a fifth of a point of each
other and the doc comment explained why — "an unvisited edge is a state that
lasts two descents". At `cp=0.02` a descent is 56 sub-decisions long and most
of it runs through nodes created moments earlier, so unvisited edges *are* the
frontier and what FPU charges them decides where the search goes. The knob went
from inert to worth five points across its range **without its best value
moving**.

**2. `tree_reuse` flipped from a wash to worth +1.94.** Its doc comment says it
is "a search-quality knob, not a speed one" and then records that the quality
does not move; at `cp=0.02` it moves. A reused node keeps the statistics it
earned as an interior node of the previous sub-decision, and with 56-deep
descents the previous sub-decision built far more of the tree the next one
needs. **Keep it on for strength now, not only for memory.**

**3. `widen_cap` survives.** `nocap` is +0.19 (-0.51..+0.89) on the real prior
*and* in the deep regime — the module header's "`max_edges` / `widen_cap` /
`widen_c` never" is the one piece of the old tuning order that re-measurement
did not overturn.

**4. Watch the pairing.** Unpaired, `d-nocap` reads +7.17 against the champion's
+5.99 and looks like a +1.2 win; paired on seed it is +0.19 +/- 0.70. Two runs
sharing a baseline and a seed sequence must be compared paired, or the
block-subset difference reads as an effect.

### 2026-09-08 05:00 — the ridge so far (v6), paired against `8192:cp=0.02`

| contrast | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| `4096:cp=0.02` - `8192:cp=0.02` | **-1.90** | -3.54..-0.27 | 39 | 0.019 |
| `32768:cp=0.02` - `8192:cp=0.02` | **+2.45** | +1.39..+3.50 | 99 | 5e-6 |
| `65536:cp=0.02` - `8192:cp=0.02` | +3.58 | -1.24..+8.40 | 8 | 0.08 |
| `32768:cp=0.01` - `32768:cp=0.02` | +0.25 | -3.51..+4.01 | 15 | 0.89 |

The budget axis is still rising at 32,768 and the `c_puct` optimum has **stopped
moving**: 0.01 and 0.02 are indistinguishable at 32,768 where at 2,048 the
optimum was 0.06 and at 8,192 it was 0.02. Consistent with the depth table —
`cp` sets depth, and once the depth is right more budget only supports it.

### 2026-09-08 05:12 — **`pmin=2` flips from a settled negative to +3.42**, and the cost frontier is measured

Paired on seed against `x-8192cp002` (353 blk), everything at
`mcts:8192:heuristic:cp=0.02`:

| flag | at `cp=2.0` | **at `cp=0.02`** | 95% CI | paired blk | p |
|---|---|---|---|---|---|
| **`pmin=2`** | **-1.35** (settled negative, 96 blk) | **+3.42** | +1.41..+5.44 | 18 | **4e-4** |
| `k=8` (`max_edges`) | +0.74 (inert) | +0.06 | -1.53..+1.64 | 19 | 0.94 |
| `noreuse` | -0.03 (inert) | **-2.00** | -3.26..-0.73 | 76 | 0.0018 |
| `cpb=4913` | — | -0.83 | -2.44..+0.79 | 54 | 0.31 |

**`prior_min_edges` is the second-largest single constant in the file.** It
shipped at 3: a node with fewer than three edges gets a *uniform* prior instead
of the softmaxed one-ply scores. At `cp=2.0` lowering it to 2 cost -1.35 and the
doc comment explained why — "a 2-edge node is resolved by three simulations, and
softmaxing two scores is sharper than the uniform it replaces". **At `cp=0.02`
it is worth +3.42.** Same mechanism as `fpu` and `tree_reuse`: a descent that is
56 sub-decisions deep instead of 8 passes through *seven times as many* narrow
nodes, and at a 2-edge node a uniform prior is the search declining to use
information it has already computed. When the descent was one turn long that
did not matter; now most of the descent is made of such nodes.

Only 18 blocks — `cp=0.02` itself read +4.68 at 13 and settled at +5.99 — so
this is a strong lead, not a verdict. More blocks running, plus the same flag at
32,768 and on the `v8` evaluator.

**`max_edges` stays inert** (+0.06 +/- 1.6) and `tree_reuse` is now confirmed
worth **+2.00** at 76 paired blocks.

#### The `v8` (current evaluator) confirmation, first reads

| contrast | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| `8192:cp=0.02` - `8192:quality` | **+10.39** | +7.68..+13.10 | 13 | 6e-17 |
| `8192:cp=0.02` - `mcts:2048:quality` (v8 null) | +10.42 | +6.52..+14.32 | 13 | 6e-9 |

**On the evaluator that will ship, the `c_puct` effect is larger, not smaller** —
+10.39 at a fixed 8,192 simulations against v6's +5.93. `d7b4e42` did not
undermine the result; it amplified it.

#### Cost, user CPU, one pinned binary, two alternated reps, 72-turn sample

| spec | ms/turn | x `mcts:2048:quality` |
|---|---|---|
| `mcts:2048:heuristic:quality` | 14.72 | 1.00 |
| `mcts:4096:heuristic:cp=0.02` | 71.60 | 4.86 |
| `mcts:8192:heuristic:cp=0.02` | 158.47 | 10.76 |
| `mcts:16384:heuristic:cp=0.02` | 366.04 | 24.86 |
| `mcts:32768:heuristic:cp=0.02` | 775.35 | 52.67 |
| `mcts:65536:heuristic:cp=0.02` | **1656.60** | **112.52** |

Cost is slightly **super-linear** in the budget (2.2x per doubling, not 2.0x):
a bigger tree transposes less and `retain_subtree` carries more.

### 2026-09-08 05:39 — `pmin=2` holds at 50 blocks, and reproduces on the `v8` evaluator

Paired on seed. v6 rows against `x-8192cp002`; v8 rows against `v8-8192cp002`.

| contrast | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| **`8192:cp=0.02,pmin=2` - `8192:cp=0.02`** (v6) | **+4.73** | +3.22..+6.24 | 50 | 4e-10 |
| **the same on `v8`** | **+2.92** | +1.06..+4.77 | 28 | 0.0013 |
| `...,q0=1,qn=16` - `cp=0.02` | **-4.18** | -6.00..-2.35 | 38 | 4e-6 |
| `...,pick=q` - `cp=0.02` | -1.60 | -3.27..+0.08 | 30 | 0.051 |
| `2048:cp=0.02` - `8192:cp=0.02` | **-4.79** | -7.09..-2.49 | 35 | 3e-5 |
| `65536:cp=0.02` - `8192:cp=0.02` | +3.15 | +0.06..+6.24 | 14 | 0.03 |

**`pmin=2` survives the evaluator change**, which is the test that mattered: it
was found on v6 and it is +2.92 on the code that will ship. Two platforms, two
independent block sets, same sign, both intervals clear of zero.

**`q_pseudo` is another flip, this time to a large negative.** At `cp=2.0`
`q0=1,qn=16` changed 2.2% of turns and measured +0.42 (inert). At `cp=0.02` it
is **-4.18**. Same story as `fpu`: anything that prices an *unvisited or barely
visited* edge is inert when the descent is 8 deep and decisive when it is 56.

**`pick=q` did not flip** — -1.60 against -2.41 at `cp=2.0`. Most-visited is
still the right root rule.

#### Where the budget ladder stops paying

Paired against `8192:cp=0.02`, all at `cp=0.02`:

| rung | vs 8192 | 95% CI | paired blk | ms/turn | x 2048:quality |
|---|---|---|---|---|---|
| 2048 | **-4.79** | -7.09..-2.49 | 35 | — | — |
| 4096 | **-1.90** | -3.54..-0.27 | 39 | 71.6 | 4.9 |
| 8192 | 0 | — | — | 158.5 | 10.8 |
| 32768 | **+2.45** | +1.39..+3.50 | 99 | 775.4 | 52.7 |
| 65536 | +3.15 | +0.06..+6.24 | 14 | 1656.6 | 112.5 |

**32,768 to 65,536 is +0.70 for 2.1x the CPU and the interval covers zero.**
The ladder is +2.9 from 2048 to 4096, +1.9 from 4096 to 8192, +2.45 from 8192 to
32768 (two doublings, so ~1.2 each), and then flat. **The ridge stops paying at
about 32,768 simulations**, which at 775 ms/turn is where the cost also starts
to hurt.

#### A caution about the `v8` null

`v8-null` (`mcts:2048:heuristic:quality` against itself) reads **-1.77
(-2.76..-0.77) at 86 blocks** — it *excludes* zero, where the v6 null is -0.10
over 374 blocks. Same baseline spec, different platform. So **v8 absolute
centred scores are offset by about -1.8 and must not be read as if zero were
the null**; every v8 claim above is a *paired* contrast between two v8 runs
sharing that baseline, where the offset cancels exactly. This is the 02:30 rule
("run a matched null for every baseline you use") biting a second time, and it
is now a platform-specific fact as well as an agent-specific one. `v8-null`
keeps running.

### 2026-09-08 05:50 — two more flips, and the budget rung confirmed on `v8`

| contrast | at `cp=2.0` | at `cp=0.02` | 95% CI | paired blk | p |
|---|---|---|---|---|---|
| `priors=mixed` | +0.38 (inert, 193 blk) | **-4.26** | -7.31..-1.21 | 13 | 0.002 |
| `pmarg` | +0.21 (inert, 86 blk) | -3.20 | -7.75..+1.35 | 10 | 0.11 |
| `pmin=2,fpu=0.4` - `pmin=2` | — | -2.97 | -6.85..+0.92 | 9 | 0.08 |

`priors=mixed` joins `priors=grad` — **`Priors::Mixed` uses the gradient on
`Take` nodes, and `Take` nodes are where the deep descent spends its length**,
so half a blind prior is most of a blind prior. `fpu` does not retune under
`pmin=2` either: 0.2 stays right.

And on the shipping evaluator:

| contrast (v8) | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| `32768:cp=0.02` - `8192:cp=0.02` | **+5.58** | +2.65..+8.50 | 8 | 6e-6 |

**On `v8` the budget rung is worth more than on v6** (+5.58 against +2.45), the
same way the `c_puct` effect was. Both of the two big search findings got
*larger* when the evaluator improved, which is what a search-side explanation
predicts: a better leaf value makes deeper lookahead worth more.

### 2026-09-08 05:53 — operational: the machine started killing arenas, and it was memory

Between 05:19 and 05:53 seven arena runs died with no output, and three
background jobs returned **exit 137 (SIGKILL)** within a minute of starting.
It looked like process management; it was not.

    vm.swapusage: total = 5120.00M  used = 3982.31M  free = 1137.69M

**Swap was 78% full.** A `mcts:32768:cp=0.02` game in flight holds ~430 MB
(`tree_reuse` grows the arena across a game's sub-decisions), so `--concurrency
3` is 1.3 GB per process and three such processes plus the eval workstream's
`evalab` exceeded 24 GB of RAM. The 02:25 entry predicted exactly this for
65,536 and the same limit binds one rung lower once several deep runs share the
machine.

**Rules for a deep-search sweep, added to the ones already in this file:**

* **`--concurrency 2` is the ceiling at 32,768 simulations**, and 65,536 wants
  its own machine or `noreuse` — which costs -2.00, so it is not free.
* **Check `sysctl vm.swapusage` before adding a job**, not just `load` and
  `%CPU`. Load average said 58 while the machine was doing 1,264% of 1,400% —
  the load number was telling me about swap wait, and I read it as scheduling.
* A run killed this way leaves its JSONL intact and `--resume` picks it up, so
  the cost is time, not data. **No duplicate writers were created** — checked
  by counting processes per `--out` path before relaunching.

### 2026-09-08 06:00 — the 32,768 rung became unmeasurable: another workstream took the RAM

A foreground probe, single-threaded, `--concurrency 1`, one block:

    arena_v8 --candidate mcts:32768:heuristic:cp=0.02,pmin=2 ... --games 4
    -> Killed: 9   (EXIT=137, after 30 s, 0 blocks written)

Not a harness problem and not a bug — **macOS jetsam**. At that moment:

| | |
|---|---|
| free RAM | **1.5 GB** of 24 |
| swap | 3,974 MB used of 5,120 |
| `evalab` processes (eval workstream) | ~10 live, 0.14-1.10 GB each, **~5 GB total** |
| a `mcts:32768:cp=0.02` game in flight | ~1.6 GB (extrapolated from 412 MB at 8,192) |

The same spec ran for two hours at `--concurrency 2` between 02:46 and 04:47,
which is why this looked like process management before it looked like memory.
**It was the neighbours.**

**What this costs.** The 32,768 rung keeps the block counts it already has —
`x-32768cp002` 120 blk, `x-32768pmin2` 8 blk, `v8-32768cp002` 16 blk — and the
paired contrasts computed from them stand. What could not be finished is
`pmin=2` *at* 32,768 to a resolvable interval. The work moved to 8,192, which
fits in ~400 MB, and the 32,768 arms are queued to retry if the RAM comes back.

**The generalisable rule:** on a shared machine, a deep-search sweep's binding
resource is RAM, not cores, and the failure is silent — the arena prints its
banner, writes nothing, and exits 137. Check `vm_stat` free pages and
`vm.swapusage` before every launch, and treat a run that produced a banner and
no blocks as *killed*, not *slow*.

### 2026-09-08 06:05 — the `v8` null is the shared-instance asymmetry, and it is agent-specific not platform-specific

Two controls on the `v8` binary, same seeds:

| control | candidate = baseline | blk | centred | 95% CI | win (null) |
|---|---|---|---|---|---|
| `v8-null-greedy` | `heuristic:full`, **stateless** | 13 | **+0.00** | +0.00..+0.00 | **0.250** |
| `v8-null` | `mcts:2048:heuristic:quality`, **stateful** | 134 | **-1.41** | -2.18..-0.65 | 0.211 |

**The seating is exact.** A stateless agent returns +0.00 with a win rate of
exactly 0.250 and an interval of zero width — the rotation cancels seat
advantage perfectly. So the -1.41 is not a harness bias; it is the thing
`bin/arena.rs:229` does, building three baseline seats from *one* agent
instance, so three seats share a `Mutex<Mcts>` — one arena, one transposition
index, one RNG — while the candidate gets its own.

On v6 that cost -0.10 over 374 blocks; on v8 it costs -1.41 over 134. The
mechanism is identical and the price is not, which is the 02:30 finding
("run a matched null for every baseline you use") in its sharpest form yet:
**the null depends on the agent, the platform *and* the evaluator, and the only
safe reading of a solo centred score is against a null measured on the same
three.** Every `v8-` claim in this file is a paired contrast between two `v8-`
runs sharing that baseline, where the offset cancels exactly.

### 2026-09-08 06:12 — the memory ceiling, stated as a limit on what this run could measure

With the eval workstream holding 26-38 `evalab` processes and swap at 3,974 MB
of 5,120, the arena is limited by RAM per budget rung, and the failure is a
silent `Killed: 9` after the banner:

| rung | approx RSS per game in flight | runnable alongside the neighbours? |
|---|---|---|
| 2,048 | 186 MB | yes, several |
| 8,192 | 412 MB | **yes, three or four** |
| 16,384 | ~800 MB | **one at a time, and not for long** |
| 32,768 | ~1.6 GB | **no** |
| 65,536 | ~3 GB | no |

A single foreground `mcts:16384:cp=0.02,pmin=2` block completed cleanly
(+17.38, one block); a second 16,384 process launched beside it was killed
within 25 seconds, and every 32,768 attempt since 05:55 was killed within 40.

**So the `pmin=2` x budget cell above 8,192 could not be finished**, and the
champion is quoted at the rung that could: `mcts:8192:heuristic:cp=0.02,pmin=2`.
What is known about the rung above is the *budget* ladder measured earlier in a
quieter window (`32768:cp=0.02` - `8192:cp=0.02` = **+2.45**, 99 paired blocks
on v6; **+5.58**, 8 paired blocks on v8) and an 8-block read of
`32768:cp=0.02,pmin=2` at +17.05 against `32768:cp=0.02`'s +9.29. Those are
consistent with the two effects stacking, and **8 blocks is not a result** — it
is the same block count that read +4.68 for `cp=0.02` before it settled at
+5.99.

### 2026-09-08 06:23 — cost of the champion, user CPU, `arena_v8`/`sprobe_v8`

`mc/cost.sh`, one spec per process, alternated, two reps, 72-turn sample.

| spec | ms/turn | x old champion | what it buys over the row above |
|---|---|---|---|
| `mcts:2048:heuristic:quality` | 14.38 | 1.00 | — |
| `mcts:8192:heuristic:cp=0.02` | 157.57 | 10.96 | **+10.39** (the `c_puct` cut, v8, 13 paired blk) |
| **`mcts:8192:heuristic:cp=0.02,pmin=2`** | **181.25** | **12.61** | **+3.06** (v8, 46 paired blk) / +4.73 (v6, 50) |
| `mcts:32768:heuristic:cp=0.02,pmin=2` | 853.33 | 59.36 | the budget rung: +2.45 (v6, 99 blk) / +5.58 (v8, 8 blk) |

**`pmin=2` costs 1.15x and pays three to five points** — by far the best ratio
of anything in this file. It is the one-ply probe firing on two-edge nodes,
which are half of all nodes but only two `eval::heuristic` calls each, so the
extra probe is cheap exactly where it turns out to matter most.

For scale on the other axis: `mcts:16384:heuristic:quality` — the same CPU spent
on *width* rather than depth — was 187.12 ms/turn and scored +0.56.

### 2026-09-08 06:30 — **the headline, on the shipping evaluator**

`arena_v8`, seed 3000000, solo unless stated. The control first, because it is
what makes the rest readable:

| run | candidate | baseline | blk | centred | 95% CI | win (null) |
|---|---|---|---|---|---|---|
| `v8-null-greedy` | `heuristic:full` | itself | 90 | **+0.00** | +0.00..+0.00 | **0.250** (0.250) |
| `v8-oldchamp-vs-greedy` | `mcts:2048:heuristic:quality` | `heuristic:full` | 183 | +3.41 | +2.72..+4.10 | 0.358 |
| **`v8-champ-vs-greedy`** | **`mcts:8192:heuristic:deeper`** | `heuristic:full` | 44 | **+13.63** | +12.11..+15.15 | **0.778** |
| **`v8-champ-vs-ab`** | the same | `minimax:8:600000::greedy:capw=25` | 26 | **+13.65** | +11.90..+15.40 | **0.774** |
| `v8-pairs-champ` | the same, `--mode pairs` | `mcts:2048:heuristic:quality` | 6 | +8.16 | +6.11..+10.21 | 0.917 (0.500) |

**Paired on seed, champion minus old champion, both against the same stateless
opponent: +10.34 (95% CI +8.50..+12.17, 44 paired blocks, p = 2e-29).**

Read the win rates. Against three copies of a stateless greedy agent an equal
player takes 25% of games; the champion this run started with takes **35.8%**
and `mcts:8192:heuristic:deeper` takes **77.8%**. Against three copies of the
alpha-beta it takes **77.4%**, where the module header records the old
champion at 40.8%.

**The control is exact.** `heuristic:full` against itself returns +0.00 with a
zero-width interval and a win rate of exactly 0.250 over 90 blocks — the seat
rotation cancels perfectly for a stateless agent — so these solo centred scores
read directly against zero with no null correction at all. That is the
strongest form of this claim available in the harness, and it is why the
stateless opponent is the one to quote.

Three opponents, three designs (stateless greedy, alpha-beta, and a 2v2 mode
where instance sharing is symmetric), one direction, all on the code that will
ship.

### 2026-09-08 06:35 — a free determinism check, and the pooled 2,048 ridge point

`mcts:2048:heuristic:cp=0.02` was raced twice by accident, into two files
(`c-cp0.02` at 01:47 and `x-2048cp002` at 06:04), same binary, same seed base.
**34 seeds are in both files and the centred scores differ by 0.000000.**

That is worth more than the row it produced. It says the arena is a pure
function of (binary, spec, seed) across a four-hour gap, so `arena_v6` was not
swapped underneath either run, and two files of the same spec may be pooled by
seed *union* (never concatenated — the shared seeds are the same games).

Pooled: **`mcts:2048:heuristic:cp=0.02` = +0.70 (CI -0.22..+1.62, 114 blocks)**.

### 2026-09-08 06:35 — the `c_puct` x budget ridge, final (v6, vs `mcts:2048:heuristic:quality`, null -0.10)

| sims | `cp=2.0` | `cp=0.06` | **`cp=0.02`** | `cp=0.01` | `cp=0.005` | ms/turn |
|---|---|---|---|---|---|---|
| 2,048 | 0 by construction | **+1.04** (264 blk) | +0.70 (114) | -0.13 (39) | -1.34 (36) | 14.4-19.3 |
| 4,096 | — | +3.98 @ `cp=0.04` (32) | **+4.64** (72) | — | — | 71.6 |
| 8,192 | +0.42 (352) | +1.94 (69) | **+5.99** (353) | +3.74 (24) | +4.12 (21) | 157.6 |
| 16,384 | +0.56 (47) | — | **+7.94** (45) | — | — | 366.0 |
| 32,768 | +0.14 (24) | — | **+9.29** (120) | +9.02 (24) | — | 775.4 |
| 65,536 | — | — | **+11.75** (14) | — | — | 1656.6 |

**The optimum `c_puct` falls with the budget and then stops falling.** 0.06 at
2,048, 0.02 at 8,192 — and at 32,768 the 0.01 and 0.02 cells are
indistinguishable (paired +0.25, CI -3.51..+4.01, 15 blk). That is what the
depth table predicts: `cp` sets the descent length (56 at 0.02, 78 at 0.01, 95
at 0.005, essentially independent of the budget), and once the length is right
more simulations only support it. **Depth is the axis; the budget is the
support.**

Paired against `8192:cp=0.02`, which is the honest form:

| rung | vs 8,192 | 95% CI | paired blk | CPU vs 8,192 |
|---|---|---|---|---|
| 2,048 | **-4.79** | -7.09..-2.49 | 35 | 0.09x |
| 4,096 | **-1.90** | -3.54..-0.27 | 39 | 0.45x |
| 32,768 | **+2.45** | +1.39..+3.50 | 99 | 4.9x |
| 65,536 | +3.15 | +0.06..+6.24 | 14 | 10.5x |

**The frontier, "best spec at each CPU budget"** (user CPU per turn, one pinned
binary, alternated; strength on the v6 platform against the old champion):

| ms/turn | x old champion | best spec | centred |
|---|---|---|---|
| 14.4 | 1.0 | `mcts:2048:heuristic:quality` | 0 |
| 19.3 | 1.3 | `mcts:2048:heuristic:cp=0.06` | +1.04 |
| 71.6 | 5.0 | `mcts:4096:heuristic:cp=0.02` | +4.64 |
| **181.3** | **12.6** | **`mcts:8192:heuristic:deeper`** | **+5.99 and +4.73 for `pmin=2` on top** |
| 366.0 | 25.5 | `mcts:16384:heuristic:cp=0.02` | +7.94 |
| **853.3** | **59.4** | **`mcts:32768:heuristic:deeper`** | +9.29, plus `pmin=2` (8 blk, unfinished) |
| 1656.6 | 115.2 | `mcts:65536:heuristic:cp=0.02` | +11.75 (14 blk) |

**Where it stops paying:** 32,768 to 65,536 is **+0.70** for 2.1x the CPU, and
the interval covers zero. Everything below 32,768 buys 1 to 3 points per
doubling. If there is no time cap, 32,768 is the rung; if there is any, 8,192
gives four fifths of the strength for a seventh of the CPU.

### 2026-09-08 06:37 — `virtual_loss` and the widening constants are **exact** no-ops, not merely inert

The last two `cp=2.0`-era knobs. Both reach the config (`mcts_label` prints
`:vl=0` and `:wc=1:wa=0.75`), and both produce **bit-identical games**:

| spec (all at 8,192 sims, `cp=0.02`) | ms/turn | agrees with the default |
|---|---|---|
| `cp=0.02` | 444.0 | 100% (itself) |
| `cp=0.02,vl=0` | 436.3 | **100.0%** of 99 turns |
| `cp=0.02,wc=1.0,wa=0.75` | 435.5 | **100.0%** of 99 turns |
| `cp=0.02,pmin=2` | 460.4 | **51.5%** — it changes *half of all turns* |

and the arena agrees: `d-w75` and `d-vl0` are identical to `x-8192cp002` on
every shared seed, to 1e-9.

**`virtual_loss` is dead code in the single-threaded search.** It is applied on
the way down and removed on backup, so it can only bite if one descent re-enters
an edge it already took — which needs a cycle through the transposition graph,
and at 56 sub-decisions deep that never happened in 99 turns. The bookkeeping is
correct and it costs ~2%; it is there for the threaded search that does not
exist.

**Progressive widening cannot bind.** `widen` admits the m-th child once
`N >= widen_c * m^widen_alpha`, and a node's median width is **2**: by the time
`N` is large enough for the two schedules to differ, PUCT at `cp = 0.02` has
already committed to the top edge. This is the third independent confirmation of
the module header's "`max_edges` / `widen_cap` / `widen_c` never", and the
strongest — the others were intervals around zero, this is *zero*.

**Read the last row against the others.** `pmin=2` changes 48.5% of played
turns; the two knobs that were "leads" from an under-powered `cp=2.0` sweep
(`wc=1.0,wa=0.75` read +1.55 at 19 blocks) change none. A flag that cannot move
a move cannot move a score, and `bin/sprobe mcts` answers that in one minute
where the arena needs an hour. **Run it before spending arena time on a knob.**

### 2026-09-08 06:45 — the 16,384 and 32,768 retries, and what is therefore left open

Six further attempts between 06:05 and 06:45, at `--concurrency` 1 and 2, with
2.6-3.7 GB reported free each time. **Every one was killed within 60 seconds.**
The eval workstream's process count went 10 -> 23 -> 26 -> **42** over the same
window, and jetsam takes the largest resident process, which is always the
deepest search. A single foreground 16,384 block *did* complete (+17.38), so
the binary is fine; what is missing is a window in which a long-running one can
live.

**Left open, and named so it can be closed later:**

1. **Does `pmin=2` stack with the budget?** Measured at 8,192 (+4.73 v6, +3.06
   v8). At 32,768 there are 8 blocks of `x-32768pmin2` reading +17.05 against
   `x-32768cp002`'s +9.29 (120 blk) — the right direction and the right size for
   stacking, and **8 blocks is not a result**. The one-minute proxy says the
   flag is still doing a great deal of work at depth: `bin/sprobe mcts` puts
   `cp=0.02,pmin=2` at **51.5% move agreement** with `cp=0.02` at 8,192
   simulations, i.e. it changes half of all turns.
2. **Where exactly the ridge tops out.** 65,536 has 14 blocks (+3.15 against
   8,192, CI +0.06..+6.24) — enough to say the ladder has flattened, not enough
   to say by how much.

**The champion is therefore quoted at 8,192**, which is measured end to end on
the shipping evaluator against three opponents, and 32,768 is recorded as the
stronger-but-unfinished rung with the ladder evidence behind it.

### 2026-09-08 06:55 — FINAL, on `arena_v8` (the shipping evaluator)

Solo mode, seed 3000000. `v8-null-greedy` (a stateless agent against itself)
returns **exactly +0.00 with a win rate of exactly 0.250 over 90 blocks**, so
rows whose baseline is `heuristic:full` read directly against zero.
`v8-null` (the stateful `mcts:2048:heuristic:quality` against itself) is -1.26
over 149 blocks, so rows against *that* baseline are quoted **paired** only.

| run | candidate | baseline | blk | centred | 95% CI | win |
|---|---|---|---|---|---|---|
| `v8-null-greedy` | `heuristic:full` | itself | 90 | **+0.00** | +0.00..+0.00 | **0.250** |
| `v8-oldchamp-vs-greedy` | `mcts:2048:heuristic:quality` | `heuristic:full` | 203 | +3.31 | +2.66..+3.96 | 0.355 |
| **`v8-champ-vs-greedy`** | **`mcts:8192:heuristic:deeper`** | `heuristic:full` | 84 | **+13.32** | +12.38..+14.27 | **0.725** |
| `v8-champ-vs-ab` | the same | `minimax:8:600000::greedy:capw=25` | 50 | **+13.40** | +12.21..+14.58 | **0.752** |
| `v8-pairs-champ` | the same, `--mode pairs` | `mcts:2048:heuristic:quality` | 14 | +7.61 | +6.31..+8.92 | 0.893 (null 0.50) |

**The two constants, each isolated, paired on seed, at a fixed 8,192 simulations:**

| contrast | centred | 95% CI | paired blk | p |
|---|---|---|---|---|
| `cp=0.02` - `cp=2.0` | **+9.85** | +8.29..+11.40 | 44 | **9e-37** |
| `pmin=2` - `pmin=3` (both at `cp=0.02`) | **+2.99** | +1.51..+4.48 | 53 | **6e-05** |
| `32768:cp=0.02` - `8192:cp=0.02` | +5.58 | +2.65..+8.50 | 8 | 6e-06 |
| **champion - old champion, both vs `heuristic:full`** | **+10.57** | **+9.32..+11.82** | **84** | **7e-63** |

**+10.57 centred, 84 paired blocks, against a common stateless opponent whose
null control is exact.** The champion takes 72.5% of its games against three
copies of that opponent; the champion this run started with takes 35.5%; an
equal agent takes 25%.

Both constants got *larger* when the evaluator improved (`cp` was +5.93 on v6
and +9.85 on v8), which is what a search-side explanation predicts: a better
leaf value makes deeper lookahead worth more.

`cargo test --release`: **181 passed, 0 failed** (including two new pins in
`tests/search.rs` — that `prior_min_edges` actually reaches a two-edge node, and
that the champion spec's flags all reach the config).

### 2026-09-08 07:00 — a memory-light proxy for the open `pmin` x budget question

The arena cell above 8,192 is unreachable, but `bin/sprobe mcts` needs one
short-lived process and answers a weaker question cheaply: **does `pmin=2` keep
changing moves as the budget rises, or does the extra budget wash it out?**
81 turns per rung, `cp=0.02` on both sides.

| sims | turns `pmin=2` plays differently | its cost ratio |
|---|---|---|
| 2,048 | **55.6%** | 1.25x |
| 8,192 | **63.0%** | 1.08x |
| 16,384 | **55.6%** | 1.05x |

**It does not wash out.** The flag still redirects more than half of all turns
at 16,384 simulations, so the mechanism — narrow nodes getting a real prior
instead of a uniform one — is not something a larger budget discovers on its
own. And it gets *cheaper* with the budget (1.25x -> 1.05x): the extra one-ply
probes are per-node and amortise over more simulations.

**What this does not show.** Changing a move is not improving it. At 2,048
simulations `cp=0.02` is already over-committed (mean descent 48 on a
2,048-simulation tree) and `pmin=2` there is unraced. So this rules out one
explanation for the effect fading — it does not establish that it stacks. The
arena cell at 32,768 remains the honest way to settle it, and it needs a
machine with ~2 GB free.

(The ms/turn printed by `sprobe` here is wall clock on a loaded machine — 1157
against the 366 that `mc/cost.sh` measures in user CPU for the same spec. Quote
the ratio from this table, never the milliseconds; user CPU is in the 06:23
entry.)

### 2026-09-08 07:12 — the `pmin` increment is drifting down as blocks accumulate; watch it

`v8-8192pmin2 - v8-8192cp002`, paired on seed, as the block count grows:

| paired blk | centred | 95% CI | p |
|---|---|---|---|
| 28 | +2.92 | +1.06..+4.77 | 0.0013 |
| 46 | +3.06 | +1.43..+4.68 | 0.00016 |
| 53 | +2.99 | +1.51..+4.48 | 5.7e-05 |
| **66** | **+2.33** | **+1.01..+3.66** | **4.9e-04** |

Every reading is inside the previous interval, so nothing is contradicted — but
the point estimate has fallen by a fifth and the direction of travel is down.
On the v6 platform the same contrast settled at +4.73 over 50 blocks. **Quote
the v8 increment as "about two to three points, still tightening"**, and take
the final value from the highest block count on file rather than from this
entry.

Unchanged and not drifting: `v8-champ-vs-greedy - v8-oldchamp-vs-greedy` =
**+10.40 (CI +9.29..+11.51, 110 paired blocks, p = 9e-77)**, and the champion
takes **72.3%** of its games against three stateless greedy agents.
