# FINDINGS — the network workstream

Log of what was tried, the numbers, and the filename→meaning mapping. Append
only; every measurement goes in the moment it lands.

**Session opened 2026-09-09.** HEAD `0fd36d9`. Pinned binaries for every
measurement live in `<scratch>/bin/` with `GITREV` beside them:

```
0fd36d986bfc6bdb6fc39c82e1e691c1cb0333fb
fca72235...  arena
f2231592...  selfplay
```

`<scratch>` = `/private/tmp/claude-501/-Users-simonchervenak-Documents-GitHub-tzgolk-in/c022ad4d-2e63-463b-bc70-9327f34a42b4/scratchpad`

---

## N1. The two seams TRAINING.md §7 calls open are already closed

`docs/TRAINING.md` §7.1 says `train/features.py` is "a much smaller hand-written
stand-in (568 wide)". It is not, any more. It calls the **real** encoder through
`src/ffi.rs`'s `tzolkin_encode_batch` over ctypes:

```
$ PYTHONPATH=train train/.venv/bin/python train/features.py data/replay
D_IN 3072   x (1024, 3072) float32   finite True
  live columns : 1913 / 3072
  range        : -1.733 .. 1.333
```

§7.2 (tensor names) is closed too:

```
$ cd train && python check_manifest.py --check
# MAIN: 82 tensors, 4332376 parameters
# SMALL: 70 tensors, 1913016 parameters
MISMATCHES: 0
```

One real bug found and fixed in `train/features.py`: its `__main__` referenced
the bare name `D_IN`, which the module only exposes through module-level
`__getattr__` (works for `features.D_IN`, not for an unqualified reference
inside the module). `NameError` on every smoke test. Now calls `d_in()`.

**Doc §7.1/§7.2 are stale; TRAINING.md should be corrected.**

## N2. `data/replay/` — the rules stamp and the codec are clean

```
$ <scratch>/bin/selfplay --verify data/replay/gen0000-000-20260907T141024Z.tzr
  rules version : 1 (current 1)
  records       : 21550717
  games         : 200000
  records/game  : 107.8
  mean score    : -14.7
  phase tags    : [0, 20014984, 0, 0, 0, 1535733, 0, 0]
  OK
```

30.9 s wall, 5.2 s user. Every record round-trips through the codec that wrote
it, on **today's** binary.

Why nothing drifted, checked rather than assumed: `git diff 944a06a..HEAD`
(944a06a is the commit in `manifest.json`) touches **neither `src/state.rs` nor
`src/tree.rs` at all**, and its only hunks in `src/record.rs` start at line 1728
— the agent/spec half. `encode_state`/`decode_state` and every `O_*` offset are
byte-identical. So:

* **rules version: valid.** `RULES_VERSION` is still 1 in `src/lib.rs`.
* **`TREE_EDGE` edge order: moot, and separately valid.** Moot because the
  buffer contains **no `TREE_EDGE` records at all** — `manifest.json` says the
  producer was `heuristic:32`, whose candidate moves come out of an RNG, so
  every record is `policy_kind = NONE`. Separately valid because `tree.rs` has
  not changed since the shards were written, so even a `TREE_EDGE` record would
  still resolve.

**The buffer is decodable. What it is not is representative — see N3.**

## N3. What the buffer actually contains, and why it is a warm start and nothing more

```
21,550,717 records over 1 shards       producer heuristic:32, git 944a06a
  mean score      : -14.8  (sd 14.7, centred sd 13.6)
  policy targets  : 0.0% usable
  day range       : 0..26
  phase tags      : Mode 92.9%, ExtraDay 7.1%, everything else 0
  root_value      : populated, r = 0.66 with the realised z_rel
```

Three facts, each of which bounds what can be learned from it:

1. **No policy targets at all.** Every record is `policy_kind = NONE`. The
   buffer can train the **value heads only**. That is TRAINING.md §6's warm
   start exactly, not a degraded mode — but it means the policy heads of any
   net trained here stay at their random initialisation.
2. **Two phases of eight.** Only `Mode` (the turn root) and `ExtraDay` appear.
   A one-ply agent decides a whole turn at `Mode`, so `Placing`, `PickWorker`,
   `Take`, `Beg`, `PityPlace` and `DraftTile` are **absent from the training
   distribution** — and those are most of the nodes a real MCTS descent
   evaluates. The phase one-hot is an input to `encode`, so the net is served
   a feature combination at play time that it never saw in training.
3. **The play is 65 points weaker than today's champion.** Mean final score in
   the buffer is **−14.7**; on the pinned `f81666d` ladder `heuristic:full`
   scores 50.4 and `mcts:32768:heuristic:deeper` scores 61.4. The `z_rel`
   target is "how a game between four weak agents ends", which is a different
   function from the one a strong search wants backed up.

## N4. The bar, measured where it is cheap: `eval::heuristic` already fits this buffer well

New binary `src/bin/valprobe.rs` (mine). It walks a shard on a stride, asks an
`Evaluator` for its `value` on each position, and scores it against the `z_rel`
the record actually realised. 40,058 positions:

| evaluator | rmse vs z_rel | Pearson r | top-1 seat | rmse by day 0-8 / 9-17 / 18-26 |
| --- | --- | --- | --- | --- |
| `eval::heuristic`, today's | **0.2900** | **+0.755** | 0.611 | 0.384 / 0.271 / 0.174 |
| `root_value` stored in the record (the evaluator that *generated* the data) | 0.3297 | +0.667 | 0.565 | 0.426 / 0.317 / 0.208 |
| a 100-step SMALL net (sanity rung) | 0.3154 | +0.701 | 0.556 | 0.379 / 0.299 / 0.255 |

Two things fall out of the middle row. Today's hand-tuned evaluator is
**0.040 rmse better** than the one that wrote these games, which is a second,
independent confirmation that `eval.rs` genuinely improved over the week — and
it is a *hard* target: the net has to beat 0.2900 on the very distribution it
is being fitted to before it has any chance inside the search.

Cost: 4.0 s user for 40k positions over both evaluators.

## N5. The missing six phases are a small problem, measured rather than feared

N3 says the buffer holds only `Mode` and `ExtraDay`; a search evaluates all
eight. `valprobe` with `VALPROBE_PHASES=1` re-evaluates the same turn-root
position under seven different phase tags and reports how far the turn holder's
value moves. 920 positions:

| evaluator | Beg | Mode | Placing{0} | Placing{2} | PickWorker | Take{w0} | PityPlace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eval::heuristic` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| a 100-step SMALL net | 0.0055 | — | 0.0138 | 0.0062 | 0.0134 | 0.0078 | 0.0129 |

`eval::heuristic` ignores the phase entirely, so its row is zero by
construction. The net's largest excursion into an unseen phase is **0.014** on
a (−1, 1) scale, against an rmse of ~0.29. So the phase one-hot is a small
enough part of a 3,072-wide input that the value head effectively ignores it,
and **the phase-coverage gap is not the binding problem.** Worth knowing before
spending a day on phase-balanced regeneration.

## N6. A held-out value fit: the net beats `eval::heuristic` after 1,000 steps

`train/train.py` now takes `--holdout FRAC` (default 0.02) and
`train/replay.py`'s `Buffer.sample` draws only from `[0, n_train)`. Records are
written a whole game at a time, so a contiguous tail is a set of **whole games**
no optimiser step ever draws from. `valprobe` takes `VALPROBE_FROM=0.98` and
walks the same tail. Without this the offline number measures memorisation.

First run, *before* the holdout landed, so ~95% out of sample by luck (1,000
steps had drawn 1.02M of 21.5M records):

| evaluator | rmse vs z_rel | r | top-1 | by day 0-8 / 9-17 / 18-26 |
| --- | --- | --- | --- | --- |
| `eval::heuristic` | 0.2927 | +0.751 | 0.610 | 0.386 / 0.273 / 0.175 |
| SMALL net, **1,000 steps** | **0.2667** | **+0.798** | **0.636** | 0.349 / 0.250 / 0.163 |

**Better than the hand-tuned evaluator on every column, after four minutes of
CPU.** That is the first evidence in this project that a trained value head is
worth anything. It is also the *easy* half of the question: fitting this
buffer's outcomes is not the same as being a better leaf inside a search that
plays 65 points better than the buffer does. The arena is the test that counts.

## N7. **The ordering flips on champion-distribution positions.** This is the finding.

N6's value fit is measured on the buffer's own positions. The evaluator's job at
play time is to score positions a `mcts:8192:heuristic:deeper` search reaches,
which are not those positions. So I generated some and re-ran the same probe.

```
<scratch>/bin/selfplay --agent mcts:8192:heuristic:deeper --games 400 \
    --out <scratch>/champ-replay --gen 10 --concurrency 24
```

First snapshot, 6,327 records over ~60 games (`<scratch>/champ-snap1.tzr`):

| | warm-start buffer (`data/replay`) | champion self-play |
| --- | --- | --- |
| mean final score | **−14.7** (sd 14.6, max 45) | **+33.7** (sd 11.7, max 60) |
| phase mix | `Mode` 93%, `ExtraDay` 7% | Beg 11%, Mode 14%, Placing 23%, PickWorker 26%, Take 19%, ExtraDay 7% |
| policy targets | 0% | **100% `TREE_EDGE`** |

And the same three evaluators, same code, on each:

| evaluator | rmse on the **buffer** | rmse on **champion positions** |
| --- | --- | --- |
| `eval::heuristic`, today's | 0.2927 | **0.2614** |
| SMALL net, 1,000 steps | **0.2667** | 0.2894 |
| `root_value` in the record | 0.3297 (old heuristic) | 0.2413 (the 8,192-sim search's own backup) |

**The order reverses.** The net wins by 0.026 rmse on the distribution it was
fitted to and loses by 0.028 on the distribution it will be used in. Pearson r
goes the same way: +0.798 vs +0.751 on the buffer, +0.635 vs +0.708 on champion
positions. So the buffer's *decodability* (N2) was never the question; its
**provenance** is, and the 48-point gap in mean final score is the whole of it.

The third row is the useful anchor. `root_value` on the champion shards is
MCTS's own backed-up value at 8,192 simulations, and it reaches 0.2413 — so a
static evaluator has about 0.02 of rmse in front of it before it is as good as
the search that would be using it. That is the size of the prize, and
`eval::heuristic` has already taken most of it.

### N7a. Caveat on the champion test set, which strengthens the conclusion

Self-play is deliberately noisy: `record::selfplay_temperature` is 1.0 for days
0-8, 0.5 for 9-17, 0.1 after, plus Dirichlet noise and playout-cap
randomisation (75% of turns run at `sims/8`). So the champion shards are
**exploratory** champion play, mean score +33.7, not the +50-ish of the
temperature-0 arena games the evaluator will actually be used in. The gap
between the buffer and the distribution that matters is therefore **larger**
than the 48 points N7 measures, not smaller.

## N8. A blend knob, and an offline sweep that costs seconds instead of core-hours

`LEARNING.md` §7.5 asks for `V = (1-λ)·v_net + λ·(something unbiased)` while the
value head is weak, with a truncated playout on the right. On this project
`eval::heuristic` is both cheaper than a playout and much stronger, so it is the
better right-hand side.

**Additive change to `src/record.rs`** (declared): `NetBatch` gains
`pub blend: f32`, applied in its two `Evaluator`/`BatchEvaluator` bodies, and
`parse_backend` accepts `PATH.safetensors@LAMBDA`. Default 0.0, so no existing
spec changes behaviour. `valprobe` takes the same spelling.

The point of doing it this way: **the two endpoints of the knob are the two arms
of the headline race.** `@1.0` is `HeuristicEvaluator`'s value exactly, `@0.0`
is the net alone, so any interior optimum is genuine complementarity and not a
rescaling — and the sweep can be run offline against a shard in seconds rather
than in the arena at core-hours per point.

λ sweep on the champion shard, 6,327 positions, 1,000-step SMALL net:

| λ | rmse | r | top-1 |
| --- | --- | --- | --- |
| 1.0 (= `eval::heuristic`) | 0.26141 | +0.7080 | 0.560 |
| **0.7** | **0.25889** | **+0.7099** | **0.564** |
| 0.5 | 0.26262 | +0.6997 | 0.549 |
| 0.3 | 0.27049 | +0.6800 | 0.546 |
| 0.0 (net alone) | 0.28936 | +0.6348 | 0.539 |

**`@1.0` reproduces the `heuristic` row to every printed digit**, which is the
control on the arithmetic. And there *is* an interior optimum — at 30% net
weight the blend beats the hand-tuned evaluator by 0.0025 rmse. Real, but small
against the 0.020 that separates `eval::heuristic` from the 8,192-simulation
search's own backed-up value.

## N9. Pipeline: everything runs end to end, including the policy path that never had data

```
$ cd train && python selftest.py ../data/replay        # 13/13 PASS
$ python train/train.py --replay data/replay --out ckpt --gen 1 --steps N --size small
$ python train/export.py --ckpt ckpt/gen0001           # -> .safetensors, 70 tensors
$ arena --candidate greedy:32:ckpt/gen0001.safetensors --baseline random --games 40
  mean centred score      32.875   (+30.100, +35.650)  +    0.00
  win rate                 1.000   ( +1.000,  +1.000)  +    0.25   [10 blocks]
```

**The smallest thing that works**: a **100-step** `Arch::SMALL` net (1,913,016
parameters, four minutes of CPU) exported to safetensors, loaded back by
`Net::load`, and played as a one-ply agent — +32.9 centred against `random`, 40
of 40 games, 10 blocks. Shards in, tensors out, checkpoint back in, games
played.

Training throughput: **6.6-7.2 steps/s** on an idle machine at batch 1,024 with
4x perspective augmentation (4,096 rows/step, ~28,000 rows/s), `Arch::SMALL`,
CPU, `OMP_NUM_THREADS=8`.

**The policy-target path now has real data behind it for the first time.**
TRAINING.md §4.3 says it is "fabricated and checked" in `selftest.py` because
"every record is `policy_kind = NONE`". On the champion shards of N7 it is 100%
`TREE_EDGE`, and `features.batch_arrays` produces live targets for all five
fixed-arity heads:

```
36,350 records   policy targets: 100.0% usable
  policy_beg         rows   51  dist sum 1.000  mask cols 3.7
  policy_mode        rows   74  dist sum 1.000  mask cols 2.0
  policy_place       rows  115  dist sum 1.000  mask cols 5.9
  policy_who         rows  129  dist sum 1.000  mask cols 2.6
  policy_extra_day   rows   45  dist sum 1.000  mask cols 2.0
```

`Take` (19% of nodes) and `DraftTile` still fall through to the value heads: the
pointer head needs `encode_choice` features the record does not carry.

## N10. The checkpoint that goes to the arena: `warm-s2053`

`Arch::SMALL`, 1,913,016 parameters, **2,053 steps** at batch 1,024 x4
perspective, lr 2e-3, `--holdout 0.02`, on the 21.1M trainable records of
`data/replay`. Stopped by Ctrl-C at 2,053 rather than the requested 20,000
because **the value head had plateaued**: `rel` (the loss term that matters —
1.5 x Huber on `v_rel` against `z_rel`) ran 0.0570 at step 250, 0.0516 at 1,000,
0.0513 at 2,000. The remaining 18,000 steps were not where anything was left.
Files: `<scratch>/ckpt/warm-s2053.{pt,safetensors}`.

Held-out probe, `VALPROBE_FROM=0.98` — the tail of whole games no optimiser step
ever drew from, 40,057 positions:

| evaluator | rmse | r | top-1 |
| --- | --- | --- | --- |
| `eval::heuristic` | 0.2924 | +0.7538 | 0.623 |
| **`warm-s2053`** | **0.2688** | **+0.7978** | **0.648** |
| `root_value` (old heuristic) | 0.3327 | +0.6645 | 0.572 |

And the λ sweep on the **whole 37,575-record champion shard**, none of which the
optimiser ever saw:

| λ | rmse | r | top-1 | day 0-8 / 9-17 / 18-26 |
| --- | --- | --- | --- | --- |
| 1.0 (= `eval::heuristic`) | 0.26166 | +0.6917 | 0.582 | 0.3380 / 0.2432 / 0.1720 |
| 0.8 | 0.25736 | +0.6972 | 0.586 | 0.3347 / 0.2389 / 0.1644 |
| **0.7** | **0.25674** | +0.6970 | **0.587** | 0.3341 / 0.2387 / 0.1628 |
| 0.6 | 0.25717 | +0.6946 | 0.585 | 0.3344 / 0.2398 / 0.1626 |
| 0.5 | 0.25867 | +0.6897 | 0.578 | 0.3354 / 0.2422 / 0.1640 |
| 0.0 (net alone) | 0.28093 | +0.6252 | 0.539 | 0.3513 / 0.2717 / 0.1911 |
| — `root_value`, i.e. the 8,192-sim search's own backup | 0.24076 | +0.7404 | 0.614 | |

So on the distribution that matters the ordering is
**blend@0.7 < heuristic < net alone**, and the whole spread from the best static
evaluator to the search's own backed-up value is 0.016 rmse. `@0.7` takes
0.005 of that 0.016 — about a third — for 30% net weight.

**These two tables are the honest summary of what this buffer can buy: the net
is a clearly better predictor of the games it was trained on and a clearly worse
one of the games it will be used in, and the only place it pays is as a minority
term in a blend.**

## N11. The races, and how they are being run

One pinned binary for every arm of every race: `<scratch>/bin2/arena`,
sha256 `8148e534...`, built at git `0fd36d9` with `PROVENANCE` beside it
recording the working tree at build time. **`src/eval.rs`, `src/mcts.rs`,
`src/tree.rs`, `src/phase.rs`, `src/encode.rs` and `src/net.rs` were all clean
at that build**, so this binary's evaluator and search are the ones
`docs/FINAL-LADDER.txt` measured (`f81666d`, two commits back, neither touching
them). Another agent is editing `src/bin/evalab.rs` and `docs/FINDINGS-eval.md`
in this tree concurrently — hence the provenance file rather than a bare rev.

The design point: the champion is `mcts:8192:heuristic:deeper`, and
`MctsConfig::priors` defaults to `Priors::OnePly`, which calls `eval::heuristic`
**directly** (`mcts.rs:2269`) rather than asking the evaluator. So swapping the
`EVAL` field of the spec changes the **leaf value and nothing else** — same
simulations, same `cp=0.02`, same `pmin=2`, same prior. That is the matched
single-variable comparison the question asks for, and it is also the only one
this buffer can support, since a value-only buffer leaves the net's policy head
at its random initialisation.

| race | candidate | baseline | file |
| --- | --- | --- | --- |
| `c8-control` | `mcts:8192:heuristic:deeper` | itself | `logs/c8-control.jsonl` |
| `h8-net` | `mcts:8192:warm-s2053.safetensors:deeper` | `mcts:8192:heuristic:deeper` | `logs/h8-net.jsonl` |
| `b8-blend07` | `mcts:8192:warm-s2053.safetensors@0.7:deeper` | `mcts:8192:heuristic:deeper` | `logs/b8-blend07.jsonl` |

All `--mode solo` (1 candidate seat, 3 baseline seats, all four searching),
`--concurrency 28`, `--resume`, one writer per file.

**Cost note:** with all four seats at 8,192 simulations these are expensive —
the net side runs at **16,600 evaluations/s at a realised batch of 9**, against
the 115,000/s `TRAINING.md` §2.1 measures at batch 256, because 28 blocks in
flight is 28 concurrent net queries and the search spends most of its time
outside `evaluate`. ~25 min of wall per game per seat, so blocks land at roughly
28 per 100 minutes. Block counts in the results below are what actually landed.

## N12. **400 champion games are worth more than 200,000 warm-start ones.** The scaling answer.

The fine-tune: `warm-s2053` continued for **900 steps** at lr 3e-4, batch 512
x4, on the 400-game champion shard (`--holdout 0.25`, so 28,181 trainable
records and 9,394 whole games never drawn from). Three and a half minutes of
CPU. `<scratch>/ckpt/ft-champ.safetensors`.

**This is also the first time the policy heads have ever had a gradient**: the
champion records are 100% `TREE_EDGE`, and all five fixed-arity heads trained —
`pi_place` 1.756 -> 1.564, `pi_beg` 0.997 -> 0.702, `pi_mode` 0.767 -> 0.524,
`pi_who` 0.952 -> 0.912, `pi_extra_day` 0.346 -> 0.300 over 900 steps.

`VALPROBE_FROM=0.75` — the held-out champion tail, 9,394 positions, no
optimiser step ever drew from them:

| evaluator | rmse | r | top-1 |
| --- | --- | --- | --- |
| `eval::heuristic` (the bar) | 0.2555 | +0.6934 | 0.599 |
| `warm-s2053` alone | 0.2766 | +0.6209 | 0.524 |
| `warm-s2053@0.7` | 0.2509 | +0.6976 | 0.592 |
| `ft-champ` alone | 0.2614 | +0.6590 | 0.559 |
| `ft-champ@0.7` | 0.2448 | +0.7113 | **0.608** |
| **`ft-champ@0.5`** | **0.2436** | **+0.7130** | 0.604 |
| `ft-champ@0.3` | 0.2473 | +0.7026 | 0.599 |
| *`root_value` — the 8,192-sim search's own backed-up value* | *0.2353* | *+0.7411* | *0.620* |

Read the gap. `eval::heuristic` sits **0.0202 rmse** above the search's own
value, which is the whole prize a static evaluator can win. The warm-start net
takes 0.0046 of it as a 30% blend term. **900 steps on 400 champion games takes
0.0119 — 59% of the prize — and moves the optimal net weight from 30% to 50%.**
Pearson r goes from +0.693 to +0.713, against the search's +0.741, and top-1
from 0.599 to 0.608 against 0.620.

Two things follow, and they are the answer to "what would it take":

1. **The 10 GB buffer is not the constraint; its provenance is.** 400 games from
   the current champion moved the held-out fit two and a half times as far as
   200,000 games from `heuristic:32` at a two-week-old evaluator.
2. **Nothing here has hit a data ceiling.** 400 games is ~28k trainable records
   for a 1.9M-parameter net. This machine generates champion self-play at
   **0.68 games/s at 24 in flight** (400 games in 590 s wall), i.e. ~2,450
   games/h, ~59,000 games/day, ~5.5M records/day. One day of generation is
   **150x** the fine-tuning set that produced the table above.

## N13. Control: identical searching agents do **not** give a zero-width interval

`docs/FINAL-LADDER.txt`'s control is `heuristic:full` against itself and reads
`+0.000, (+0.000, +0.000)` because that agent is deterministic and the seeds
match. A searching agent is not: `AgentSpec` hands each instance its own MCTS
seed from an `AtomicU64`, and the candidate and baseline specs are separate
objects, so the two sides of a self-race are not bit-identical. The noise floor
therefore has to be measured, not assumed.

`cc-control`: **`mcts:512:heuristic` vs `mcts:512:heuristic`**, solo, 200 games:

```
  mean centred score      -0.428   ( -1.727,  +0.876)  .    0.00
  win rate                 0.246   ( +0.203,  +0.289)  .    0.25
  mean game length          27.0 rounds (a complete game is 27)
  NOT DISTINGUISHABLE.  p = 0.5100.   50 blocks.
```

Cost **230.1 s user CPU** for 50 blocks, 674 MB peak RSS at `--concurrency 32`.

So the instrument is honest on a searching pair — the centred score is
indistinguishable from zero and the win rate from 0.250 — and the **noise floor
at 50 blocks is about ±1.3 points**. Nothing smaller than that should be called
an effect at that block count, whatever the point estimate says.

## N14. Operational notes worth keeping

* **The heartbeat line is invisible in a redirected log.** `arena` writes it with
  `\r` and no newline; Rust block-buffers stdout when it is not a TTY, so a
  redirected run shows one stale progress line and nothing else for an hour.
  The `--out` JSONL is the only live progress signal — `wc -l` it.
* **`pkill -f NAME` is dangerous here.** Killing `b8-blend07` also took down the
  `h8-net` race started in the same shell, because the launcher's command line
  contained both names. An hour of an 8,192-simulation race went with it (it had
  not yet completed a block, so nothing but time). Match on the `--out` path, and
  check `ps` after every kill.
* **Concurrency is the net's throughput knob and it is capped by memory.** At
  `--concurrency 28`, four searching seats at 8,192 simulations, the realised
  batch is **9-11** and the net serves 15,000-16,600 evaluations/s, against
  115,000/s at batch 256 in `TRAINING.md` §2.1. The search spends most of a
  simulation outside `evaluate`, so blocks in flight buys batch size only
  sublinearly. Peak RSS 0.7-2.0 GB per arena process at that setting.
* **The net computes priors it cannot use.** Under `Priors::OnePly` (the
  default, and what `deeper` keeps) `Mcts::priors_for` discards
  `Evaluation::priors` and recomputes the prior with `eval::heuristic`. The
  network's five policy heads and its pointer head run on every leaf and are
  thrown away. That is pure overhead in the net arm of every race below, and it
  is also why an untrained policy head does not poison the comparison.

## N15. Two details worth having beside the races

**The baseline arm *is* the published champion.** `git diff f81666d 0fd36d9`
touches only `rs/src/ui.rs` and two docs, so the `mcts:8192:heuristic:deeper`
that is the baseline of every race here is bit-for-bit the agent
`docs/FINAL-LADDER.txt` measured at **+9.21 [+8.60, +9.83], win 0.564** against
`heuristic:full`. Any centred score below is therefore directly interpretable as
a shift of that +9.21.

**Calibration** (`valprobe` now prints the slope of the outcome regressed on the
prediction; >1 means compressed, <1 means over-confident). Champion holdout:

| evaluator | sd of prediction | sd of target | slope |
| --- | --- | --- | --- |
| `eval::heuristic` | 0.292 | 0.347 | **0.83** |
| `warm-s2053` | 0.264 | 0.347 | 0.82 |
| `ft-champ` | 0.234 | 0.347 | **0.98** |
| `ft-champ@0.5` | 0.250 | 0.347 | 0.99 |
| `root_value` (the search) | 0.289 | 0.347 | 0.89 |

`eval::heuristic` is **over-confident by about 20%** — it moves 1.0 for an
outcome that moves 0.83 — and 900 steps of champion fine-tuning fixes that
almost exactly (0.98). Since `c_puct` was tuned against the heuristic's scale,
a net that is better calibrated is not automatically better *inside that
search*: `cp=0.02` was fitted to a value function 20% too loud.

## N16. Artifact index — filename to meaning

Durable copies are under **`rs/data/net/`** (`rs/data` is gitignored at the repo
root, so nothing here reaches a commit). The scratchpad copies are the working
originals and will not survive the session.

| path | what it is |
| --- | --- |
| `data/net/ckpt/warm-s2053.safetensors` | `Arch::SMALL`, 2,053 steps on `data/replay` (200k `heuristic:32` games). The warm start. **This is "the checkpoint".** |
| `data/net/ckpt/warm-s2053.pt` | same, torch side, with optimiser state — `train.py --init` / `--resume` takes this base name |
| `data/net/ckpt/ft-champ.safetensors` | `warm-s2053` + 900 steps on 300 champion games. Best static evaluator measured. |
| `data/net/ckpt/ft-champ.pt` | ditto, torch side |
| `data/net/champ-replay/gen0010-000-*.tzr` | **400 games of `mcts:8192:heuristic:deeper` self-play**, 37,575 records, 100% `TREE_EDGE`. The seed of a real generation 1. |
| `data/net/bin/arena-0fd36d9` + `PROVENANCE` | the pinned binary every number here was measured on, with the working tree recorded |
| `<scratch>/logs/cc-control.jsonl` | control: `mcts:512:heuristic` vs itself, 50 blocks |
| `<scratch>/logs/h8-net.jsonl` | `mcts:8192:warm-s2053:deeper` vs `mcts:8192:heuristic:deeper` |
| `<scratch>/logs/b8-ft05.jsonl` | `mcts:8192:ft-champ@0.5:deeper` vs `mcts:8192:heuristic:deeper` |
| `<scratch>/logs/h2-cp.jsonl` | `mcts:2048:warm-s2053:cp=0.06` vs `mcts:2048:heuristic:cp=0.06` |

Code I touched: `src/bin/valprobe.rs` (new, mine), `src/record.rs` (**additive
only** — `NetBatch::blend`, its two call sites, and the `@LAMBDA` suffix in
`parse_backend`; default 0.0 so no existing spec changes), `train/features.py`
(a `NameError` in `__main__`), `train/replay.py` + `train/train.py` (the
`--holdout` split), `docs/FINDINGS-net.md` (this file).

## N17. `cargo test --release` after the `record.rs` change

**186 passed, 0 failed, 7 ignored** — the same count the session started with.
The `NetBatch::blend` addition and the `@LAMBDA` spec suffix break
nothing; the record round-trip, the `TREE_EDGE` regeneration test and the tree
tests all pass. (Run under `nice -n 15` alongside three arena races, so the
timings in it mean nothing.)

## N18. Two points of a data-scaling curve, and a warning about the recipe

Same fine-tuning recipe (900 steps, batch 512 x4, lr 3e-4, from `warm-s2053`),
only the number of champion games varied by moving `--holdout`. All evaluated on
the same held-out tail, `VALPROBE_FROM=0.75`, 9,394 positions:

| champion games trained on | records | best λ | rmse | share of the 0.0202 gap to `root_value` closed |
| --- | --- | --- | --- | --- |
| `eval::heuristic` (the bar) | — | — | 0.2555 | 0% |
| 0 — `warm-s2053@0.7` | 0 | 0.7 | 0.2509 | 23% |
| **25** — `ft-g25@0.7` | 2,349 | 0.7 | 0.2532 | **11%** |
| **300** — `ft-champ@0.5` | 28,181 | 0.5 | **0.2436** | **59%** |
| *`root_value`, the search's own backup* | — | — | *0.2353* | *100%* |

**25 games is worse than no fine-tuning at all.** Its training loss collapsed —
`rel` 0.0021, `rank` 0.0194, an order of magnitude under the 300-game run — and
alone it scores rmse 0.3196 with r +0.43, well below where it started. 900 steps
at batch 512 over 2,349 records is 196 epochs; it memorised them. The recipe has
to scale with the data, and a fixed step count is a trap at the small end.

The useful reading is the 0 -> 300 leg: **28k champion records, three and a half
minutes of CPU, closes 59% of the distance between the hand-tuned evaluator and
the 8,192-simulation search's own value.** Nothing about that curve looks
saturated at 300 games.

## N19. **The answer at 2,048 simulations: the net alone loses, and badly.**

`h2-cp`: **`mcts:2048:warm-s2053.safetensors:cp=0.06`** against
**`mcts:2048:heuristic:cp=0.06`**, solo, one pinned binary, `cp=0.06` because
that is the documented optimum at this budget and both arms carry it.
**Complete: 100 blocks / 400 games.**

```
  metric                  value        95% CI              null
  ----------------------------------------------------------------
  mean centred score      -6.916   ( -7.568,  -6.263)  -    0.00
  points vs baseline      -9.221   (-10.090,  -8.351)  -    0.00
  margin vs best rival   -18.177   (-19.304, -17.051)        n/a
  win rate                 0.044   ( +0.024,  +0.064)  -    0.25
  ----------------------------------------------------------------
  candidate mean score     31.11                        baseline   40.33
  mean game length          27.0 rounds (a complete game is 27)
  WEAKER.  p = 0.0000.  100 blocks.
```

Cost **6,673 s user CPU** (3,819 s wall, `--concurrency 32`), 1.18 GB peak RSS.
The net side: 7,392,878 batches of **mean size 8.8** (max 32), 1,459 µs mean
queue wait, **85.2 µs per evaluation inside the net** — against the 9.8 µs
`net.rs` benchmarks in isolation and the ~20 µs `TRAINING.md` §2.1 sees at batch
256. Blocks in flight is the only way to fill that batch and it costs memory;
this is the single largest efficiency loss in the whole exercise.

**Decisive, and in the predicted direction.** The effect is −6.9 points against a
measured noise floor of ±1.3 at 50 blocks (N13), and the win rate is 0.047 where
an equal agent takes 0.250 — the net-led seat wins one game in twenty-one. The
offline probe (N7, N10) said the warm-start net is a *worse* predictor than
`eval::heuristic` on champion-distribution positions by 0.019 rmse; inside the
search that is worth about seven points of centred score.

So: **`data/replay/` as it stands cannot produce a leaf evaluator that beats
`eval::heuristic`.** Not "not yet" — the value head trained on it is a strictly
worse evaluator on the positions the search visits, and no amount of further
optimisation on that buffer changes what the buffer's `z_rel` means.

## N20. **The headline, at 8,192 simulations: the net alone loses, the blend wins.**

Both against **`mcts:8192:heuristic:deeper`** — the reigning champion, and on
this binary bit-for-bit the agent `FINAL-LADDER.txt` scores at +9.21 against
`heuristic:full` (N15). Solo mode, all four seats searching at 8,192, one pinned
binary, only the `EVAL` field of the spec differs.

**First reading, one look, blocks still accumulating:**

| candidate (all `mcts:8192:…:deeper`) | blocks | centred | 95% CI | win (null 0.250) | cand / base score |
| --- | --- | --- | --- | --- | --- |
| `warm-s2053.safetensors` (the net alone) | 16 | **−2.84** | (−4.76, −0.92) | 0.148 | 41.91 / 45.69 |
| `ft-champ.safetensors@0.5` (blend) | 25 | **+1.69** | (+0.38, +2.99) | 0.280 | 46.70 / 44.45 |

Mean game length 27.00, 0 aborted, in both.

Two things, and the second is the result this session was looking for.

**The net alone loses by less at 8,192 than at 2,048** — −2.84 here against
−6.92 at 2,048 (N19). That is the same mechanism `OVERNIGHT.md` names for why
the champion's own margin fell when `eval.rs` improved, run backwards: a deeper
search does more of the work itself, so the leaf evaluator's quality matters
less. It does not rescue the warm-start net — the interval is entirely below
zero either way.

**The blend beats the hand-tuned evaluator inside the champion's own search.**
+1.69 centred with the interval clear of zero, and a win rate of 0.280 where an
equal agent takes 0.250. Held to this project's own standard — "never call an
effect smaller than its interval an improvement" — +1.69 against a half-width of
1.30 clears the bar, **but only just, at 25 blocks, on one look at accumulating
data.** Treat it as promising and unconfirmed until the block count doubles.
Corroboration is running at 2,048 (`b2-ft05`), where blocks are 4x cheaper.

## N21. **Confirmed: the blend beats the hand-tuned evaluator; the net alone does not.**

All four races on one pinned binary (`bin2/arena`, `8148e534...`, git `0fd36d9`,
`eval.rs`/`mcts.rs`/`tree.rs` clean at build), solo mode, all four seats
searching, **only the `EVAL` field of the agent spec differs between candidate
and baseline**. Centred score and its interval are the arena's own running
figures, read from its per-block line.

| race | candidate | baseline | blocks | centred ± 95% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| `cc-control` | `mcts:512:heuristic` | itself | 50 | **−0.43 ± 1.30** | not distinguishable — the noise floor |
| `h2-cp` | `mcts:2048:warm-s2053:cp=0.06` | `mcts:2048:heuristic:cp=0.06` | **100** | **−6.92 ± 0.65** | **WEAKER** |
| `h8-net` | `mcts:8192:warm-s2053:deeper` | `mcts:8192:heuristic:deeper` | 24 | **−2.71 ± 1.50** | **WEAKER** |
| `b2-ft05` | `mcts:2048:ft-champ@0.5:cp=0.06` | `mcts:2048:heuristic:cp=0.06` | **64** | **+5.34 ± 1.03** | **STRONGER** |
| `b8-ft05` | `mcts:8192:ft-champ@0.5:deeper` | `mcts:8192:heuristic:deeper` | 28 | **+1.76 ± 1.22** | **STRONGER** |

(`b2-ft05` read +4.55 ± 1.33 at 30 blocks, +5.54 ± 1.00 at 62, +5.34 ± 1.03 at
64 — stable and tightening, which is what a real effect looks like as blocks
accumulate. `h8-net` was stopped at 24 blocks by Ctrl-C once its interval was
clear of zero, to give the two blend races the cores.)

Every race: mean game length 27.00 rounds, 0 aborted.

**Read the two columns together.** At 2,048 simulations the swing from the
warm-start net alone to the champion-fine-tuned blend is **−6.92 → +5.34, a
12.3-point move**, and the sign flips. At 8,192 it is −2.71 → +1.76, a 4.5-point
move, and the sign flips again. Both intervals on the blend rows are clear of
zero and both are wider than the −0.43 ± 1.30 control.

**The effect is smaller at the larger budget, in both directions.** −6.92 at
2,048 and −2.71 at 8,192 for the same bad evaluator; +4.55 and +1.76 for the
same good one. That is the mechanism `OVERNIGHT.md` records for why the
champion's margin over greedy *fell* when `eval.rs` got better, seen from the
other side: **a deeper search substitutes for its leaf evaluator**, so the
evaluator is worth less the more simulations sit on top of it. It is also why a
trained value head is worth more to the cheap agents than to the champion.

### The one-line answer

`data/replay/` alone produces an evaluator that costs the champion **2.71
points**. The same net plus **400 games of the champion's own self-play** and a
50/50 blend with `eval::heuristic` gains it **1.76**. The buffer is not
worthless — it is the warm start, and it is *only* the warm start.

## N22. `b2-ft05` complete — 100 blocks, the blend wins decisively

```
mcts:2048:ft-champ.safetensors@0.5:cp=0.06   vs   mcts:2048:heuristic:cp=0.06
100 blocks / 400 games

  metric                  value        95% CI              null
  ----------------------------------------------------------------
  mean centred score       5.069   ( +4.301,  +5.836)  +    0.00
  points vs baseline       6.758   ( +5.735,  +7.782)  +    0.00
  margin vs best rival    -1.755   ( -3.016,  -0.494)        n/a
  win rate                 0.431   ( +0.381,  +0.482)  +    0.25
  ----------------------------------------------------------------
  candidate mean score     46.03                        baseline   39.27
  mean game length          27.0 rounds (a complete game is 27)
  STRONGER.
```

Cost **7,264.8 s user CPU**, 5,498 s wall at `--concurrency 32`, 1.13 GB peak
RSS. Net side: 8,155,841 batches of mean size 8.3, 1,987 µs queue wait, 107.4 µs
per evaluation inside the net.

**The candidate seat wins 43.1% of its games against three copies of the
hand-tuned evaluator's own search, where an equal agent wins 25%**, and finishes
6.8 points ahead on raw score. Against the noise floor of ±1.30 at 50 blocks
(N13), and with the interval half-width down to 0.77 at 100 blocks, this is not
marginal.

For scale: `FINAL-LADDER.txt` puts the whole of `mcts:8192:heuristic:deeper` over
one-ply `heuristic:full` at +9.21. A **+5.07** swing from changing only the leaf
value of a 2,048-simulation search is over half of that, bought with 400 games
of self-play and four minutes of training.

---

# Session 2 — champion self-play as the training distribution

**Opened 2026-09-09 16:38.** HEAD `f9789b3`, working tree **clean** at pin time.
Pinned binaries in `<scratch>/bin3/` with `PROVENANCE` beside them:

```
f9789b3bad57e0285b966b3595a4f4e48e968a85
7c216f6f...  arena
5e5fc38a...  selfplay
d8206ce6...  valprobe
```

## N23. The old pin is **invalid** — `9bfcb6e` changed the rules under it

`<scratch>/bin2/arena` and `data/net/bin/arena-0fd36d9` were built at `0fd36d9`.
Since then `9bfcb6e` ("Rules: three research-adjacent bugs") landed and it
touches **`src/spaces/chichen.rs`, `src/spaces/tikal.rs` and `src/options.rs`** —
Theology-2 spending a just-granted block, a double build using a just-granted
architecture level, and card #30 at zero corn. `RULES_VERSION` is still 1, so
nothing warns. **Every number in N19-N22 was measured on a binary that plays a
slightly different game from the one the champion shards were generated with**
(`2846dda`, which is after the fix). Those numbers stand as a record of what was
true then; nothing new may be compared against them across the boundary.

Everything below is on `bin3/`, built at `f9789b3` from a clean tree, which is
`2846dda` + `docs/`, `src/ui.rs`, `src/bin/tui.rs`, `tests/` only — so **the
pinned binary and the champion data agree on the rules, the codec and the tree.**

## N24. `data/replay-champ/` verifies, and it is the champion's own distribution

The producer is live (`selfplay --agent mcts:8192:heuristic:deeper --games 20000
--concurrency 24 --keep-gb 6`, ~12.8 of 14 cores). Two hazards handled before
reading a byte:

* **Never read `gen0020.part`** — it is being appended to.
* `--keep-gb 6` deletes shards and the manifest is rewritten hourly, so the four
  sealed shards were **hard-linked** into `<scratch>/champ-frozen/` with a frozen
  `manifest.json` listing exactly those four. Hard links cost nothing and survive
  the producer unlinking its own names. This is the experiment's fixed test set.

`bin3/selfplay --verify`, each shard, on today's binary:

| shard | records | games | rec/game | mean score | verify |
| --- | --- | --- | --- | --- | --- |
| `gen0020-000` | 93,891 | 1,017 | 92.3 | **+33.8** | OK |
| `gen0020-001` | 105,973 | 1,134 | 93.5 | +33.5 | OK |
| `gen0020-002` | 190,486 | 2,058 | 92.6 | +33.6 | OK |
| `gen0020-003` | 333,891 | 3,584 | 93.2 | +33.6 | OK |
| **total** | **724,241** | **7,793** | 92.9 | +33.6 | — |

`rules version : 1 (current 1)` on all four; totals match `manifest.json` to the
record. Phase mix, from the tag histogram, identical across shards:

| Beg | Mode | Placing | PickWorker | Take | ExtraDay |
| --- | --- | --- | --- | --- | --- |
| 11.2% | 13.8% | 23.5% | 25.6% | 18.4% | 7.4% |

against `data/replay/`'s `Mode` 92.9% / `ExtraDay` 7.1%. **Policy targets 100.0%
usable** (20,118 of 20,118 in a sampled batch) against the old buffer's 0.0%.
So N7's two complaints — provenance and phase coverage — are both answered by
this directory, and it is **19.5x the 400 games** N12's scaling claim rested on.

Cost: 0.19 s user CPU to verify all four shards.

## N25. Control on the new pin: the instrument is still honest

`s2-control`: **`mcts:512:heuristic` against itself**, solo, `bin3/arena`,
`--concurrency 8`, 50 blocks / 200 games.

```
centred  +0.012  ( -1.410, +1.434)   win 0.307 (0.254, 0.360)   cand 46.30 base 46.29
days 27.00, 0 aborted
```

Centred score indistinguishable from zero; **the noise floor is ±1.42 at 48
blocks**, against N13's ±1.30 at 50 on the old pin. Nothing smaller than that
gets called an effect at that block count.

One caveat worth carrying: the **win rate came out 0.307 with a lower bound of
0.254**, i.e. marginally above its 0.250 null on a race between two copies of
the same agent, while the centred score was exactly zero. So on this instrument
**the centred score is the trustworthy statistic and the win rate is not**, at
least at this block count. Every verdict below is read off `centred`.

Cost: ~10 blocks/min at 512 simulations, concurrency 8, on a machine whose other
12.8 cores are running the self-play generator.

## N26. **(a) ANSWERED. The ordering flips back, at step 1,000.**

`bin3/valprobe`, probe set = the last 10% of `gen0020-003` (33,390 positions).
The training buffer is the four frozen shards with `--holdout 0.05`, so the
optimiser's window ends at 95% of 724,241 records = 89.15% of that shard.
**`VALPROBE_FROM=0.90` is strictly inside the held-out tail**: whole games, no
optimiser step ever drew from one of them.

`ch-scratch` at **step 1,000 of 12,000** — `Arch::SMALL`, from random init, lr
2e-3, batch 1,024 x4 perspective, champion data only, four minutes of CPU:

| evaluator | rmse | r | top-1 | slope | day 0-8 / 9-17 / 18-26 |
| --- | --- | --- | --- | --- | --- |
| `eval::heuristic`, today's | 0.2722 | +0.6757 | 0.587 | 0.81 | 0.3492 / 0.2596 / 0.1778 |
| **the net alone** | **0.2659** | **+0.6829** | **0.598** | 1.15 | 0.3273 / 0.2595 / 0.1907 |
| **net@0.5** | **0.2577** | **+0.7013** | **0.604** | **1.02** | 0.3279 / 0.2480 / 0.1700 |
| net@0.7 | 0.2608 | +0.6939 | 0.599 | 0.93 | 0.3340 / 0.2499 / 0.1696 |
| *`root_value` — the 8,192-sim search's own backup* | *0.2527* | *+0.7213* | *0.625* | *0.88* | |

**Compare directly with N7.** There, on champion positions, `eval::heuristic`
scored 0.2614 and the buffer-trained net 0.2894 — the net **lost by 0.0280**.
Here the net **wins by 0.0063**, on r as well as on rmse as well as on top-1.
The distribution was the whole of it, exactly as N7 said.

Sizing it against the prize: `eval::heuristic` sits **0.0195 rmse** above the
search's own backed-up value, which is all a static evaluator can win.

| | share of the 0.0195 closed |
| --- | --- |
| the net alone, 1,000 steps | **32%** |
| net@0.5 | **74%** |

For scale, N12's best — a warm start plus 900 steps on 400 champion games —
closed 59%, and it needed a 50% heuristic blend to do it. **1,000 steps from
random initialisation on 7,793 champion games beats the hand-tuned evaluator
outright**, with no blend at all, which nothing on `data/replay` ever did at any
step count.

Calibration reads the same way: the net's slope is 1.15 and `eval::heuristic`'s
is 0.81 (over-confident by ~20%, unchanged since N15); the 50/50 blend lands on
**1.02**, which is why it is the best of the three.

Cost: **10.1 s user CPU** for 33,390 positions across five evaluators.

## N27. Operational: **`nice` is not free on a machine this saturated**

The self-play generator holds ~12.8 of 14 cores at nice 0. An arena started at
`nice -n 14` alongside it got **83.5% of one core** and produced zero blocks in
two minutes. Restarted at nice 0 it competes fairly. Everything measured below
runs at nice 0; the training job sits at nice 10 and still gets a full core,
because it is single-process and only ever wants one.

Memory is the constraint the user named, and it is real but not where it was
feared: `PhysMem 23G used, 104M unused, 9.9G compressor` with one arena at
`--concurrency 6` costing **122 MB RSS**. So the limit on how many races can run
at once is the *core* count, not the RSS — 122 MB apiece leaves plenty of room,
and pushing `--concurrency` higher buys almost nothing because the realised
evaluation batch is set by how much of a simulation happens outside `evaluate`
(N14, N19), not by blocks in flight.

## N28. **The policy targets were scrambled.** `edge index != head slot`.

`valprobe` gained a policy probe (`VALPROBE_POLICY=1`, mine). For every
held-out `TREE_EDGE` record it rebuilds the visit distribution over edges and
scores three priors against it: the **net's own** (`Net::evaluate(..).priors`,
which is exactly what `Priors::Evaluator` hands `priors_for`), the **one-ply
`eval::heuristic` softmax** at `prior_temp = 1` that `Priors::OnePly` actually
uses, and **uniform**. Seconds, not core-hours, and — see N30 — it predicts the
arena.

It also checks something nothing had ever checked. `train/features.py` built the
fixed-arity policy targets by scattering visit count `c` at **edge index** `e`
into **head slot** `e`, and by masking slots `< n_edges`. That is right only if a
head's legal edges are a *prefix* of its slots. `encode::edges` is the authority
on the map — it is the same call the net's own prior path makes at play time —
and it disagrees:

```
features.py's `edge index == head slot` assumption, vs encode::edges:
phase        fixed nodes   mismatched
Beg                 979        37.7%
Mode               1197         0.0%
Placing            1985        32.3%
PickWorker         2129       100.0%
ExtraDay            603         0.0%
```

`PickWorker`'s slots are worker ids and its edges are *this player's* on-board
workers, so the map is essentially never the identity. `Beg`'s edges are the
temples you can actually step down. `Placing`'s are the gears you can afford.
Only `Mode` and `ExtraDay` — the two heads with a fixed, always-complete edge
list — were ever right.

**And the consequence is exactly visible in the policy quality**, same probe,
8,348 held-out positions, net = 1,000 steps on champion data:

| phase | share | edges | H(target) | CE net | CE one-ply | CE uniform | top-1 net | top-1 one-ply |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mode | 14% | 2.00 | 0.221 | **0.575** | 0.693 | 0.693 | **0.746** | 0.722 |
| ExtraDay | 7% | 2.00 | 0.218 | **0.685** | 0.771 | 0.693 | 0.546 | 0.685 |
| Beg | 11% | 3.55 | 0.210 | 0.790 | **0.647** | 1.254 | 0.719 | **0.770** |
| Placing | 24% | 5.88 | 0.681 | 1.627 | **1.603** | 1.759 | 0.327 | **0.339** |
| PickWorker | 26% | 2.64 | 0.384 | 0.984 | **0.941** | 0.945 | 0.375 | **0.423** |
| **Take** | 18% | 17.12 | 0.385 | **2.074** | **0.817** | 2.011 | **0.178** | **0.826** |
| ALL | | 5.84 | 0.399 | 1.219 | **0.995** | 1.302 | 0.437 | **0.574** |

Read the first two rows against the next three. **The net beats the one-ply
prior on exactly the two heads whose targets were correct, and loses on all
three whose targets were scrambled.** That is as clean a confirmation as the
data can give, and it was invisible from the loss curve: a scrambled target
still has a decreasing cross-entropy, it just teaches the head to rank the wrong
move.

**`Take` is a separate and larger hole.** 18% of nodes, 17 edges apiece — the
widest phase — and it has no fixed-arity head at all. The pointer head runs at
inference and has never had a gradient (`features.py` has no per-candidate
features to build a target from), so the net's prior there is **worse than
uniform** (2.074 against 2.011) while the one-ply prior is 0.817. Top-1 0.178
against 0.826.

## N29. The fix, and the test that pins it

* **`src/ffi.rs` — new `tzolkin_edge_slots`** (additive; a new `#[no_mangle]`
  symbol, nothing existing touched). `n` records in, `(kind, arity, slots)` out,
  straight from `encode::edges`. Exported rather than restated, for the same
  reason `tzolkin_encode_batch` exists: two copies of the rule is how it stopped
  agreeing the first time.
* **`train/features.py`** — targets and legality mask now built from that map.
  An out-of-range visit index is **dropped** rather than clamped onto the last
  slot (clamping credits a move that was never visited). A row whose edge list
  does not reconstruct is reported unusable rather than mapped wrongly.
* **`train/features.py`, second bug, found while fixing the first: the 4x
  perspective augmentation was being applied to the policy targets.** It is
  legitimate for the value heads — the encoder takes the querying player
  independently and every seat has a value target. It is not legitimate here:
  `net.rs` only ever queries the policy from `phase.mover(turn)`'s perspective,
  and from any other seat **the input does not identify whose move is being
  predicted**, so the head was being asked to emit one distribution from four
  inputs, three of which cannot determine it. **75% of the policy gradient was
  unlearnable noise.** Now `usable &= (movers == rec["mover"])`; those rows still
  train the value heads, as before.
* **`train/selftest.py`** — the three checks that encoded the old assumption are
  replaced. The fabricated batch now sits on *real* `Placing` nodes that really
  have 7 edges (an invented `phase_tag`/`n_edges` pair is now correctly refused,
  which is the point). New checks: no mass outside the mask; the augmented
  selection is 1-in-4 and it is the mover's row; and
  `test_slot_map_is_not_the_identity`, which prints the off-diagonal rate per
  phase and **fails if every phase reports 0%** — so the targets can never
  quietly go back to being built from `n_edges`.

`python selftest.py <champ-frozen>`: **all checks passed**, 16 of 16.

## N30. The offline policy probe predicts the arena, and it is brutal

`smoke-pri`, on `bin3/arena`: `mcts:512:<net>:deeper,pri=eval` against
`mcts:512:heuristic:deeper`, solo, 11 blocks — with the **pre-fix** net.

```
centred  -20.773  (-23.678, -17.867)   win 0.000   cand 12.30  base 39.99
days 27.00, 0 aborted
```

Against a noise floor of ±1.4 (N25) that is a sixteen-sigma catastrophe, and the
candidate lost every one of 44 games. The offline probe said the net's prior was
worse than uniform at `Take` and worse than one-ply at three phases out of five;
the arena charges **21 points** for it. **The probe and the arena agree, and the
probe costs 20 seconds where the race cost 440.** Every prior arm from here gets
screened offline first.

Cost: 440 s wall for 10 blocks at 512 simulations, `--concurrency 6`, 221 MB RSS
— **7x the wall of the same block count with `heuristic` as the evaluator**,
because the realised evaluation batch is capped by blocks in flight.

## N31. **`Take` is trainable now.** The pointer head has a gradient for the first time.

TRAINING.md and N9 both say the pointer head is blocked because "the records do
not carry per-candidate features". They do not need to: `encode::edges` **builds
those features already**, on the same call and in the same edge order the net's
own prior path uses at play time. The gap was an ABI, not data.

* **`src/ffi.rs` — new `tzolkin_edge_choices`** (additive). Per row it returns
  the kind, the board `cell` the `Take` is resolving, and
  `[max_edges, D_CHOICE]` candidate features in the engine's edge order.
* **`train/features.py`** — a `policy_ptr` target: `(sel, dist, weight, mask,
  cand, cell, tag)`. **Capped at 64 candidates**, which is `net.rs`'s `TOP_K`:
  at or under the cap stage 1 prunes nothing, so the cross-entropy trained here
  is *exactly* the softmax the search reads. The cap keeps **92.4%** of `Take`
  nodes (mean width 21.8, p99 228, max 9,038).
* **`train/model.py`** — `Pointer.forward` gained `cell` and `tag`. This was a
  second latent skew: `net.rs`'s `finish` builds the stage-2 query as
  `q_deep(h) + cell_emb[cell] + step_emb[tag]`, and the Python forward omitted
  both, so those two parameter blocks would have shipped at their
  `N(0, 0.02)` initialisation and arrived at inference as **noise added to every
  pointer query**. `Net.forward` gained `ptr_rows` so the head runs on the ~18%
  of the batch that is `Take` rather than on a mostly-padded full one.
  `loss_fn` gained `pi_deep` (weight 1.0) and `pi_fast` (0.25 — stage 1 only
  bites above `TOP_K`).
* **`train/arch.py`** — `D_CHOICE = 96` named once; the FFI length-checks
  against it, so a disagreement is a `-2` return rather than a silent reshape.

`check_manifest.py --check`: **MISMATCHES: 0** — SMALL still 70 tensors,
1,913,016 parameters, so nothing about the exported file changed. The pointer
parameters were always in it; they had never been trained.

Coverage, one sampled batch of 1,024 records: beg 126, mode 135, place 230, who
272, extra_day 73, **ptr 173** — **1,009 of 1,024 records now carry a policy
target**, against 836 before. Trial run, 60 steps: `pi_deep` 1.834 -> 1.593,
`pi_fast` 1.89 -> 1.41 (shown scaled by its 0.25 weight). Throughput unchanged
at ~4 steps/s.

## N32. **(b) OFFLINE. The net's prior beats the champion's prior in every phase.**

`p2000` — `Arch::SMALL`, from random init, **2,000 steps**, champion data only,
correct slot map, mover-only policy rows, pointer head trained. Ten minutes of
CPU. Same held-out tail, 16,591 policy nodes.

| phase | share | CE net | CE one-ply | CE uniform | top-1 net | top-1 one-ply |
| --- | --- | --- | --- | --- | --- | --- |
| Beg | 11% | **0.6269** | 0.6382 | 1.2531 | 0.776 | 0.788 |
| Mode | 14% | **0.5237** | 0.6931 | 0.6931 | **0.768** | 0.727 |
| Placing | 24% | **1.4362** | 1.6077 | 1.7571 | **0.458** | 0.337 |
| PickWorker | 26% | **0.8006** | 0.9392 | 0.9431 | **0.626** | 0.417 |
| **Take** | 18% | **0.8000** | 0.8196 | 1.9848 | 0.800 | 0.820 |
| ExtraDay | 7% | **0.5640** | 0.7335 | 0.6931 | **0.727** | 0.677 |
| **ALL** | | **0.8768** | 0.9947 | 1.3027 | **0.661** | 0.573 |

**Every phase.** `Priors::OnePly` is `OVERNIGHT.md`'s largest single measured
effect — it nearly tripled the champion's margin over greedy, from +4.15 to
+11.12 — and a 1.9M-parameter net trained for ten minutes ranks the edges better
than it does, on the visit distribution of the very search that produced the
data. Overall top-1 agreement with the 8,192-simulation search's own argmax goes
**0.573 -> 0.661**.

Read `Take` against N28: **2.0721 -> 0.8000**, from worse than uniform to better
than one-ply, entirely from giving the pointer head a gradient. It is the widest
phase and 18% of nodes, and it was the whole of the overall loss in N28.

### N32a. The value head has *regressed* at this step count, and that is the trade

Same checkpoint, value probe, 33,390 held-out positions:

| evaluator | rmse | r | top-1 | slope |
| --- | --- | --- | --- | --- |
| `eval::heuristic` | 0.2722 | +0.6757 | 0.587 | 0.81 |
| `p2000` alone | **0.2771** | +0.6498 | 0.571 | 1.18 |
| `p2000@0.3` | 0.2637 | +0.6887 | **0.601** | 1.13 |
| **`p2000@0.5`** | **0.2603** | **+0.6945** | 0.601 | **1.05** |
| `p2000@0.7` | 0.2616 | +0.6907 | 0.597 | 0.95 |
| *`root_value`* | *0.2527* | *+0.7213* | *0.625* | *0.88* |

N26 had the value head *beating* `eval::heuristic` outright at 1,000 steps, when
five of its six policy heads were being trained on scrambled targets and the
sixth did not exist — i.e. when almost none of the trunk's capacity was going
into the policy. With six working heads competing for a **1,913,016-parameter**
trunk the value head is 0.005 rmse behind the hand-tuned evaluator at 2,000
steps. `@0.5` still closes **61%** of the 0.0195 gap to the search's own backed-up
value, and remains the right value arm.

That is a capacity result, not a data result, and it names the next experiment:
`Arch::MAIN` is 4,332,376 parameters and this pipeline builds it from the same
command.

## N33. The races, and what a block costs on a machine that is already full

All on `bin3/arena` (`7c216f6f...`, git `f9789b3`, clean tree). Solo mode,
1 candidate seat against 3 baseline seats, all four searching, `--resume`, one
writer per `--out`.

`bin3/arena` predates the two `src/ffi.rs` additions of N29/N31, which is
deliberate and harmless: those add exported C symbols the arena never calls and
change no game code. It is the binary built from `f9789b3` with a clean tree,
which is what a measurement wants.

| race | candidate | baseline | isolates |
| --- | --- | --- | --- |
| `p8-prior` | `mcts:8192:p2000@1.0:deeper,pri=eval` | `mcts:8192:heuristic:deeper` | **the policy head alone** |
| `p2-prior` | `mcts:2048:p2000@1.0:cp=0.06,pri=eval` | `mcts:2048:heuristic:cp=0.06` | the same, 4x cheaper |

**`@1.0` is the trick that makes this a single-variable comparison.**
`NetBatch::mix` touches only `Evaluation::value`, never `Evaluation::priors`, so
`@1.0` reproduces `HeuristicEvaluator`'s value *exactly* (N8's control on the
arithmetic) while `pri=eval` routes `Evaluation::priors` — the net's five
fixed-arity heads and its pointer head — into `priors_for`. Candidate and
baseline therefore differ **in the prior and in nothing else**: same simulations,
same `c_puct`, same leaf value, same everything.

**Cost, measured.** The machine is running the self-play generator at ~9.2 of 14
cores and the trainer at 1; each arena process gets **about one core**, so:

| | s/block, 1 core, `--concurrency 10` | RSS |
| --- | --- | --- |
| `mcts:512` heuristic both sides (N25) | 6 | 122 MB |
| `mcts:512` net one side (N30) | 44 | 221 MB |
| `mcts:2048` net one side | ~180 | 481 MB |
| `mcts:8192` net one side | ~700 | 651 MB |

Two arena processes is the ceiling here and it is set by **memory**, not cores:
`PhysMem` reads 109 MB unused with 10 GB in the compressor, and the trees at
8,192 simulations are 48-65 MB per block in flight. Raising `--concurrency` is
the only lever that would claim a larger share of a saturated scheduler, and it
is exactly the lever memory forbids.

## N34. `cargo test --release` with the FFI and trainer changes in the tree

**200 passed, 0 failed, 7 ignored.** Per binary: lib 30, `encode` 26, `rules` 82,
`search` 36, `tree` 24, `zz_width` 1, doc-test 1. Run under `nice -n 19 --jobs 2`
alongside the generator and two races, so the timings in it mean nothing.

Files I changed, and the rule each one sits under:

| file | mine? | change |
| --- | --- | --- |
| `docs/FINDINGS-net.md` | yes | this log |
| `src/bin/valprobe.rs` | yes | the `VALPROBE_POLICY` probe |
| `train/features.py` | yes | slot map, mover-only policy rows, pointer targets |
| `train/model.py` | yes | `cell`/`tag` on the pointer query, `ptr_rows`, `pi_deep`/`pi_fast` |
| `train/arch.py` | yes | `D_CHOICE` named once |
| `train/selftest.py` | yes | the checks that pinned the old assumption, replaced |
| `train/train.py` | yes | pointer tensors threaded through `make_batch` |
| **`src/ffi.rs`** | **not listed either way** | **purely additive**: two new `#[no_mangle]` functions, `tzolkin_edge_slots` and `tzolkin_edge_choices`. Nothing existing is touched, no game code is reachable from them except the `encode::edges` they export, and no in-tree Rust caller uses them. Declared here because it is outside the files named as mine. |

`src/bin/rlab.rs` and `docs/FINDINGS-research-value.md` are also dirty in this
tree; they belong to the research workstream, not to me, and I have not touched
them. They compile, which is why the count above is 200 rather than the 196 this
session opened with.

### N33a. Which two races, and why those two

CPU, not memory, decided this in the end: each arena process gets **one core**
against the generator's nine, so the machine supports two races, and the two have
to carry the whole question between them.

| kept | spec | what it buys |
| --- | --- | --- |
| `p2-prior` | `mcts:2048:p2000@1.0:cp=0.06,pri=eval` vs `mcts:2048:heuristic:cp=0.06` | **(b)** the policy head alone, at 4x the block rate and — on N21's evidence that effects run ~2.5x larger at 2,048 — roughly 10x the statistical power per CPU-second |
| `f8-full` | `mcts:8192:p2000@0.5:deeper,pri=eval` vs `mcts:8192:heuristic:deeper` | **(c)** the strongest candidate the offline probes support, at the bar the question names |

Dropped: `p8-prior` (the isolation at 8,192 — covered at 2,048 and offline) and
two 512-simulation screens (covered by `p2-prior`, which is a better measurement
for the same money).

`f8-full` carries the **blend at 0.5 and the net's prior together**, because
that is the strongest agent the offline evidence supports: `@0.5` is the best
value column of N32a, and N32 says the net's prior beats `Priors::OnePly` in
every phase. If it loses, the decomposition is still recoverable — `p2-prior`
isolates the prior and N32a isolates the value.

**Operational, again, and it caught me a second time:** `pkill -f` on the
`--out` path killed the *poll loops watching that path* as well as the race,
exactly as N14 warns. Nothing was lost this time because the loops were only
sleeping. PIDs now live in `<scratch>/logs2/PIDS` and things get killed by PID.

## N35. **The value head and the policy heads are fighting over 1.9M parameters, and the policy is winning**

Four `Arch::SMALL` checkpoints, all champion data, all probed on the same
held-out tail (33,390 positions). The only thing that varies down the table is
**how much real policy learning the trunk is doing**:

| checkpoint | steps | policy heads with usable targets | value rmse | r | slope |
| --- | --- | --- | --- | --- | --- |
| `eval::heuristic` — the bar | — | — | 0.2722 | +0.6757 | 0.81 |
| `smoke` | 1,000 | 5, on **scrambled** targets — almost no real signal | **0.2659** | +0.6838 | 1.15 |
| `fx1000` | 1,000 | 5, correct | 0.2686 | +0.6711 | 1.08 |
| `p2000` | 2,000 | **6**, correct + pointer | 0.2771 | +0.6498 | 1.18 |
| `p4500` | 4,500 | 6, correct + pointer | **0.2812** | +0.6353 | 1.18 |
| *`root_value` — the search's own backup* | — | — | *0.2527* | *+0.7213* | *0.88* |

It is monotone, and the two rows that share a step count settle the direction:
`smoke` and `fx1000` are both 1,000 steps and differ **only** in whether the
policy targets were scrambled — 0.2659 against 0.2686. Then the pointer head
lands and the value gets worse again, and **more steps make it worse still**
(2,000 -> 4,500 is 0.2771 -> 0.2812), which rules out "it just needs longer".
The predicted `sd` shrinks all the way down the table, 0.224 -> 0.195 against a
target sd of 0.361: the value head is being squeezed out.

So N26's headline — "the net beats `eval::heuristic` outright" — was measured on
a net that was spending essentially none of its trunk on the policy. **With six
working policy heads a 1,913,016-parameter trunk cannot do both.**

That is a capacity result, not a data result, and it changes the answer to "what
would it take": **not more games. More parameters.** `Arch::MAIN` is 4,332,376
and the pipeline builds it from the same command, so that is the next run.

Two things keep this from being bad news:

* **`@0.5` is unaffected.** The blend column barely moves down the table —
  `p2000@0.5` 0.2603, `p4500@0.5` 0.2612, against `eval::heuristic`'s 0.2722 —
  still closing ~57-61% of the 0.0195 gap to the search's own value.
* **The trade is probably the right way round.** `OVERNIGHT.md` prices the prior
  as the largest single effect ever measured in this project (+4.15 -> +11.12
  against `heuristic:full` from the prior alone), and N32 says the net's prior
  now beats it in every phase. A trunk that spends itself on the policy and
  leans on `eval::heuristic` for half its value is spending itself on the more
  valuable of the two.

`ch-scratch` SMALL was stopped at **step 4,571** (`<scratch>/ckpt3/small-final.pt`,
resumable) and the core given to an `Arch::MAIN` run on the same data.

## N36. **(b) IN THE ARENA. The policy head alone is worth +9.6 points.**

`p2-prior`, first look, **10 blocks / 40 games**, `bin3/arena`, solo, all four
seats searching at 2,048 simulations:

```
mcts:2048:p2000@1.0:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06

  centred     +9.619   ( +7.654, +11.583)     null 0.00
  vs_base    +12.825
  win          0.600   ( +0.433,  +0.767)     null 0.25
  cand 50.33   base 37.50
  days 27.00, 0 aborted
```

**Only the prior differs.** `@1.0` makes `NetBatch::mix` overwrite the net's
value with `HeuristicEvaluator`'s, exactly (N8's control on the arithmetic), and
`NetBatch::mix` never touches `Evaluation::priors`. Same simulations, same
`c_puct`, same leaf value, same everything else. The candidate seat is the
champion's search with the champion's evaluator and **the network's policy head
in place of `Priors::OnePly`**.

For scale, and it is worth pausing on: `docs/FINAL-LADDER.txt` puts the *entire*
advantage of `mcts:8192:heuristic:deeper` over one-ply `heuristic:full` at
**+9.21**. A +9.6 swing from replacing one softmax is larger than that. It is
also nearly twice N22's +5.07, which was the best thing this workstream had
produced and came from the leaf *value*.

Ten blocks is a first look, on a half-width of 1.97 against a measured noise
floor of ±1.39 (N25), so the sign and the rough size are safe and the third
digit is not. The race continues.

**The offline probe called it.** N32 said the net's prior beats `Priors::OnePly`
on cross-entropy in every phase and lifts top-1 agreement with the search's own
argmax from 0.573 to 0.661, for 10 minutes of CPU. N30 said the *pre-fix* prior
was catastrophic and the arena charged 21 points for it. Both times the 20-second
probe predicted the multi-hour race, in sign and in rough magnitude.

## N37. Artifact index for this session — filename to meaning

`<scratch>` = `/private/tmp/claude-501/-Users-simonchervenak-Documents-GitHub-tzgolk-in/c022ad4d-2e63-463b-bc70-9327f34a42b4/scratchpad`

| path | what it is |
| --- | --- |
| `<scratch>/bin3/{arena,selfplay,valprobe}` + `PROVENANCE` | the pinned binaries. `arena` and `selfplay` are `f9789b3` from a clean tree; `valprobe` was rebuilt after its policy probe landed, and is not used for any arena number |
| `<scratch>/bin3/libtzolkin.dylib` | the cdylib the trainer is pinned to via `TZOLKIN_LIB`, so another agent's `cargo build` cannot move the encoder under a running job |
| `<scratch>/champ-frozen/` | **the fixed test set.** Hard links to the four sealed `gen0020` shards + a frozen manifest. 724,241 records, 7,793 games, mean score +33.6. Never includes `gen0020.part` |
| `<scratch>/ckpt2/smoke.safetensors` | SMALL, 1,000 steps, **scrambled** policy targets. Kept as the "before" of N28 |
| `<scratch>/ckpt2/fx1000.safetensors` | SMALL, 1,000 steps, slot map fixed, no pointer head |
| **`<scratch>/ckpt3/p2000.safetensors`** | **SMALL, 2,000 steps, everything fixed + pointer head. The checkpoint the races use.** |
| `<scratch>/ckpt3/p4500.safetensors` | the same run at 4,500 steps — better policy, worse value (N35) |
| `<scratch>/ckpt3/small-final.pt` | that run stopped at step 4,571, resumable with `--resume` |
| `<scratch>/ckpt4/gen0001.*` | the `Arch::MAIN` run, 4,332,376 parameters, same data |
| `<scratch>/logs2/s2-control.jsonl` | the noise floor, 50 blocks |
| `<scratch>/logs2/smoke-pri.jsonl` | the **pre-fix** prior at 512 sims: -21.5, 20 blocks |
| `<scratch>/logs2/p2-prior.jsonl` | **the policy head alone at 2,048** |
| `<scratch>/logs2/f8-full.jsonl` | **`@0.5` + net prior at 8,192, against the champion** |
| `<scratch>/logs2/PIDS` | what is running and how to kill it *without* `pkill -f` |
| `<scratch>/logs2/summ.py` | read-only block summariser. It parses JSONL and plays nothing — the one that contaminated a workstream was an `arena` invocation pointed at a live file |

## N38. `p2-prior` at 20 blocks, and it is stable

```
mcts:2048:p2000@1.0:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06
 10 blocks   centred  +9.619  ( +7.654, +11.583)   win 0.600
 20 blocks   centred  +9.347  ( +7.716, +10.977)   win 0.625   cand 51.01  base 38.55
```

Point estimate steady to a third of a point while the block count doubled and
the interval tightened — the shape a real effect has, and the shape N21 records
for `b2-ft05` (+4.55 at 30, +5.54 at 62, +5.34 at 64). Days 27.00, 0 aborted.

## N39. `Arch::MAIN` — the value head stops being squeezed

4,332,376 parameters, same data, same recipe, **3,000 steps** (1.9 steps/s
against SMALL's 3.2, so about the same wall clock as SMALL's 5,000).

| evaluator | rmse | r | top-1 | **sd of prediction** | slope |
| --- | --- | --- | --- | --- | --- |
| `eval::heuristic` | 0.2723 | +0.6755 | 0.588 | 0.300 | 0.81 |
| SMALL `p2000` (2,000 steps) | 0.2771 | +0.6498 | 0.573 | **0.195** | 1.18 |
| SMALL `p4500` (4,500 steps) | 0.2812 | +0.6353 | 0.556 | **0.195** | 1.18 |
| **MAIN `m3000`** | **0.2741** | +0.6547 | 0.573 | **0.259** | **0.91** |
| MAIN `m3000@0.5` | **0.2605** | +0.6947 | 0.603 | 0.268 | 0.94 |
| *`root_value`* | *0.2526* | *+0.7214* | *0.624* | *0.295* | *0.88* |

**The diagnostic is the `sd` column, not the rmse.** Against a target sd of
0.361, SMALL's value head predicts with sd 0.195 and a slope of 1.18 — it has
given up on the extremes, which is what a head does when it is out of capacity.
MAIN predicts with **sd 0.259 and slope 0.91**, much closer to the search's own
0.295 / 0.88, at the *same or better* policy quality (`Mode` CE 0.514 against
SMALL's 0.524, `Placing` 1.429 against 1.436). So N35's squeeze is capacity, and
2.3x the parameters visibly relieves it.

It has not yet turned into a better rmse than `eval::heuristic` — 0.2741 against
0.2723 at 3,000 steps — and the run continues to 8,000. `@0.5` is unchanged at
0.2605.

### N38a. `p2-prior` final — **24 blocks, +9.24 [+7.82, +10.66]**

```
mcts:2048:p2000@1.0:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06
 10 blk  +9.619 ( +7.654, +11.583)  win 0.600
 20 blk  +9.347 ( +7.716, +10.977)  win 0.625
 24 blk  +9.237 ( +7.815, +10.659)  win 0.635   cand 50.94  base 38.62
```

Monotone, tightening, nowhere near the ±1.39 noise floor. Days 27.00, 0 aborted,
96 games. Cost **3,529 s user CPU**, 160 s/block, 304 MB peak RSS at
`--concurrency 10`.

Stopped here on purpose: the sign and size are settled and the core is worth
more spent on the **full** candidate, whose 2,048 twin (`f2-full`,
`@0.5,pri=eval`) starts now against the same baseline and the same seat layout.
Between them they separate "the prior alone" from "the prior plus a blended
value" at one budget, and `f8-full` carries the same agent to the bar.

## N40. **The full candidate at 2,048: +14.7, winning five games in six.**

`f2-full`, first look, **6 blocks / 24 games**:

```
mcts:2048:p2000@0.5:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06

  centred    +14.740   (+11.591, +17.889)     null 0.00
  vs_base    +19.653
  win          0.833   ( +0.730,  +0.937)     null 0.25
  cand 56.25   base 36.60
  days 27.00, 0 aborted
```

Six blocks is a half-width of 3.1 and the third digit means nothing, but the
effect is ten times the noise floor.

**The decomposition at 2,048 is clean and it adds up**, all against the same
baseline, the same seat layout, the same binary:

| what changes from the champion | centred | blocks |
| --- | --- | --- |
| the leaf value only, `ft-champ@0.5` — N22, previous session, old rules | +5.07 | 100 |
| **the prior only**, `p2000@1.0,pri=eval` | **+9.24** | 24 |
| **both**, `p2000@0.5,pri=eval` | **+14.74** | 6 |

+9.24 and +5.07 sum to +14.31 against a measured +14.74, so on this evidence
**the value and the prior are close to independent contributions**, and the
prior is worth nearly twice the value. That ordering is the one
`OVERNIGHT.md` predicts from the other direction — the prior was the largest
single effect ever measured in this project — and it is the reason N28's
scrambled targets mattered so much more than N35's squeezed value head.

## N41. **(c) THE BAR IS CLEARED. +12.4 against the champion at 8,192 simulations.**

`f8-full`, first look, **6 blocks / 24 games**, `bin3/arena`, git `f9789b3`,
solo mode, all four seats searching at 8,192 simulations, only the evaluator
field of the spec differs:

```
mcts:8192:p2000@0.5:deeper,pri=eval   vs   mcts:8192:heuristic:deeper

  centred    +12.427   ( +9.415, +15.439)     null 0.00
  vs_base    +16.569
  win          0.750   ( +0.623,  +0.877)     null 0.25
  cand 61.25   base 44.68
  days 27.00, 0 aborted
```

The baseline is the reigning champion. `docs/FINAL-LADDER.txt` scores that agent
at **+9.21 [+8.60, +9.83]** over one-ply `heuristic:full` at a 0.564 win rate,
and `mcts:32768:heuristic:deeper` at +11.83. **So a 1,913,016-parameter network
trained for ten minutes on 7,793 games of the champion's own self-play beats the
champion by more than the champion beats greedy, and by more than a 4x
simulation increase buys.** It takes three games in four against three copies of
it.

Six blocks is a half-width of 3.0 — nine times the ±1.39 noise floor of N25, and
the interval is entirely clear of zero — but it is six blocks, and the race
continues. Cost so far: 5,294 s user CPU.

**The budget scaling is the one N21 predicts.** Same net, same recipe, same
baseline family:

| budget | candidate | centred | blocks |
| --- | --- | --- | --- |
| 2,048 | `p2000@0.5:cp=0.06,pri=eval` | **+14.74** | 6 |
| 8,192 | `p2000@0.5:deeper,pri=eval` | **+12.43** | 6 |

Smaller at the larger budget, because a deeper search substitutes for what its
evaluator and prior give it — and still +12.4 at the budget the question names.

## N42. `Arch::MAIN`, 8,000 steps: better on every policy head, and honestly calibrated

4,332,376 parameters, 8,000 steps at 1.9 steps/s (~70 min, one core), same
frozen champion data, same recipe. Against SMALL `p2000` on the same held-out
16,591 policy nodes:

| phase | CE MAIN | CE SMALL | CE one-ply | T1 MAIN | T1 SMALL | T1 one-ply |
| --- | --- | --- | --- | --- | --- | --- |
| Beg | **0.5840** | 0.6269 | 0.6382 | **0.804** | 0.776 | 0.788 |
| Mode | **0.4839** | 0.5237 | 0.6931 | **0.799** | 0.768 | 0.727 |
| Placing | **1.3863** | 1.4362 | 1.6077 | **0.487** | 0.458 | 0.337 |
| PickWorker | **0.7470** | 0.8006 | 0.9392 | **0.680** | 0.626 | 0.417 |
| Take | **0.7465** | 0.8000 | 0.8196 | 0.814 | 0.800 | **0.820** |
| ExtraDay | **0.5312** | 0.5640 | 0.7335 | **0.741** | 0.727 | 0.677 |
| **ALL** | **0.8288** | 0.8768 | 0.9947 | **0.693** | 0.661 | 0.573 |

Better than SMALL on all six and better than `Priors::OnePly` on all six, and it
now beats one-ply on **top-1 at `Beg`** as well, which SMALL did not.
Agreement with the 8,192-simulation search's own argmax: **0.693** against the
champion prior's 0.573.

Value, same checkpoint:

| evaluator | rmse | r | top-1 | sd | slope |
| --- | --- | --- | --- | --- | --- |
| `eval::heuristic` | **0.2722** | +0.6757 | 0.587 | 0.300 | 0.81 |
| MAIN alone | 0.2746 | +0.6504 | 0.571 | 0.241 | **0.98** |
| **MAIN@0.5** | **0.2593** | **+0.6967** | **0.604** | 0.258 | 0.98 |
| *`root_value`* | *0.2527* | *+0.7213* | *0.625* | *0.295* | *0.88* |

`MAIN@0.5` is the best static evaluator this project has measured: **0.2593**,
closing **66%** of the 0.0195 gap between the hand-tuned evaluator and the
8,192-simulation search's own backed-up value, with a calibration slope of 0.98
where `eval::heuristic` is 0.81.

The value head alone is still 0.0024 behind `eval::heuristic` on rmse. Its
`sd` is 0.241 against SMALL's 0.195 and the target's 0.361, so the squeeze of
N35 is much reduced but not gone: **at 8,000 steps on 688k records, capacity is
still the binding constraint on the value head, and it is not data.**

`MAIN` is not what the races below run — they were started on SMALL `p2000` and
a checkpoint swap mid-race would invalidate them. It is the better net on every
offline measure and 2.3x the inference cost, and racing it is the obvious next
job.

## N43. **The control that makes N36 and N41 mean what they say**

The claim in those two is "only the prior changed". Here is the check.
`u2-uniform` puts `pri=eval` on the **`heuristic` backend**, whose
`Evaluation::priors` is empty — so `priors_for` falls through to
`node_for`'s uniform fallback and the candidate is the champion's search with a
**flat prior**:

```
mcts:2048:heuristic:pri=eval:cp=0.06   vs   mcts:2048:heuristic:pt=1:pmin=3:cp=0.06
12 blocks / 48 games

  centred    -20.198   (-22.874, -17.522)     null 0.00
  win          0.000                          null 0.25
  cand 14.92   base 41.85
  days 27.00, 0 aborted
```

Three things fall out, and the third is the one that matters.

1. **`pri=eval` really does replace the prior.** Turning it on with an evaluator
   that has no priors costs 20.2 points and every single game. The flag is doing
   what the name says, not something subtler.
2. **It prices the champion's own prior.** `OVERNIGHT.md` measured uniform at
   +4.15 and one-ply at +11.12 against `heuristic:full`, a ~7-point gap on that
   baseline; measured directly, head to head at 2,048, **`Priors::OnePly` is
   worth +20.2 over uniform**.
3. **So the scale of N36 is legible.** Uniform -> one-ply is +20.2. One-ply ->
   the net's policy head is a further **+9.24** (N38a). The network's prior is
   worth **46% as much again** as the largest single effect this project had
   ever measured, on top of it.

It also explains N30 exactly. The pre-fix net's prior scored −21.5 at 512
simulations — statistically indistinguishable from having no prior at all. The
scrambled targets did not make the policy head bad; they made it **worth
nothing**, which is what its own probe said when its `Take` cross-entropy came
out above uniform.

## N44. `f2-full` at 12 blocks, `f8-full` at 10 — both holding

```
f2-full   mcts:2048:p2000@0.5:cp=0.06,pri=eval  vs  mcts:2048:heuristic:cp=0.06
   6 blk  +14.740  (+11.591, +17.889)  win 0.833
  12 blk  +14.938  (+12.816, +17.059)  win 0.854   cand 58.08  base 38.17

f8-full   mcts:8192:p2000@0.5:deeper,pri=eval   vs  mcts:8192:heuristic:deeper
   6 blk  +12.427  ( +9.415, +15.439)  win 0.750
  10 blk  +14.056  (+11.548, +16.565)  win 0.825   cand 62.52  base 43.78
```

### N43a. `u2-uniform` final — **78 blocks, −21.92 [−23.21, −20.63]**

```
mcts:2048:heuristic:pri=eval:cp=0.06   vs   mcts:2048:heuristic:cp=0.06
 12 blk  -20.198  (-22.874, -17.522)
 78 blk  -21.921  (-23.211, -20.630)   win 0.000   cand 13.15  base 42.38
```

312 games, not one of them won, days 27.00, 0 aborted. Stopped at 78 blocks; the
question it answers was settled by twelve. **The champion's one-ply prior is
worth +21.9 over a flat one at 2,048 simulations**, and that is the yardstick
the +9.24 of N38a is measured against.

### N44a. `f2-full` final — **18 blocks, +14.25 [+12.35, +16.15]**

```
mcts:2048:p2000@0.5:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06
   6 blk  +14.740  (+11.591, +17.889)  win 0.833
  12 blk  +14.938  (+12.816, +17.059)  win 0.854
  18 blk  +14.250  (+12.352, +16.148)  win 0.826   cand 57.33  base 38.33
```

72 games, days 27.00, 0 aborted. Cost 3,367 s user CPU. Stopped at 18 blocks and
the cores handed to the two 8,192-simulation races, which are the ones the
question is actually about.

**The whole 2,048 picture, one baseline, one binary, one seat layout:**

| candidate, all `mcts:2048:…:cp=0.06` | what differs from the champion | centred | 95% CI | blocks | win |
| --- | --- | --- | --- | --- | --- |
| `heuristic:pri=eval` | prior -> **uniform** | **−21.92** | (−23.21, −20.63) | 78 | 0.000 |
| — the champion itself — | — | 0 | — | — | 0.250 |
| `p2000@1.0,pri=eval` | prior -> **the net** | **+9.24** | (+7.82, +10.66) | 24 | 0.635 |
| `p2000@0.5,pri=eval` | prior -> net **and** value -> 50/50 blend | **+14.25** | (+12.35, +16.15) | 18 | 0.826 |

A 36-point span from one field of the agent spec, and the noise floor is ±1.39.

## N45. The answer, in one place

**(a) Does training on champion data flip N7's ordering?** Yes, and it took
1,000 steps. On champion-distribution positions the buffer-trained net lost to
`eval::heuristic` by 0.0280 rmse (N7); a net trained from random initialisation
on 7,793 champion games **wins by 0.0063** (N26). The distribution was the whole
of it, exactly as N7 said. The ordering then reverses again once six working
policy heads compete for a 1.9M-parameter trunk (N35), which is a *capacity*
effect, not a data effect — `@0.5` remains ahead of `eval::heuristic` at every
checkpoint, and `MAIN@0.5` reaches **0.2593 against 0.2722**, closing 66% of the
gap to the search's own backed-up value (N42).

**(b) What is the policy head worth?** More than anything else in this project.
Offline it beats `Priors::OnePly` in **every phase** and lifts top-1 agreement
with the search's own argmax from 0.573 to 0.661 for SMALL, 0.693 for MAIN
(N32, N42). In the arena, with the leaf value held *identical* by `@1.0`,
**+9.24 [+7.82, +10.66] over the champion at 2,048 simulations, 24 blocks**
(N38a) — on a scale where the champion's own one-ply prior is worth +21.9 over
uniform (N43a).

**(c) The checkpoint against `heuristic` inside the same search.**

```
mcts:8192:p2000@0.5:deeper,pri=eval   vs   mcts:8192:heuristic:deeper
  centred  +14.06  (+11.55, +16.57)   win 0.825 (null 0.250)   10 blocks / 40 games
  cand 62.52  base 43.78    days 27.00, 0 aborted
```

The baseline is the reigning champion, which `FINAL-LADDER.txt` scores at +9.21
over `heuristic:full`. **The net beats the champion by more than the champion
beats greedy**, and by more than `mcts:32768` buys (+11.83).

**(d) How much more data would it take?** None. The binding constraint moved off
data during this session: 7,793 games already carry six phases and 100%
`TREE_EDGE` targets, and the two things that were actually wrong were a
**scrambled target map** (N28) and a **missing head** (N31), neither of which
more games would have fixed. What binds now is **parameters** — N35's value-head
squeeze and N42's `sd` 0.195 -> 0.241 -> 0.295 ladder.

**(e) What did not work** — N28 (scrambled slots), N30 (the arena charged 21
points for them), N32a/N35 (the value head squeezed out of a SMALL trunk), N23
(a pinned binary invalidated by a rules commit), N14/N33a (`pkill -f`, twice).

## N46. `f8-full` — **20 blocks, +13.25 [+11.52, +14.98]**

A single arena at `--concurrency 10` writes its blocks in bursts of ten and, on
one core, a burst is 88 minutes. A **second process on a disjoint seed range**
(`--seed 8500000` against `8000000`, `--out f8-full-b.jsonl`) plays different
games of the same race and doubles the rate. Blocks are pooled by seed in
`<scratch>/logs2/pool.py`, which is read-only. This is **not** the F6 failure
mode — that was two runners appending to *one* file, which duplicated rows and
narrowed the interval by sqrt(2). Here the seed ranges cannot overlap, the files
are separate, and the pooler de-duplicates by seed anyway.

**Correction, checked rather than assumed:** at the time of writing the second
runner has produced **zero** blocks (it needs ~70 minutes of CPU for its first
burst of six), so **all 20 blocks below come from the primary runner alone** —
seeds 8000000..8000438, all distinct. The pooling is set up and verified, it has
simply not contributed yet. Arena advances the block seed by ~22 per block, so
the primary's 500 planned blocks reach ~8011000 and cannot collide with the
second runner's 8500000.

```
mcts:8192:p2000@0.5:deeper,pri=eval   vs   mcts:8192:heuristic:deeper

  10 blk   +14.056  (+11.548, +16.565)  win 0.825
  17 blk   +13.640  (+11.851, +15.428)  win 0.824
  20 blk   +13.250  (+11.519, +14.981)  win 0.800
           cand 61.99   base 44.32   days 27.00   80 games   0 aborted
```

Point estimate steady inside a point across a doubling of the block count, the
interval tightening, and the lower bound **+11.5** — still above the +9.21 that
`FINAL-LADDER.txt` gives for the whole of the champion over `heuristic:full`.

## N47. **The ladder rung: +20.76 against `heuristic:full`, where the champion is +9.21**

The obvious worry about N41/N46 is that the net was trained on games the
baseline generated, so beating that baseline could be opponent modelling rather
than strength. `l8-ladder` answers it by changing the opponent entirely:
**`heuristic:full`** is one-ply greedy over every legal move — a different agent
class, not the data generator, and the rung `docs/FINAL-LADDER.txt` publishes.

```
mcts:8192:p2000@0.5:deeper,pri=eval   vs   heuristic:full
7 blocks / 28 games

  centred    +20.759   (+17.336, +24.181)     null 0.00
  vs_base    +27.679
  win          0.893                          null 0.25
  cand 73.00   base 45.32
  days 27.00, 0 aborted
```

Placed on the published ladder, measured against the same baseline:

| agent | centred vs `heuristic:full` | raw score | source |
| --- | --- | --- | --- |
| `mcts:8192:heuristic:deeper` — the champion | **+9.21** [+8.60, +9.83] | 61.4-ish | `FINAL-LADDER.txt` |
| `mcts:32768:heuristic:deeper` | +11.83 | — | `FINAL-LADDER.txt` |
| **`mcts:8192:p2000@0.5:deeper,pri=eval`** | **+20.76** [+17.34, +24.18] | **73.00** | here, 7 blocks |

**It is not opponent modelling.** Against an agent that generated none of its
training data the net-augmented search is worth more than twice the champion's
own margin, and 1.75x what a 4x simulation increase buys. Seven blocks is a
half-width of 3.4 and the two numbers are from different sessions and different
rules commits (N23), so the comparison is indicative rather than exact — but the
gap is 11.5 points and the noise floor is 1.4.

### N47a. `l8-ladder` final — **8 blocks, +20.90 [+17.92, +23.88]**, raw score 73.00

```
mcts:8192:p2000@0.5:deeper,pri=eval   vs   heuristic:full
  7 blk  +20.759  (+17.336, +24.181)  win 0.893
  8 blk  +20.898  (+17.922, +23.875)  win 0.906   cand 73.00  base 45.14
```

32 games, days 27.00, 0 aborted, cost 5,121 s user CPU. Stopped at 8 blocks and
the core given to `m8-main`, which puts the `Arch::MAIN` checkpoint of N42 —
the better net on every offline measure — at the same bar.

## N48. One handicap the candidate carries, for the record

Under `Priors::Evaluator` the raw one-ply score vector is empty, so
`Mcts::q_init_from` returns an empty vector and **every edge gets `q_init = 0`**.
The baseline, on `Priors::OnePly`, gets the real `q_init` offsets that order its
unvisited edges. So the candidate in `p2-prior`, `f2-full`, `f8-full`,
`l8-ladder` and `m8-main` is not merely swapping one prior for another — it is
also **giving up `q_init` entirely**, and still wins by 13.25 at the bar.

That cuts one way only: it makes the measured effect a *lower* bound on what the
policy head is worth, and it names a free improvement — a `Priors` variant that
takes the evaluator's priors and keeps the one-ply `q_init` would cost one
`one_ply` call per node and is not expressible in `mcts.rs` today.

Both arms are otherwise identical: `mcts8192/…@0.5:pri=eval:cp=0.02` against
`mcts8192/heuristic:pt=1:pmin=2:cp=0.02`, same simulations, same `c_puct`, same
seat layout, same binary. (`pt` and `pmin` are absent from the candidate's label
because `mcts_label` suppresses them when `priors == Evaluator`, where they have
no effect — `record.rs:2305`.)

### N46a. `f8-full` at 30 blocks — **+13.20 [+11.87, +14.52]**, win 0.804

```
  10 blk   +14.056  (+11.548, +16.565)  win 0.825
  17 blk   +13.640  (+11.851, +15.428)  win 0.824
  20 blk   +13.250  (+11.519, +14.981)  win 0.800
  30 blk   +13.196  (+11.871, +14.521)  win 0.804   cand 61.93  base 44.34
```

120 games, days 27.00, 0 aborted. The half-width is now **1.33**, which is the
noise floor N25 measured at 50 blocks, and the point estimate has moved 0.86 of
a point across a tripling of the block count.

## N49. **`Arch::MAIN` in the arena: +17.23, three points above SMALL at the same budget**

`m2-main`, first look, **6 blocks / 24 games**. Same baseline, same budget, same
binary, same seat layout as `f2-full` — the *only* difference is which
checkpoint the candidate carries:

```
mcts:2048:m8000@0.5:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06

  centred    +17.229   (+15.155, +19.303)     null 0.00
  win          0.896                          null 0.25
  cand 59.25   base 36.28     days 27.00, 0 aborted
```

| checkpoint | parameters | steps | centred at 2,048 | blocks |
| --- | --- | --- | --- | --- |
| SMALL `p2000` | 1,913,016 | 2,000 | +14.25 [+12.35, +16.15] | 18 |
| **MAIN `m8000`** | **4,332,376** | 8,000 | **+17.23** [+15.16, +19.30] | 6 |

**The capacity story closes in the arena.** N35 found the value head being
squeezed out of a 1.9M-parameter trunk by six working policy heads; N42 found
2.3x the parameters relieved the squeeze (`sd` 0.195 -> 0.241 against a target
0.361) and improved every policy head; and here that is worth **+3.0 points** on
top of an already-decisive +14.25. Six blocks, so the third digit is noise, but
the intervals do not overlap.

`m8-main` — the same checkpoint at 8,192 — was started and abandoned: `MAIN` is
2.3x the inference cost, and on one core a first burst of six blocks at 8,192
needs ~2.7 hours. The 2,048 reading is the same comparison for a quarter of the
money.

### N46b. `f8-full` at 40 blocks — **+13.59 [+12.36, +14.83]**, win 0.809

```
  10 blk   +14.056  (+11.548, +16.565)  win 0.825
  20 blk   +13.250  (+11.519, +14.981)  win 0.800
  30 blk   +13.196  (+11.871, +14.521)  win 0.804
  40 blk   +13.591  (+12.355, +14.826)  win 0.809   cand 62.36  base 44.24
```

160 games, days 27.00, 0 aborted. Half-width **1.24**.

---

# Session 14 (continued after a usage-limit kill mid-sentence at N49)

## N50. Collecting the races that were still running — and a correction to N49

Read off disk before touching anything, `summ.py` over `logs2/*.jsonl`:

```
f8-full     blk 40  centred +13.600 (+12.368,+14.832)  win 0.831  cand 62.19 base 44.05
f8-full-b   blk 12  centred +13.125 (+10.473,+15.777)  win 0.792  cand 62.25 base 44.75
  POOLED    blk 52  centred +13.490 (+12.374,+14.607)  win 0.822  208 games, 0 aborted
m2-main     blk 12  centred +15.307 (+12.889,+17.726)  win 0.802  cand 57.27 base 36.86
f2-full     blk 18  centred +14.250 (+12.352,+16.148)  win 0.826  cand 57.33 base 38.33
l8-ladder   blk  8  centred +20.898 (+17.922,+23.875)  win 0.906  cand 73.00 base 45.14
p2-prior    blk 24  centred  +9.237 ( +7.815,+10.659)  win 0.635
s2-control  blk 50  centred  -0.016 ( -1.387, +1.354)  win 0.305   <- the null, still null
u2-uniform  blk 78  centred -21.921 (-23.211,-20.630)  win 0.000
```

**(a) `f8-full-b` has now contributed, and the pooling works.** Twelve blocks
from seeds 8500000.., zero duplicates against the primary's 8000000.. range,
and the two independent runners agree to within 0.5 of a point (+13.60 against
+13.13, intervals overlapping almost entirely). Pooled, **52 blocks / 208 games,
+13.49 [+12.37, +14.61]**, half-width **1.12** — below the 1.33 noise floor N25
measured at 50 blocks. This is the number for the champion-budget race and it is
finished; the point estimate has moved 0.57 of a point across 10 -> 52 blocks.

**(b) N49 was six blocks and the third digit was not the only thing that was
noise.** `m2-main` has doubled to 12 blocks and the MAIN advantage has halved:

| checkpoint | params | centred at 2,048 | blocks | interval |
| --- | --- | --- | --- | --- |
| SMALL `p2000` | 1,913,016 | +14.25 | 18 | [+12.35, +16.15] |
| MAIN `m8000` | 4,332,376 | **+15.31** (was +17.23 at 6 blk) | 12 | [+12.89, +17.73] |

The intervals now overlap over almost their whole length. **N49's "+3.0 points,
intervals do not overlap" does not survive doubling the block count** — the gap
is +1.06 and the pooled half-width on the difference is larger than the gap.
MAIN is not yet shown to be worth anything in the arena. The offline case (N42:
`sd` 0.195 -> 0.241, every policy head better, top-1 0.661 -> 0.693) stands
unchanged; what does not stand is the claim that it converts to three points of
score. Logged here before doing anything else, because N49 as written is wrong.

## N51. The second generation is the wrong next input, because the *first* one was only 39% consumed

Before deciding anything about generating more games I counted what is already
on disk. `data/replay-champ/manifest.json`:

```
generation 20, agent mcts8192/heuristic:pt=1:pmin=2:cp=0.02, commit 901257d
games   20,000
records 1,856,372   over 8 shards, gen0020-000 .. gen0020-007
```

And the header of both training runs that produced the checkpoints this whole
session is built on (`train-ptr.log` and `train-main.log`, identical first line):

```
724,241 records over 4 shards  (688,028 trainable, 5.0% held out)
  newest shard : gen0020-003-20260909T202303Z.tzr
```

**`p2000` and `m8000` were trained on shards 000-003 only — 724,241 of
1,856,372 records, 39%.** Not by a choice anyone made: shards 004-007 were
written at 21:23, 22:23, 23:23 and 00:19, and the training ran 17:33-18:43. The
self-play the brief describes as "finished at 20,000 games" finished *after* the
nets were trained, and **2.56x the data has never had a gradient taken on it.**

So N45(d)'s "how much more data would it take? None" was answered against 688k
records, and the experiment that tests it has been sitting on disk unrun and
costs no self-play at all.

**Decision: do not generate a second generation yet.** A generation of games
from `p2000`/`m8000` is 20,000 games of 8,192-simulation search — the single
most expensive thing this project can do — and it would be bought *before*
finding out whether the data already generated is exhausted. The order is:

1. Retrain `MAIN` on the full buffer, same recipe, same step count, **only the
   buffer changes** — the controlled version of "is it data or parameters?".
2. If the value head's `sd` climbs off 0.241 toward the target's 0.361 and the
   arena moves, **data was still binding**, N45(d) is wrong, and a second
   generation is worth its cost.
3. If it does not move, data is genuinely saturated at this architecture, and
   the second generation is worth its cost for a *different* reason (better
   targets from a stronger searcher), which is a claim that then has to be
   argued rather than assumed.

Set up so both nets are scored on records neither has seen: shard
`gen0020-007` (142 MB, ~277k records) is withheld from the new run entirely and
becomes the common holdout for `valprobe`.

## N52. `train.py` now reads the holdout it has always reserved

`Buffer.__init__` has carved a contiguous tail of whole games out of the
sampling range since the loader was written, and `train.py` never once looked at
it. Every number in `train-ptr.log` and `train-main.log` — every number this
session's architecture conclusions were read off — is a **training** loss, which
is the one quantity that cannot distinguish a net that is learning from a net
that is memorising. That is a bad instrument to answer N51 with, so it is fixed
before N51's run is read.

* `Buffer.sample_holdout(n, rng)` (`train/replay.py`) draws from
  `[n_train, n)`. It **raises** on a buffer opened with `holdout=0.0` rather
  than falling back to `sample`: a validation number that is quietly the
  training set is worse than none.
* `evaluate()` (`train/train.py`) averages `loss_fn` over `--eval-batches`
  held-out batches under `no_grad` and `net.eval()`. The rng is **re-seeded
  identically on every call**, so it is the same records at every checkpoint and
  a move in the number is a move in the net, not a resample.
* `--eval-every N` (default 500, 0 disables), `--eval-batches K` (default 8).
  Readings print as `HELD <step>` lines, append to `<base>.held.jsonl`, and the
  last one goes into `<base>.json` beside the training loss.
* `forward()` / `to_dev()` factored out so the eval path and the training path
  cannot drift — the pointer-row routing is written once.

Checked: the final evaluation is suppressed when `--steps` is a multiple of
`--eval-every` (it was double-logging step 8000); the no-holdout buffer warns
once and skips; and `selftest.py` gained `test_holdout_is_disjoint`, which
asserts the two index ranges do not overlap and that the no-holdout buffer
refuses. `train/selftest.py <replay7>` — all checks pass.

**N51's run was restarted to pick this up.** It was 450 of 8,000 steps in, the
seeds are fixed (`torch.manual_seed(1234+gen)`, `default_rng(1234+gen)`) and
`evaluate` draws from its own generator under `no_grad`, so the restarted run
reproduces the same training trajectory and adds the curve. 145 seconds of CPU
to make the experiment readable.

## N53. **The held-out probes were not held out. `VALPROBE_FROM` was pointed at the wrong shard.**

N51's experiment needed a "before" reading on records neither checkpoint had
seen, so I probed `gen0020-007` — written at 00:19, after both trainings had
finished at 18:43, and in no training buffer. Both nets are equally blind to it.

```
valprobe gen0020-007 20000 heuristic p2000@0.5 m8000@0.5      21,349 positions

  eval::heuristic     rmse 0.2699  r +0.6784  top1 0.585  sd 0.296  slope 0.83
  SMALL p2000@0.5     rmse 0.2605  r +0.6926  top1 0.595  sd 0.235  slope 1.06
  MAIN  m8000@0.5     rmse 0.2604  r +0.6918  top1 0.594  sd 0.254  slope 0.98
  root_value (search) rmse 0.2511  r +0.7231  top1 0.626  sd 0.292  slope 0.89
```

and the policy heads on the same 21,193 nodes:

| | CE ALL | T1 ALL |
| --- | --- | --- |
| SMALL `p2000` | 0.8775 | 0.655 |
| MAIN `m8000` | **0.8733** | **0.664** |
| `Priors::OnePly` | 0.9821 | 0.583 |

**2.3x the parameters buys 0.0001 rmse and 0.0042 nats.** N42 measured the same
two checkpoints on what it called a held-out set and got 0.2593 against SMALL's
worse figure and **CE 0.8288 against 0.8768, T1 0.693 against 0.661** — ten
times the policy gap.

Note which side moved. **SMALL scores 0.8775 here against N42's 0.8768 — the
same net, the same number.** MAIN goes 0.8288 -> 0.8733. A distribution shift
between the two probe sets would move both. Only the big net moved, which is
the signature of a probe set the big net had already fitted.

### N53a. Where the holdout actually is

`Buffer.open` reverses the file list so the **newest shard is index 0**, then
concatenates. The holdout is `[n_train, n)` — the tail of the *concatenation* —
which is therefore the tail of the **oldest** shard, not the newest. Measured,
not reasoned:

```
shard order as the Buffer indexes them, the 4-shard buffer both nets trained on:
  [        0 ..   333,891)  gen0020-003   333,891 records
  [  333,891 ..   524,377)  gen0020-002   190,486
  [  524,377 ..   630,350)  gen0020-001   105,973
  [  630,350 ..   724,241)  gen0020-000    93,891

  n = 724,241   n_train = 688,028   holdout = [688,028, 724,241)
  the held-out tail lies in shard: gen0020-000     <- the OLDEST

  gen0020-003 occupies global [0, 333,891)
  VALPROBE_FROM=0.90 on gen0020-003 probes global [300,501, 333,891)
  inside the training window [0, 688,028)?  YES -- the probe was training data
```

N26 states the reasoning that went wrong, in as many words: "the optimiser's
window ends at 95% of 724,241 records = 89.15% of that shard", taking for
granted that the tail of the buffer is the tail of `gen0020-003`. It is the tail
of `gen0020-000`. So `VALPROBE_FROM=0.90` on `gen0020-003` — the probe behind
**N26, N32, N35 and N42** — walked records the optimiser drew from freely.

The training code was never wrong. `sample` really does draw from `[0, n_train)`
and the reserved tail really is whole games no step ever sees. What was wrong is
that every probe was aimed at the wrong file, and `--holdout`'s own help text
says "fraction of the **newest** records reserved for evaluation", which is the
opposite of what the ordering produces. Nothing in the output said which file to
aim at, so nothing caught it.

### N53b. What this invalidates, and what it does not

**Invalid — offline, training-set numbers, not generalisation:** the MAIN-vs-
SMALL policy table in N42; the value tables in N26, N32, N35, N42 insofar as
they claim to be held out. The *ordering* against `eval::heuristic` may well
survive — `eval::heuristic` has no parameters and cannot memorise, so a net
beating it on training data is weak evidence but not zero — but the margins are
not generalisation margins and the SMALL-vs-MAIN comparisons are worthless.

**Still valid — nothing in the arena touches this.** `f8-full`, `f2-full`,
`l8-ladder`, `p2-prior`, `u2-uniform`, `s2-control`, `m2-main` play games from
fresh seeds against a live opponent. The headline stands untouched: **52 blocks,
+13.49 [+12.37, +14.61]** at the champion's own budget. So does +20.90 against
`heuristic:full` and +9.24 for the policy head alone.

**And it explains two things that had not made sense.** N49's +3.0 for MAIN
collapsed to +1.06 with overlapping intervals when the block count doubled
(N50b); and N45(d) concluded "parameters bind, not data" on the strength of the
N35/N42 ladder. Both were reading a memorisation gradient. **On honest held-out
data MAIN and SMALL are the same net**, and the claim that capacity is the
binding constraint has no evidence behind it.

### N53c. The same two checkpoints on three probe sets — the overfit, isolated

`valprobe`, the policy probe, `p2000` and `m8000` unchanged throughout. Only the
records change.

| probe set | what it is | SMALL CE | MAIN CE | MAIN's edge | SMALL T1 | MAIN T1 |
| --- | --- | --- | --- | --- | --- | --- |
| `gen0020-003` tail, `FROM=0.90` | **training data** (what N26/N42 ran) | 0.8780 | **0.8274** | **0.0506** | 0.658 | 0.690 |
| `gen0020-000` tail, `FROM=0.6143` | **the real holdout** of that buffer | 0.8802 | 0.8792 | 0.0010 | 0.652 | 0.659 |
| `gen0020-007`, all of it | a shard written after training ended | 0.8775 | 0.8733 | 0.0042 | 0.655 | 0.664 |

**Read the SMALL row across: 0.8780, 0.8802, 0.8775.** The 1.9M-parameter net
scores the same on data it trained on and on data it has never seen — a spread
of 0.003 nats over three sets, which is what "not overfitting" looks like. The
4.3M net is **0.0506 better on the records it trained on and 0.001 better
everywhere else.**

The value probe says the same thing more quietly, and once with the sign
flipped:

| probe set | heuristic | SMALL@0.5 | MAIN@0.5 | root_value |
| --- | --- | --- | --- | --- |
| training data | 0.2722 | 0.2603 | **0.2593** | 0.2527 |
| the real holdout | 0.2738 | **0.2633** | 0.2637 | 0.2539 |
| unseen shard 007 | 0.2699 | 0.2605 | **0.2604** | 0.2511 |

The training-data row reproduces **N42's value table to four decimals**
(0.2722 / 0.2593 / 0.2527), which is the check that this is the probe N42 ran.
On the real holdout MAIN is 0.0004 *worse* than SMALL.

**So `Arch::MAIN`'s entire measured advantage was memorisation of 688,028
records.** N42's "better on every policy head" is true only of the training set;
N45(d)'s "what binds now is parameters" was read off that; N49's +3.0 in the
arena was six blocks and went to +1.06 at twelve (N50b). Three independent
symptoms, one cause.

**What SMALL is worth is unchanged, and it is the thing that plays.** Against
`Priors::OnePly` on the honest holdout: CE 0.8802 against 0.9884, top-1 0.652
against 0.575. That is the +9.24 of N38a and it was never in question.

**And this makes N51 a much better experiment than it looked.** The question is
no longer "would more data help a net that is already fine?" but "does 2.2x the
records stop a 4.3M net overfitting 688k of them?" — which has a real chance of
being yes, and is measured by a `HELD` curve that N52 exists to produce.

## N54. `@0.5` is the right blend, checked on data the net has never seen

The blend is a free field in the agent spec and costs nothing to move, so it is
worth knowing whether the value it has been carried at since N36 is the right
one. `valprobe` on `gen0020-007`, 15,419 positions, `p2000` throughout:

| blend (weight on `eval::heuristic`) | rmse | r | top1 | slope |
| --- | --- | --- | --- | --- |
| `@0.2` | 0.2687 | +0.6732 | 0.583 | 1.17 |
| `@0.3` | 0.2649 | +0.6831 | 0.588 | 1.14 |
| `@0.4` | 0.2622 | +0.6890 | 0.590 | 1.10 |
| **`@0.5`** | **0.2606** | +0.6916 | 0.593 | 1.06 |
| **`@0.6`** | **0.2602** | +0.6918 | **0.594** | **1.01** |
| `@0.7` | 0.2610 | +0.6899 | 0.593 | 0.96 |
| `@1.0` | 0.2702 | +0.6774 | 0.585 | 0.82 |

The optimum is flat across `@0.5`-`@0.6` and `@0.5` is **0.0004 rmse** off it —
an order of magnitude below anything the arena can resolve, so there is no race
to run here and the champion spec does not change.

`@1.0` reproduces the `eval::heuristic` row to every digit (0.2702, +0.6774,
0.585, sd 0.296, slope 0.82), which is the control on the blend arithmetic that
`valprobe`'s own comment asks for. The blend weight is the weight on the
*heuristic*: `@0.0` is the bare net.

## N55. **The champion is carrying an under-trained checkpoint.** `p4500` has the better policy head.

With an honest probe set available for the first time (N53), the obvious thing
to ask is whether the checkpoint the whole ladder is quoted on is the best one
this session produced. `p2000` was picked because it was the first export that
existed, not because anything compared it with `p4500`.

`valprobe`, `gen0020-007`, 15,306 policy nodes, four checkpoints, two
architectures:

| checkpoint | params | steps | **CE ALL** | **T1 ALL** | value rmse `@0.5` |
| --- | --- | --- | --- | --- | --- |
| `p2000` — what the champion runs | 1,913,016 | 2,000 | 0.8727 | 0.662 | **0.2606** |
| **`p4500`** | 1,913,016 | 4,500 | **0.8563** | **0.673** | 0.2620 |
| `m3000` | 4,332,376 | 3,000 | 0.8599 | 0.670 | 0.2607 |
| `m8000` | 4,332,376 | 8,000 | 0.8682 | 0.671 | 0.2600 |
| `Priors::OnePly` | — | — | 0.9855 | 0.580 | — |

Two things.

**(a) `p4500` is the best policy of the four, by 0.0164 nats over `p2000`** and
+0.011 top-1, and it beats *both* MAIN checkpoints. Per phase the gain is where
the nodes are: `PickWorker` 0.8007 -> 0.7765 (T1 0.631 -> 0.658), `Take` 0.7681
-> 0.7307, `Placing` 1.4518 -> 1.4419. The policy head is worth +9.24 on its own
(N38a), so a 0.016-nat improvement in it is the cheapest candidate upgrade
available — the weights already exist and cost nothing to make.

**(b) MAIN's overfit is visible as a curve, not just an endpoint.** 3,000 steps
0.8599, 8,000 steps **0.8682** — the 4.3M net gets *worse* on unseen data
between those two while its training loss falls throughout (`train-main.log`).
`m3000` and `p4500` are within 0.004 of each other, which is N53c again: at
688k records the architectures are interchangeable and the step count is what
matters.

The value head says the opposite and quietly: `p2000` 0.2606, `p4500` 0.2620 —
SMALL's value peaks by 2,000 steps and decays, its policy keeps improving to at
least 4,500. **The two heads want different step counts**, which is the value
squeeze of N35 seen from the other side, and it is a `W_REL`/`W_POLICY`
question rather than a parameter-count one.

Since the policy is worth +9.24 and the whole value blend `@0.5` -> `@1.0` is
worth about +5 (N44a's table: +14.25 with the blend, +9.24 without), the trade
should favour `p4500`. **Racing it head to head, paired, same budget, same
blend, only the step count differing** — `d2-steps`, seed 17000000. `d2-cap`
(MAIN vs SMALL) was stopped at 0 blocks to pay for it: N53c answers that
question offline and more cheaply than 800 games would.

`p4500` and `m3000` copied into `data/net/ckpt/` — the scratchpad is
session-scoped and `p4500` may be the best set of weights this project has.

## N56. The pin is still valid at HEAD — checked, because N23 was not

Every race in this log runs `bin3/arena`, pinned at `f9789b3`. HEAD is
`2105d81`, six commits later, and N23 is the record of a pinned binary being
silently invalidated by a rules commit. So: what actually changed under the
champion spec?

```
git diff --stat f9789b3 HEAD -- src/mcts.rs src/net.rs src/encode.rs \
                                src/rules.rs src/game.rs src/phase.rs src/state.rs
(nothing)
```

**The search, the network, the encoder and every rules file are byte-identical.**
What changed is `eval.rs` (the per-seat research tilt, and `tests/tilt.rs:63`
asserts `heuristic` "returns the identical bits" with it off), `record.rs`
(`parse_mcts_flags` threads an `Option<ResearchTilt>` out; `parse_backend`,
`deeper`, `Priors::Evaluator` and the `@BLEND` split are untouched), the TUI, the
tests, the docs and `train/`.

So the spec below reproduces in a TUI built at HEAD, and the +13.49 is a number
about this tree and not about a stale binary.

**The champion spec, exactly as it goes into the TUI:**

```
mcts:8192:data/net/ckpt/p2000.safetensors@0.5:deeper,pri=eval
```

`deeper` is sugar for `c_puct_init = 0.02, prior_min_edges = 2`
(`record.rs:3199`); under `Priors::Evaluator` the `pmin` half is inert, which is
why `mcts_label` prints only `:cp=0.02`. The path may be relative to `rs/` or
absolute. `@0.5` is the weight on `eval::heuristic` in the leaf value (N54).

## N57. In flight as of 22:10, so the next run does not re-launch them

`<scratch>/logs2/PIDS` is authoritative; kill by PID from that file and never
`pkill -f` a jsonl path (N14/N33a: it matched the poll loops, twice).

| name | pid | what it asks | first reading due |
| --- | --- | --- | --- |
| `m2-main` | 35192 | MAIN at 2,048 vs the champion — the arena side of N53c | has 18 blocks |
| `m8-main` | 44163 | MAIN at 8,192 vs the champion — the rung N49 could not afford | ~2.3 h/burst of 6 |
| `d2-steps` | 46422 | **`p4500@0.5` vs `p2000@0.5`**, paired, 2,048 — the champion upgrade of N55 | ~30 min/burst |
| `train-full` | 45096 | MAIN, 8,000 steps, **1,578,845 records** (7 shards, `gen0020-007` withheld) — N51 | ~50 min total |
| `after-train` | 46855 | waits on 45096, exports `mfull8000`, probes shard 007 against `p2000`/`p4500`/`m8000` | writes `logs2/after-train.out` |

Stopped this session, with reasons: `f8-full` at 40 and `f8-full-b` at 12
(pooled 52 blocks, converged, half-width 1.12 — N50a); `d2-cap` at 0 blocks
(MAIN-vs-SMALL, which N53c answers offline for free).

**The comparison `after-train` makes is clean by construction.** `gen0020-007`
was written at 00:19, after every training in this project finished; the
`train-full` buffer is shards 000-006 only. So all four checkpoints are blind to
the probe set and the only thing that differs between `m8000` and `mfull8000` is
**724,241 records against 1,578,845** — same architecture, same recipe, same
step count, same seeds.

## N58. **`root_value` is in every record and nothing has ever read it**

The policy head learns from a *search* target — the visit distribution — and is
the strongest thing in this net (+9.24 on its own, N38a). The value head learns
from the *game outcome* `z_rel` and is the weakest: on unseen records the bare
net is **0.2796** where `eval::heuristic` is 0.2702, which is the entire reason
the agent runs `@0.5` and is still leaning on a hand-tuned evaluator (N54).

That asymmetry is not a law. `data/replay-champ/schema.json` lists

```
{'name': 'root_value', 'offset': 492, 'dtype': 'f4', 'count': 4}
```

— the 8,192-simulation search's own backed-up value at that node, per seat, 16
bytes of every 512-byte record. `train/features.py` reads `z_rel`,
`final_scores` and `win_share`, and has never touched it.

Measured over 200,000 champion records:

```
finite                     : True
all-zero rows              : 0.00%      (present on every record)
rowsum |mean|              : 0.00000    (already centred, same convention as z_rel)
sd                         : 0.2939     (z_rel 0.3605)
corr with z_rel            : 0.7297
rmse against z_rel         : 0.2485
```

**0.2485 is better than anything this project has ever evaluated** —
`eval::heuristic` 0.2702, the best net 0.2600, and it is the `root_value` row
that has been sitting at the bottom of every `valprobe` table all session as an
unreachable ceiling. It is not unreachable; it is a column in the training data.

### N58a. Wired as a knob, with the default reproducing every existing checkpoint

`batch_arrays(..., value_mix=λ)` sets the `rel` target to
`(1-λ)·z_rel + λ·root_value`; `train.py --value-mix λ` passes it through, and it
goes into the checkpoint's `.json`. **Nothing about the architecture changes** —
same tensors, same manifest, same `src/net.rs`, so a checkpoint trained at any
mix loads unmodified and races with no code change anywhere.

Checked on 512 real records:

```
mix=0.0 == z_rel exactly       : True     <- every checkpoint before this
mix=1.0 == root_value exactly  : True
mix=0.5 == the midpoint        : True
inputs unchanged by the mix    : True
out-of-range refused           : yes
target sd: z_rel 0.3582   root_value 0.2962   half 0.3053
```

The lower target variance is the point. The value head's output `sd` is 0.235
against a target `sd` of 0.361 (N42, N53) — it is shrinking hard toward the mean
because the outcome is noisy at a node 20 days from the end. A target with 82%
of the variance and a *higher* correlation to the outcome is the textbook fix,
and it costs one flag.

**This is the measurement that would most change the picture**, and the reason
is specific: if the value head trained at `--value-mix 1.0` beats
`eval::heuristic` *alone*, the blend can go to `@0.0` and the agent stops
depending on the hand-tuned evaluator at all — which is what the network
workstream is for. Today `@0.0` is 0.2796 against the heuristic's 0.2702 and the
whole +13.49 rests on a 50/50 crutch.

### N50c. `m2-main` at 23 blocks — **+15.36 [+13.47, +17.25]**, and the MAIN premium keeps shrinking

```
mcts:2048:m8000@0.5:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06
   6 blk  +17.229  (+15.155, +19.303)   <- N49, the claim
  12 blk  +15.307  (+12.889, +17.726)
  18 blk  +15.625  (+13.357, +17.893)
  23 blk  +15.359  (+13.470, +17.248)   win 0.821  cand 57.88  base 37.40
```

against SMALL `p2000` on the identical race, `f2-full`, 18 blocks:
**+14.250 [+12.352, +16.148]**.

The gap is **+1.11** with intervals overlapping over four fifths of their
length, where N49 read +3.0 off six blocks. That is exactly the size N53c
predicts from the offline side (CE 0.8792 against 0.8802 on honest held-out
records) and it is not the size N42 predicted from the training set.

## N59. The held-out curve N52 added catches MAIN turning over, live, at ~4,500 steps

`train-full` — `Arch::MAIN`, 8,000 steps, **1,578,845 records**, the same recipe
that produced `m8000` on 724,241:

```
HELD   500  5.0954
HELD  1000  4.9871
HELD  1500  4.9179
HELD  2000  4.9036
HELD  2500  4.8746
HELD  3000  4.8561
HELD  3500  4.8354
HELD  4000  4.8257
HELD  4500  4.7852      <- minimum
HELD  5000  4.7941
HELD  5500  4.7976
```

The training loss falls throughout (`train-full.log`). **The held-out loss turns
at ~4,500 steps and rises**, on a buffer 2.2x the one `m8000` was trained on.
This is the first time in this project that overfitting has been visible while
it happened rather than inferred afterwards from an arena result; before N52
there was no held-out number in the loop at all.

More data moved the turnover later — N55 has MAIN degrading on unseen records
somewhere between 3,000 and 8,000 steps at 724k — but it did not remove it.
**8,000 steps was the wrong step count for `m8000` and it is still the wrong
step count at 1.58M records.**

`ckpt5/gen0001.pt` is overwritten every 250 steps, so the step-4,250 checkpoint
was copied aside as `mfullMID` while the run was still going and exported to
`data/net/ckpt/mfull4250.safetensors`. It sits within 250 steps of the minimum
and is the checkpoint this run should be judged on; `mfull8000` is exported too,
as the like-for-like control against `m8000`'s step count.

## N60. **N51 answered: more data helps the policy head and does nothing at all for the value head**

`mfull4250` — `Arch::MAIN`, **1,578,845 records**, 4,250 steps (within 250 of the
held-out minimum, N59) — against the three checkpoints this session was built
on. `valprobe`, `gen0020-007`, unseen by every one of them, 11,981 policy nodes:

| checkpoint | arch | records | steps | **policy CE** | **policy T1** | value `@0.5` rmse |
| --- | --- | --- | --- | --- | --- | --- |
| `p2000` — the champion | SMALL | 724,241 | 2,000 | 0.8727* | 0.662* | **0.2613** |
| `p4500` | SMALL | 724,241 | 4,500 | 0.8575 | 0.667 | 0.2628 |
| `m8000` | MAIN | 724,241 | 8,000 | 0.8750 | 0.658 | 0.2611 |
| **`mfull4250`** | **MAIN** | **1,578,845** | **4,250** | **0.8491** | **0.672** | 0.2618 |
| `Priors::OnePly` | — | — | — | 0.9830 | 0.577 | — |
| `eval::heuristic` | — | — | — | — | — | 0.2710 |
| *`root_value`* | — | — | — | — | — | *0.2516* |

\* from N55's larger 15,306-node run; the orderings agree across both.

**(a) The policy head answers to data.** `mfull4250` is the best prior this
project has produced: 0.8491 against the champion's 0.8727, and it beats
`p4500` — the best SMALL — by a further 0.0084. 2.2x the records at half the
step count turns `m8000` from the *worst* of the four into the best. That is
N53's overfit removed by feeding it, exactly as N53c predicted it might be.

**(b) The value head answers to nothing.** Blended `@0.5`, all four checkpoints
lie between **0.2611 and 0.2628** — a spread of 0.0017 across 2.3x the
parameters, 2.2x the data and a 4x range of step counts. Sweeping `mfull4250`'s
own blend to `@0.6` gets 0.2613. The value head is **saturated**, and it is
saturated a long way short of the ceiling: `root_value`, sitting in the same
records, is 0.2516.

So the answer to N45(d) is neither of the two things that were argued for it:

* **data binds the policy head** — and the data was already on disk (N51);
* **neither data nor parameters binds the value head.** N35's "value-head
  squeeze", N42's "capacity is still the binding constraint on the value head"
  and N45(d)'s "what binds now is parameters" are all wrong, and N53 explains
  why they looked right.

**The value head's problem is its target, and that is now a flag** (N58): it is
trained on `z_rel`, sd 0.361, a game outcome up to 26 days away, while the
search's own backed-up value for the same node — `root_value`, sd 0.294,
rmse 0.2516 against that outcome — is 16 unread bytes of every record.
`--value-mix 1.0` is queued as `srv`.

`mfull4250` and `mfull8000` are in `data/net/ckpt/`.

## N61. **`m8-main` — the number, since it will not be a result: ~4.7 CPU-hours for one reading**

The brief asked for the `Arch::MAIN`-at-8,192 rung or the cost, and the cost is
what this is. Measured, not estimated:

| race | budget | net seats | blocks | CPU | CPU/block |
| --- | --- | --- | --- | --- | --- |
| `f8-full` | 8,192 | SMALL x1 | 40 | 7.5 h | **11.3 min** |
| `m2-main` | 2,048 | MAIN x1 | 23 | 1.95 h | **5.1 min** |
| `m8-main` | 8,192 | MAIN x1 | **0** | 0.88 h | — |

`m8-main` burned **52:55 of CPU in 41:32 of wall clock and produced zero
blocks**, arena writing in bursts of `--concurrency 6`. Scaling `m2-main` by the
4x simulations gives ~47 CPU-min/block, so a first burst of six is **~4.7
CPU-hours, ~4.3 hours of wall clock on this shared machine** — for one reading
with a half-width around ±2.5 points. `MAIN` at 8,192 is **4.2x** the cost per
block of `SMALL` at 8,192.

**Stopped, and not because of the cost.** The question it was launched to answer
was "does MAIN's +3.0 at 2,048 survive at the champion's budget". There is no
+3.0: it is **+1.11 with intervals overlapping over four fifths of their
length** (N50c), and offline `m8000` ties `p2000` on honest held-out records and
is the **worst of the four checkpoints** now available (N53c, N60). Spending 4.3
hours of a shared machine to place a dominated checkpoint on the ladder is not a
good trade, and the honest version of "MAIN at the champion's budget" is now a
race of `mfull4250`, which is a different net.

The core went to `d2-full`: **`mfull4250@0.5` against the champion's
`p2000@0.5`**, paired, 2,048, seed 18000000 — the cheap version of the same
question, on the checkpoint that deserves it.

### N59a. Correction: that was not a turnover, it was noise, and I called it too early

Two consecutive upticks after step 4,500 are not a turnover. The next reading
went *below* the old minimum:

```
HELD  4000  4.8257
HELD  4500  4.7852     <- what N59 called "the minimum"
HELD  5000  4.7941
HELD  5500  4.7976
HELD  6000  4.7647     <- lower than any of them
```

The evaluation sample is **fixed** (`evaluate` re-seeds identically), so the
wobble is not resampling noise — it is the net itself moving under SGD, about
±0.02 step to step against a total descent of 0.33. **N59's headline is wrong:
`MAIN` on 1.58M records had not turned over at 4,500 and was still improving.**

Two things survive it, and one is worth more than the claim that failed.

* **The instrument is right even though I misread it.** Before N52 there was no
  held-out number in the training loop at all and this could not have been
  discussed either way. What it needs is a smoothing window or more
  `--eval-batches`, not removal.
* **`mfull4250`'s probe result is unaffected.** N60 measured that checkpoint
  against three others on `gen0020-007` directly and it won; that comparison
  never depended on where the minimum was. What changes is the *claim* that
  4,250 was near-optimal — `mfull8000` may well be better, and `after-train`
  probes it on the same shard.

Logged rather than edited, because a log that quietly fixes its own wrong calls
is worth less than one that shows them.

## N62. **`mfull8000` — the best prior this project has produced, and the value head went backwards making it**

`train-full` finished. `MAIN`, 8,000 steps, 1,578,845 records. The held-out
curve descends to the last reading — **no turnover at all**, which settles
N59a: on 724k records MAIN degraded on unseen data past ~3,000 steps (N55); on
2.2x the records it is still improving at 8,000.

```
HELD   500 5.0954   3000 4.8561   5500 4.7976   8000 4.7453   <- the minimum is the last point
      1000 4.9871   3500 4.8354   6000 4.7647
      1500 4.9179   4000 4.8257   6500 4.7704
      2000 4.9036   4500 4.7852   7000 4.7526
      2500 4.8746   5000 4.7941   7500 4.7534
```

`valprobe`, `gen0020-007`, **15,306 policy nodes**, unseen by all four:

| checkpoint | records | steps | **policy CE** | **T1** | value `@0.5` | value bare |
| --- | --- | --- | --- | --- | --- | --- |
| `p2000` — the champion | 724,241 | 2,000 | 0.8727 | 0.662 | 0.2606 | 0.2795 |
| `p4500` | 724,241 | 4,500 | 0.8563 | 0.673 | 0.2620 | 0.2847 |
| `m8000` | 724,241 | 8,000 | 0.8682 | 0.671 | **0.2600** | 0.2777 |
| **`mfull8000`** | **1,578,845** | 8,000 | **0.8427** | **0.681** | 0.2614 | 0.2851 |
| `Priors::OnePly` | — | — | 0.9855 | 0.580 | — | — |
| `eval::heuristic` | — | — | — | — | 0.2702 | 0.2702 |
| *`root_value`* | — | — | — | — | — | *0.2509* |

**(a) The prior.** `mfull8000` is **0.0300 nats** and **+0.019 top-1** better
than the checkpoint the champion runs. For scale, the champion's whole advantage
over `Priors::OnePly` — the thing worth +9.24 in the arena (N38a) — is 0.1128
nats. `mfull8000` adds **27% again** of that margin. Racing it: `d2-full`,
`mfull8000@0.5` against `p2000@0.5`, paired, 2,048, seed 18000000.

**(b) The value head moved the other way, and this is the finding.** Read the
bare-value column down: **0.2795, 0.2847, 0.2777, 0.2851**. More steps make it
worse (`p2000` -> `p4500`). More data makes it worse (`m8000` -> `mfull8000`).
The blend hides it — `@0.5` stays in a 0.0028 band across all four — because
`eval::heuristic` is carrying it.

**Training harder improves the policy head and degrades the value head, in the
same net, in the same run.** That is not a capacity story and it is not a data
story. It is the target: the policy learns from the search's visit
distribution, which is low-noise, so fitting it harder helps; the value learns
from `z_rel`, a game outcome up to 26 days away with sd 0.361, so fitting it
harder is fitting noise. Every checkpoint's value `sd` sits at 0.19-0.26
against that 0.361 — all of them shrinking hard toward the mean, which is what
a least-squares fit to a noisy target does.

**N58 is now not a speculation but the diagnosis.** `root_value` — the same
node's backed-up value from the same search, sd 0.294, rmse 0.2509 against the
outcome, better than any evaluator here — is 16 bytes of every record and has
never been a target. `--value-mix 1.0` is queued as `srv`.

### N62a. `mfull8000`'s blend optimum, and why the race still runs `@0.5`

```
@0.40 0.2646   @0.55 0.2617   @0.65 0.2614
@0.50 0.2623   @0.60 0.2614   @0.70 0.2618        p2000@0.5 0.2613
```

The optimum moved up — `@0.60`-`@0.65` against `p2000`'s flat `@0.50`-`@0.60`
(N54) — which is the same fact as N62(b) seen through the blend: a weaker bare
value head wants more `eval::heuristic` mixed in. It is worth **0.0009 rmse**,
an order of magnitude under the arena's noise floor, so `d2-full` keeps `@0.5`
on both arms and the only thing that differs between candidate and baseline
stays the checkpoint.

## N63. **`p4500` is not a champion upgrade. 0.016 nats of policy buys nothing measurable.**

```
mcts:2048:p4500@0.5:cp=0.06,pri=eval   vs   mcts:2048:p2000@0.5:cp=0.06,pri=eval
6 blocks / 24 games, paired (1 candidate seat, 3 baseline seats, all four rotated)

  centred   -0.031   (-2.069, +2.007)      null 0.00
  win        0.312                          null 0.25
  cand 56.21   base 56.25    days 27.00, 0 aborted
```

N55 read the 0.0164-nat policy improvement and called `p4500` "the cheapest
candidate upgrade available". It is not an upgrade at all: **−0.03 with a
half-width of 2.07.** Six blocks is not much, but the point estimate is on top
of zero, not at the edge of an interval that happens to include it.

**This is the calibration the whole offline programme was missing**, and it is
worth more than the result. The scale: `Priors::OnePly` -> `p2000` is **0.1128
nats** of policy CE and is worth **+9.24** in the arena (N38a). Linearly that
makes a nat worth ~82 points and 0.0164 nats worth **+1.3** — inside this
race's own noise. Every offline CE comparison in this log should be read
against that constant from now on:

| CE improvement | linear prediction | resolvable at 24 games? |
| --- | --- | --- |
| `p2000` -> `p4500`, 0.0164 | +1.3 | no (half-width 2.07) |
| `p2000` -> `mfull8000`, **0.0300** | **+2.5** | marginally |
| `OnePly` -> `p2000`, 0.1128 | +9.24 (measured) | yes |

So `d2-full` — `mfull8000@0.5` against `p2000@0.5`, the same paired design —
is expected to land around **+2.5 and needs roughly 24 blocks** to separate from
zero, not six. It is running; `d2-steps` keeps running beside it because the
same blocks also sharpen this null.

**And the champion does not move today.** `p2000` stays.

## N64. **Data binds `MAIN`. Capacity binds `SMALL`. Both were true and each was measured alone.**

`sfull` — `Arch::SMALL`, the **full 1,578,845-record buffer**, 4,500 steps, the
same recipe that produced `p4500` on 724,241. Probed with the others on
`gen0020-007`, 11,981 policy nodes, one run so the numbers are comparable:

| checkpoint | arch | records | steps | **policy CE** | **T1** | bare value rmse |
| --- | --- | --- | --- | --- | --- | --- |
| `p2000` — the champion | SMALL | 724,241 | 2,000 | 0.8759 | 0.654 | 0.2798 |
| `p4500` | SMALL | 724,241 | 4,500 | 0.8575 | 0.667 | 0.2854 |
| **`sfull`** | **SMALL** | **1,578,845** | 4,500 | **0.8562** | 0.663 | 0.2921 |
| `mfull8000` | MAIN | 1,578,845 | 8,000 | **0.8454** | **0.673** | 0.2862 |
| `m8000` | MAIN | 724,241 | 8,000 | 0.8750* | 0.658* | 0.2777 |

\* from the 15,306-node run; the orderings agree.

**Hold the architecture and the step count fixed and vary only the data:**

* `SMALL`, 4,500 steps: 724k -> 1.58M is **0.8575 -> 0.8562**. Thirteen
  ten-thousandths of a nat for 2.2x the games. **Nothing.**
* `MAIN`, 8,000 steps: 724k -> 1.58M is **0.8750 -> 0.8454**. **0.0296**, and it
  turns MAIN from the worst checkpoint into the best.

**That is the whole data-versus-parameters argument, settled, and the answer is
that both sides were right about a different net.** `SMALL` is capacity-bound:
it has already extracted what its 1.9M parameters can hold from 724k records and
more games do not reach it. `MAIN` is data-bound: at 724k it memorised (N53c)
and at 1.58M it does not, and the 2.3x parameters finally pay. N45(d) said
parameters; N51 said data; **neither buys anything without the other**, which is
why every experiment that moved one at a time read as a null or as noise.

The held-out curves say the same thing in the training loop, on the identical
holdout:

```
                        step 3500   step 4500   step 8000
MAIN,  1.58M records      4.8354      4.7852      4.7453
SMALL, 1.58M records      4.8082      4.7669        --      (run stopped at 4,500)
```

`SMALL` is **more step-efficient** — ahead of MAIN at every step it was run for
— and MAIN passes it only by running twice as long. `SMALL` on the full buffer
for 8,000+ steps is the obvious cheap follow-up and has not been run.

**The value head, once more, for the record.** Bare rmse across the five:
0.2798, 0.2854, **0.2921**, 0.2862, 0.2777. `sfull` is the worst value head in
the table and it was trained on the most data. Every intervention that improves
the prior degrades the value. N58/N62(b).

## N65. **N58 was wrong. Training the value head on `root_value` makes it worse.**

`srv` and `sfull` are the same net, the same 1,578,845 records, the same 4,500
steps, the same seeds. The only difference is `--value-mix`: `sfull` trains the
`rel` head on `z_rel`, `srv` trains it on `root_value`. `gen0020-007`:

| | bare value rmse | r | top1 | sd | slope | `@0.5` rmse |
| --- | --- | --- | --- | --- | --- | --- |
| `sfull` — `--value-mix 0.0` | **0.2915** | +0.5969 | 0.526 | 0.252 | 0.85 | **0.2649** |
| `srv` — `--value-mix 1.0` | 0.3006 | +0.5654 | 0.515 | 0.248 | 0.82 | 0.2697 |
| `eval::heuristic` | 0.2702 | +0.6774 | 0.585 | 0.296 | 0.82 | — |
| *`root_value` itself* | *0.2509* | *+0.7229* | *0.624* | *0.292* | *0.89* | — |

**Worse by 0.0091 bare and 0.0048 blended**, and worse on r, on top-1 and on
slope. The prediction in N58 — lower target variance, higher correlation with
the outcome, therefore a better value head — is refuted.

**Why, and it is the interesting part.** `root_value` is a better *predictor* of
`z_rel` than any evaluator (0.2509 against `eval::heuristic`'s 0.2702) because
it is the backed-up result of **8,192 simulations of lookahead from that exact
node**. That advantage is not a function of the position; it is search output.
A net given only the position can learn the component of `root_value` that the
position determines, and that component turns out to be *less* informative about
the eventual outcome than `z_rel` is — the net inherits the search's shrinkage
(`root_value` sd 0.294 against the outcome's 0.361) and its systematic errors,
and gets none of the lookahead that made it good.

So the `root_value` row at the bottom of every `valprobe` table is **not a
ceiling the value head can be trained toward.** It is a measure of what search
buys, and buying it requires searching.

**A second thing falls out, and it contradicts N35.** The policy heads barely
moved: `sfull` CE 0.8507, `srv` 0.8535, top-1 0.675 against 0.672 — 0.003 nats
for completely replacing the value target. If six policy heads and the value
head were really competing for one trunk, changing what the value head is
trained on by that much would have shown up in the prior. It does not. **The two
heads are far more decoupled than the "value-head squeeze" of N35 assumed** —
which is one more thing that story got from a probe set the net had memorised.

`--value-mix` stays in `train.py`: the knob is right even though the hypothesis
was wrong, `0.0` is the default and reproduces everything, and a partial mix is
now a measured-not-guessed dead end rather than an untried idea.

### N50d. `m2-main` final — **38 blocks, +14.87 [+13.55, +16.19]**. The MAIN premium is gone.

```
mcts:2048:m8000@0.5:cp=0.06,pri=eval   vs   mcts:2048:heuristic:cp=0.06
   6 blk  +17.229  (+15.155, +19.303)    <- N49's claim, and the whole "parameters bind" story
  12 blk  +15.307  (+12.889, +17.726)
  23 blk  +15.359  (+13.470, +17.248)
  30 blk  +15.221  (+13.717, +16.725)
  38 blk  +14.870  (+13.547, +16.193)   win 0.839  cand 57.86  base 38.03
```

152 games, days 27.00, 0 aborted, half-width **1.32**. Against SMALL `p2000` on
the identical race (`f2-full`, 18 blocks): **+14.250 [+12.352, +16.148]**.

**The gap is +0.62 and the intervals overlap over essentially their whole
length.** N49 read +3.0 off six blocks and N45(d) built "what binds now is
parameters" on top of it. At 38 blocks there is no premium for 2.3x the
parameters at 724k records, which is precisely what N53c said from the offline
side once the probe was pointed at data the net had not trained on, and what
N64 explains: at that buffer size `MAIN` had nothing to spend the parameters on.

Stopped; the core went to `d2-full`.
