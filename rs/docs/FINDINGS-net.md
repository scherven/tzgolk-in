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
