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
