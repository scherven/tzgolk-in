# Training

How to run a generation, what the numbers mean, and how to tell "it is working"
from "it has stalled".

`LEARNING.md` is the design of the network and the loop; `SEARCH.md` is the
design of the tree. This document is the operating manual for the three things
that turn those designs into a run: `selfplay`, `arena`, and `train/`.

The one-line summary: **`arena` is the instrument. Run it every few hours. If
its centred-score interval is not moving up, nothing else you do matters.**

---

## 0. The three commands

```bash
# 1. generate games                         (hours; interruptible)
cargo run --release --bin selfplay -- --agent mcts:3200 --games 20000 \
    --out replay --gen 0

# 2. train on them                          (minutes; interruptible, resumable)
python train/train.py --replay replay --out ckpt --gen 0 --steps 3000 \
    --window 8000000

# 3. ask whether it got better              (minutes; interruptible)
cargo run --release --bin arena -- \
    --candidate ckpt/gen0000.safetensors --baseline heuristic:32 --games 600
```

All three run today. Step 1 plays real tree search — `src/mcts.rs` landed —
against whatever evaluator you name: `heuristic` until a network is trained, a
`.safetensors` checkpoint after. Step 2 runs on the value heads and on the
`policy_kind = TREE_EDGE` targets a searching agent writes. The one seam still
open is the encoder, §7.1.

Two defaults there are deliberately **not** the ones `LEARNING.md` §6.6
specifies, and §9 says why: **3,200 simulations instead of 800, and an 8 M-record
window instead of 1.5 M.** Both come from `COMPUTE.md` §7. At the throughput the
batched driver reaches, the loop as originally specified would overflow its own
replay buffer several times per generation and throw the excess away; the right
place to spend the surplus is search depth, not game count.

---

## 1. The benchmark: `arena`

### 1.1 What it does

Plays a candidate against a baseline over N four-player games and reports four
numbers with confidence intervals:

| metric | null | what it is |
|---|---|---|
| **mean centred score** | 0 | `score[candidate] - mean(all four scores)`. **The headline.** |
| points vs baseline | 0 | `score[candidate] - mean(score[baseline seats])`. The same signal in points. |
| margin vs best rival | *none* | `score[candidate] - max(score[rivals])`. Descriptive only — see §1.4. |
| win rate | 0.25 or 0.50 | share of games won, ties split. |

```
  metric                  value        95% CI              null
  ----------------------------------------------------------------
  mean centred score      17.432   (+16.758, +18.106)  +    0.00
  points vs baseline      23.243   (+22.344, +24.141)  +    0.00
  margin vs best rival     4.115   ( +3.402,  +4.828)        n/a
  win rate                 0.744   ( +0.717,  +0.770)  +    0.25
```

`+` means the interval is entirely above the null, `-` entirely below, `.`
straddling it. The report then says outright which of the three it is, and when
it cannot tell, how many more games would settle it.

### 1.2 Seating, and why it is a rotation block

Turn order in Tzolk'in is worth real points: the first-player marker moves, and
the corn surcharge to place on a higher space means seat 0 and seat 3 do not
face the same board. Measuring an agent in one seat measures the seat.

So the unit of work is a **rotation block**: the same game seed played once with
the candidate in each of the four seats.

* Seat advantage cancels *inside* the block, exactly, by construction.
* The four games share a seed, so they share the shuffled decks, the monument
  row and the dealt starting tiles. This is a matched design and it removes a
  large slice of variance that is nothing to do with agent strength.

The second point has a consequence that the statistics must respect: **games
within a block are correlated**, so the block, not the game, is the independent
unit. Every `n` in every interval above is a count of blocks. Treating 1,200
games as 1,200 independent samples would shrink the interval by up to a factor
of two and produce confident nonsense.

### 1.3 What the win rate means

`--mode solo` (default) is one candidate against three baselines. **An agent
exactly as strong as its baseline wins 25% of games, not 50%.** The report says
so every time, because it is the single easiest number in this project to
misread.

`--mode pairs` is two candidates against two baselines over all six distinct
seatings of `{C,C,B,B}`; its win rate is the share of games won by *either*
candidate seat and its null is 50%. `LEARNING.md` §6.8 asks for both, and for
good reason: `solo` is the configuration that exposes an agent which has only
learned to play against copies of itself, because three of the four players are
something else.

### 1.4 Why `margin vs best rival` has no null

It is one score against the maximum of three draws, so it is negative even
between identical agents — about **−11 points** in this game. There is no
analytic null to compare it to, so the report prints no marker and no null
column for it. Read it *across* runs: if it goes from −11 to −4 over ten
generations, the agent is closing on the field. Comparing it to zero is
meaningless.

Verify this for yourself in thirty seconds — it is the best single check that
the harness is honest:

```bash
cargo run --release --bin arena -- --candidate heuristic:24 --baseline heuristic:24 --games 1200
#   mean centred score      -0.029   ( -0.553,  +0.495)  .    0.00
#   win rate                 0.245   ( +0.222,  +0.268)  .    0.25
```

Identical agents give a centred score indistinguishable from zero and a win rate
indistinguishable from 25%. If that ever stops being true, the seating rotation
has broken and every other number is suspect.

### 1.5 Sample size

Centred score has a per-game standard deviation of roughly 12–25 depending on
how good the players are. Per **block** it is about half that, because the seat
rotation averages four correlated games. In practice:

| you want to detect | blocks | games |
|---|---|---|
| 10 points (a new agent vs random) | ~10 | 40 |
| 3 points (one generation vs the last, early) | ~150 | 600 |
| 1 point (one generation vs the last, late) | ~1,300 | 5,200 |

`--games 600` is the standard panel run. When the answer comes back "not
distinguishable", the report prints the block count that *would* resolve the
effect it measured; if that number is enormous, the honest conclusion is that
the two agents are the same and no amount of games will say otherwise.

### 1.6 The anchor panel

Do not only compare to the previous generation. Four-player games are markedly
less transitive than two-player ones — A beats B beats C beats A is common — and
a loop that only ever compares to the previous champion can walk in a circle
while every comparison says "improved". Run a fixed panel every ten generations
(`LEARNING.md` §6.8):

```bash
G=ckpt/gen0040.safetensors
for base in random heuristic:32 ckpt/gen0000.safetensors ckpt/best.safetensors; do
  cargo run --release --bin arena -- --candidate $G --baseline $base \
      --games 600 --out eval/g40-vs-$(basename $base).jsonl --resume
done
```

| anchor | what it catches |
|---|---|
| `random` | the absolute floor; **must never be lost to** |
| `heuristic:32` | is the net beating a one-ply heuristic? |
| `mcts:800` | is the net beating raw search with a uniform prior? |
| generation 0 | monotone progress from the warm start |
| best so far | the actual champion comparison |

### 1.7 Interruption and resume

Blocks are appended to `--out` as JSONL and flushed as they complete. Ctrl-C
finishes the blocks in flight, prints the summary over everything on disk, and
exits 130; a second Ctrl-C aborts immediately. `--resume` skips seeds already on
file, so a panel run can be spread over a week of interruptions and still report
one interval. The progress file is the source of truth for both the summary and
the resume, so a run stopped and restarted five times reports the same interval
as one that ran straight through — including the mean game length, which is the
report's rules-bug detector and which is therefore stored per block rather than
recomputed.

The p-value in the footer assumes a single look at the data. A run you restart
five times and read after each is five looks, and the nominal 5% is really
closer to 15%. Use the interval, not the p-value, for anything you act on.

### 1.8 Baselines and cost

`random` and `heuristic:K` are built in and need no files — they are the floor
and the one-ply rung of the anchor panel, and both run at thousands of games per
second, so a 600-game panel against either is seconds. A checkpoint or an
`mcts:` candidate is far more expensive: `mcts:3200` against `heuristic:32` is
about 13 games/s here, so 600 games is roughly a minute, and two searching
agents against each other is about 6 games/s.

The arena takes the same concurrency flags as `selfplay` and for the same
reason (§2.1): `--concurrency` blocks in flight (128 by default when either side
searches), `--batch`, `--batchers`, `--linger-us`, `--no-batch`. Each side gets
its own batcher pool, because the candidate and the baseline are usually
different networks.



## 2. Self-play: `selfplay`

```bash
VECLIB_MAXIMUM_THREADS=1 cargo run --release --bin selfplay -- \
    --agent mcts:3200 --games 20000 --out replay --gen 3 \
    --snapshot-hours 3
```

Output:

```
replay/
  gen0003-000-20260907T031200Z.tzr    sealed shards, chronologically sortable
  gen0003-001-20260907T061200Z.tzr
  gen0003.part                        in flight; sealed on interval and at exit
  latest                              one line: the newest sealed shard
  manifest.json                       generation, rules version, agent, git commit
  schema.json                         the record layout, for train/replay.py
```

### 2.1 Concurrency: games in flight, not threads

**The driver runs one OS thread per concurrent game, defaulting to 512 of them,
and none of them is a rayon worker.** That is the whole of `COMPUTE.md` §2.6 and
it is worth understanding before touching the knobs, because the obvious
setting — "one game per core" — is the one that throws the throughput away.

A tree search spends almost all of its time inside one call: `Evaluator::
evaluate`, one position at a time. Measured here on `Arch::SMALL` with
Accelerate, `VECLIB_MAXIMUM_THREADS=1`, one core:

| batch | evals/s |
|---|---|
| 1 | 18,900 |
| 32 | 67,000 |
| 128 | 102,000 |
| 256 | **115,000** |

**6.1x, on one core, for free.** The batch has to come from somewhere, and it
cannot come from inside one tree: `SEARCH.md` §3.5 wants parallel descents, but
a turn's first nodes have width 1–7 and parallel descents collide there
immediately, and virtual loss costs real search quality. It comes instead from
the number of *games* in flight, which is a free parameter bounded only by
memory — and inter-game batching costs **nothing** in search quality
(`COMPUTE.md` §2.5). With one descent per tree, each tree's simulation completes
before its next begins: no virtual loss, no stale statistics, and a search that
is bit-for-bit what a single-threaded run from the same seed would produce. The
only thing that changes is latency per game, which self-play does not care
about.

So: `--concurrency` game threads, each with its own tree and its own slot in the
batch, all blocking inside `evaluate`; and one to four **batcher threads outside
the pool** draining the queue into `Net::evaluate_batch`. Nothing in `mcts.rs`
changed, nothing became a future, and `Mcts::search` is still a synchronous
`&mut self` call.

| flag | default | what it is |
|---|---|---|
| `--concurrency N` | 512 searching, else one per core | games in flight, one OS thread each |
| `--batch N` | `min(256, concurrency)` | ceiling on one forward pass |
| `--batchers N` | `cores/5`, at most 4 | inference threads, outside the pool |
| `--linger-us N` | 200 | how long a short batch waits for more |
| `--no-batch` | off | one evaluation per forward pass |

**Measured on this machine** — 14 cores, 2,048 games of
`--agent mcts:64:net-random:small`, which is real tree search over an untrained
network, so the mix of phases and node widths is the one a real run sees:

| shape | games/s | evals/s |
|---|---|---|
| one game per core, unbatched (the old `par_iter`) | 12.3 | ~82,000 |
| 512 in flight, 3 batchers | 19.6 | 131,000 |
| 1,024 in flight, 3 batchers | 22.9 | 153,000 |
| 2,048 in flight, 3 batchers | **26.2** | **175,000** |
| 2,048 in flight, 10 batchers | 22.8 | 152,000 |

Read that honestly. **The whole-machine gain is 2.1x, not the 20x `COMPUTE.md`
§0 quotes**, and the difference is not a bug: §2.2's 20.3x is a *per-core*
figure, and the shape it replaces was already running batch-1 inference on all
fourteen cores. Per core the batching is worth the 6.1x in the first table;
across the machine the baseline was already parallel, so what is left is the
2.1x above. Plan capacity from **175,000 evaluations/s on this Mac**, not from
`COMPUTE.md` §5's 350,698.

Batcher utilisation sits at 82% in every batched row, so the machine — not the
queue — is the limit. The per-evaluation cost inside the network at these batch
sizes is ~19 µs against the 9.8 µs the isolated benchmark shows, which is what
concurrent GEMMs on shared memory bandwidth cost, plus the `Take` nodes: see the
report note on `Query::candidates` in §7.4.

Three operational notes:

* **Set `VECLIB_MAXIMUM_THREADS=1`.** Otherwise Accelerate spawns its own pool
  and fights the batchers for the same cores. `selfplay` warns if it is unset
  and a network is in play.
* **More batchers is not better.** Three is the measured optimum here; ten is
  13% *slower*, because each batcher takes a slice of the same queue and ten
  slices are each too small to amortise anything. Watch the mean batch size.
* **`--concurrency` is the knob that matters, and it was still paying at
  2,048.** It is also the one that costs memory: ~1 MB of tree per game at 800
  simulations, four times that at the default 3,200. 512 is the default because
  it is where the curve starts to flatten and because 512 x 4 MB is comfortable;
  if the box has RAM to spare, raise it before touching anything else.

### 2.2 What the run prints, and what to watch

```
  compute  : 512 games in flight, batch <= 256 over 3 inference threads, 200 us linger
  ...
  batches 87150 of mean size 78.5 (max 256), mean queue wait 4077 us, 21.1 us per eval inside the net
  114127 evaluations/s over the whole run; batcher utilisation 80%
```

`COMPUTE.md` §2.6: *"if the mean batch size is not close to the configured
maximum, none of §5's numbers are happening, and it is the only symptom you will
get."* So:

* **mean batch size** far below `--batch` → the concurrency is not there. Raise
  `--concurrency`, or lower `--batchers` so fewer of them split the queue. The
  driver prints a warning when it drops under half.
* **batcher utilisation** is the fraction of the run each batcher spent inside
  `evaluate_batch`. Near 100% means the evaluator is the bottleneck and only a
  faster one (bigger batches, a GPU) helps. Well under 50% means the batchers
  are starved and `--concurrency` is the knob.
* **us per eval inside the net** is the honest per-evaluation cost at the batch
  size actually realised. Compare it to the table above.

### 2.3 Durability

**Two levels, deliberately separated.** Every finished game is written and
`flush`ed before its thread moves on, so a Ctrl-C, a panic or a `kill -9` loses
nothing that had ended. `fsync` runs every `--fsync-games` (default 8), which
bounds what a power cut can take; one fsync per game is affordable at MCTS rates
but not at the 1,300 games/s the one-ply agent reaches, hence the knob.

**Interrupt granularity is now the concurrency.** With 512 games in flight, a
Ctrl-C abandons up to 512 partial games rather than one per core, and it takes
up to one game's duration to wind up — the handler asks every worker to stop
*between* games, never mid-game, so nothing half-written reaches the shard. At
`mcts:3200` with 512 games sharing the machine a game is over a minute, so
expect Ctrl-C to take about that long to return; a second Ctrl-C aborts
immediately and loses those partial games instead. Both are cheap: 512 partial
games is well under a minute of aggregate work.

Memory, measured at the defaults: **3.3 GB resident** for 512 games in flight at
3,200 simulations. Scale it linearly in `--concurrency` and in the simulation
count when sizing a machine.

**Snapshots.** Every `--snapshot-hours` (default 3) the part file is sealed into
a timestamped `.tzr`, `latest` is rewritten by temp-then-rename, and
`manifest.json` is refreshed the same way. A run left going overnight leaves a
trail of complete, immediately trainable shards rather than one file that is
only usable when the run ends.

**Verify a shard** after the first generation and after any change to
`src/state.rs`:

```bash
cargo run --release --bin selfplay -- --verify replay/$(cat replay/latest)
```

It re-decodes every record through the same codec that wrote it. If `GameState`
has changed shape underneath the buffer, this is where it shows up, rather than
in a silently mistrained value head.

### 2.4 Agent specs

The same grammar in both binaries; it is `AgentSpec` in `src/record.rs`.

| spec | what it is |
|---|---|
| `random` | the `sample_legal_move` rollout policy — the floor of the anchor panel |
| `heuristic` / `heuristic:K` | one-ply greedy over K sampled candidate turns |
| `greedy:K:EVAL` | one-ply greedy over any evaluator |
| `mcts:SIMS` / `mcts:SIMS:EVAL` | tree search; `EVAL` defaults to `heuristic` |
| `net-random[:small\|main]` | an untrained network — not a player, a stopwatch |
| `PATH.safetensors` | a checkpoint; shorthand for `mcts:3200:PATH` |

`random` and `heuristic:K` write `policy_kind = NONE` records: their candidate
moves come out of an RNG, so their indices are not regenerable at training time
and they teach the **value heads only**. That is not a degraded mode — §6.

A searching agent writes `policy_kind = TREE_EDGE`, and only from the 25% of
turns that get the full playout budget (`LEARNING.md` §6.3's playout-cap
randomisation; the other 75% run at `SIMS/8` and are not written). Forced nodes
— a single legal edge, which `mcts.rs` collapses without an evaluation — are
skipped too, since there is no distribution to teach. Measured: **~85 records
per game**, against ~103 for the one-ply agent, which writes one per turn.


## 3. The record format

Another program has to read this, so here it is exactly. Everything is
**little-endian**. The authority is `src/record.rs`; `schema.json` is generated
from it, and `train/replay.py` builds its numpy dtype from `schema.json` rather
than from a second copy of this table. Regenerate at any time with
`cargo run --release --bin selfplay -- --print-schema`.

### 3.1 Shard header (64 bytes, once per file)

| off | size | field |
|---|---|---|
| 0 | 4 | magic `TZZR` |
| 4 | 2 | format version (1) |
| 6 | 2 | header bytes (64) |
| 8 | 4 | record bytes (512) |
| 12 | 4 | `tzolkin::RULES_VERSION` |
| 16 | 2 | live state bytes (293 today) |
| 18 | 2 | players (4) |
| 20 | 2 | max visit pairs (24) |
| 22 | 2 | flags, reserved |
| 24 | 8 | created, unix seconds |
| 32 | 4 | generation |
| 36 | 28 | producer string, NUL-padded |

### 3.2 Record (512 bytes, fixed)

| off | size | type | field |
|---|---|---|---|
| 0 | 320 | u8 | **state slot** — first 293 live, rest zero (§3.3) |
| 320 | 8 | u64 | game id (the game's seed) |
| 328 | 2 | u16 | node index within the game |
| 330 | 1 | u8 | `turn` — whose turn it is |
| 331 | 1 | u8 | `mover` — `phase.mover(turn)`, whose decision this is |
| 332 | 1 | u8 | `Phase::tag()` |
| 333 | 5 | u8 | phase args, including `PickWorker`'s retrieval counter (§3.4) |
| 338 | 1 | u8 | `policy_kind` — 0 none, 1 tree edge index |
| 339 | 1 | u8 | flags — bit 0 turn root, bit 1 full budget |
| 340 | 2 | u16 | `n_edges` — **all** legal edges at this node, before any cap the tree applied |
| 342 | 2 | u16 | populated visit pairs |
| 344 | 4 | u32 | total visits |
| 348 | 96 | u16×48 | 24 × (edge index, visit count), best first; the index is into `tree::legal_steps` (§3.5) |
| 444 | 4 | f32 | policy weight, `log(1+N)/log(1+800)` |
| 448 | 2 | u16 | `day` (a cheap filter key; also in the state) |
| 450 | 1 | u8 | temperature × 100 |
| 451 | 1 | — | pad |
| 452 | 8 | i16×4 | final scores, **absolute seat order** |
| 460 | 16 | f32×4 | `z_rel = tanh((score - mean)/25)`, seat order |
| 476 | 16 | f32×4 | `win_share`, `1/\|winners\|`, seat order |
| 492 | 16 | f32×4 | `root_value` — the agent's own estimate, seat order |
| 508 | 4 | u32 | `RULES_VERSION` |

Every multi-byte field is naturally aligned, so a numpy structured dtype maps
straight onto it.

Four design notes, each of which another implementer needs:

* **Fixed size.** A shard is `np.memmap`ed with a structured dtype: no parsing,
  no framing, no allocation, and a shard truncated by a kill is valid up to its
  last whole record. Two shards concatenate.
* **The trailer is duplicated per record.** Final scores, `z_rel` and
  `win_share` are the same for every record of a game — 40 bytes of the 512.
  Paying that removes the loader's need to join records back to games, and it is
  why a game is buffered in memory and written in one call, which is in turn why
  an interrupted run loses at most the game in flight.
* **Everything per-player is in absolute seat order.** The rotation into
  perspective order is the encoder's job, and it happens per query — that is
  what lets one record be presented four times with four different queriers
  (`LEARNING.md` §6.4).
* **`RULES_VERSION` is on every record, not just the header.** Shards get
  concatenated and sub-sampled; the stamp has to survive that. It is what turns
  "the whole buffer is suspect" into "the records before generation N are"
  (`LEARNING.md` §8.4).

### 3.3 State codec (293 bytes today, inside a 320-byte slot)

`GameState` does not derive `Serialize` and `src/state.rs` is not this code's to
edit, so the codec is written by hand, field by field, in `encode_state` /
`decode_state`. Explicitly, not as a `transmute`: `GameState` is `repr(Rust)`,
so its padding and field order are the compiler's business, and a replay buffer
that a `rustc` upgrade can silently reinterpret is not a replay buffer. The pair
is pinned by a round-trip test over real mid-game and terminal states.

| off | size | field |
|---|---|---|
| 0 | 76 | 4 × player (19 bytes: colour, corn, res[4], points i16, corn/wood tiles, free workers, discount, may_skip_day, buildings u32, monuments u16) |
| 76 | 48 | 24 × worker (tag: 0 locked / 1 available / 2 on gear / 3 first-player space; arg: `gear*16 + pos`) |
| 124 | 55 | 5 gears × `MAX_GEAR_SPACES` occupancy bytes, `0xFF` empty |
| 179 | 12 | `temples[3][4]` |
| 191 | 16 | `research[4][4]` |
| 207 | 16 | 8 × Palenque `(corn, wood)` |
| 223 | 2 | `chichen_filled` u16 |
| 225 | 15 | age-1 deck: ids[14], cursor |
| 240 | 19 | age-2 deck: ids[18], cursor |
| 259 | 14 | monument deck: ids[13], cursor |
| 273 | 6 | `buildings_up` (0 = empty; ids are 1-based) |
| 279 | 6 | `monuments_up` |
| 285 | 8 | first-player space (`0xFF` none), accumulated corn, skulls, current, first player, age, day, over |

**The state slot is fixed at 320 bytes whatever the live width.** Chichen Itza
went from 10 worker spaces to 11 during the week this was written; with a fixed
slot, that grew `state_bytes` from 288 to 293 and moved *nothing else*. A
compile-time assertion breaks the build if the state ever exceeds 320, which is
the moment to bump the slot and the format version together rather than the
moment to discover silent corruption.

**Note that `Deck::ids` is stored and must not be encoded.** It is the undrawn
pile in shuffled order. It is in the record because the buffer should outlive
encoder changes; a network fed it learns to read the future and will not
transfer to play against humans (`LEARNING.md` §1.6). `train/selftest.py` pins
this: it shuffles the undrawn tail of every deck and asserts the encoding does
not move.

### 3.4 Phase args

Five bytes, keyed by `phase_tag`. `schema.json` carries this table too, under
`"phase_args"`, so the loader need not restate it.

| tag | phase | args |
|---|---|---|
| 0 | `Beg` | — |
| 1 | `Mode` | — |
| 2 | `Placing { n }` | `[n]` — workers already placed this turn |
| 3 | `PickWorker` | `[retrieved_so_far]` — **see below** |
| 4 | `Take { worker }` | `[worker]` |
| 5 | `ExtraDay { claimer }` | `[claimer]` |
| 6 | `PityPlace` | — |
| 7 | `DraftTile { dealt, kept }` | `[d0, d1, d2, d3, kept]` |

**`PickWorker`'s byte 0 is load-bearing and was nearly missed.** `phase.rs` says
`(GameState, Phase)` is a complete search node, and for edge *generation* it
very nearly is — but `tree::legal_steps` takes a fourth argument, the count of
workers already retrieved this turn, and uses it at exactly one node:
`StopRetrieving` is legal only once at least one worker has been resolved. A
loader that regenerated the enumeration without it would get a list one edge
short at every `PickWorker` node with `retrieved_so_far > 0`, and **every stored
index after the missing edge would name the wrong step**. It is stored in the one
phase whose arg bytes are otherwise unused, so it cost no layout change.

To regenerate the enumeration for a record, in Rust:

```rust
let legal = tree::legal_steps(&state, phase, turn, done);   // `done` = that byte
// visits[i].0 indexes `legal`
```

### 3.5 `policy_kind`, and the contract it names

| value | meaning |
|---|---|
| 0 `NONE` | No usable policy target. The record teaches the **value heads only**. |
| 1 `TREE_EDGE` | `visits[i].0` indexes `tree::legal_steps(state, phase, turn, done)`, which the loader regenerates. |

`TREE_EDGE` requires that the index be **regenerable at training time from the
record alone** — from `(state, phase, turn, done)`, with no access to the network
that produced the search. `SEARCH.md` §5b note 3 commits the generator to a
deterministic order: `choices_for_worker` ends in `options::dedup`, which sorts,
and `visit_legal_moves` documents its traversal order.

**That is not the order `mcts.rs` reports its visit counts in, and assuming it
was would have silently corrupted every wide node.** `Mcts` sorts a node's edge
array by prior and truncates it whenever the node is wider than `max_edges`
(32), so for those nodes `SearchResult::visits` is indexed by an order that
depends on the network's priors — which change every generation and which no
loader can reconstruct. The adapter in `src/record.rs` therefore does not store
the tree's index at all: `SearchResult::visits` carries the `Step` itself, so
`search_node` looks each step up in a fresh `tree::legal_steps` call and stores
*that* index. It costs one `legal_steps` call per written node, on the 25% of
turns that are written. `n_edges` is likewise the full legal count, not the
tree's truncated one.

There is a test in `src/record.rs`
(`a_search_agent_plays_legal_moves_and_records_regenerable_targets`) that
regenerates the enumeration for every node a search agent writes and asserts
that `n_edges` matches and every stored index is in range. It is the thing
standing between a reordering in `tree.rs` and a silently mislabelled buffer;
run it after any change there.

Storing indices rather than a dense distribution is what keeps the record fixed
size even for the pointer head, where the "index" is a position in a candidate
list of up to several thousand.


## 4. Training: `train/`

```
train/
  replay.py       shard reader: memmap + structured dtype, built from schema.json
  features.py     THE ENCODER SEAM (§7.1) + batch assembly. numpy only, no torch.
  model.py        trunk and heads, LEARNING.md §4.1 / §2.2 / §3.1
  train.py        the optimiser loop: AdamW, the §6.6 schedule, resumable
  export.py       THE WEIGHT-FORMAT SEAM (§7.2)
  selftest.py     checks that need neither torch nor a trained net
```

Install: `pip install -r train/requirements.txt`. CPU only, and deliberately —
`LEARNING.md` §5.2 documents small nets that train fine on CPU and **silently
fail to learn on MPS with no error**, and a 26 GFLOP optimiser step does not
need a GPU. On the Windows machine `--device cuda` works, but diff its loss
curve against CPU for the first few hundred steps before trusting it.

Smoke-test the data path with no torch installed at all:

```bash
python train/features.py replay     # encode a batch, print its shape and range
python train/selftest.py replay     # the checks in §4.3
```

### 4.1 One training run

```bash
python train/train.py --replay replay --out ckpt --gen 3 --steps 3000 --size small
```

* **Batch 1024**, sampled uniformly from a window of the newest shards up to
  1.5M records (`--window`).
* **4× perspective augmentation** on by default: every position is presented
  once per querying player, each giving a full value target. Legitimate because
  the encoder takes the querying player independently of whose turn it is, and it
  trains exactly the off-turn query a max^n backup makes. This is the only
  augmentation the domain offers — Go gets 8× from the dihedral group, this game
  gets 4× on the value heads and nothing on the policy.
* **AdamW**, LR constant-with-drops per §6.6 (2e-3 to generation 60, 1e-3 to
  150, 3e-4 after), decay 1e-4 excluding LayerNorm and biases, gradient clip at
  global norm 1.0, 200-step linear warmup.
* **Ctrl-C** saves the model, the optimiser moments and the step counter, then
  exits 130. `--resume` continues. The moments are not optional: a resume
  without them silently restarts Adam and loses a generation's progress. A
  periodic checkpoint also lands every `--checkpoint-every` steps.

`--size small` (384 wide, 4 blocks) for generations 1–30 and `--size main`
(512 wide, 6 blocks) after. The swap is a from-scratch retrain on the
accumulated buffer, not net surgery; ~4,000 steps, about six minutes.

### 4.2 The loss

`LEARNING.md` §6.5, implemented in `model.loss_fn`. Only the head matching a
record's phase contributes a policy term; the value terms contribute on every
record.

| term | weight | note |
|---|---|---|
| `Huber(v_rel, z_rel)` | 1.5 | the primary target; what search backs up |
| `Huber(v_score, z_score)` | 0.5 | `(score - 75)/50`, clipped to ±2 |
| `CE(v_rank, one-hot place)` | 0.3 | ties split evenly; gives `P(win)` free |
| masked CE per fixed policy head | 0.5–1.0 | weighted by `log(1+N)/log(1+800)` |

Huber and not MSE throughout: early self-play produces scores in the −70..+40
range and MSE lets those outliers dominate the gradient. The policy mask is
applied **inside** the loss, not only at inference, so no gradient ever reaches
an illegal action.

Not yet implemented, and flagged as such in `model.py`: the score-decomposition
head (§3.2's highest-value-per-byte auxiliary) and the `Take` pointer head. The
first needs the six end-of-game score components in the record — a natural
format-version-2 addition, and the 32 reserved bytes are there for it. The
second needs `encode_choice` from `src/encode.rs`; `TakeHead` is written and
wired, waiting on the features.

### 4.3 `selftest.py`

Four things worth pinning before a long run, none of which need torch:

* **The policy-target path**, which no record on disk exercises yet — every
  record is `policy_kind = NONE` until MCTS lands, so the code that turns visit
  pairs into a normalised masked distribution is fabricated and checked here.
* **Perspective relativity** — slot 0 of every target is the querying player,
  and the encoding actually changes when the querier does.
* **Hidden information** — shuffle the undrawn tail of every deck and assert the
  encoding does not move.
* **The two phase tables** in `features.py` and `model.py` agreeing.

---

## 5. Is it working?

### 5.1 The gate before anything else

Before a single generation, confirm the harness measures what you think.

**Does the one-ply ladder separate?**

```bash
cargo run --release --bin arena -- --candidate heuristic:1  --baseline random --games 800
cargo run --release --bin arena -- --candidate heuristic:16 --baseline random --games 800
```

Observed on this machine:

| candidate | mean centred score | win rate |
|---|---|---|
| `heuristic:1` | −0.09 (−0.80, +0.62) | 0.249 |
| `heuristic:4` | +14.22 (+13.56, +14.88) | 0.652 |
| `heuristic:16` | +17.43 (+16.76, +18.11) | 0.744 |
| `heuristic:32` | +16.84 (+15.58, +18.10) | 0.752 |
| `heuristic:64` | +17.62 (+16.91, +18.33) | 0.765 |

`heuristic:1` samples one candidate move and therefore has no choice to make: it
*is* the rollout policy, and the arena correctly calls it indistinguishable from
`random`. That row is the harness checking itself.

**Does more search buy strength?** This is the stronger gate, because it
exercises `mcts.rs`, the batching, the step→`Move` reconstruction and the arena
at once. Every one of these uses the *same* uniform-prior heuristic evaluator, so
the only thing changing is depth.

```bash
for s in 200 800 3200; do
  cargo run --release --bin arena -- --candidate mcts:$s --baseline heuristic:32 --games 120
done
```

| candidate | vs `heuristic:32`, mean centred score | win rate |
|---|---|---|
| `mcts:48` | −7.15 (−12.56, −1.73) | 0.083 |
| `mcts:200` | +0.81 (−0.79, +2.40) | 0.296 |
| `mcts:800` | +7.36 (+5.42, +9.30) | 0.542 |
| `mcts:3200` | +9.92 (+7.73, +12.11) | 0.625 |

and directly, `mcts:3200` vs `mcts:800`: **+1.99 (+0.03, +3.95)**.

Two things to take from that. First, it is monotone and its intervals separate,
which is what a working instrument looks like end to end. Second, the crossover
is around 200 simulations — below that, search with a uniform prior is *worse*
than one-ply greedy, because a turn is a chain of ~8 sub-decisions and 200
simulations spread over them is 25 each. This is `SEARCH.md` §3.6's complaint
made concrete, and it is the empirical case for §9's "spend the surplus on
depth": the last table is what four times the search buys.


### 5.2 What "it is working" looks like

Per generation, against the fixed panel of §1.6:

* **Centred score against generation 0 rises and its interval stays clear of
  zero.** This is the signal. Early generations should move several points at a
  time; by generation 50 expect one to two.
* **The floor is never lost.** Centred score against `random` stays large and
  positive. A generation that drops toward zero here has broken something, not
  learned something.
* **Value loss falls and keeps falling slowly.** A value loss that plateaus
  inside a generation while `--steps` is still running means the buffer is
  exhausted: cut steps or generate more games.
* **The `root_value` in fresh records tracks the realised `z_rel`.** They are
  both in the shard; a growing gap is a calibration warning and usually means
  the value head is being trained on stale data.
* **Mean absolute final score rises.** The rollout policy scores about −20 and
  one-ply heuristic play about 0. Real play is 50–150. Watch this in the
  `selfplay` progress line; it is the cheapest liveness signal there is.
* **The mean batch size stays near `--batch`.** Not a learning signal, but the
  one that says the machine is being used. §2.2.



### 5.3 What "it has stalled" looks like

| symptom | likely cause | what to do |
|---|---|---|
| Centred score vs the previous generation hovers at zero for 3+ generations, but vs generation 0 is still rising | normal; late-run gains are ~1 point | compare to generation N−10, not N−1 |
| Centred score vs generation 0 flat, training loss still falling | overfitting the buffer | cut `--steps`, widen `--window`, or generate more games |
| Beats generation N−1 but loses to N−5 | four-player non-transitivity — the loop is walking in a circle | this is what the fixed panel is for; keep a rolling best-by-anchor pointer and roll back |
| Loses to `random` | broken encoder, broken target rotation, or a mistrained head | run `train/selftest.py`; check `selfplay --verify` |
| Policy entropy collapses; self-play games look identical | Dirichlet noise not reaching the sub-decision nodes | `SEARCH.md` §3.7: noise at *every* node of the root player's turn, not only at `Beg` |
| Mean game length below 27 rounds | rules bug | `selfplay --check`, then `cargo test` |
| Everything looks fine but the agent plays obvious nonsense | the encoder and the trainer disagree | this is what the Rust batcher of §7.1 exists to prevent |
| `selfplay` throughput collapses; mean batch size near 1 | the concurrency is not reaching the batcher | raise `--concurrency`, lower `--batchers`; §2.2 |
| `selfplay` is slower than it was with no network at all | `VECLIB_MAXIMUM_THREADS` is unset and Accelerate is fighting the batchers | export it as 1; the driver warns |
| every record has `policy_kind = 0` | the agent is `random` or `heuristic:K`, which have no regenerable policy | use `mcts:SIMS` or a checkpoint; §2.4 |
| the loader's regenerated edge list is a different length than `n_edges` | it is ignoring `PickWorker`'s retrieval counter | §3.4 |

### 5.4 A target worth holding yourself to

`LEARNING.md` §7.3, stated plainly: one week on this machine will not produce
near-optimal play. A defensible target is **beats a rollout-MCTS anchor in
better than 80% of four-player matches, and beats the warm-started net by more
than 10 points of mean centred score**. That is achievable. If after fifty
generations the panel says less than that, the constraint is games, not
cleverness — see §7.4 of `LEARNING.md` for what actually buys more.

---

## 6. The warm start

`LEARNING.md` §6.7 calls this the highest-leverage two hours in the plan, and it
is available *today*, before any network exists:

```bash
# ~1M positions paired with the final scores of a plausible continuation
cargo run --release --bin selfplay -- --agent heuristic:32 --games 10000 \
    --out replay --gen 0

python train/train.py --replay replay --out ckpt --gen 0 --steps 4000 --size small
```

Those records are `policy_kind = NONE` — the one-ply agents draw their
candidates from an RNG, so the indices are not regenerable — which means they
teach the **value heads only**. That is not a degraded mode: positions paired
with the final scores of a plausible continuation is exactly what §6.7 asks for.
A randomly-initialised value head captures nothing; one trained on this captures
that corn, resources, temple position and workers-in-play are worth something.

Then gate on it. If the warm-started net does not beat `heuristic:32` in the
arena by a wide margin, something is wrong with the encoder, the target or the
loss. Finding that on day 1 rather than day 5 is the entire point.

---

## 7. The seams

### 7.1 The encoder — `train/features.py`  *(open)*

`src/encode.rs` owns the real encoder: `LEARNING.md` §1 specifies a 3,072-wide
flat vector. `train/features.py` is a **much smaller hand-written stand-in**
(568 wide) over the fields the replay format exposes, so that the training loop,
the checkpointing, the loss and the schedule are all debugged before the real
encoder arrives.

To close it, replace `encode_batch` with either a subprocess call to the Rust
`batcher` binary of `LEARNING.md` §5.1 or a binding to `tzolkin::encode::encode`.
Keep the signature; nothing else in `train/` depends on which encoder is in use
except the value of `D_IN`.

**Do the Rust batcher rather than a second Python encoder.** "The encoder used
to train is bit-identical to the encoder used to play" is the single correctness
property most worth having here, and it is only free if there is one encoder.

The stand-in honours both of the real encoder's contracts, and any replacement
must too: perspective rotation by `(owner - p) mod 4`, and never reading
`Deck::ids`.

### 7.2 The weight format — `train/export.py`  *(open on the naming)*

`LEARNING.md` §5.1 settles the container: **safetensors, both directions**. What
is not settled is the tensor names `src/net.rs` looks for.

`export.py` writes the weights plus a **manifest JSON** listing every tensor's
name, shape and dtype. `net.rs` now has `Net::tensor_manifest(arch)`, which
emits the names and shapes it expects; diff the two and fix `rename()` in
`export.py`, which is the only place names are decided. A dependency-free `.tzw`
container (u32 header length, JSON header, 64-byte-aligned f32 blobs) is written
when `safetensors` is not installed; prefer safetensors.

### 7.3 The search — `src/record.rs`  *(closed)*

`src/mcts.rs` landed and the adapter is written: `SearchAgent` in
`src/record.rs`. `mcts:SIMS`, `mcts:SIMS:EVAL` and a bare `.safetensors` path all
work in both binaries today.

Three things about the adapter that are worth knowing, because each of them is a
place a plausible implementation would have been wrong:

* **It searches on a copy and hands back a `Move`.** `Mcts` advances the state
  by `tree::apply_step`; `play_game` advances it by `moves::apply_move`. The
  adapter runs the search over a copy, reconstructs the move with
  `tree::move_from_path` and `tree::retag_workers`, and lets `play_game` apply
  it. So self-play and the arena go through exactly the `apply_move` that 49
  rules tests cover, and any divergence between the two representations shows up
  as an illegal move rather than as a quietly different game.
* **One agent instance per thread, built by `AgentSpec::instance`.** `Agent::
  play_turn` takes `&self` and `Mcts::search_at` takes `&mut self`. A shared
  `Mutex<Mcts>` would serialise every game in the run, so the drivers build one
  instance per worker — its own tree arena, its own slot in the evaluation
  batch, its own MCTS seed. Anything genuinely shared (the weights, the batcher
  pool) sits behind an `Arc` in the spec.
* **The stored edge index is re-derived, not the tree's.** §3.5. This is the one
  that would have corrupted the buffer silently.

### 7.4 Where the batching lives

`src/record.rs` owns `BatchQueue`: the queue, the parked workers, the batcher
threads, and the statistics. It talks to a network only through
`Net::evaluate_batch`, and to everything else through `phase::Evaluator`, so it
does not care what is behind it.

`src/net.rs` has at times carried its own `BatchedEvaluator` with the same
design. **If that one is permanent, delete `BatchQueue` and use it** — the
change is confined to `Backend::evaluator` in `src/record.rs` and is about ten
lines. Two batching implementations is one too many; the reason there are
currently two is that they were written in parallel.

**One thing the queue cannot do, and it costs measurable throughput.**
`net::Query` has a `candidates` field, and `net.rs` says filling it "skips a
`choices_for_worker` re-derivation, which is ~35 µs — more than two forward
passes". Nothing fills it, because `phase::Evaluator::evaluate` — which is
frozen — carries only `(state, phase, turn, n_edges)`, and the batcher is on the
far side of that boundary from the search that already built the candidate list.
Measured here, a `Take` node with 4 candidates costs 13% more than a `Placing`
node at batch 256 (93,400 vs 107,400 evals/s), and `Take` is 16% of the nodes a
real game evaluates. Closing that needs a change in `phase.rs` or in `mcts.rs`
and belongs to whoever owns them; it is worth roughly 3–5% end to end.

---

## 8. A generation, end to end

```bash
export VECLIB_MAXIMUM_THREADS=1
G=3
P=$(printf 'ckpt/gen%04d' $((G-1)))

# 1. self-play against the previous checkpoint      (10-15 min; interruptible)
cargo run --release --bin selfplay -- \
    --agent $P.safetensors --games 20000 \
    --out replay --gen $G --snapshot-hours 1

# 2. train                                          (minutes; resumable)
python train/train.py --replay replay --out ckpt --gen $G --steps 3000 \
    --window 8000000 --init $P
python train/export.py --ckpt $(printf 'ckpt/gen%04d' $G)

# 3. did it get better?                             (minutes; interruptible)
cargo run --release --bin arena -- \
    --candidate $(printf 'ckpt/gen%04d' $G).safetensors --baseline $P.safetensors \
    --games 600 --out $(printf 'eval/gen%04d' $G).jsonl --resume
```

Every one of those commands is interruptible and picks up where it left off.
`LEARNING.md` §6.9 puts the worst case at one generation, ~35 minutes; the
checkpointing here makes it much less — one game per thread for `selfplay`, 250
steps for `train.py`, one block for `arena`.

**What each step should print if it is working:**

| step | healthy | wrong |
|---|---|---|
| `selfplay` | mean batch size within ~2x of `--batch`; batcher utilisation 60–90%; mean final score rising generation on generation | mean batch near 1 — the concurrency is not there, §2.2 |
| `selfplay --verify` | `OK`, ~150 records/game, `rules version : 3 (current 3)` | any codec round-trip failure means `state.rs` moved under the buffer |
| `train.py` | value loss falling; policy loss falling once `TREE_EDGE` records are in the window | policy loss flat while `policy_kind` is all 0 — you are training on a one-ply agent's records, which carry no policy |
| `arena` | interval clear of zero, mean game length 27.0 | mean game length below 27 is a rules bug and invalidates everything above it |

Two operational rules from `LEARNING.md`, worth repeating because they cost a
week each when ignored:

* **Do not start the real run until the rules are frozen.** A rules change
  invalidates the learned value function even when every tensor shape survives.
  `manifest.json` records the git commit for exactly this reason: when the rules
  move, the buffer can be truncated at the right generation rather than thrown
  away.
* **Do not upgrade macOS mid-project.**

---

## 9. The compute budget, and why it is not "more games"

`COMPUTE.md` §7 is the surprising part of that document and it changes the
defaults in this one, so it is restated here with the numbers measured on this
machine rather than the ones modelled.

**The obvious move is wrong.** Batching makes self-play several times faster; the
obvious thing to do with that is generate several times more games. But
`LEARNING.md` §6.3 writes records only from the 25% of turns that get the full
playout budget, ~5 qualifying nodes each, so 20,000 games is ~3 M records — and
§6.6's replay buffer holds 1.5 M across *twenty* generations. **One generation
would overflow the entire window by 2x.** Games stopped being the scarce
resource; target quality became it.

So the defaults here differ from `LEARNING.md` §6.6 in three places:

| knob | `LEARNING.md` §6.6 | here | why |
|---|---|---|---|
| simulations, full | 800 | **3,200** | §5.1: +2.0 points of centred score, measured, and it is the only knob that answers `SEARCH.md` §3.6's complaint that factoring buys `S/8` turns of lookahead |
| simulations, reduced | 100 | **400** | `SIMS/8`, unchanged as a ratio |
| replay window | 1.5 M records | **8 M** (`--window`) | 8 M x 512 B is 4.1 GB on a 24 GB machine — the old figure was 768 MB and very conservative |
| generation length | 30 min | **10–15 min** | at these rates 30 minutes is far more data than one policy-improvement step needs, and shorter cycles mean fresher weights |

**What that costs and what it buys.** Four times the search is four times the
cost per game, so games/hour falls 4x. From §2.1's measured 175,000
evaluations/s and `LEARNING.md` §7.1's 108 turns per game:

| search setting | evals/game | games/h, this Mac | games/week (130 h) |
|---|---|---|---|
| 800/100 (mean 275 sims) | 29,700 | ~21,000 | 2.7 M |
| **3,200/400 (mean 1,100)** | **118,800** | **~5,300** | **0.69 M** |

Add the Windows box at `COMPUTE.md` §5's modelled 2–5x and the plan lands
between **1.3 M and 3.3 M games per week, each with four times the search behind
it.** That is the low end of `COMPUTE.md` §7's 2–4.5 M, because §5's Mac row is
about 2x optimistic against the end-to-end pipeline — see §2.1.

Take the depth. `COMPUTE.md` §6 is right that none of this reaches strong play,
and the binding constraint at four-player non-transitivity with 27-round credit
assignment is target quality, not sample count.

**Two things this makes urgent that nothing currently does.**

* **Disk.** Budget **10–25 GB per day** of replay data and a retention policy
  that deletes shards older than the training window. Nothing in `LEARNING.md`
  §6, `record.rs` or `train/` deletes anything today. Shard filenames sort
  chronologically precisely so that `ls | head -n -N | xargs rm` is a correct
  retention policy.
* **RAM for the window.** `train/replay.py` memory-maps shards, so an 8 M-record
  window is 4.1 GB of page cache rather than 4.1 GB of heap. It is fine on 24 GB
  and it is not fine on 8.
