# Generation throughput — measurements as they landed

Workstream: move generation (`src/moves.rs`, `src/options.rs`, `src/spaces/*`).
Baseline commit `41e9d70`, 169 tests green.

Why this file exists: three agent runs were killed by usage limits and only what
was on disk survived. Every number goes in the moment it lands.

## The harness

`src/bin/movestats.rs`, `GENBENCH=1` mode:

```
GENBENCH=1 REPS=120 DESCENTS=4 CAT=1 /usr/bin/time ./target/release/movestats 24 heuristic:32
```

* Phase 1 plays 24 seeded games with `heuristic:32` and keeps all 2,380 turn
  positions. From each, `take_nodes` walks the turn chain four times the way a
  descent does and keeps every `Take` node it stands on: **12,813 nodes,
  227,065 edges, mean 17.7, p50 4, p90 40, p99 220, max 1118.** That is the
  MCTS-facing workload — `tree::legal_steps`'s `Take` arm is a
  `choices_for_worker` and nothing else.
* `cfw` times `moves::choices_for_worker` (the outer `dominated_dedup`);
  `cat` times `spaces::choices_at` alone (the inner one).
* **`GENBENCH check`** prints an order-sensitive digest of every generated list.
  A throughput change must leave it bit-identical — the same *sequence*, not
  just the same set, because `choices_for_worker` ends in a sort and the
  `TREE_EDGE` index space is that order.
* Cost in **user CPU from `/usr/bin/time`**, never wall clock; the machine is
  shared (load average 499 while these ran). ~0.7 s of the total is fixed
  overhead (phase 1 plus the two untimed checksum passes).

## Where generation's time actually goes — baseline profile

`sample` over 20 s of the `GENBENCH` loop at `41e9d70` (15,403 samples), self
time:

| share | frame |
|---|---|
| 21.4% | `SlicePartialOrd::partial_compare` — `Choice`'s derived `Ord` on the effect slice |
| 21.0% | `Iterator::cmp_by` — the **filtered effect iterators** compared inside `dominated_dedup`'s comparator and its `group` closure |
| 14.9% | the `sort_unstable_by` comparator body |
| 13.1% | `dominated_dedup` itself (the sweep, `wealth`) |
| 12.0% | sort machinery (`smallsort`, `quicksort`, `drift::sort`) |
| 12.0% | malloc/free |
| 4.7% | `Effect::cmp` |
| 4.8% | memmove/memset |
| <10% | **all actual option construction** (`recurse`, `pay_blocks`, `building_choices`, `corn_exchange`, `tikal::at_d`) |

**Meaning: roughly three quarters of generation is `dominated_dedup` and the
comparisons it drives. Building the options is under a tenth of it.** The
profile the MCTS owner saw from the other side (`Choice` sort/compare/eq 24.7%
of the whole search) is this, seen through `spaces::choices_at`.

## Measurements

All timings below are **interleaved** runs of pinned binaries through
`<scratch>/gen/bench.py`, four rounds round-robin, median user CPU. Interleaving
is not optional: the same binary measured 23.7 s at load average 32 and 31.4 s
at load average 499, while two variants measured against each other in the same
round agree to under 2%.

| # | what varied | median user CPU | vs `41e9d70` | checksum | binary |
|---|---|---|---|---|---|
| v0 | `41e9d70`, unchanged | 23.73 s | 1.000x | `cfw=54076aa3b8879166` | `ms_v0` |
| v1 | `dominated_dedup` on a precomputed `Key` | 15.64 s | **1.51x** | same | `ms_v1` |
| v2 | v1 + lexicographic tie-break moved out of the sort into the sweep | 15.61 s | 1.52x | same | `ms_v2` |
| v3 | v2 + `dedup` sorts unstably | 15.26 s | **1.56x** | same | `ms_v3` |
| v4 | v3 + `Choice` compared through packed `u32`s | 16.16 s | 1.47x — **worse** | same | `ms_v4` |
| v5 | v4 + `choices_for_worker` skips the inner dominance pass | 13.59 s | 1.74x | same | `ms_v5` |
| v6 | v3 + v5 (i.e. v5 without the packed compare) | 12.89 s | 1.84x | same | `ms_v6` |
| v7 | v6 + the builders dedup without sorting | 12.57 s | 1.88x | same | `ms_v7` |
| v8 | v7 + `Effects` inline capacity 12 / 16 | 12.59 / 12.22 s | 1.87x / **1.92x** — not landed, see 7 | same | `ms_v8_12`, `ms_v8_16` |
| v9 | v7 + a stack-array path for lists of <= 16 | 12.67 s | 1.86x — **worse** | same | `ms_v9` |
| v10 | v9 without the stack path (the `sweep`/`compact` split alone) | 12.87 s | 1.83x — **worse** | same | `ms_v10` |
| v13 | **shipped**: v7's shape, restored after v9/v10 | **12.55 s** | **1.87x** | same | `ms_v13` |

Net of the ~0.7 s fixed overhead the shipped shape is **22.78 → 11.85 s,
1.92x**, with every generated list bit-identical to `41e9d70`'s and
`cargo test --release` at 174 passed / 0 failed.

**End to end, this is 1.317x on the whole MCTS search** — see 6.

### 1. `dominated_dedup` on a precomputed key — 1.55x, output bit-identical

*What varied.* The old pass sorted `Vec<Choice>` with a comparator that, for
every one of the O(n log n) comparisons, built two filtered effect iterators and
compared them lexicographically, then re-ran the same filtered compare in the
`group` closure during the sweep, then recomputed `wealth` per element. The new
pass summarises each choice **once** — a 64-bit digest of the structural
(non-wealth) effects in order, plus the five-axis wealth vector and its sum —
and sorts a `u32` index array on that. `Effect` packs into a `u32` that is
order-isomorphic to its derived `Ord` (`options::pack`). The digest is an
accelerator, never the rule: before any prune fires, `same_group` confirms the
real structural key, so a 64-bit collision costs a missed prune and can never
drop a legal move. The four working vectors moved to a thread-local scratch.

*Numbers.* 24 games / 4 descents / 120 reps, 27.2M edges regenerated.
Baseline 31.45 and 31.02 s user; after 20.25 and 20.61 s user. Netting the
~0.7 s fixed overhead: **30.5 → 19.7 s, 1.55x.**

*Answer-preserving.* `GENBENCH check cfw=54076aa3b8879166 cat=50ca6503b2397e51`
before and after — the same digest, so every generated list is the same
sequence of the same choices. `cargo test --release`: 169 passed, 0 failed.

*Meaning.* The dominance rule was never the cost; re-deriving its key inside a
comparator was. Same rule, same output, two thirds of the time.

### 2. The tie-break belongs in the sweep, not the sort — neutral (v1 → v2)

*What varied.* The index sort's third key was `Choice`'s own `Ord`, there so
that two spellings of one position always leave the same survivor. It is only
ever needed when two choices have the *same* group and the *same* wealth vector
— a dominator's total is at least the dominated one's and the sweep runs in
total order, so equal totals plus componentwise `>=` forces equality — so the
sweep can settle it in the one place it matters instead of paying a `Choice`
compare on every one of the n log n sort steps.

*Numbers.* 15.64 → 15.61 s. **No measurable effect**, well inside the run-to-run
spread. Kept anyway: it is the correct place for the rule and it stops the
comparator touching effect lists at all.

### 3. `dedup` sorts unstably — 1.52x → 1.56x (v2 → v3)

*What varied.* `dedup` used `v.sort()`. The only elements its comparator calls
equal are *identical* choices, so stability is not a question, and the stable
sort's scratch buffer was showing up as `driftsort` plus a malloc at every
`research_choices`, `corn_exchange` and `tikal::at_d` call.

*Numbers.* 15.61 → 15.26 s, **1.023x**. Small but repeatable across all four
rounds and free.

### 4. Comparing `Choice` through packed `u32`s — **did not work** (v3 → v4)

*What varied.* `pack` turns an `Effect` into a `u32` that orders exactly as the
derive does, so a lexicographic compare over packed words is the same order with
an integer compare per element instead of a match on a pair of 14-variant enums.
The baseline profile put `SlicePartialOrd::partial_compare` at 21% and
`Effect::cmp` at 5%, so this looked like the obvious next lever.

*Numbers.* 15.26 → 16.16 s: **6% slower**, consistently, in every round.

*Meaning.* The derived `Effect::cmp` is already cheap — `Effect` is four bytes
and LLVM compares the tag and the payload directly — while `pack` adds a
jump table per element on top of it. The 21% is the *volume* of comparisons, not
the price of one. `pack` stays, because the group digest needs it, but nothing
compares through it.

### 5. `choices_for_worker` skips the inner dominance pass — 1.56x → 1.84x

*What varied.* `choices_for_worker` stacks one `choices_at` per step-down fee
and runs `dominated_dedup` over the concatenation; `choices_at` had already run
`dominated_dedup` over each space's list on the way out. The inner pass cannot
change the answer: dominance is transitive, so a choice beaten inside one
space's list is beaten inside the union, and the lexicographic tie-break between
two spellings of one position picks the same survivor over a subset as over the
whole. It was doing the outer pass's work twice on overlapping data. The inner
call is now `spaces::raw_at`; `spaces::choices_at` keeps the dominance pass for
its other callers, so its own output is unchanged.

*Numbers.* 15.26 → 12.89 s, **1.18x on top of v3, 1.84x on `41e9d70`**.
`GENBENCH check cfw` unchanged, which is the whole argument made empirically:
27.2M regenerated edges, same sequence.

*Meaning.* Two dominance passes over the same data cost more than the smaller
input the first one bought.

### 6. What it is worth to the search — 1.317x of whole-search CPU

*What varied.* Two `arena` binaries built back to back, one with `41e9d70`'s
`options.rs`/`moves.rs`/`spaces/mod.rs` and one with this workstream's, run
round-robin at the reigning champion spec.

```
GAMES=32 SPEC=mcts:2048:heuristic:quality THREADS=8 arenabench.py 3 arena_v0 arena_v6
arena_v0  n=3 median=41.16 s user  min=41.15 max=41.37
arena_v6  n=3 median=31.26 s user  min=30.78 max=31.37   1.317x
```

*Meaning.* Generation is **53% of MCTS runtime**, not the 30% the search-side
profile attributed to `Choice` sort plus `dominated_dedup`: solve
`1/((1-f) + f/1.84) = 1.317` for the share `f` that got 1.84x faster. The
missing quarter is the allocator (14.6%, "mostly `options.rs`"), `options::`
generation (5.6%) and the memmove/SmallVec lines the search-side table split off.

The search does the same thing — same games, same moves, bit-identical
generation — in **76% of the CPU**. At a fixed simulation budget that is worth
exactly zero strength, by construction; what it buys is 1.317x the simulations
at equal wall time, which on the measured budget curve (+3.04 for 8x, i.e.
+1.01 a doubling) is worth about **+0.40 centred**. Confirming an effect that
size directly needs a few thousand blocks; see 9.

### 7. `Effects` inline capacity — 2.8%, measured, deliberately not landed

*What varied.* `Effects` is `SmallVec<[Effect; 8]>`. **15.2% of generated
choices spill to the heap** (227,065 choices measured; the length histogram runs
out to 23, with 4,465 at nine, 7,270 at ten, 5,086 at eleven), because every
step-down fee prefixes one more effect and every `chain` -- the Uxmal mirror,
the Chichen theology discount, Tikal's double build -- concatenates two lists.

*Numbers.* Capacity 12: 12.59 s, **no effect**. Capacity 16: 12.22 s, **1.028x**.

*Not landed, and why.* The one-line change lives in `src/effect.rs`, which this
workstream does not own, and it widens `Choice` from 40 bytes to 72 --
`mcts.rs:1116` reasons explicitly about an edge being an inline
`SmallVec<[Effect; 8]>`. 2.8% of generation is 1.5% of the search; a 1.8x
widening of every stored edge could easily cost more than that in the tree.
**Left for the `effect.rs`/`mcts.rs` owner to decide with the tree's memory in
view.**

### 8. A stack path for short lists — **did not work**

*What varied.* The median list reaching `dominated_dedup` holds six choices
(mean 23, p90 49, p99 281, max 1881), so most calls reach the shared thread-local
buffers, clear three vectors and `resize` a fourth in order to sweep six
elements. v9 gave lists of <= 16 a path with everything in stack arrays; v10
isolated the refactor that came with it.

*Numbers.* v7 12.57 s, v9 12.67 s (**0.8% worse**), v10 12.87 s (**2.4%
worse**). Splitting the body into `sweep` and `compact` helpers cost more than
the stack arrays saved back.

*Meaning.* Zeroing a `[Key; 16]` (512 bytes) costs more than reusing a warm
`Vec`, and the straight-line body optimises better than the split one. The
shipped code says so in a comment so nobody tries it a third time.
