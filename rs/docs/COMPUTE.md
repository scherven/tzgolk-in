# Compute

Where the self-play cycles come from, now that renting them is off the table.

`LEARNING.md` §7.4 named "buy self-play compute" as the highest-leverage item on
its list and justified it with: *"This workload is CPU-bound and does not GPU-
accelerate."* That sentence is the load-bearing one, and this document exists
because **it is wrong** — not as a matter of opinion, but against measurements
taken on this machine, of this network's real shapes, and of the encoder that
landed in `src/encode.rs` while this was being written.

The correction is worth roughly 30x, and most of it is available for zero
dollars and zero new hardware.

---

## 0. The recommendation, in priority order

**1. Put a batching evaluator behind the `Evaluator` trait. Do it this week,
before `mcts.rs` grows a second user.** This is the whole ballgame. Measured on
this machine: the same forward pass runs at **5,491 evaluations/s at batch 1 and
111,469/s at batch 256 — 20.3x, on one core, with no new hardware.** Right now
`phase::Evaluator::evaluate` takes one position and returns one `Evaluation`,
and `Net`'s impl of it calls `Net::evaluate_batch` with a slice of length one
(`net.rs:1471`). The fast path exists and the trait boundary throws it away.
Cost to fix: ~150 lines, and **zero lines inside `mcts.rs`** (§2.6).

**2. Then run self-play on the Windows box as well as the Mac.** The port is
small: the only platform-specific code in `src/` is `net.rs`'s Accelerate block,
which already has a portable fallback, and `record.rs`'s `SIGINT` constants,
which already handle Windows. `cargo build --release` works there today. The GPU
is worth **2–5x the Mac** — but the more valuable thing it buys is a machine you
can pin at 100% for a week without losing your dev laptop.

**3. Write off Colab entirely; use Kaggle for tuning, never for self-play.**
Colab's free tier explicitly prohibits **"chess training"** and *"running
distributed computing workers"*, and blocks accounts automatically for ordinary
ML training — this is a terms problem, not a performance one (§3.3). Kaggle is
well-behaved and has **no weekly CPU quota**, but every usable self-play session
overflows its 20 GB output limit, because self-play's whole job is turning
compute into bytes. Point it at the inverse shape instead: anchor panels,
`c_puct` sweeps, and `SEARCH.md` §3.7's Dirichlet ablation all consume days of
compute and emit a table (§3.4).

**4. Spend most of the surplus on deeper search, not on more games.** At the
throughputs in §5 the training loop as specified in `LEARNING.md` §6.6 would be
discarding most of what self-play produces — a 30-minute generation would
overflow the entire 1.5 M-record buffer several times over. Games stop being the
scarce resource and target quality becomes it (§7).

What this buys, honestly: **6–19 M self-play games per week** instead of
`LEARNING.md` §7.2's 200 k–600 k. That is a genuine ~30x, it clears AlphaGo
Zero's 4.9 M *game count* — and it is still **not** strong play, for reasons that
have nothing to do with compute and that §6 sets out plainly.

---

## 1. What the measurements say

### 1.1 Provenance

Everything in this section was measured today, on this machine (M4 Pro, 10 P +
4 E cores, 24 GB, macOS 15.3.1), against a release build of the working tree.
Engine numbers come from `src/encode.rs`, `src/moves.rs` and `src/state.rs` as
they stand; network numbers come from a standalone C harness that reproduces the
§4 architecture's exact GEMM shapes over 16.4 MB of distinct weights, so the
weight traffic is real rather than a hot 1 MB buffer measured 20,000 times.

Nothing in `rs/` was modified to take these; the build was done on a copy.

### 1.2 The engine side is nearly free

| operation | cost | source |
|---|---|---|
| `encode(g, p, phase, out)` -> `[f32; 3072]` | **0.63 µs** | measured, 200 k calls, mid/late-game states |
| `edges(...)` for `Beg` / `Mode` / `Placing` / `PickWorker` | 0.003–0.027 µs | measured |
| `choices_for_worker` at a `Take` node | 0.593 µs | measured; width p50 2, p90 6, p99 54, max 249 |
| `encode_choices` | 0.062 µs per candidate | measured (matches §2.3's "~70 ns") |
| `GameState` copy (312 B) | 0.005 µs | measured |
| transposition lookup, `HashMap` + SipHash | 0.586 µs | measured, 320-byte key |
| transposition lookup, FxHash | 0.406 µs | measured |

Add PUCT descent and backup over ~15–25 levels and the **CPU-side cost of
producing one leaf is ~2.5–4 µs.** Call it 3 µs.

Two asides worth acting on. First, the transposition key is 320 bytes and the
default hasher spends **0.51 µs** on it — comparable to the entire state
encode. Use FxHash or ahash for `Mcts::index`, and keep the lookup at expansion
only (following `edge.child` during descent costs nothing). Doing it per level
instead of per simulation would make the index the largest single cost in the
search. Second, `Evaluation.priors` is a `Vec<f32>`: one heap allocation per
evaluation, at a few hundred thousand evaluations per second. In the batched
path, hand out slices of one arena-allocated buffer instead.

### 1.3 The network side is 3–8x more expensive than all of that

The whole forward pass — 29 GEMMs, the factored stem, six residual blocks,
every head, with LayerNorm and GELU between — measured single-threaded through
Accelerate:

| batch | µs / forward | evals/s (1 thread) | GFLOP/s |
|---|---|---|---|
| 1 | 182.1 | **5,491** | 46 |
| 8 | 296.3 | 27,002 | 228 |
| 32 | 434.3 | 73,676 | 622 |
| 64 | 727.7 | 87,950 | 742 |
| 128 | 1,282.3 | 99,822 | 843 |
| 256 | 2,296.6 | **111,469** | 941 |
| 512 | 4,297.7 | 119,133 | 1,005 |

At batch 256 the network costs **9.0 µs of core time per evaluation**, against
~3 µs for everything the search does around it. At batch 1 it costs **182 µs**.

**So `LEARNING.md` §0(a) — "move generation is the bottleneck, not the network"
— is no longer true, and the thing that made it untrue is `SEARCH.md`.** §0(a)
compared a flat `legal_moves()` call against one forward pass. Factoring the
turn (`SEARCH.md` §2.4) replaced that flat enumeration with ~8 sub-decision
nodes whose edge lists cost 0.0026 ms *for the whole turn* — a 3,100x reduction
(`SEARCH.md` §1.2) — while leaving the per-evaluation network cost exactly
where it was, and multiplying the number of evaluations per turn by 8. The two
documents were written in parallel and the arithmetic never got redone across
the boundary. `SEARCH.md` §3.5 already says the right thing ("the search is
evaluation-bound by two to three orders of magnitude"); `LEARNING.md` §0(a) is
the stale half.

Everything else in this document follows from that one correction.

### 1.4 The Mac's GEMM ceiling is 3.5 TFLOP/s and two threads reach it

`64 x 512 x 512` SGEMM issued from N threads, `VECLIB_MAXIMUM_THREADS=1`:

| threads | aggregate GFLOP/s | scaling |
|---|---|---|
| 1 | 1,705 | 1.00x |
| 2 | 3,428 | **2.01x** |
| 4 | 3,504 | 2.06x |
| 8 | 3,841 | 2.25x |

This is the two-AMX-cluster structure `LEARNING.md` §4.4 predicted, confirmed:
**GEMM throughput on this machine is capped at ~3.5–3.8 TFLOP/s and is fully
claimed by two threads.** The eighth core adds 10%, not 300%.

The whole-network figures scale better than raw GEMM (1 thread 102,218 evals/s
-> 8 threads 350,698) precisely because the LayerNorm/GELU half *is* ordinary
vector work and does spread across cores. The mix matters: the pass is roughly
half AMX-bound GEMM and half core-bound elementwise.

Two corrections to §4.4 while I am here. Its single-GEMM table understates this
machine — it reports 1,109 GFLOP/s at `256^3` and a 3.9 µs call floor where I
measure 1,626 GFLOP/s and 1.34 µs, most likely because Accelerate's own thread
pool was left enabled. And its derived "**~65,000 evals/s** for the whole
machine" is **5.4x too low**: the measured figure is 350,698. Set
`VECLIB_MAXIMUM_THREADS=1` as §4.4 says, and the GEMM saturates at **batch 32**,
not 128–256 — beyond 32 the per-call GFLOP/s is flat (1,717 at 32, 1,745 at
256). Batch above 32 buys amortised *fixed* costs (queue handoff, elementwise
tiling, and on a GPU the kernel launch), not GEMM efficiency.

---

## 2. Batching

### 2.1 What batch size is reachable

The question is not "how many leaves can one tree offer at once" — factoring
makes that number small and awkward, because a turn's first few nodes have width
1–7 and parallel descents collide immediately (`SEARCH.md` §3.5). The question is
**how many independent games are in flight**, and that is a free parameter.

`SEARCH.md` §3.1 sizes a search tree at ~1.4 MB per move at 800 simulations.
With playout-cap randomisation the average is nearer 0.5 MB. So:

| concurrent games | resident trees | reachable batch |
|---|---|---|
| 64 | ~90 MB | 64 |
| 256 | ~360 MB | 256 |
| 1,024 | ~1.4 GB | 1,024 |

On 24 GB, **256 concurrent games is comfortable and 1,024 is possible.** Batch
size is limited by memory and by nothing else, and §1.4 says we only need 32–64
to saturate the GEMM anyway. Everything above that is insurance against a
half-full batch.

This is exactly what KataGo does. Its self-play docs say the game-thread count
is *"very large, probably far larger than the number of cores on your machine…
This is intentional, as each thread only currently runs synchronously with
respect to neural net queries, so a large number of parallel games is needed to
take advantage of batching"* — and its analysis-engine guidance is to *"keep
numSearchThreads small, and instead parallelize across positions, so you can
reduce conflict between threads and improve the overall throughput."**
([KataGo SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md),
[Analysis_Engine.md](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md),
accessed Sept 2026.)

### 2.2 What batching buys on the CPU alone — measured

Before any GPU enters the picture:

| | evals/s | games/hour |
|---|---|---|
| batch 1, 1 thread (what the trait does today) | 5,491 | 665 |
| batch 256, 1 thread | 111,469 | 13,511 |
| batch 128, 2 threads | 197,619 | 23,954 |
| batch 128, 8 threads | **350,698** | **42,509** |

(Games/hour throughout this document uses `LEARNING.md` §7.1's 29,700
evaluations per game: 108 turns x a mean 275 simulations under playout-cap
randomisation.)

**The single largest available speedup on this project — 20x — costs no money,
no hardware, and no GPU.** It is a queue.

Note also that the whole-machine optimum is lopsided in a way that is worth
knowing: because the network costs ~9–23 µs of core time per evaluation and the
search costs ~3 µs, **roughly 8 of 10 performance cores should be running
inference**, with the search threads fitting into the remainder and the four
efficiency cores. Any split that reserves most of the machine for move
generation is optimising §0(a)'s world, not this one.

### 2.3 What a GPU buys on top — modelled, not measured

I have no NVIDIA hardware here, so this section is arithmetic and is labelled
as such. The model is: time per batched forward pass is
`max(compute, weight-stream, PCIe-in) + launch overhead`.

Published specs ([nanoreview](https://nanoreview.net/en/gpu-compare/geforce-rtx-4070-vs-geforce-rtx-3060),
[topcpu](https://www.topcpu.net/en/gpu-c/geforce-rtx-3060-vs-geforce-rtx-4070),
accessed Sept 2026; L2 figures from
[NVIDIA's Ada and Ampere whitepapers](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)):

| card | FP32 peak | bandwidth | L2 |
|---|---|---|---|
| RTX 3060 12 GB | 12.74 TFLOP/s | 360 GB/s | 3 MB |
| RTX 3080 10 GB | 29.77 TFLOP/s | 760 GB/s | 5 MB |
| RTX 4070 | 29.15 TFLOP/s | 504 GB/s | **36 MB** |
| RTX 4090 | 82.58 TFLOP/s | 1008 GB/s | **72 MB** |

Applying a 30% realisation factor for FP32 on `M x 512 x 512` chains, and a 25%
factor against roughly-2x-FP32 tensor throughput for an FP16 path with FP32
accumulation:

| card | FP32 path | | FP16 tensor path | |
|---|---|---|---|---|
| | evals/s | games/h | evals/s | games/h |
| RTX 3060 | 453,000 | 54,900 | 755,000 | 91,500 |
| RTX 3080 | 1,058,000 | 128,000 | 1,764,000 | 214,000 |
| RTX 4070 | 1,036,000 | 126,000 | 1,727,000 | 209,000 |
| RTX 4090 | 2,935,000 | 356,000 | 4,892,000 | 593,000 |

**Against the Mac's measured 350,698 evals/s, even the weakest card on that list
is ~1.3x on the plain FP32 path and ~2.2x with tensor cores.** A 4070 or 3080 is
~5x. Uncertainty here is at least ±2x in the realisation factors and I would not
defend any individual cell; the ordering and the rough magnitude are what
matter.

### 2.4 Where the GPU stops helping — and it is sooner than the specs suggest

Three ceilings sit below the FLOP numbers, and on the fast cards they bind first.

**Kernel launch overhead is the one that decides whether this works at all.**
The forward pass is ~29 GEMMs plus ~26 elementwise ops — call it 55 kernels.
Launched individually, at the 5–10 µs per launch that is typical when CPU
dispatch rivals the GPU work, that is **275–550 µs per forward pass regardless
of batch size**, which on its own caps a 4070 at well under what the Mac already
does. **CUDA Graphs are not an optimisation here, they are a precondition**: a
graph replay *"skips all layers of argument setup and kernel dispatch… and
submits the whole graph with one `cudaGraphLaunch`"*
([PyTorch blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
[CUDA Graphs overview](https://ai-infrastructure.net/cuda-graphs/), accessed
Sept 2026). This matters more on Windows than on Linux, since consumer cards run
under WDDM and cannot use the lower-overhead TCC path.

**Weight streaming sets a fixed floor per pass.** 4.22 M parameters is 8.4 MB in
FP16: 23 µs on a 3060, 8 µs on a 4090. Below batch ~30 you are paying that floor
for very few evaluations. On a 4070 or 4090 the entire net fits in L2 (36 / 72
MB), so after the first pass the weights never leave the chip — a real
advantage of the Ada cards that has nothing to do with their TFLOPs.

**PCIe becomes the binding constraint before the fast cards do.** The input is
3,072 floats. Even in FP16 that is 6.1 KB per evaluation:

| throughput | PCIe input traffic |
|---|---|
| 500 k evals/s | 3.1 GB/s |
| 1 M evals/s | 6.1 GB/s |
| 2 M evals/s | 12.3 GB/s — saturates PCIe 3.0 x16 |
| 4 M evals/s | 24.6 GB/s — saturates PCIe 4.0 x16 |

So: **send FP16, use pinned memory, and double-buffer on a separate copy stream**
so the transfer overlaps compute. And note that a 4090's headline 4.9 M evals/s
is unreachable across any consumer PCIe link.

**The CPU has to produce the leaves.** At ~3 µs per leaf on an Apple P-core, and
allowing a 1.6x derate for a desktop x86 core doing this branchy pointer work:

| cores | leaves/s | games/h |
|---|---|---|
| 6 | 1.5 M | 182,000 |
| 8 | 2.0 M | 242,000 |
| 12 | 3.0 M | 364,000 |
| 16 | 4.0 M | 485,000 |

Put the four ceilings together and the conclusion is sharp:

> **On the Windows box the GPU model barely matters. What matters is the core
> count.** A 3060 is GPU-bound and delivers ~750 k evals/s. A 4070, 3080 or 4090
> is bound by PCIe and by how fast 8–16 cores can descend trees, and they all
> land in the same 1.5–2.5 M evals/s band. The gap between a 3060 and a 4090 in
> *this* workload is maybe 2.5x, not the 6.5x their FP32 specs imply.

The GPU's real contribution is not that it is fast. It is that it **takes the
network off the CPU entirely.** That matters much more on Windows than here,
because there is no AMX on x86: a Zen 4 core with AVX-512 realises perhaps
200 GFLOP/s on SGEMM, so eight of them come to ~1.6 TFLOP/s — **less than half
this laptop's 3.5.** CPU-only self-play on a typical gaming desktop would be
*slower* than the Mac. With the GPU doing the network, every core goes to search
and the same box is 3–5x the Mac.

### 2.5 The cost in search quality — and how to pay none of it

`SEARCH.md` §3.5 recommends tree parallelism with virtual loss, and the
literature is clear that this costs something: parallel descents *"violate the
path-dependent nature of sequential UCT, resulting in higher search overhead"*,
virtual loss recovers most but not all of it, and at least one domain study
finds lock-free tree parallelism with virtual loss still degrading results
([Chaslot et al., *Parallel Monte-Carlo Tree Search*](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf);
[Mirsoleimani et al., *An Analysis of Virtual Loss in Parallel MCTS*](https://liacs.leidenuniv.nl/~plaata1/papers/paper_ICAART17.pdf)).
`SEARCH.md` is right that the cost is *worse* here than usual, because a turn's
first nodes have width 1–7 and threads have nowhere to diverge.

**None of that applies to the batching this document recommends, because the
batch is assembled across games rather than within a tree.**

| | source of the batch | virtual loss | statistics seen during descent | search quality |
|---|---|---|---|---|
| intra-tree | N descents in one tree | required | stale by up to N-1 simulations | measurably degraded |
| **inter-game** | 1 descent in each of N trees | **not needed** | **exact** | **identical to sequential** |

With one descent in flight per tree, each tree's simulation completes — leaf
evaluated, value backed up — before that tree's next descent begins. The tree
never observes a virtual loss it has to undo, never selects against statistics
that are missing an in-flight result, and produces bit-for-bit the search a
single-threaded run would produce from the same seed. The only thing that
changes is that the tree sits idle while its leaf is queued, which is
irrelevant when 255 other trees are using the machine.

**The tradeoff is real but it is not search quality — it is latency per game.**
Each individual game advances N times slower in wall-clock terms while N games
advance at once. For self-play that is a non-issue: you want games per hour, not
a fast single game. For the anchor panel (`LEARNING.md` §6.8), for interactive
play, and for anything where one position must be searched quickly, you still
want intra-tree parallelism and you still pay virtual loss's cost there. That is
a tiny fraction of total compute, and `mcts.rs` already implements the virtual
loss bookkeeping for it.

Three second-order costs of inter-game batching that are worth naming:

- **Memory**, §2.1: ~0.5–1.4 MB of tree per concurrent game. Bounded and cheap.
- **Interrupt granularity.** `LEARNING.md` §6.9 sizes the worst-case crash loss
  at 32 games. With 256 in flight it is 256 partial games. Still under a minute
  of work at these rates, but the `SIGINT` handler should drain all in-flight
  games rather than one.
- **Correlated staleness.** 256 games started against the same checkpoint all
  finish against it too, so a generation's data is slightly more homogeneous
  than 256 sequential games would be. Dirichlet noise is per-game and already
  handles diversity; this is a real effect but a small one.

### 2.6 The architecture change, and why it is small

The task framed this as "does self-play need to become an async/batched pipeline
rather than a synchronous loop, and how invasive is that given `Mcts::search` is
a synchronous call?" The answer is **yes to batched, no to async, and it does
not touch `Mcts` at all.**

`mcts.rs` landed while this was being written. It is honest about where it
stands: *"this is single-threaded… there is no batching."* `Mcts<E: Evaluator>`
owns its evaluator, and `search(&mut self, …) -> SearchResult` blocks until the
simulations are done. That shape is fine. **Do not make it resumable, and do not
introduce futures.**

The change is entirely behind the trait:

```
N OS threads, one Mcts each, all holding Arc<BatchedEvaluator>
        |
        |  Evaluator::evaluate(...)  -- synchronous, as today
        v
  BatchedEvaluator: push (Query, oneshot slot) onto a queue; park on a condvar
        |
        v
  1-2 batcher threads: drain up to B requests, or wait <= 200 us for a partial
                       batch; call Net::evaluate_batch once; write results back;
                       broadcast
```

That is the KataGo design, and it is why their configs run far more game threads
than cores. Concretely:

- **`mcts.rs`: no changes.** `search` stays synchronous and `&mut self`. It
  blocks inside `evaluate` instead of computing there.
- **`phase.rs`: no signature change required.** The batch-1 `evaluate` is a
  perfectly good client of a batching backend. (Adding an
  `evaluate_batch` default method is still worth doing so the trainer and the
  arena can use it directly, but self-play does not need it.)
- **New: ~150 lines** for the queue, the parking, and the batcher loop.
- **`src/bin/selfplay.rs`: one line, and it is the important one.** The driver
  landed while this was being written and runs `seeds.par_iter().for_each(...)`
  — one game per rayon task. Rayon's default pool is
  `available_parallelism()`, so **that structure caps the batch at ~14 on this
  machine**, which by §2.2's table is roughly 27,000 evals/s per thread instead
  of 111,000. The concurrency that produces the batch has to come from the
  *game* count, not the core count.

Four implementation notes, all of them things that will otherwise cost a day:

1. **Size the game pool by memory, not by cores.** Either
   `ThreadPoolBuilder::num_threads(256)`, or plain `std::thread` for the game
   workers. 256 mostly-parked OS threads is the intended design, not a smell —
   it is what KataGo's "far larger than the number of cores" config means.
2. **The batcher must not be a rayon worker.** If every thread in the pool is
   parked waiting for a batch and the batcher is itself queued behind them, the
   pool deadlocks. Give it a dedicated `std::thread` outside the pool. Blocking
   inside the rayon tasks is then safe, because they all block symmetrically on
   something that is guaranteed to be running.
3. **Set `VECLIB_MAXIMUM_THREADS=1`** (as `net.rs` already warns) or Accelerate
   will spawn its own pool and fight the game threads for the same cores.
4. **Wake waiters with one broadcast per batch**, not 256 individual signals. At
   350 k evals/s the futex traffic is otherwise a measurable tax.

**The reason this is priority 1 is that it is nearly free today and gets more
expensive every day.** As of this writing `net.rs`'s `evaluate_batch` has
exactly one caller — its own batch-1 shim; `mcts.rs` is single-threaded by its
own admission; and `selfplay.rs` has just committed to one-game-per-core.
`src/` gained an encoder, a tree, an MCTS, a network, a record format and a
self-play driver in the time it took to write this document. Every one of those
is a place the batch-1 assumption can set, and it is setting now.

Finally, **instrument the batch.** Log a histogram of realised batch sizes and
queue-wait times from the first day. If the mean batch size is not close to the
configured maximum, none of §5's numbers are happening, and it is the only
symptom you will get.

---

## 3. The three sources of free compute

### 3.1 The Mac

**Verdict: keep it, run self-play on it, but do not make it the only machine.**

Measured ceiling **350,698 evals/s = 42,509 games/h**. After pipeline losses —
queue stalls, allocator pressure, arena cache misses that my standalone harness
does not model, the ragged pointer head, and the difference between a model of
the network and the network — plan on **45–60% of that: 19,000–25,000 games/h.**

The specific things I have *not* modelled and that will cost something:

- The two-stage pointer head over a variable candidate list. Batching ragged
  rows is real work; the `K = 32` cap (`SEARCH.md` §1.3, `mcts.rs`'s
  `max_edges`) bounds it, which is why it is a tax and not a disaster.
- Contention for memory bandwidth between the inference threads and 256 search
  threads pointer-chasing through growing arenas.
- **Thermals.** `hw.perflevel*` says 10 P + 4 E and 24 GB, which is both the
  MacBook Pro M4 Pro and the Mac mini M4 Pro. If it is the laptop, a multi-day
  all-core run will throttle — plan on losing 10–25% — and, more to the point,
  you will not want to use your dev machine while it is doing this. **That is the
  strongest argument for the Windows box, and it is not a throughput argument.**

Do **not** put the network on the Mac's own GPU. The M4 Pro's 20-core GPU is
around [9.2 TFLOP/s FP32](https://www.cpu-monkey.com/en/igpu-apple_m4_pro_20_core)
(third-party figure; Apple publishes none) against the 3.5 TFLOP/s of AMX
measured in §1.4, so the headroom is perhaps 1.5–2x after realisation — and
`LEARNING.md` §5.2 documents, with issue numbers, that every Metal path
reachable from Rust today carries either a correctness risk or a maturity risk.
A 2x that costs a week of debugging Metal is a bad trade when the Windows box
offers 3–5x for a day of porting. Revisit only if the Windows box never happens.

Training stays here and stays on the CPU — `TRAINING.md` §4.1 reaches the same
conclusion independently. `LEARNING.md` §0(b) is still right and the new numbers
make it more so: ~26 GFLOP per optimiser step at batch 1024 is
~25 ms against measured AMX throughput, so 3,000 steps is under two minutes.
The data loader is not a bottleneck either — regenerating masks and candidates
costs ~0.6 µs plus 0.63 µs per encode, so 4x perspective augmentation over
3 M samples is about 12 seconds of core time. **Nothing about training wants a
GPU.** This matters for §3.3 and §3.4: Colab and Kaggle cannot help with the
part of the pipeline they are actually good at, because that part is already
free.

### 3.2 The Windows box

**Verdict: do this. It is the second-biggest win after batching, and the port is
small.**

**Portability is already handled.** I grepped `src/` for `cfg(`, `target_os`,
`target_arch`, `libc`, `std::os::unix`, `#[link]` and hard-coded paths. There
are exactly two hits and both are deliberate: `net.rs`'s Accelerate block behind
`cfg(target_os = "macos")`, and `record.rs`'s `cfg(unix)` / `cfg(windows)`
`SIGINT` constants — which is to say the replay writer was already written to
run on Windows. No POSIX assumptions, no hard-coded paths, no platform APIs
anywhere else. `net.rs` states the intent outright:

> *"The doc names Accelerate `cblas_sgemm` as the backend. That was right for one
> machine; the same weights now have to run on a Windows box with a large GPU and
> possibly in Colab. So every matmul goes through `Gemm`, which has a portable
> pure-Rust implementation that is correct everywhere and an Accelerate one
> behind `cfg(target_os = "macos")`."*

So `cargo build --release` on Windows works today and gives a correct — if slow
— self-play binary. The whole port is one more `Gemm` implementation.

**Which backend.** Two options, in order of preference:

1. **`ort` (ONNX Runtime) with the CUDA execution provider.** `LEARNING.md` §5.2
   already surveyed `ort` and called it *"the most mature crate in this survey by
   a distance"*, rejecting it only because CoreML's ANE has a 0.2–0.4 ms
   per-call floor. **That objection is specific to CoreML and does not transfer
   to CUDA.** The CUDA EP is enabled by a Cargo feature, supports
   `enable_cuda_graph`, and supports IO binding to device buffers — which §2.4
   says are both mandatory
   ([ort docs](https://crates.io/crates/ort),
   [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
   accessed Sept 2026). Export from PyTorch to ONNX is one call, and the
   safetensors path stays as the source of truth.
2. **`cudarc` + cuBLASLt by hand**, mirroring the structure of the Accelerate
   backend. ~300 lines, no ONNX in the loop, full control over graph capture.
   Take this if `ort`'s CUDA packaging on Windows proves painful, which it
   sometimes is.

Either way, implement `Gemm` *and* the elementwise ops on the device — do not
round-trip activations across PCIe between layers. The correctness gate is
cheap and mandatory: **run the same 10,000 positions through the Mac
(Accelerate), the portable fallback, and the CUDA backend, and assert agreement
to ~1e-4.** A silently-wrong GPU backend generating a week of replay data is the
worst failure mode available in this project.

**Topology.** Self-play on Windows, self-play on the Mac, **training on the
Mac.** Reasons: training is minutes and wants to sit next to the code and the
checkpoints; the Mac is where you are; and the replay data flows one way
(§4). Do not try to train on both.

**What to expect**, using §2.4's ceilings and a 45–60% derate:

| Windows configuration | ceiling | plan on |
|---|---|---|
| 3060-class, FP32 path only | 54,500 games/h | 25,000–33,000 |
| 3060-class, FP16 tensor path | 91,900 games/h | 41,000–55,000 |
| 3080 / 4070-class, 8-core CPU | 206,000 games/h | 93,000–124,000 |
| 4090-class, 16-core CPU | 388,000 games/h | 175,000–233,000 |

The honest reading of that table: **get the FP32 path working first**, because
even it beats the entire Mac, and the FP16 tensor path is a 1.7x on top that can
wait. And if the box turns out to be 6-core, the top two rows collapse toward
each other — core count, again, not the card.

### 3.3 Google Colab free tier

**Verdict: no. Not "impractical" — the free tier's terms name this workload.**

Google publishes almost no concrete free-tier numbers, and says so: *"Colab is
able to provide resources free of charge in part by having dynamic usage limits
that sometimes fluctuate, and by not providing guaranteed or unlimited
resources."* ([Colab FAQ](https://research.google.com/colaboratory/faq.html),
live doc, accessed Sept 2026). So most of what follows is community-observed,
and flagged as such.

| | free tier | source |
|---|---|---|
| GPU | T4, 15 GB — **best-effort, model not guaranteed** | community; FAQ says only that types *"vary over time"* |
| CPU / RAM | **2 vCPU**, ~13 GB | community ([saturncloud](https://saturncloud.io/blog/whats-the-hardware-spec-for-google-colaboratory/), updated 2026-05-01); Google publishes neither |
| max session | *"at most 12 hours, depending on availability and your usage patterns"* | FAQ, accessed Sept 2026 |
| idle timeout | *"Runtimes will time out if you are idle."* **No number published** | FAQ. The widely-quoted 90 min is folklore |
| background execution | **Paid only** | [developers.google.com/colab](https://developers.google.com/colab), accessed Sept 2026 |
| quotas / cooldown | *"Colab does not publish these limits"* | FAQ |
| disk | ~35–78 GB, ephemeral, destroyed with the VM | community |

Any one of the following would be enough to rule it out:

**It is explicitly prohibited.** The FAQ carries a list of things *"disallowed
from managed Colab runtimes running free of charge… and may be terminated at any
time without warning"*, and that list names **"chess training"** and **"running
distributed computing workers"** outright. Board-game self-play RL is the exact
analogue of the former and the intended shape of this pipeline is the latter.
Working around it with multiple accounts is separately and explicitly banned.
This is not a gray area, and I am not going to recommend a plan whose first step
is breaking the terms of the service it runs on.

**Enforcement is automated and catches ordinary ML training.** Google's own
issue tracker has accounts blocked mid-training with *"This account has been
blocked from accessing Colab runtimes due to suspected abusive activity"* —
[colabtools#6041](https://github.com/googlecolab/colabtools/issues/6041)
(2026-07-02, still open) is a user training a speech model on a T4;
[#6047](https://github.com/googlecolab/colabtools/issues/6047) (2026-07-06) was
not even on a GPU. Appeals succeed in a day or two, which is not a schedule you
want a training run to depend on.

**2 vCPUs cannot feed the T4.** §2.4's table says the CPU has to produce leaves
at ~4 µs each; two shared Xeon vCPUs give perhaps 300–500 k evals/s at absolute
best, and that is before the T4 is considered.

**Unattended multi-day execution is a paid feature that does not reliably work
even when paid for.** Background execution is Pro/Pro+ only, and the tracker is
full of 2025–2026 reports of paying users' sessions dying in 20 minutes to 5
hours ([#5793](https://github.com/googlecolab/colabtools/issues/5793),
[#5939](https://github.com/googlecolab/colabtools/issues/5939),
[#5950](https://github.com/googlecolab/colabtools/issues/5950),
[#5979](https://github.com/googlecolab/colabtools/issues/5979)).

For completeness, since the question was whether the paid tiers are worth naming:
Colab Pro is **$9.99/mo for 100 compute units**, Pro+ **$49.99/mo for 500**
([Google Workspace admin KB](https://knowledge.workspace.google.com/admin/getting-started/editions/google-workspace-add-ons),
accessed Sept 2026); a T4 burns ~1.19 CU/hr, so Pro is about 84 T4-hours. Given
the reliability record above and the hard no-money constraint, **no, they are not
worth naming.** The Windows box is free and better.

And note the finding from §3.1 that removes Colab's one remaining rationale:
**training does not need a GPU either.** 3,000 optimiser steps is under two
minutes on this laptop. There is nothing left for Colab to do.

### 3.4 Kaggle Notebooks

**Verdict: genuinely usable, but not for self-play. Use it for ablations and
evaluation panels — work with a high compute-in / bytes-out ratio.**

Kaggle is the better-behaved sibling in every respect that matters: real
published quotas, a documented headless API, an AUP that permits personal ML
work, and — the finding that surprised me — **no weekly quota on CPU at all.**

| | Kaggle | source |
|---|---|---|
| GPU quota | **30 h/week**, *"or sometimes higher depending on demand"* | [docs/efficient-gpu-usage](https://www.kaggle.com/docs/efficient-gpu-usage), accessed Sept 2026 |
| GPU type | **T4 x2** (2 x 16 GB). ⚠ **P100 retires 2026-09-15** — nine days from now | [product-announcements/735239](https://www.kaggle.com/discussions/product-announcements/735239), Kaggle staff, ~2026-08-14 |
| TPU quota | 20 h/week, 9 h/session, v5e-8 | [docs/tpu](https://www.kaggle.com/docs/tpu) + [product-announcements/607202](https://www.kaggle.com/discussions/product-announcements/607202) |
| **CPU quota** | **none documented** | quota docs cover GPU and TPU only |
| session length | **12 h** CPU/GPU, 9 h TPU — batch *and* interactive | [docs/notebooks](https://www.kaggle.com/docs/notebooks), accessed Sept 2026 |
| concurrency | **5 batch + 5 interactive**; 2 GPU | Kaggle staff, [product-feedback/483684](https://www.kaggle.com/discussions/product-feedback/483684) |
| cores / RAM | **4 cores / 30 GB** (CPU-only); 4 / 29 GB (T4 x2) | docs/notebooks |
| `/kaggle/working` | **20 GB**, auto-saved and reattachable | docs/notebooks |
| private datasets | 200 GB each, **200 GB total**, max 50 top-level files | [docs/datasets](https://www.kaggle.com/docs/datasets) |
| quota reset | **Saturday 00:00 UTC**, fixed not rolling | Kaggle staff, [product-feedback/173129](https://www.kaggle.com/product-feedback/173129) |
| idle timeout | ⚠ Kaggle's own docs say **20 min** in one place and **60 min** in another. Design for 20. Batch sessions have **none** | docs/notebooks vs docs/efficient-gpu-usage |

Three mechanical things work in our favour and are worth recording, because they
are what make the ablation use case real:

- **Compiled Rust binaries run.** `/kaggle/input` is read-only and strips the
  exec bit, so the recipe is: ship the binary in a private Dataset, copy it to
  `/kaggle/working` or `/tmp`, `chmod 0o755`, exec. There is abundant public
  precedent, including Rust game agents for Kaggle's own simulation competitions.
- **The whole loop is drivable headless from the laptop.** `kaggle kernels push`
  *"pushes new code/notebook and metadata to a kernel, then runs the kernel"*,
  then `kernels status` and `kernels output`. Kaggle explicitly endorses this:
  *"Consider using the Kaggle-API to avoid interactive sessions entirely."*
  (Gotchas: `enable_internet` and `enable_gpu` both default to **false** in
  `kernel-metadata.json`, and `--dir-mode` defaults to `skip`, which silently
  drops subdirectories on dataset upload.)
- **Interactive output is not saved.** Only Save Version / Save & Run All
  produces durable, reattachable output. Anything you care about must come from
  a batch session.

**Why it still fails for self-play: the data cannot get out.**

A T4 x2 session's real limit is its 4 CPU cores, not its GPUs — at ~4 µs/leaf
that is ~1.0 M evals/s of feed against ~1.9 M evals/s of FP16 tensor capacity.
So call it 600–800 k evals/s net, or **33,000–58,000 games/h** after the usual
derate, which over the 30 h weekly quota is **1.0–1.7 M games/week**. That is a
real contribution, comparable to what the Mac produces.

Except:

| games/h | replay produced | per 12-h session | `/kaggle/working` |
|---|---|---|---|
| 33,000 | 2.28 GB/h | 27.4 GB | 20 GB |
| 45,000 | 3.11 GB/h | 37.3 GB | 20 GB |
| 58,000 | 4.01 GB/h | 48.1 GB | 20 GB |

(`record.rs` fixes `RECORD_BYTES = 512`; ~135 records/game under
`LEARNING.md` §6.3's write policy and playout-cap randomisation.)

**Every usable session overflows the output limit by 1.4–2.4x**, and even if
compression bought a factor of four you would then be pulling 7–12 GB per
session down the Kaggle API — which uses *"dynamic rate limiting"* with no
published numbers and 3-hour token expiry. Self-play's defining property is that
it converts compute into a large volume of bytes, and that is the one thing
Kaggle is bad at.

**So use it for the inverse shape.** Several genuinely useful jobs consume a lot
of compute and emit almost nothing, and every one of them is currently
unscheduled in `LEARNING.md` and `SEARCH.md`:

- **The anchor panel** (`LEARNING.md` §6.8). ~600 games per run, output is a
  JSON scoreboard. Runs against a fixed checkpoint you upload once.
- **`SEARCH.md` §3.7's Dirichlet ablation**, which the document itself flags as
  *"the item in this document most worth an ablation"* — noise at the `Beg` node
  only vs. the whole turn chain, compared on policy entropy. Output: two numbers.
- **`c_puct`, FPU, `max_edges` and playout-cap sweeps.** A grid of 20
  configurations x a few hundred games each is days of Mac time and a couple of
  Kaggle sessions, and it emits a table.
- **`src/bin/rollout_quality.rs`**, which `LEARNING.md` §7.5 says must run before
  committing to the rollout-blending hybrid.

That is a real and unqualified win: it takes the tuning work off the critical
path of the machines doing self-play, which is exactly the work that otherwise
never gets done because it competes with data generation.

**The one caution**: Kaggle's AUP (version June 22 2025, accessed Sept 2026)
prohibits *"server farming"* and *"activity unrelated to ML data science."*
Training a game agent is plainly ML, so the substance is fine — but saturating
all five batch slots round the clock indefinitely starts to look like the
former, and Kaggle has demonstrably cut concurrency caps *"to mitigate abuse."*
Use it in bursts for named experiments. Do not use a second account.

### 3.5 Splitting the work across all three

**Verdict: two machines, yes. Three, no.**

Mac + Windows is worth it because the coordination cost is genuinely near zero.
The self-play binary is already the right shape for it — `LEARNING.md` §7.4
noticed this: *"weights in, replay shards out, no shared state."* Each machine
pulls the current checkpoint, plays whole games, writes numbered shards. There
is no cross-machine synchronisation inside a generation, no shared arena, no
consistency requirement. The failure mode of one machine dying is "half the data
this generation", not a corrupt run.

The things you must get right, and they are all cheap:

- **Stamp every shard** with `RULES_VERSION` (already in `lib.rs`), the
  generation number, the checkpoint hash, *and the producing host and backend*.
  When the Mac and the CUDA backend disagree, you will want to be able to find
  and drop exactly the affected shards.
- **Numbered, host-prefixed shard names** so two producers never collide:
  `replay/gen_0042/mac_0001.bin`, `replay/gen_0042/win_0001.bin`.
- **Let generations be ragged.** Do not make the trainer wait for both machines;
  train on whatever shards have landed. A machine that is 20 minutes behind
  contributes to the next generation instead of stalling this one.

Adding Colab or Kaggle as a **third self-play producer** is where it stops being
worth it, and the reason is not throughput. It is that a hosted notebook is the
only component in the system whose failures are *external, silent and
uncorrelated with anything you control* — a session reclaimed at 40 minutes, an
undocumented quota, a policy change, an automated abuse block. Every one of them
turns into "why is generation 84 short" at 3 a.m. Two machines you own have
failure modes you can see.

**But the split that is worth making is by job, not by machine.** Kaggle should
not produce self-play games (§3.4); it should absorb the tuning and evaluation
work that would otherwise steal time from the two machines that do:

| job | where | why |
|---|---|---|
| self-play | Windows (most) + Mac | bytes-out heavy; needs local disk and a fast link to the trainer |
| training | Mac | minutes per generation, sits next to the code and the checkpoints |
| anchor panel, ablations, hyperparameter sweeps | Kaggle | compute-in, tiny-answer-out; the exact inverse of self-play |
| anything | Colab | never (§3.3) |

That is a heterogeneous setup that does *not* multiply failure modes, because
the pieces are decoupled: if Kaggle vanishes for a week you lose some tuning
data, not a generation.

---

## 4. Moving bytes between the machines, for free

The traffic is wildly asymmetric, and noticing that is most of the design:

| what | size | direction | rate |
|---|---|---|---|
| weights (self-play workers need only these) | **16.9 MB** fp32 / 8.4 MB fp16 | trainer -> workers | once per generation |
| full resumable checkpoint (+ Adam moments) | **50.6 MB** | stays on the trainer | once per generation |
| **replay shards** | **512 B/record** (`record.rs`), ~135 records/game | workers -> trainer | **1.2–7.1 GB/hour** |

Weights are trivial: 300 generations of weights-only is 5 GB in total. Replay is
the entire problem, and at the §5 rates it is **30–171 GB/day**, or
**160–930 GB across a week.**

**The answer is a LAN, and it is not close.** At the combined rates in §5,
replay flows at **0.35–2.0 MB/s** — nothing at all on gigabit ethernet or decent
Wi-Fi, and *painful-to-impossible* on a typical home upload link. So:

**Put both machines on the same network and use [Syncthing](https://syncthing.net).**
It is free and open source with no paid tier, peer-to-peer with no server, TLS
between devices that have explicitly approved each other, has no storage limits,
and runs natively on both Windows and macOS. On a LAN, files never leave the
network. Configure two folders, not one:

- `ckpt/` — **bidirectional**, tiny, sync everything.
- `replay/` — **"Send Only" on Windows, "Receive Only" on the Mac.** One-way
  removes the entire class of bug where the trainer's retention policy deletes a
  shard and Syncthing helpfully deletes it on the producer too.
- Set a Syncthing ignore pattern for `*.part`, so `LEARNING.md` §6.9's
  in-progress shards do not sync half-written. Only the `rename` to the final
  name should trigger a transfer.

Alternatives, if Syncthing is not wanted: Windows has **OpenSSH Server built in**
since Windows 10 1809, so `rsync`/`scp` over SSH works with nothing installed,
and a plain SMB share works too. Both are fine; Syncthing is better because it
is continuous and restartable rather than something you have to remember.

**What not to do.** Do not route replay data through any free cloud tier. Google
Drive's free quota is 15 GB — under a day of output — and Colab's own FAQ warns
about per-user operation-count and bandwidth quotas and about mount failures
past ~10,000 items in a directory. Kaggle's 200 GB private-dataset quota is
large enough on paper but has a 50-top-level-file cap and undocumented dynamic
API rate limits. GitHub is fine for **checkpoints** (16.9 MB is well under the
100 MB per-file limit) and hopeless for replay.

**Add a retention policy on day one.** `LEARNING.md` §6 never deletes anything,
and neither does `record.rs`'s shard writer. At these rates the replay directory
reaches 160–930 GB in a week. Delete shards older than the training window, and
do it from the trainer, which is why the `replay/` sync is one-way.

---

## 5. Revised throughput

The honest revision of `LEARNING.md` §7.1–7.2. Everything on the Mac row is
derived from measurements taken today; everything on the Windows rows is
modelled from published specs and should be read with a ±2x error bar.

| setup | measured/modelled ceiling | **plan on** | per week (130 h) |
|---|---|---|---|
| Mac, batch-1 evaluator (today's trait) | 665 games/h | — | 0.09 M |
| Mac, batched, 8 inference threads | 42,500 games/h | **19,000–25,000** | 2.5–3.3 M |
| Windows, 3060-class, FP32 path | 54,500 | 25,000–33,000 | 3.3–4.3 M |
| Windows, 3060-class, FP16 tensor | 91,900 | 41,000–55,000 | 5.3–7.2 M |
| Windows, 3080/4070-class, 8-core | 206,000 | 93,000–124,000 | 12–16 M |
| Windows, 4090-class, 16-core | 388,000 | 175,000–233,000 | 23–30 M |
| **Mac + Windows (3060-class)** | | **44,000–80,000** | **5.7–10.4 M** |
| **Mac + Windows (3080/4070-class)** | | **112,000–149,000** | **15–19 M** |
| (Kaggle, if it could ship the data — it cannot) | | 33,000–58,000 for 30 h/wk | 1.0–1.7 M |

**Headline: 6–19 M games/week across the two machines you already own, against
`LEARNING.md` §7.2's 200 k–600 k.** That is consistently about **30x** at either
end of both ranges, for the price of a queue and a CUDA backend.

It decomposes cleanly, which is the best evidence that it is not wishful:
**~4–5x from the Mac alone** (mostly because §4.4's 65,000 evals/s figure was
5.4x low, plus real batching), and **~3–7x again from adding the Windows box.**
Neither factor requires anything clever.

Three caveats that belong next to that number rather than in a footnote:

1. **The ±2x is real, and asymmetric.** The Mac row's components are measured
   but their *composition* is not — no batched self-play driver exists yet. The
   Windows rows depend on realisation factors I could not measure and on a GPU
   whose model is unknown. If the box turns out to be 6-core with a 3060, the
   combined figure is nearer 45,000 games/h than 130,000, and the headline
   becomes 6 M rather than 19 M.
2. **These numbers are for the current search settings**, and §7 argues you
   should not keep them. Trading 4x of this for 4x the search depth per move is
   the better use of it, which lands the realistic plan at **2–4 M games/week
   with four times the search behind every target.**
3. **Disk and the training loop bind before compute does.** See §7.

---

## 6. Is that strong play?

**No. Closer than `LEARNING.md` §7.3 concluded, and still no.**

`LEARNING.md` §7.3 is worth rereading, because **five of its six reasons survive
this document completely intact.** More compute answers exactly one of them
(reference scale) and does nothing about the other five. Repeating them, because
they are the real answer to the question:

2. The successful laptop-scale replications are Connect Four and 6x6 Othello:
   branching under 100, two players, zero-sum, perfect information. This game is
   worse along every one of those axes.
3. Credit assignment over 27 rounds. A Tikal entry-space worker pays off five
   rounds later; a temple investment pays at rounds 14 and 27.
4. Four-player non-transitivity. Self-play in a non-zero-sum four-player game can
   cycle rather than climb, and **more games make cycling faster, not less
   likely.** `LEARNING.md` §6.8's fixed anchor panel is the only defence and
   nothing in this document strengthens it.
5. No symmetry augmentation on the policy — 4x on the value head only.
6. The `RetrieveWhat` action space is the hard part and each of its cases appears
   in a small fraction of games.

And there is a sixth that this document adds, which is really `LEARNING.md`
§8.4 with more force: **more compute per hour raises the cost of a rules change.**
At 100,000 games/h a rules change that invalidates the value function throws
away an order of magnitude more work than it did at 5,000. Freeze the rules
first. This was already the largest live risk; the speedup makes it larger.

On reason 1 — reference scale — the picture genuinely does change, but not as
much as the game counts suggest, and this is the number to keep in mind:

| | per evaluation | per game |
|---|---|---|
| AlphaGo Zero (40-block, 256-ch ResNet, 1,600 sims/move, ~200 moves) | ~34 GFLOP | ~10.9 PFLOP |
| this project (4.22 M-param MLP, 275 sims/turn, 108 turns) | 0.0084 GFLOP | 0.25 TFLOP |
| **ratio** | **~4,000x** | **~43,000x** |

(The AGZ per-evaluation figure is my estimate from its published architecture,
not a measured number; treat it as order-of-magnitude.)

So 10 M of our games is roughly **0.005%** of AlphaGo Zero's self-play compute.
Passing its 4.9 M game count is a real milestone for the training loop — there
are that many policy-improvement steps' worth of data — but it is not evidence
of comparable strength, because each of our games contains ~43,000x less search.
A game at 275 simulations spread over a ~7.3-node sub-decision chain is,
per `SEARCH.md` §3.6, about 34 turn-equivalents of lookahead. That is a shallow
game, played a great many times.

**The defensible target, revised.** `LEARNING.md` §7.3 offered: beats the
rollout-MCTS anchor in >80% of matches, and beats the warm-started supervised
net by >10 points of mean centred score. With 10–30x the compute, spent as §7
recommends, I would raise that to:

> **Beats the rollout-MCTS anchor in >95% of four-player matches; beats the
> warm-started supervised net by >25 points of mean centred score; never loses a
> match to `sample_legal_move`; and holds up in 1-vs-3 seatings, not just 2-vs-2.**

That last clause is the one that matters and the one compute does not buy. An
agent that has only learned to play against copies of itself looks fine in
2-vs-2 and falls apart at 1-vs-3 (`LEARNING.md` §6.8). Measure it.

What is *not* on the table, at any amount of free compute: near-optimal play,
confident long-horizon planning, or any claim about superhuman strength — which
in a four-player non-transitive game is not even cleanly defined.

---

## 7. Spend the surplus on depth, not on games

This is the most actionable consequence of §5 and it changes the training loop.

At 44,000–150,000 games/h, a 30-minute generation produces 22,000–75,000 games.
At `LEARNING.md` §6.3's write policy — records only from the 25% of turns that
get the full playout budget, ~5 qualifying nodes each — that is **3.0–10 M
records per generation, against §6.6's total buffer cap of 1.5 M.** One
generation would overflow the entire 20-generation window by 2–7x. The loop as
specified would be generating data at several times the rate it can absorb it,
and throwing the excess away.

Three ways to spend the surplus, best first:

**1. Raise simulations per turn.** This is where the money should go. Take the
full budget from 800 to 3,200 and the reduced budget from 100 to 400, and games
per hour falls 4x while every policy target gets four times the search behind it.
`SEARCH.md` §3.6's central complaint — that factoring buys `S/8` turns of
lookahead instead of `S` — is answered by raising `S`, and this is the only item
on the list that answers it. At 4-player non-transitivity with 27-round credit
assignment, **target quality is the binding constraint, not sample count.**
`LEARNING.md` §7.1 already names the reverse trade ("dropping 800 to 600 takes
games/hour from 4,700 to 6,400… worth taking early, and worth reversing late").
The free compute lets you reverse it much earlier and much further.

**2. Raise the buffer and lengthen the window.** 1.5 M records at `record.rs`'s
512 bytes is **768 MB** on a 24 GB machine — very conservative. 8–16 M records
(4.1–8.2 GB) is comfortable and gives the value head a longer, more diverse
window. `record.rs` chose fixed 512-byte records specifically so a shard is a
`numpy.memmap` with a structured dtype, so growing the window costs nothing in
loader complexity.

**3. Only then, more games.** After the first two, whatever is left.

`TRAINING.md` landed while this was being written and already exposes all three
as flags, so none of this needs new code: `--window` (defaulting to the same
1.5 M records), `--steps`, and `selfplay --snapshot` for the generation length.
Raise `--window` first; it is the cheapest of the three and the one the new
rates most obviously invalidate.

A reasonable concrete split: **~4x the search depth, ~4x the buffer, and ~3–6x
the games** relative to `LEARNING.md` §7. That lands at roughly 15,000–35,000
games/h across both machines, **2–4.5 M games per week**, each game with four
times the search — which I would take over 19 M shallow ones without
hesitating.

Two knock-on effects to plan for:

- **Shorten generations to 10–15 minutes** rather than 30. The point of a
  generation is a policy-improvement step; at these rates 30 minutes of data is
  far more than one step needs, and shorter cycles mean fresher weights.
- **Disk.** Even at the reduced rates, budget **25–50 GB per day** of replay
  data and a retention policy that deletes shards older than the training
  window. Nothing in `LEARNING.md` §6 or `record.rs` currently deletes anything.

---

## 8. What this changes in the other two documents

Small, specific list, so the corrections do not get lost:

| document | claim | correction |
|---|---|---|
| `LEARNING.md` §0(a) | "Move generation is the bottleneck, not the network" | **False under `SEARCH.md`'s factored tree.** Measured: ~3 µs of search per leaf against 9 µs of network at batch 256, and 182 µs at batch 1. §1.3 |
| `LEARNING.md` §4.4 | "~65,000 evals/s" for the whole machine | **5.4x low.** Measured 350,698 evals/s at batch 128, 8 threads. §1.3 |
| `LEARNING.md` §4.4 | single-GEMM table (1,109 GFLOP/s at 256³, 3.9 µs floor) | Measured 1,626 GFLOP/s and 1.34 µs with `VECLIB_MAXIMUM_THREADS=1`. §1.4 |
| `LEARNING.md` §4.4 | "accumulate at least 64… target batch 128–256" | GEMM saturates at **batch 32**. Larger batches amortise fixed costs, not GEMM. §1.4 |
| `LEARNING.md` §5.4 | "only reach for the GPU if the search delivers batches of 256+" | The batch is a free parameter once games are concurrent (§2.1), so the condition is trivially met. But the Mac's *own* GPU is still not worth it (§3.1). |
| `LEARNING.md` §5.2 | `ort` rejected | Rejected for CoreML's ANE call floor, which does not apply to CUDA. **Take `ort` on Windows.** §3.2 |
| `LEARNING.md` §6.6 | buffer 1.5 M records, 20 generations, 30-min generations | Overflowed several times over per generation at the new rates. Raise to 8–16 M, shorten generations. §7 |
| `LEARNING.md` §7.1–7.2 | 3,000–6,000 games/h; 200 k–600 k games/week | **19,000–25,000 games/h on the Mac alone**; 6–19 M/week across both machines at current search settings. §5 |
| `LEARNING.md` §7.4 item 1 | "Buy self-play compute… nothing else on this list is close" | Off the table, and mostly unnecessary. Batching plus the Windows box recovers 10–30x for nothing. |
| `SEARCH.md` §3.5 | "target a batch of 32–128 leaves" via tree parallelism | Right number, wrong mechanism for self-play. Get it from concurrent games, not concurrent descents, and pay no virtual-loss cost. §2.5 |
| `phase.rs` | `Evaluator` has only a batch-1 `evaluate` | Not a signature problem — a batching backend behind it is fine (§2.6). But add an `evaluate_batch` default method so the trainer and arena can use `Net`'s directly instead of going through the shim at `net.rs:1471`. |
| `selfplay.rs` | `seeds.par_iter().for_each(...)` — one game per rayon task | Caps the batch at the core count (~14). Concurrency must come from the game count: 128–256 workers, batcher on a thread outside the pool. §2.6 |
| `mcts.rs` | `HashMap` with the default hasher | 0.586 µs per lookup on a 320-byte key. Use FxHash: 0.406 µs. §1.2 |
| `mcts.rs` / `phase.rs` | `Evaluation.priors: Vec<f32>` | One heap allocation per evaluation, at 350 k/s. Hand out slices of an arena in the batched path. §1.2 |

---

## 9. Risks

**The unbuilt-pipeline risk dominates everything else here.** Every throughput
number in §5 is for a batched self-play driver that does not exist yet. The
measurements underneath it are real, but the composition is not, and a
first-cut pipeline that stalls on a half-full batch or serialises on an
allocator will land at half these figures until profiled. Budget a day for that,
and instrument batch-size histograms and queue-wait times from the start —
if the mean batch size is not close to the configured maximum, nothing else in
this document is happening.

**A wrong GPU backend is worse than a slow one.** §3.2's cross-backend agreement
test is not optional. Silently-wrong replay data is unrecoverable and would not
show up as anything except an agent that stops improving.

**The rules are still moving.** `src/` gained an encoder, a tree, a network and
an MCTS *during the writing of this document*, and `MAX_GEAR_SPACES` has now
moved twice. `LEARNING.md` §8.4's advice — freeze the rules, tag the commit,
stamp `RULES_VERSION` into every record — is more important at 100,000 games/h
than at 5,000, not less.

**Thermals and machine availability.** A week-long pin at 100% on a laptop is
not a thing you will enjoy. Verify which Mac this is before planning around it.

**Non-transitivity is the one risk that gets worse with more compute.** Faster
self-play means faster cycling if the population is cycling. The anchor panel
(`LEARNING.md` §6.8) is the only instrument that detects it; run it every 10
generations as specified, and do not let the extra throughput tempt you into
skipping it.
