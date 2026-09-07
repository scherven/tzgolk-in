# Agents in flight

**Status 2026-09-07:** all four were killed by a session limit; agent 4 had
already reported. Agents 1-3 were relaunched with review-and-finish briefs
carrying `docs/COMPUTE.md`'s findings. If they die again, reuse those briefs.

Four agents were running when this was written. Subagents do not survive a
session restart, so this records what each owns, what has already landed, and
what a relaunch would need to say.

**Their code is on disk and passing tests.** A restart loses their final reports
— the synthesis of what they built, what they deviated from, and what they need
from outside their files — not the work itself. So a relaunch is a *review and
finish* brief, not a rebuild.

Before relaunching any of them: read the file they own, run `cargo test
--release`, and check the git diff. They may have finished.

---

## 1. Factored tree + MCTS

**Owns** `src/tree.rs` (584), `src/mcts.rs` (726), `tests/tree.rs` (832) —
19 tests passing, 2 ignored.

**Spec** `docs/SEARCH.md` §2.7 (node types), §2.6 (progressive widening),
§3 (MCTS core), §4 (four-player backup), §6 (interface).

**Landed** `legal_steps` / `apply_step` over `Phase`/`Step`; PUCT; maxⁿ backup.

**Outstanding on relaunch**
- Confirm the §2.8 equivalence test is real and thorough: the set of
  `GameState`s reachable by walking every path from `Beg` to a commit edge must
  equal the set reachable via `legal_moves` + `apply_move`. **States, not
  moves** — the ordering memo means one state can back several move objects.
  Ask what it found; a divergence is the single most important result.
- Why are 2 tests ignored?
- Measured node widths, versus the ~9,000 worst case SEARCH.md predicts.
- Whether tree parallelism (virtual loss) actually works or is only scaffolded.

## 2. Encoder + network inference

**Owns** `src/encode.rs` (1,127), `src/net.rs` (1,682), `tests/encode.rs` (740)
— 18 tests passing, 1 ignored.

**Spec** `docs/LEARNING.md` (encoding, policy head, value head, size, stack).

**Landed** flat encoder, the pointer head for `Take`, a `Gemm` abstraction.

**Outstanding on relaunch**
- The exact on-disk weight format, in enough detail for the training side to
  write files it can read. This is the seam between agents 2 and 3.
- Measured throughput, batch-1 and batched, and on what machine.
- Confirm the encoder never reads `Deck::ids` — `GameState` holds the shuffled
  *undrawn* deck, so a net fed the raw struct reads the future. There should be
  a test that fails if someone wires it in.
- What a CUDA backend would need to touch (see agent 4).

## 3. Self-play + arena benchmark

**Owns** `src/bin/selfplay.rs` (430), `src/bin/arena.rs` (640),
`src/record.rs` (1,783), `train/`, `docs/TRAINING.md` (635) — 8 tests each in
the two binaries.

**Landed** all of the above exist and compile.

**Outstanding on relaunch**
- The record format, precisely — agent 2 must interoperate with it.
- The arena's statistical design: four-player seating scheme, and what its win
  rate actually means. An evaluation with no error bar cannot decide whether
  generation 40 beats generation 39.
- Confirm every record carries `tzolkin::RULES_VERSION`.
- How to run one generation end to end.

## 4. Compute on free hardware — **DONE, do not relaunch**

**Owns** `docs/COMPUTE.md` (1,030). No source files. Reported in full; read the
doc rather than relaunching.

Headline: LEARNING.md's "this workload does not GPU-accelerate" was **wrong**,
and wrong because it predated SEARCH.md's factoring. Factoring made edge
generation ~3,100x cheaper while leaving per-evaluation network cost unchanged,
so the network is now 3-8x everything else. Batching alone is worth **20.3x on
CPU with no GPU**. Revised estimate: **6-19 M games/week** across the two
machines, against LEARNING.md's 200k-600k. Still not strong play, and the doc
says so.

Two actionable defects it found in other agents' in-flight code:
- `net.rs:1471` — `Net::evaluate_batch` exists but the `Evaluator` boundary
  calls it with a slice of **one**.
- `selfplay.rs` — `seeds.par_iter()` gives one game per rayon task, capping the
  achievable batch at the core count (~14).

**Old brief, kept only for provenance**

**Landed** the doc, plus a deep-research pass on Colab free tier that already
reported: T4, **2 vCPU**, ~12.7 GB RAM, ~90-minute idle kill where "idle" means
no cell currently executing with unacknowledged output (GPU utilisation and
subprocesses do not count), 12 h official ceiling, **background execution is
paid-only**, and Colab's own FAQ names long unattended CPU-pegged loops and
"distributed computing workers" as free-tier termination triggers, with a
documented 2026 pattern of automated blocks hitting legitimate academic runs.
Conclusion so far: unattended multi-day self-play on Colab free is a policy
problem, not a throughput one.

**Outstanding on relaunch**
- The central question: LEARNING.md claims this workload does not
  GPU-accelerate because MCTS is latency-bound at batch 1. Batching leaf
  evaluations across many concurrent self-play games turns it throughput-bound.
  At what batch size does the user's Windows GPU beat the Mac's CPU, and by how
  much?
- What batching costs in *search quality* — virtual loss and delayed evaluation
  change the search.
- Whether self-play must become an async pipeline rather than a synchronous
  `Mcts::search` call, and how invasive that is.
- A revised games-per-week estimate, and whether it reaches strong play. If not,
  say so.

---

## Context a relaunch needs

- Rules are **frozen** after the current audit round. `RULES_VERSION` is stamped
  into every self-play record so a later change invalidates only the records
  before it.
- No money is being spent. Apple Silicon Mac, plus a Windows machine with a
  GPU.
- `src/phase.rs` is the shared contract — `Phase`, `Step`, `Evaluation`,
  `Evaluator`, and `HeuristicEvaluator` as a working stand-in. **No agent may
  modify it**, nor `src/lib.rs` (modules pre-declared) or `Cargo.toml` (deps
  pre-added: rand, smallvec, ratatui, crossterm, serde, serde_json, rayon).
- Agents must not edit each other's files. Anything they need elsewhere goes in
  their report.

---

## Outstanding cross-file work (2026-09-07)

Agents 1 (tree/mcts) and 2 (encode/net) have reported and are done. Agent 3
(selfplay/arena/train) is still running. What follows is work that crosses file
ownership and so was reported rather than fixed.

### Done since their reports

- `phase.rs` grew `Query { state, phase, turn, edges: &[Step] }`,
  `Evaluator::evaluate_many` (the method to implement) and `evaluate_edges`,
  all additive with defaults. This was the trait-level blocker behind
  COMPUTE.md's batch-of-one finding — the cause was upstream of `net.rs`.
- `mcts.rs:563` now calls `evaluate_edges` with the real edge list. Previously
  it passed a count, so the evaluator re-derived the list and matched by length
  alone: **measured to abstain to a uniform prior on 15% of real nodes**, and it
  would have scrambled the policy silently on any right-length list in the wrong
  order.

### Closed 2026-09-07

- **Tree reuse** (`mcts::retain_subtree`). `search_at` used to clear the arena
  on every call, discarding the subtree under the edge just played -- including
  the node about to be searched. It now retains that subtree, renumbering it and
  rebuilding the index, moving nodes rather than cloning them (an arena entry
  owns a `Box<[Edge]>`).

  Deliberately **only within a turn**: §3.7 noises every node of the root
  player's turn, so a node carrying `in_root_turn` was noised under this turn
  and its priors are the right ones; a node from outside it was not, and reusing
  it would search un-noised priors as though they had been noised. Cross-turn
  reuse is a further win and needs the noise question answered first.

  `TZOLKIN_NO_TREE_REUSE=1` switches it off, which is how it was measured and
  how a reuse bug can be told from a search bug.

- **The encoder seam — the one that mattered.** `train/features.py` was an
  independent numpy reimplementation of `src/encode.rs`. Two encoders is a
  train/inference skew waiting to happen and an invisible one: nothing crashes,
  the network just learns against features it is never served. Closed by
  exporting the real encoder over a C ABI (`src/ffi.rs`, `crate-type` now
  includes `cdylib`) and having `features.py` call it. A precomputed feature
  file would not have worked — the 4x perspective augmentation needs
  `encode(state, p, phase)` for all four `p`. Verified against real records:
  1,869 rows, all finite, all distinct, perspective rotation confirmed.
- **`record.rs` is now a proper lib module**, not `#[path]`-included into two
  binaries. It was being compiled twice; `src/ffi.rs` needs it anyway. Test
  count went 124 -> 111 for exactly that reason: its 13 tests ran twice before.
- **`FxHashMap` for the transposition index** (`mcts.rs`). The key is a whole
  320-byte `GameState`.
- **Shard retention**: `ShardWriter::prune` plus `selfplay --keep-gb N`, run
  after each seal. `latest` is never deleted even if it alone exceeds budget.
- **`train/model.py` rewritten against the manifest**; `train/arch.py` holds the
  torch-free shapes; `train/check_manifest.py --check` reports **0 mismatches**.

### Still open

1. ~~**`train/` does not produce a loadable checkpoint.**~~ **FIXED
   2026-09-07.** `train/model.py` was rewritten against the manifest, the
   architecture constants were split into a torch-free `train/arch.py`, and
   `train/check_manifest.py --check` now diffs the Python shapes against the
   Rust ones — **0 mismatches**, 82 tensors / 4,332,376 params (MAIN) and
   70 / 1,913,016 (SMALL). The `export.py` dot-to-slash rename is gone. The
   remaining verification is the real round-trip through `Net::load`, which
   needs torch installed; `train/selftest.py` is the vehicle.

   The seven original mismatches, for the record:
   `export.py:52`'s `k.replace(".", "/")` rename; the single-`Linear` stem where
   the factored `stem.board/player/bslot/mslot/global` + `fuse` + `fuse.norm` is
   required; `Block` having one LayerNorm and biased `fc1`/`fc2` where two norms
   and unbiased projections are required; a missing `value.fc` and a missing
   `value.decomp` head; the policy head named `placing` rather than `place`;
   `q_fast`/`q_deep` declared `bias=False` where biases are required, `key` as an
   `nn.Sequential` rather than `ptr.key1`/`ptr.key2`, and no `cell_emb`/`step_emb`;
   and `F.gelu` defaulting to the erf form where `approximate='tanh'` is required.
   Agent 3's brief predates the weight format existing, so it probably has not
   fixed these.
2. **`record.rs`: `EvalRequest` carries `n_edges`, not `&[Step]`.** Anything
   still routed through `record::BatchQueue` re-derives the edge list.
   `net::BatchedEvaluator` does not; selfplay and arena already use it.
3. **Two batching evaluators still exist.** `record::BatchQueue` is the one
   `selfplay` and `arena` actually use; `net::BatchedEvaluator` has no caller
   outside `net.rs`. Deleting the unused one is ~300 lines plus its tests —
   left deliberately rather than done hastily.
5. **`phase.rs`: `Evaluation.priors` is a `Vec<f32>`** — the last heap
   allocation per evaluation. `SmallVec<[f32; 8]>` would make it zero for the
   p50 width of 2 and every fixed-arity head. **Deliberately not done while
   agent 3 was running**, since it is a breaking type change.
4. **The transposition hash.** 0.586 us through SipHash on the 320-byte key
   versus 0.406 with FxHash — comparable to a whole state encode. Needs
   `rustc-hash` in `Cargo.toml`; **held back to avoid a full rebuild under a
   running agent.**
5. **`mcts.rs` tree reuse.** `search_at` clears the arena every call, so the
   subtree containing the child just played is rebuilt across all 5-8
   sub-decisions of a turn. Agent 1 calls this the largest cheap win left.
6. **CUDA**, if wanted: a `cuda` feature plus `cudarc`/`cust` in `Cargo.toml`.
   Agent 2's write-up above `default_gemm` has the design — CUDA Graphs are a
   precondition, and the seam is `Net::evaluate_batch` with the pointer head
   left on the CPU.

### Two corrections to COMPUTE.md

Corroborated over two runs at different machine loads.

- **Batching through the trait is worth 10.4-11.0x, not 20.3x**
  (~6,400-6,550 -> 68,000-70,000 evals/s end to end).
- COMPUTE.md's throughput came from a standalone C harness modelling only the
  GEMM shapes. The real network reaches **55-60%** of it, because LayerNorm/GELU,
  the per-node priors and the ragged pointer head do not amortise the way pure
  GEMM does. **Its games/week figures should be scaled by ~0.55-0.60.**
- Two batcher threads is the right number here; one tops out at 38-41k evals/s,
  matching COMPUTE.md's own finding that GEMM is fully claimed by two threads.
  Mean realised batch 73-75 against a configured max of 128 at 256 concurrent
  games, with ~900 us of queue wait.
