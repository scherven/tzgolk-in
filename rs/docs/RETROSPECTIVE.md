# Building a board game engine + AlphaZero agent: what to reuse

Notes toward a skill that takes a rules document and produces a fuzzed engine, a
TUI, a search agent and a training loop. Written from the Tzolk'in build, which
went Go port → Rust rewrite → rules audit → search/learning design → agents.

---

## 1. The one decision everything else rested on

**Make the game state `Copy + Eq + Hash` with no heap allocation, and make
effects data rather than closures.**

The Go original expressed board actions as closures (`Execute: func(*Game,
*Player)`), which is genuinely the right shape for *generation* — a space
answering "what can this player do here, right now?". The mistake was letting
the same closures do *execution*. Turning the execution half into a closed
13-variant `Effect` enum is what unlocked everything downstream, and the payoffs
kept arriving in places nobody predicted:

| Property | What it bought |
|---|---|
| `Copy`, 312 bytes | Snapshot is `let s = *state;`. Replaced a `Freeze`/`Save`/`Load` table that deep-cloned the game **609,124 times in a two-minute run** |
| `Hash` | Transposition tables — *and* the retrieval-ordering memo, which is the only reason enumerating every legal ordering is tractable at all |
| `Eq` | Dedup moves by resulting state, so equivalent orderings collapse |
| Data effects | The move log cannot lie: a choice's label is *derived* from its effects |
| Data effects | Moves are testable, comparable, serialisable as training targets |
| `Copy` again | The TUI's preview pane is a state diff — apply to a copy, compare |

Every "what if" in the codebase — probe states in generation, the double-build on
Tikal, the arena, the preview — is a two-line memcpy because of this.

**For the skill:** state this as a hard constraint in the engine brief, before
any code. Retrofitting it is a rewrite.

## 2. What went well

**Measure before optimising, and re-measure after architecture changes.** Two
measurements redirected the whole project:

- `apply_move` is **3,680× cheaper** than `legal_moves`. So engine
  micro-optimisation was worthless; the action space was the problem.
- A random rollout's *bias*, not its noise, makes it useless as an evaluator
  here: every candidate move landed within 3.8 points of the same terrible
  outcome, because random play starves. That killed classical MCTS as the
  destination and pointed at a learned value function.

And the trap: **a measurement can be invalidated by a later design change.** The
learning design concluded "this workload does not GPU-accelerate" from a
correct measurement — taken before the action space was factored. Factoring made
edge generation ~3,100× cheaper without changing per-evaluation network cost, so
the network went from a rounding error to 3–8× everything else, and batching
turned out to be worth **20.3× on CPU alone**. Estimated throughput moved from
200k–600k games/week to 6–19M.

**Invariant fuzz as the correctness gate.** N seeded games played to completion,
validating the whole state after every turn plus independently re-checking every
generated move. Cheap to write, ran 10,000 games in 23 s, and caught real bugs
on the first game. This is the single highest value-per-line artefact in the
project and should be scaffolded automatically.

**One named regression test per bug.** Naming tests after the defect
(`foresight_survives_a_full_space`) made "which of my fixes were wrong?"
answerable later, which mattered a lot — see below.

**An adversarial audit against primary sources.** A rules audit reading the
actual rulebook found **30 issues** that ordinary code review had not, including
three latent panics and a game that was a full round too short.

**Defining the shared contract before fanning out to agents.** `phase.rs` — the
`Phase`/`Step`/`Evaluator` types plus a working heuristic stand-in — was written
first, by hand, and declared off-limits. Modules were pre-declared in `lib.rs`
and dependencies pre-added so nobody contended on those files either. Three
agents then built against it in parallel without a merge conflict.

## 3. What went badly

**I "fixed" correct behaviour four times.** Temple tie-breaking, the
first-player self-pass, the gear sizes, and Tikal's two-temple action were all
*right* in the original and I broke them, each time from a plausible reading of
the code rather than of the rules. In one case I wrote a test asserting the
violation.

> **Rule for the skill:** when porting, the original implementation is evidence.
> A deviation from it requires a rulebook citation, not a plausible story. Any
> "this looks like a bug" in ported logic goes on a list to verify, not a list
> to fix.

**Audits produce confident false positives.** Two of the audit's findings were
wrong: the gear sizes (it reasoned correctly from the rulebook but the code was
already right), and a claim that the starting tiles were missing effects — the
rulebook's glossary is headed "Starting Wealth Tiles **and Building Effects**"
and is shared between the two, so the "missing" symbols belong to the buildings.
I acted on both before verifying, and built an entire choice-expansion mechanism
for tiles that turned out to be unnecessary and had to be reverted.

> **Rule for the skill:** audit findings are *hypotheses*. Before acting on one
> that contradicts working code, get a second source — ideally the human looking
> at the physical components. A five-minute question would have saved two
> rounds of churn on both of these.

**Human-input points were discovered late.** The gear count and the tile
compositions both required the physical game, and both surfaced mid-audit. They
should have been asked at the *start* of the rules phase.

**Changing shared APIs under running agents caused thrash.** I renamed a method
and changed a data type while three agents were mid-write; they adapted, then I
reverted and broke them again. Batch breaking changes, or make them before the
fan-out.

**Long agent runs are fragile.** Each ran 20–28 minutes and all three were
killed simultaneously by a session limit. Their *code* survived on disk; their
*reports* — which carry the interface seams between them — did not. Mitigation:
have agents write findings to a file as they go, not only in a final report.

## 4. The sequence that would have worked

Human checkpoints marked **[H]**.

0. **Ingest the rules document → a structured spec.** Enumerate: state
   components, the closed effect vocabulary, phases/turn structure, scoring,
   and every value that is *printed on components rather than in the rulebook*.
   **[H] Approve the spec, and answer the component questions now** — costs,
   board counts, card compositions. This is the step whose omission cost the
   most.
1. **Vocabulary and state layout.** `Effect` enum, ids, the `Copy + Hash` state.
   No game logic yet.
2. **Data tables**, with anything unverifiable explicitly flagged.
3. **Engine + invariant fuzz.** Gate: N seeded games to completion with
   per-turn validation. Do not proceed while this is red.
4. **Adversarial rules audit against primary sources** → fix → **re-audit until
   clean**. Expect two rounds and expect false positives. **[H] Adjudicate
   contradictions between audit and code.**
5. **Freeze the rules. Stamp a `RULES_VERSION`** into every future training
   record — ten lines that turn "the whole replay buffer is suspect" into "the
   records before generation N are".
6. **Action factorization + a sampler.** Decompose a turn into a chain of small
   decisions. Then `sample_legal_move` — draw one legal move without
   enumerating. This is the biggest single speedup available (192× here) and,
   more importantly, *the sampler's recursion is the action model the policy
   head will parameterise*.
7. **Equivalence test.** The set of states reachable by walking the factored
   tree must equal the set reachable by flat enumeration. States, not moves.
   This is what makes a reimplementation of legality safe.
8. **Search**, then **encoder + network**, then **self-play + arena**.
9. **TUI** — any time after step 3, and useful early for eyeballing the engine.

**[H] decisions that recur:** the value target (score-relative vs win
probability — depends on the data budget, not on taste); compute budget and
hardware; whether to accept a strength ceiling; anything the rulebook leaves
genuinely ambiguous.

## 5. How much of this is actually generic?

The proposal is a set of crates a new game imports, supplying only its rules.
That is **mostly** right, with one important caveat.

### Genuinely game-independent

```
bg-core     State/Phase/Step/Evaluator traits, PlayerId, Transition, RULES_VERSION
bg-fuzz     the seeded invariant-fuzz driver, generic over State + a Validate fn
bg-mcts     PUCT, progressive widening, n-player maxⁿ backup, transposition table
bg-net      Gemm abstraction (portable fallback + Accelerate/CUDA), layers,
            versioned weight format, batched inference
bg-train    self-play driver, replay buffer, checkpointing, the arena with
            seat-rotation and confidence intervals, PyTorch-side glue
```

`bg-mcts` needs almost nothing game-specific: given `State: Copy + Eq + Hash`,
`legal_steps`, `apply_step`, `is_over` and `scores`, the search is the same
search. `bg-fuzz` and `bg-train` likewise. `bg-net` cares only about tensor
shapes, so it parameterises over the encoder's output width and the head
structure.

A sketch of the boundary:

```rust
pub trait Game: Sized + 'static {
    type State: Copy + Eq + Hash + Send + Sync;
    type Phase: Copy + Eq + Hash + Send + Sync;
    type Step:  Clone + Eq + Hash + Ord + Send + Sync;

    const N_PLAYERS: usize;
    const RULES_VERSION: u32;

    fn legal_steps(s: &Self::State, ph: Self::Phase, turn: PlayerId) -> Vec<Self::Step>;
    fn apply_step(s: &mut Self::State, ph: Self::Phase, turn: PlayerId, st: &Self::Step)
        -> Transition<Self::Phase>;
    fn is_over(s: &Self::State) -> bool;
    fn scores(s: &Self::State) -> Vec<f32>;
    fn validate(s: &Self::State) -> Result<(), String>;   // for bg-fuzz
}
```

### Game-specific, and irreducibly so

- **The rules engine.** Obviously.
- **The phase factorization.** This is the part that cannot be mechanised. The
  decomposition of a turn into a decision chain is where most of the search
  design work lives, it depends on the shape of the game's action space, and it
  is what took the widest node here from ~1.9M edges to ~9,000. A rules document
  does not tell you this; deriving it is judgement.
- **The encoder.** Layout is game-specific, though the *method* transfers well:
  rotate to the current player's perspective; share weights over exchangeable
  axes (seats, card slots) rather than over spatial ones; flat rather than
  convolutional unless the board genuinely has translation invariance; and
  **never encode hidden information** — the state struct here holds the shuffled
  undrawn deck, so a net fed the raw struct would be reading the future.
- **The heuristic stand-in**, needed before a net exists.
- **The invariants.** Conservation laws are per-game.
- **The TUI.** Rendering is inherently bespoke, though the *frame* — board pane,
  player table, move list, preview-by-diff, headless render test — is a reusable
  layout.

### The honest summary

"All a new game supplies is the rules engine" is optimistic by roughly a factor
of two: it also supplies the factorization, the encoder, the heuristic, the
invariants and the UI. But the genuinely hard, genuinely reusable
infrastructure — MCTS, batched inference, self-play, replay, the arena's
statistics, the fuzz driver — is perhaps 70% of the line count and close to 100%
of the fiddly correctness risk. Extracting it is worth doing.

**Two things to build into the framework from the start**, both learned here the
expensive way: a `RULES_VERSION` stamp on every record, and an evaluator
interface that is **batched by construction** (`evaluate_batch(&[...])`), so
nobody accidentally ships a per-position call and leaves 20× on the table.

## 6. Numbers, for calibration

| | |
|---|---|
| Go original | 4,462 lines, 0 tests, never completed one game in 2 min |
| Rust engine at rules-freeze | ~4,400 lines, 49 rules tests |
| Rules issues found by audit | 30, plus 12 in the second pass |
| Of those, introduced by my own "fixes" | 4 |
| Invariant fuzz | 10,000 games, 0 failures, 23 s |
| `legal_moves` vs `apply_move` | 3,680× |
| Sampling vs enumerate-then-pick | 192–324× |
| Flat vs factored worst-case node | ~1.9M → ~9,000 edges |
| Batching (CPU only, no GPU) | 20.3× |
| Revised self-play estimate | 200k–600k → 6–19M games/week |
