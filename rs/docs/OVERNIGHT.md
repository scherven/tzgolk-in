# Overnight run — **PAUSED 2026-09-08 09:40, resumes tomorrow**

The 17:50 deadline below is **void**: the user stopped the run near a weekly
usage limit and will restart it another day. No agents are running; the tick
and reaper monitors are stopped. Read `## Where this stopped` at the bottom
before doing anything.

Original plan follows.

# Overnight run — deadline **2026-09-08 17:50 EDT**, hard

Sole use of the machine (14 cores) from 2026-09-07 23:50. The deliverable at the
deadline is **the strongest agent we have, playable and legible in the TUI**:
the user wants to step through a game, see the moves it is making, and see its
plan if it has one.

If you are a fresh session picking this up: read this file, then
`docs/FINDINGS-eval.md`, then `git log --oneline -12`. Orchestration scripts are
in `<scratch>/orch/` (`status.sh` prints progress and ETAs for everything
running; `heartbeat.sh N` sleeps N minutes then prints status and exits, which
is what wakes a turn-based session). Re-create them from this description if the
scratchpad is gone.

## Decisions the user made before leaving

* **`data/replay/` (10 GB) is expendable.** The `phase.rs` sub-space factoring
  of wide `Take` nodes is authorised even though it invalidates the
  `TREE_EDGE` index space. Reordering `tree::legal_steps` is likewise allowed.
* **Strength at any cost.** No per-turn time cap. If a turn takes seconds the
  TUI must show progress rather than appear frozen.

## Standing rules

* **Log durably, immediately.** Every measurement goes into a findings file the
  moment it lands, with what varied, the baseline, the number with CI and block
  count, the output filename, and one line of meaning. Three agent runs were
  killed by usage limits today; the only work that survived was what had been
  written to disk. Assume the same will happen again.
* **Commit at every boundary**, after `cargo test --release` is green.
* **Arena runs cost CPU, not tokens.** Agent turns cost tokens and the session
  has hit usage limits repeatedly. Prefer long CPU experiments over many short
  agent turns; keep concurrent agents low.
* **User CPU, never wall clock**, for any cost claim: the same pair measured
  1.52x and 1.03x on consecutive wall-clock runs under load, while
  `/usr/bin/time` user CPU repeated to 3%.
* **Measure from a private pinned copy of the binary, never `target/release/`.**
  This is not a nicety. `cargo test` or a build by *any* other agent replaces
  the shared binary underneath a running sweep and silently moves the platform
  for both arms. It produced an impossible alternating pattern
  (av 0.0 -> +7.5, 0.2 -> +14.4, 0.3 -> +7.2, 0.4 -> +14.0) and a conclusion
  that had already reached a doc comment before it was caught and retracted
  (`FINDINGS-eval.md` F13/F14). `cp target/release/arena <mine>/arena` first,
  record the git rev beside it, and fingerprint the platform in the results so
  a mixed run is detectable rather than merely wrong.
* **One writer per output file.** Two runners appending to one JSONL produced
  deterministic duplicate rows: the mean stayed right and the interval narrowed
  by sqrt(2), which is the failure mode that looks like success (F6).
* The block is the independent unit, not the game; null win rate is 25%; never
  call an effect smaller than its interval an improvement.

## Phase plan, with a hard reserve

| phase | window | content |
| --- | --- | --- |
| A | 00:00-02:00 | land known wins: `prior_temp` 1.0, term re-pricing, `Priors::Gradient` as the prior. Establish the reigning best spec. |
| B | 02:00-08:00 | MCTS parameter sweeps on top of a real prior: `c_puct`, `fpu_reduction`, `pmin`, and the sims curve, which only becomes live once the prior is not flat. Throughput: `Choice` sort + `dominated_dedup` are ~30% of runtime. |
| C | 08:00-12:00 | structural: `phase.rs` sub-space factoring; plan priors; whatever the agents propose. |
| D | 12:00-15:30 | **the headline race** — best greedy vs best MCTS at high block count, plus the full ladder. |
| E | 15:30-17:50 | **RESERVED, do not spend.** TUI work, final verification, the write-up. |

Phase E is not optional. The deliverable is a thing the user can *see*, not a
number in a log.

## The reigning champion

Updated whenever something beats it. Always give the spec, not a description.

| when | spec | beats | by |
| --- | --- | --- | --- |
| start | `mcts:2048:heuristic:quality` | `mcts:2048` (old defaults) | +8.84 [+7.24, +10.44], 202 blk |
| 00:25 | **`mcts:2048`** — quality prior shipped as the default in `41e9d70`, so the bare spec *is* the champion | `heuristic:full` | **+11.12** [+8.88, +13.36], 100 blk, win 0.403 |
| 01:10 | same spec, but `94d85f3` re-priced the evaluator under *both* sides (+21.09 greedy / +7.63 mcts:1024 against the evaluator it replaced) | — | the +11.12 went stale exactly as predicted: the same spec now measures **+3.5** against `heuristic:full` |
| 04:30 | **`mcts:8192:heuristic:cp=0.02`** | `mcts:2048:heuristic:quality` | **+6.5** centred, 44% of games against three copies (null 25%); **+11.9** against `heuristic:full` |

### `c_puct` was pinning the search to one turn of lookahead

`c_puct_init` shipped at 2.0 and wants ~0.02 at 8,192 simulations. At 2.0 the
descent stopped after ~8 sub-decisions — **one turn** — so every extra
simulation was re-deciding what the one-ply prior had already decided. At 0.02
the mean descent is ~52 sub-decisions, about six turns, and the budget starts
paying: 8,192 sims are worth +0.62 at `cp=2.0` and **+6.5** at `cp=0.02`.

**This is why the simulation axis kept measuring flat.** It was not that
lookahead is worthless in this game; it was that the search could not perform
any. The two knobs are one knob — the best `c_puct` falls as the budget rises
(0.06 at 2,048, 0.02 at 8,192) — which is why every 1-D sweep before this found
"about a point" whichever constant it moved. It is also *cheaper* than the width
it beats: 146 ms/turn against `mcts:16384:quality`'s 187 ms.

### The prior is worth more than everything else measured so far

Same race, same 100 blocks, same baseline — only the prior differs:

| MCTS prior | vs `heuristic:full` | 95% CI | win rate (null 0.250) |
| --- | --- | --- | --- |
| uniform (the old default) | +4.15 | [+2.22, +6.08] | 0.279 |
| one-ply at `prior_temp = 1` | **+11.12** | [+8.88, +13.36] | **0.403** |

Nearly tripling the margin over greedy, from one default. It also corroborates
the +8.84 measured internally against a different baseline, by a different
route, which is the kind of agreement worth more than either number alone.

## Known, measured, not yet landed

* `prior_temp` default is **4.0**; 1.0 measures **+8.84** where 4.0 measures
  +1.01 (202 blocks). Left at 4.0 only because an experiment was running.
* `board=0.5` is **+10.48 greedy / +9.04 MCTS** (150 blk) — largest single
  effect measured. Corroborated independently by the plan workstream's refit
  (+11.85 from board alone, +8.71 engine, **-3.86 temple**).
* A gradient softmax as the *prior* (not just the truncation key) is **+3.21**
  over uniform (240 blk) and would be near-free through `Mcts::gradient`.
* The simulation budget is dead under a flat prior (16x = -1.08, interval
  excludes anything above +0.64) and worth **+3.04 for 8x** once it is real.
* `Choice` sort/compare/eq 24.7% + `options::dominated_dedup` 4.9% of MCTS
  runtime. Biggest throughput lever; no owner.
* `docs/TRAINING.md:552-554` describes truncation behaviour that is now wrong
  twice over.

## Runs cut for the deadline

The `status.sh` ETA column exists to be acted on. Killed at 02:20, with their
partial JSONL left in place so `--resume` can finish them if the picture
changes:

| run | state when cut | why |
| --- | --- | --- |
| `mcts:32768:cp=0.01` | 2/600 blocks after 12 min | ETA 3,533 min against 930 remaining |
| `mcts:65536:cp=0.02` | 0/600 after 2.5 min | a single block had not completed |
| `mcts:2720` vs `2048` | 5/3,500 after 1.4 min | 3,500 blocks is a 979-min target |

Load was 74 on 14 cores, so these were not merely doomed, they were taking
cores from runs that can finish. **The very high sims arms are the ones to cut
first**: cost scales with the budget while the measured return on it is small
even now that the axis is live, so they are the worst ratio on the machine.

## Deliverable check — done at 05:55, 11h56m out

`uidump 7 9 132 44 --agent mcts:8192:heuristic:cp=0.02` renders, and the Moves
panel shows *the search's own ranking* rather than a one-ply fallback. So the
path the user asked for works end to end with the new champion. Two things to
fix in phase E:

* The shortlist is dominated by **restatements of one retrieval** — `w0[+3
  corn] w1[-3 corn, G+1]` then the same with `w2[skip]`, `w3[skip]`, and both,
  in each order. These are genuinely distinct moves (picking a worker up with
  `skip` still returns it to hand), so `Move::same_effect` is right to keep
  them and this is a *display* problem, not a correctness one. The panel should
  collapse trailing no-op pickups, or the ranking should show one row per
  distinct outcome.
* Scores run 55.7 then 5.5, 4.7 — a 10x gap to the runner-up. Worth checking
  the units are what the panel claims before the user reads them.

## Freeze schedule — set 06:50

The final numbers must be measured on the code that ships, so changes stop
before the races that produce them.

| time | gate |
| --- | --- |
| **13:30** | **feature freeze.** No further `eval.rs` / `mcts.rs` landings. Agents finish what is measuring and report. |
| 13:30-15:30 | the headline races on the frozen build: champion vs `heuristic:full`, champion vs the old champion, and the ladder. One pinned binary, high block count. |
| 15:30-17:50 | TUI, verification, write-up. Reserved; do not spend. |

A landing after 13:30 invalidates every race started before it, and there is
not time to re-run them.

## Coupling worth knowing

`src/mcts.rs:1573` calls `eval::heuristic` **directly** to order edges, so the
evaluator is not only MCTS's leaf value -- it is also its prior. An `eval.rs`
change therefore moves MCTS on two axes at once, and any measurement of one
that rebuilds the binary is measuring the other too.

`plan::PlanEvaluator::raw` weights `eval::components`, so the evaluator
rescales compose *multiplicatively* with its fit. After `94d85f3` its board
weight is effectively shrunk ~4x, past the far edge of the measured plateau.
**`plan.rs` needs refitting before its numbers mean anything again**, and F8
makes the falsifiable prediction that its refitted temple weight comes back
positive.

## Open

* ~~Is MCTS actually stronger than `heuristic:full`?~~ **Answered.** With the
  *old* uniform-prior default, `mcts:2048` beats `heuristic:full` by **+4.15**
  [+2.22, +6.08], 100 blocks, p < 0.0001, win rate 0.279 against a null of
  0.250. Real but not a landslide -- and that is the version whose prior was
  flat. The quality-prior rerun is at `<scratch>/vs_greedy/quality.jsonl`.
* Is `temple_outlook` negative because of a weight or a bug?


## Where this stopped

**State: green.** `cargo build --release` clean, `cargo test --release`
**181 passed, 0 failed**. HEAD is the commit this note is in.

**The champion is `mcts:8192:heuristic:deeper`** (`cp=0.02,pmin=2`),
+13.32 centred against `heuristic:full` at a **0.725** win rate where an equal
agent takes 0.250; +10.40 [+9.29, +11.51] over the old champion on a common
opponent, 110 paired blocks. ~181 ms/turn of user CPU.

**Unfinished, in priority order:**

1. **The TUI is half-refactored and it is the deliverable.** An agent died
   mid-change having added `App::thinking` and moved `App::agent` to `Arc` so a
   worker thread can search off the draw loop. I completed the mechanical half
   (both binaries and the test now compile and pass) but **the feature itself is
   not built**: nothing sets `thinking`, so there is still no progress display,
   and the two display bugs it was spawned for are open — the shortlist is
   mostly one retrieval restated with trailing no-op pickups, and the score
   column's units are unlabelled. `docs/FINDINGS-tui.md` has its notes.
2. ~~**`board = 0.90` is probably worth taking**~~ — **done, and the +2.45 was
   on a pre-`48c39bd` `mcts.rs`.** Re-measured on the shipped search at **600**
   blocks it is +0.48 [+0.19, +0.76] *, still free on both shallow agents, and
   it landed together with `TEMPLE_SCALE` 1.4 -> 1.2 — the pair is worth **+1.71
   [+1.34, +2.08] * on `mcts:1024:cp=0.05`** and costs `greedy:64` −0.30.
   `docs/FINDINGS-eval.md` F41-F48. **Note for anyone quoting a deep number from
   before 2026-09-09: `mcts.rs` changed by 830 lines between `6afe5fc` and
   `48c39bd`, so `--agent mcts:N` named a different search then (F39).**
3. **The headline race never ran on a frozen build.** Every number above was
   measured while `eval.rs` was still moving. Before believing the champion's
   margin, re-run it on one pinned binary.
4. `pmin=2` x budget above 8,192 is open — memory, not CPU, is the binding
   constraint there (deep trees are 300-800 MB resident and jetsam kills them).

**The one thing to carry forward above any result:** every workstream's log
(`FINDINGS-eval.md` ~93 KB, `-mcts.md`, `-generation.md`, `-tui.md`) survived
eight agent deaths by usage limit. Nothing else did.
