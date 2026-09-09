# Does the evaluator's **per-track** research shape matter to the agent that ships?

> **Status: PREPARED, NOT RUN.** Everything below is written before any game
> has been played. Nothing in this file is a measurement; it is the design, the
> arms, the decision rule and the cost, pre-registered so that the run that
> happens next cannot be re-interpreted after the fact.
>
> **Two constraints this workstream was given and has kept.**
> 1. **`src/eval.rs` is not edited.** Not one character. Two arena races are
>    live with `eval::heuristic` on the baseline side
>    (`mcts:8192:heuristic:deeper` and `mcts:2048:heuristic:cp=0.06`, pinned
>    binaries in `<scratch>/bin3/`), plus a 20,000-game self-play and a
>    training job. The arms below exist only as **unapplied `.patch` files**,
>    and the runner applies them into a *pristine `git archive` copy* of the
>    tree, never into the working tree. See §7 for the proof-of-cleanliness log.
> 2. **No measurement run was started.** `cargo build` and `cargo test
>    --release` only. Machine at pin time: load average 71/81/90 on 14 cores;
>    `selfplay` at 536% CPU, two `arena` at 152% and 102%, `train.py` at 96%.
>
> **HEAD at pin: `5a8f9d19df1ae5d0416fe46f3fb155d40068e377`.**
> Working tree at pin: `src/eval.rs` clean (`git diff` empty).

## 0. The question, in one paragraph

`docs/FINDINGS-research-value.md` R12/R13 measured what each maxed research
track is causally worth by granting it free and never asking the evaluator:
Architecture **+23.52** (win 0.90), Theology **+15.96**, Agriculture **+7.49**,
Extraction **+4.51** — a **5.2× spread**. `eval.rs`'s `research_step_value`
prices the same four tracks at 1.55 / 1.95 / 1.95 / 2.00 — a **1.12× spread**,
and it ranks Extraction *equal to* Architecture. No setting of `RESEARCH_SCALE`
can express that, which is why F50–F57's sweep of it found no cell
distinguishably positive on any agent. R16 proposes two **scale-neutral**
changes: redistribute the four per-track sums in the measured proportions,
total held at 7.45; and delete the `lvl == 2 && horizon > 0.15 → +0.4` step at
`src/eval.rs:627`, which R18 shows is 56–61% of the evaluator's entire opinion
of a track's depth while the causal ladder puts 45–65% on level 3.

This file prepares the experiment that decides both.

## 0.1 The whole thing, in one screen

**Start it:**

```
TRACKRACE_CONC=4 /private/tmp/claude-501/-Users-simonchervenak-Documents-GitHub-tzgolk-in/\
c022ad4d-2e63-463b-bc70-9327f34a42b4/scratchpad/ts/run.sh all
```

`run.sh check` and `run.sh build` are the same script's first two phases and
play no games; `run.sh status` reads whatever is on disk through the
pre-registered rule.

| | |
| --- | --- |
| **arms** | `redist` (R16's table), `nostep2` (delete the `lvl==2` step), **`both`** (primary), **`mirror`** (the exact reflection of `redist` about HEAD — Extraction up, Architecture down), `step2x` (double the step). Plus `head`, the null. |
| **vehicle** | six unapplied `.patch` files. `mcts.rs` calls `eval::heuristic` **directly** for the default prior and the default edge ordering, so an injected `Evaluator` would move the leaf value only and miss the half of the mechanism that decides *which track*. §1. |
| **two evaluators, one process** | the arm is a property of the **seat** — a thread-local `[u8; 4]` in `eval.rs`, read by `engine_value`. Arm 0 is bit-identical to HEAD, proved by diffing two binaries' `--verify` output. §1.1, §3.1. |
| **decides at** | `mcts:2048:heuristic:cp=0.06`, 400 blocks. Confirms at `mcts:8192:heuristic:deeper`, 200 blocks, on the winner only, **before** it lands. §4. |
| **endpoints** | centred score (decided at half-width ≤ 0.40) **and** `shift` = ΔArchitecture − ΔExtraction levels (decided at ≤ 0.02). Both primary. §6.1. |
| **lands if** | score lower bound ≥ −0.10, `shift` entirely above +0.05, total levels within ±0.15, **and** the paired `both − mirror` difference entirely above 0. |
| **refutes R16 if** | `shift` is inert; or score is distinguishably negative; or `both` and `mirror` land on top of each other (F55's failure mode — that voids `both` outright). |
| **inconclusive if** | the armed seat reaches level ≥ 2 in fewer than 5% of games: the redistribution is fitted to *maxed* tracks and those games never priced them. §6.2. |
| **cost** | **17.1 s** user CPU/block at 2,048 and **122.6 s** at 8,192, measured on this machine today. Core programme (null + `both` + `mirror`) **4.3 CPU-h**; all six arms **9.1**; with the 8,192 confirmation **15.9**. §7. |
| **blend arm** | specified and deferred: `NetBatch::mix` computes the heuristic half of a blend on a **batcher thread with no arm mask**, so measuring it correctly needs `--no-batch`, which is a 20× loss and puts the arm near **250 CPU-hours**. And the sign cannot flip, only the magnitude. §5. |
| **most likely wrong answer** | the table is fitted to **maxed** tracks, the games are decided at **level 1**, and there is no causal level-1 number for Extraction or Theology at all — the two the redistribution moves furthest. §8. |
| **state** | **nothing has been run.** `<scratch>/ts/out/` is empty; `src/eval.rs` is byte-identical to HEAD. §9.3. |

## 1. The vehicle, and the two candidate vehicles that were rejected

This is the load-bearing design decision and it was settled by reading
`mcts.rs`, not by preference.

**`eval::heuristic` enters the shipping agent in three places, and only one of
them goes through the `Evaluator` trait.**

| call site | `src/mcts.rs` | reached by a custom `Evaluator`? |
| --- | --- | --- |
| leaf value | `phase::HeuristicEvaluator::evaluate` → `eval::heuristic` | **yes** |
| **prior over edges** (`Priors::OnePly`, the default) | `one_ply()` — `crate::eval::heuristic(&next, mover)` at line 2269 | **no** — hard-coded |
| **edge ordering** (`EdgeOrder::Gradient`, the default) | `Gradient::new` — `crate::eval::heuristic` at lines 953/957 | **no** — hard-coded |

`HeuristicEvaluator` returns **uniform** priors (`vec![1.0 / n_edges; n]`), so
the priors *always* come from `Priors::OnePly`, which calls `eval::heuristic`
directly. `MctsConfig::default()` sets `priors: Priors::OnePly` and `ordering:
EdgeOrder::Gradient`, and `deeper` (`cp=0.02,pmin=2`) changes neither.

**Consequence.** A measurement vehicle that injects a variant `Evaluator`
changes the *leaf value only* and leaves the prior and the edge ordering on
HEAD's track weights. Edge ordering is exactly where "which track does it pick"
is decided — `Gradient` prices `Effect::AdvanceResearch(Science::ALL[i])` by
probing `eval::heuristic` — so such a vehicle would systematically
under-measure the one thing under test. **The whole binary has to move
together, which means the change has to be in `eval.rs`, which means a patch.**

**Rejected vehicle 1 — `src/bin/evalab.rs`.** It exists for exactly this job
and holds a parameterised copy of the evaluator, but that copy is pinned at
**`6a99f2f`** and `src/eval.rs` has moved a long way since: `CORN_INCOME_PER_ROUND`
1.9 → 0.0, `ACTION_VALUE` 1.2 → 0.2, `RESEARCH_SCALE` 0.5 → **0.05**, new
`BOARD_SCALE 0.9` and `TEMPLE_SCALE 1.2`, `space_value_scaled` / `GEAR_SCALE` /
`hungry`, and the temple `ceiling` rule. `V::HEAD` no longer reproduces
`eval::heuristic` and `--eqcheck` would not read `0e0`. Re-syncing it is a
larger and more error-prone job than the patch, and it would still only move
the leaf value (above).

**Rejected vehicle 2 — a delta evaluator** (`variant = eval::heuristic + Δ`,
where Δ re-prices the research block; exact, because the block is added
linearly into `engine_value`). Cheap and unimpeachable arithmetic — and it
fails for the same reason: the prior and the ordering never see it.

### 1.1 How two evaluators live in one process: **the arm is a property of the seat**

A patched binary has one evaluator, so `arena --candidate X --baseline Y` in a
patched binary compares an agent with itself and reads 0 by construction. The
contrast has to be inside the process.

`eval::heuristic(g, p)` already takes the player. So the harness patch makes
the arm a **per-seat** property: `static ARM: [AtomicU8; 4]` in `eval.rs`, set
once per game, and `heuristic(g, p)` reads `ARM[p]`. Nothing else changes —
`mcts.rs`, `record.rs`, `arena.rs` are untouched.

This is not a hack, it is the honest model. Every call site passes the player
whose position is being estimated:

* `one_ply` scores a child with `eval::heuristic(&next, mover)` — the mover's
  arm, which is the seat actually choosing.
* `Gradient::new(g, p)` probes for the mover — the mover's arm.
* `HeuristicEvaluator` builds all four raw values and centres them — each seat
  estimated with **that seat's own** evaluator, which is what "these two agents
  are playing each other" means.

So a block is: one seed, four games, the arm seat rotating through all four
positions, three HEAD seats opposite it. Identical variance structure to
`bin/arena --mode solo`; null centred score 0, **null win rate 0.25**.

## 2. The arms

Six patches, generated from **one table** by `<scratch>/ts/gen.py`, which is the
single source of truth. Five are treatments; `head` is the table as committed.

| arm | `research_step_value` | `lvl == 2` step | what it isolates |
| --- | --- | --- | --- |
| `head` | as committed | 0.4 | the null. Bit-identical to the unpatched file. |
| **`redist`** | R16's table, verbatim | 0.4 | the per-track redistribution **alone** |
| **`nostep2`** | as committed | **deleted** | R18's step deletion **alone** |
| **`both`** | R16's table | **deleted** | **the primary arm** — R16's whole proposal |
| **`mirror`** | the reflection of `redist` about `head` | 0.4 | **the anti-shape control.** Extraction *up*, Architecture *down*. |
| `step2x` | as committed | **0.8** | anti-control for `nostep2` — double the step instead of deleting it |

### 2.1 The table

```
              Agriculture      Extraction      Architecture      Theology     total
head        0.35 0.45 0.75   0.55 0.65 0.75   0.30 0.85 0.80   0.45 0.65 0.90   7.45
redist      0.25 0.31 0.52   0.18 0.22 0.25   0.52 1.48 1.40   0.52 0.75 1.04   7.44
mirror      0.46 0.59 0.97   0.92 1.08 1.25   0.08 0.21 0.21   0.38 0.55 0.76   7.46
per-track sums:  A            R                C                T
head        1.55             1.95             1.95             2.00
redist      1.08  (0.70x)    0.65  (0.33x)    3.40  (1.75x)    2.31  (1.16x)
mirror      2.02  (1.30x)    3.25  (1.67x)    0.50  (0.26x)    1.69  (0.85x)
```

`redist` is R16's published table, unaltered. **`mirror` is its exact
reflection about `head`**: `sum_mirror[t] = 2 · sum_head[t] − sum_redist[t]`,
so it has the same total, the same `‖Δ‖`, and the opposite direction on every
track. That is a stronger anti-control than "swap two tracks" — it is the
*negative of the treatment vector*, so "the shape matters" and "any change of
this size moves the score" are separated by construction rather than by
argument. Within-track shares are kept in all three, because R18 measured them
right.

The ±0.01 in the totals (7.45 / 7.44 / 7.46) is 2-decimal rounding on R16's
published numbers. At `RESEARCH_SCALE = 0.05` and `uses ≤ 7` it is worth
**≤ 0.0035 points**. Recorded rather than fudged, because R16's table is quoted
verbatim and fidelity to the proposal is worth more than the third decimal.

### 2.2 What the arms are actually worth to the evaluator — measured, no games

`trackrace --probe` writes a research row into one seat of a fresh game and
differences `eval::heuristic` across the arms. Deterministic, game-free, and
reproducible on a loaded machine (`<scratch>/ts/logs/probe.txt`):

```
  research row (seat 0)                      head     redist   nostep2      both    mirror    step2x
  [0,0,0,0] empty                         11.1700   +0.0000   +0.0000   +0.0000   +0.0000   +0.0000
  [1,0,0,0] A1                            11.2925   -0.0350   +0.0000   -0.0350   +0.0385   +0.0000
  [0,1,0,0] R1  Extraction 1              11.3625   -0.1295   +0.0000   -0.1295   +0.1295   +0.0000
  [0,0,1,0] C1  Architecture 1            11.2750   +0.0770   +0.0000   +0.0770   -0.0770   +0.0000
  [0,0,2,0] C2  the lvl==2 step is live   11.9725   +0.2975   -0.4000   -0.1025   -0.3010   +0.4000
  [0,0,3,0] C3  Architecture maxed        12.8650   +0.5075   +0.0000   +0.5075   -0.5075   +0.0000
  [0,3,0,0] R3  Extraction maxed          12.8650   -0.4550   +0.0000   -0.4550   +0.4550   +0.0000
```

Three things fall straight out of this table, and all three shaped the design.

1. **`redist` and `mirror` are exact negatives of each other** (+0.5075 /
   −0.5075, +0.0770 / −0.0770, −0.1295 / +0.1295). The anti-control is as clean
   as it can be made.
2. **R13's headline, visible in one line of arithmetic:** at `head`, a maxed
   Architecture track and a maxed Extraction track are both worth **12.8650**.
   The evaluator cannot tell them apart *at all* — not approximately, exactly —
   while causally they are +23.52 and +4.51.
3. **The perturbation is small where the agent lives.** The biggest arm effect
   is 0.51 points on a maxed track; on the level-1 step the agent almost always
   sees it is **0.04 to 0.13 points**, against an estimate of ~11 and a final
   score of ~45. **A large score effect was never the prediction, and R16 says
   so itself.** This is why the decision rule below has a behavioural endpoint
   as a co-primary and a "does no harm" gate on score, rather than betting
   everything on centred score.

Note also that `lvl == 2` is an **equality**, not `>=`: the +0.4 is paid while a
track sits at exactly level 2 and is *taken away again* on reaching level 3.
That is R18's complaint stated at its sharpest — the evaluator charges the agent
0.4 points for finishing a track.

## 3. The runner, and how the binaries are pinned

`<scratch>/ts/run.sh`. **One command starts the whole thing:**

```
/private/tmp/claude-501/-Users-simonchervenak-Documents-GitHub-tzgolk-in/\
c022ad4d-2e63-463b-bc70-9327f34a42b4/scratchpad/ts/run.sh all
```

`TRACKRACE_CONC=3 ... run.sh all` to take fewer cores; the default is 4.
`run.sh check` / `build` / `status` do their phases alone, and neither plays a
game. `run.sh run ARM N [AGENT]` runs one arm; every arm is resumable and every
`--out` file has exactly one writer.

Per phase:

* **`check`** — `git apply --check` on all six patches; assert
  `git diff rs/src/eval.rs` is **empty before and after**; then `tables.py`, the
  three-way numeric cross-check (§3.2).
* **`build`** — `git archive <PIN> | tar -x` into `<scratch>/ts/tree`, drop in
  the one file this workstream owns (`src/bin/trackrace.rs`, untracked so the
  archive has no copy), apply `harness.patch` **inside the archive with
  `patch(1)`**, and build with a private `CARGO_TARGET_DIR` and
  `RUSTFLAGS="--cfg trackarm"`. Copy the result to `<scratch>/ts/bin/`. Also
  build an unpatched HEAD binary beside it.
* **`run`** — the pinned binary, never `target/release/`, which other
  workstreams rebuild.

**The working tree is never patched.** This is deliberately stronger than the
brief's "apply, verify, revert": there is no revert step that a usage limit can
interrupt half-way, and `git diff src/eval.rs` cannot be non-empty because
nothing ever writes to it. `git apply --check` is still run against HEAD, so
"the patch applies to the repo as it stands" is still proved.

`<scratch>/ts/bin/PROVENANCE` records, for every run: build time, the pinned
rev, rustc version, the **sha256 of each binary**, the sha256 of every patch,
and the sha256 of `trackrace.rs`.

### 3.1 The three checks the build will not proceed without

1. **`--table`** — the twelve constants compiled into the patched `eval.rs`
   must equal the ones compiled into `bin/trackrace`. Checked at runtime, in
   the binary, on every invocation.
2. **`--probe`** — `eval::heuristic` under every arm on one position. Proves
   the arms are not no-ops and prints by how much. **No games**, so it
   reproduces under any load.
3. **`--verify 6 --agent heuristic:8`**, run on *both* binaries, output diffed:
   this driver against `record::play_game` (0 disagreements), and the patched
   binary against the unpatched one score-for-score. That second diff is the
   proof that **arm 0 is bit-identical to HEAD**, which is what makes the null
   run a real platform check rather than a tautology.

**`heuristic:8`, not `heuristic:full`, and this cost a false alarm to learn.**
The first `build` failed its own self-check: `--verify 3 --agent heuristic:full`
reported one disagreement on seed 4000002. It is not a driver bug.
`Candidates::All` goes through `eval::rank_all_within` under a **2,500 ms
wall-clock** budget (`eval::FULL_BUDGET`) and falls back to sampled top-ups when
it trips; at load 128 the trip point is not reproducible. This is exactly
`FINDINGS-research-value.md` **R17.5**, re-confirmed on a second driver.
`Candidates::Sampled` draws from the passed rng and has no deadline, so it
verifies deterministically. `bin/trackrace` now prints the note itself when
`--verify` fails.

### 3.2 One table, three places, one check

The twelve constants exist in `bin/trackrace.rs` (what races), in
`harness.patch` (what the racing binary is built from), and in each
`arm-*.patch` (**what would actually land**). A number right in the first two
and wrong in the third means the experiment measured one thing and the repo
shipped another, and nothing downstream would catch it. `tables.py` reconstructs
each arm's table by replaying the landing patch's `+` lines onto HEAD's and
diffs all three; `run.sh check` fails on any mismatch. It passes now.

## 4. Which budget decides, and why

**`mcts:2048:heuristic:cp=0.06` decides. `mcts:8192:heuristic:deeper` confirms
the winner before it lands.**

The trap is real and this file does not pretend otherwise: `board 0.9 /
temple 1.1` read **−0.71 greedy** and **+1.70 deep**, and F34g's rule — a knob
whose whole effect is on the one-ply agent is a knob the shipping agent does not
want — was written because of it. So the deciding agent must be a *searching*
agent. The question is only which one.

1. **The greedy/deep disagreement is between one ply and a search, not between
   two search depths.** F57 swept `RESEARCH_SCALE` on three agents:
   `greedy:64`'s curve is twice as steep as `mcts:1024:cp=0.05`'s at every rung,
   and `mcts:256` and `mcts:1024` agree with each other. F56a is the same story
   for the Tikal space values: a `greedy:64`-only effect that is free on
   `mcts:256` **and** free on `mcts:1024`. Nothing in `FINDINGS-eval.md` shows a
   sign flip *between* two searching agents.
2. **Cost, measured on this machine, this binary, today** (§7): 17.1 s user CPU
   per block at 2,048, **122.6 s** at 8,192 `deeper` — a factor of **7.2**, not
   4, because `cp=0.02` digs much deeper trees per simulation. Deciding at 8,192
   buys one seventh of the blocks for the same CPU.
3. **An under-powered deep race is how F55 happens.** F55's `uses`-shape result
   was only trustworthy because it had 800 blocks on `greedy:64`; its
   `mcts:1024` companion (F55a, ~72 blocks) is quoted as "thin". §2.2 says the
   perturbation here is ≤ 0.13 points on the positions the agent actually
   reaches, and a small effect measured at 55 blocks is indistinguishable from a
   large one measured at 55 blocks. **Seven times the blocks is worth more than
   the extra depth.**
4. **Both live races already carry 2,048 as their leading indicator** — the
   net-blend result was +5.07 at 2,048 and +1.76 at 8,192, same sign, and the
   2,048 race finished first and called it.

The confirmation run is not optional: the winning arm is re-raced at 200 blocks
of `mcts:8192:heuristic:deeper` (6.8 CPU-hours) **before** the landing patch is
applied, because that is where it ships. If the confirmation contradicts the
2,048 result the change does not land, and that contradiction is itself the most
interesting result the experiment could produce.

## 5. The net blend: **specified, and deliberately deferred**, for a mechanical reason

`ft-champ.safetensors@0.5` beat `eval::heuristic` inside the same search by
**+5.07** at 2,048 and **+1.76** at 8,192. If it ships, `eval::heuristic` is
half the leaf value and — with `pri=eval`, which both live races use on the net
side — **none** of the prior. A track-weight change would then be worth well
under half of what it is worth today.

The blend arm is fully specified: same six arms, agent
`mcts:2048:<net>@0.5:cp=0.06,pri=eval` in all four seats, armed seat rotating.
It is deferred, and the reason is not budget alone:

1. **It cannot be measured correctly with batching on.** `record::NetBatch::mix`
   computes the `eval::heuristic` half of the blend inside `evaluate_batch`,
   which runs on a **batcher thread**. A batcher thread carries no arm mask, so
   the heuristic half of the blend would be silently computed at HEAD for every
   seat, and the arm would reach only the priors and the edge ordering.
   `bin/trackrace` refuses a non-heuristic backend outright rather than measure
   that.
2. **Measuring it correctly costs 20×.** `--no-batch` puts the net on the search
   thread, where the mask is right — and `record.rs` names that path "giving up
   the 20x" (COMPUTE.md §0: 5,491 evals/s unbatched against 111,469 batched).
   From the on-disk calibration, a 2,048-sim block with **one** net seat costs
   72.6 s user CPU against this file's measured 17.1 s for a pure-heuristic
   block, so one net seat is +55.5 s; four net seats unbatched is of order
   **4,500 s a block**, or ~250 CPU-hours for 200 blocks. That is not a thing to
   queue behind a 20,000-game self-play.
3. **The sign cannot flip, only the magnitude.** At `@0.5` the heuristic
   contributes half the leaf and none of the prior, so a track-weight change is
   worth *less*, not *differently*. A positive result at 100% stays positive at
   50%; a null at 100% is a stronger null at 50%. **Deferring is safe in the
   direction that matters**, and the landing gate (§6) is written so that
   nothing lands on a marginal positive that halving would erase: it requires a
   *behavioural* effect plus "does no harm" on score, and both survive being
   halved.

The one case where deferring is not safe is if the blend ships **and** the agent
stops using `pri=eval` for the heuristic side, which would put the heuristic
back in the prior at full strength. If that happens, re-run this experiment; the
runner takes an `--agent` argument and nothing else changes.

## 6. The uptake counters, and where they come from

R16 predicts **almost no change in how much the agent researches** — every track
still costs more than it is worth — and a change in **which track it picks**. A
race that reads only final score is therefore underpowered against the thing
actually being claimed. These are logged beside score, per block, by
`bin/trackrace`:

| field | what it is | why |
| --- | --- | --- |
| `c_lv[4]` / `b_lv[4]` | final level per track, armed seat / mean of the three HEAD seats | the raw material for everything below |
| `c_lv14[4]` / `b_lv14[4]` | the same at **day 14** | F52b's `early` column: raising `RESEARCH_SCALE` put 81% of the extra research before day 14. A shape change should move *which*, not *when*. |
| `c_max[4]` / `b_max[4]` | reached level 3 on that track | the redistribution is fitted to **maxed** tracks |
| `c_any[4]`, `c_firstsum[4]` | started the track at all; sum of first-advance days over the tracks it did start | "day of first advance", with the denominator kept separate so a block with no advance cannot contribute a fabricated day |
| `c_reach2`, `c_reach3`, `b_reach2` | reached level ≥2 / ≥3 on **any** track | the gate in §6.2 — whether these games exercised the deep half of the table at all |
| `c_bld`, `b_bld` | buildings held at the end | Architecture's whole mechanism is builds (R12: a granted C3 seat builds **9.14** a game against a baseline **2.71**), so this is the cheapest independent confirmation that a shift toward C is real and not a bookkeeping artefact |

**Where they come from.** `record::play_game` returns a `GameResult` with no
final state, so `bin/trackrace` carries its own copy of the round flow — the
same copy `bin/rlab` needs, for the same reason — reading `state.research`
directly at the end and at the day-14 boundary, and watching the per-turn
research delta to date the first advance. `--verify` plays the same seeds
through both drivers and compares scores; it reads **0 disagreements in 6
seeds** on both binaries (§3.1).

Two derived block-level scalars carry the rule:

* **`shift` = (c_lv[Architecture] − b_lv[Architecture]) − (c_lv[Extraction] −
  b_lv[Extraction])**, in levels per player-game. This is R16's directional
  prediction reduced to one number: the agent should move what little research
  it does *from* Extraction (benefit/price 0.29) *toward* Architecture (0.81).
  Null 0; `mirror` must drive it the other way.
* **`tot` = Σ c_lv − Σ b_lv**, total levels. R16 predicts ≈ 0. If it moves, the
  arm is not scale-neutral in practice and the result is confounded with the
  `RESEARCH_SCALE` axis that F54 and F57 already swept and rejected.

## 6.1 The pre-registered decision rule

Written **before any data existed**, and executable: it lives in `RULE` at the
top of `<scratch>/ts/an.py`, which prints the verdict itself so that no one has
to squint at an interval and decide afterwards what they meant.

**Unit.** The rotation block — one seed, four games, the armed seat in each of
the four positions, three HEAD seats opposite. The block, not the game.
**Null centred score 0. Null win rate 0.25.**

**Blocks.** 400 per arm at 2,048 sims (`head` 100, `step2x` 200). From the
2,048-sim on-disk precedent — N22's 100 blocks gave a 95% half-width of **0.77**
on centred score, so SD per block ≈ **3.9** — 400 blocks target a half-width of
**≈ 0.38**.

The `head` run is not only a platform check: with the mask all zeros the four
seats are the same agent, so its centred scores **measure SD per block directly**
on this binary and this machine. `an.py` recomputes the required block count
from that rather than from N22's precedent, and prints it for any arm not yet
decided. If SD comes back much above 3.9, the honest response is fewer arms at
more blocks, not more arms at the registered 400 — cut in §7.3's order.

**Endpoints.** Two, and both are primary. One is the shipping question and one
is the mechanism.

| | endpoint | decided when | null |
| --- | --- | --- | --- |
| shipping | mean centred score | 95% half-width ≤ **0.40** | 0 |
| mechanism | `shift` | 95% half-width ≤ **0.02** | 0 |

`an.py` prints, for an arm not yet decided, the number of blocks that would
decide it.

**The primary arm is `both`.** `redist` and `nostep2` are its decomposition;
`mirror` is its control; `step2x` is `nostep2`'s control.

### What LANDS

All four, together:

1. **Does no harm**: the centred-score interval's lower bound ≥ **−0.10**.
2. **Does the predicted thing**: the `shift` interval lies entirely above
   **+0.05** levels.
3. **Stays scale-neutral**: |`tot`| ≤ **0.15** levels.
4. **Beats its own anti-control**: the *paired* difference `both − mirror` on
   centred score (same seeds, matched blocks) lies entirely above 0.

If in addition the centred-score interval is entirely above 0 **and** its mean
is ≥ **+0.40**, the change lands *and* claims points. Below that it lands as a
correctness fix with a measured behavioural effect and no measured cost, which
is what R16 asked for and all it asked for.

### What REFUTES R16

Any one of:

* **`shift` inert** at half-width ≤ 0.02 — re-pricing the tracks does not change
  which track the agent picks. That is R16's mechanism failing at the first
  step, and no score result can rescue it.
* **Centred score distinguishably negative** on `both` or `redist`.
* **`both` and `mirror` on top of each other** — the paired difference's
  interval containing 0 at half-width ≤ 0.40. This is **F55's exact failure
  mode**, and it voids any positive reading of `both` alone. `an.py` prints
  "Do not land it." F55 is why the `uses` time-shape was not landed and the
  anti-shape control is why the `BOARD_SCALE`/`TEMPLE_SCALE` result was believed;
  the same discipline decides this one.

### 6.2 What is INCONCLUSIVE rather than refuting

**If the armed seat reaches level ≥ 2 on any track in fewer than 5% of games, a
null is not a refutation.** `eval.rs`'s own note says a track reaches level 3 in
**1.9% of player-games**; R16's redistribution is fitted to *maxed* tracks. If
the games never went there, the deep half of the table was never priced and the
experiment answered a narrower question than it was asked. `an.py` says so in
those words and refuses to print a refutation.

(The single calibration block in §7 came back with `c_reach2 = 0.50` and
`c_reach3 = 0.25` at 2,048 sims, and 0.50 / 0.50 at 8,192 — far above the gate.
**n = 1 block, so this is not evidence**, but it suggests uptake on the current
evaluator is higher than F52b's 1.24 levels and that the gate will pass. The
`head` null run measures it properly, at 100 blocks, before anything else runs.)

### 6.3 Looks

**One look, at the registered block count.** No stopping early on a favourable
interval. `docs/TRAINING.md` on repeated looks. If a usage limit kills the run,
whatever is on disk is reported with its actual `n` and the "decided" test
applied honestly — which is the whole reason `an.py` prints "NOT YET DECIDED"
and a required block count rather than a p-value.

## 7. Cost, and the running order

### 7.1 The calibration

Two blocks, `--concurrency 1`, on the pinned binary, on this machine at **load
128** with the self-play, both arena races and the training job running.
`/usr/bin/time -l`:

| agent | user CPU / block | peak RSS | s / game |
| --- | --- | --- | --- |
| `mcts:2048:heuristic:cp=0.06` | **17.1 s** | 386 MB | 4.3 |
| `mcts:8192:heuristic:deeper` | **122.6 s** | 672 MB | 30.7 |

`deeper` is **7.2×** the 2,048 cost, not 4× — `cp=0.02` descends much further
per simulation, exactly as `mcts.rs` reports (182 ms/turn at 8,192 `deeper`
against 18.8 for `mcts:2048:...:quality`).

Cross-checks against what is already on disk. F50 measured
`mcts:1024:cp=0.05` at **6.8 s/block** on an older rev; doubling to 2,048 gives
13.6 s against the 17.1 s measured here, and the current default carries
`pmin=3` where F50's did not — the two agree to within the difference in the
agents. The 2,048-sim net race in `FINDINGS-net.md` N22 cost **7,264.8 s user
CPU for 100 blocks** = 72.6 s/block **with one net seat of four and batching
on**; against this file's 17.1 s pure-heuristic block that puts the **net-agent
overhead at +55.5 s per block per net seat**, which is the number §5 scales
from. It is not a property of the search.

### 7.2 The bill

| # | step | blocks | agent | CPU-hours |
| --- | --- | --- | --- | --- |
| 0 | `check` + `build` + `--table` + `--probe` + `--verify` | — | — | **0.02** |
| 1 | **`head`** — the null, measured not assumed | 100 | 2,048 | **0.48** |
| 2 | **`both`** — the primary arm | 400 | 2,048 | **1.90** |
| 3 | **`mirror`** — the anti-shape control | 400 | 2,048 | **1.90** |
| 4 | `redist` — decomposition | 400 | 2,048 | **1.90** |
| 5 | `nostep2` — decomposition | 400 | 2,048 | **1.90** |
| 6 | `step2x` — `nostep2`'s anti-control | 200 | 2,048 | **0.95** |
| 7 | confirmation of the winner, **before it lands** | 200 | 8,192 | **6.81** |
| — | *blend arm, correctly measured* | *200* | *2,048 @0.5, no batch* | *~250 — deferred, §5* |

**Core (0–3): 4.3 CPU-hours.** Full 2,048 programme (0–6): **9.1**. With the
8,192 confirmation: **15.9**.

At the default `TRACKRACE_CONC=4` that is roughly 2.3 hours of wall clock for
the core and 4 hours for 0–6 *if four cores were free*. Today none are, so
expect 3–5× that, and `TRACKRACE_CONC=2` if the self-play still has 536% of the
machine.

### 7.3 The order, and the cut line

The order is the cut order read backwards: **everything below a line can be
dropped without invalidating what is above it.**

```
0  check + build              never cut — it is the proof the arms are not no-ops
1  head    100 blk            never cut — R4 and R17.5 both say the null is not zero
2  both    400 blk            never cut — the primary arm
3  mirror  400 blk            NEVER CUT — without it, step 2 is uninterpretable
--------------------------------------------------- the core stops here (4.3 CPU-h)
4  redist  400 blk            cut if short: tells you WHICH half of `both` did it
5  nostep2 400 blk            cut first among the decomposition — R18 rates it second
6  step2x  200 blk            cut before 5; it only controls 5
7  8,192   200 blk            run only on a winner, and run it BEFORE landing
```

`mirror` is above the cut line on purpose. An experiment that reports `both` and
not `mirror` has measured "a change of this magnitude moves the score", which
F55 showed is a different claim from the one being made, and it would cost the
next session the whole thing again.

Seeds start at **4,000,000** — clear of the live races (8,000,000 and 9,000,000),
`bin/rlab` (700,000) and `arena`'s default (1,000,000). Every arm plays the
*same* seed set, which is what makes `an.py`'s paired `both − mirror` comparison
legitimate and much tighter than differencing two intervals.

## 8. The single most likely way this returns a wrong answer

**The table is fitted to maxed tracks; the games are decided at level 1; and
three of the four level-1 numbers are extrapolations that were never measured.**

R13 measured four *maxed* tracks causally — Architecture +23.52, Theology
+15.96, Agriculture +7.49, Extraction +4.51 — and R16 redistributes the four
per-track **sums** in those proportions, keeping within-track shares. But the
decision the agent actually faces, in nearly every game, is *"start Architecture,
or start Extraction, or neither"*. What that decision sees is the level-1 row,
and the redistribution moves it hard: **Architecture 1 from 0.30 to 0.52 (1.7×
up), Extraction 1 from 0.55 to 0.18 (3.1× down)**.

Those two numbers are **derived from maxed-track ratios, not measured at level
1**. R13's causal depth ladders exist for exactly two tracks: Agriculture
(1.28 / 2.59 / 7.49) and Architecture (4.37 / 13.05 / 23.52). **There is no
causal level-1 number for Extraction and none for Theology at all.** And
Extraction is the track the redistribution cuts hardest, on the strength of a
maxed value (+4.51) that is *almost identical to a single level of Architecture*
(+4.37) — which is as consistent with "Extraction is front-loaded and level 1 is
most of it" as with "Extraction is worthless". One extra block per gather, by
type, is the most immediately usable of the twelve steps.

So the failure mode is specific and it is not a null-result problem: the
experiment can get the maxed proportions right, get the level-1 proportions
wrong, run games that only ever exercise level 1, and come back with
"`redist` is inert" or "`redist` costs points" — about a table that was never
wrong where it was tested. **That reading would refute the extrapolation and be
recorded as refuting R16.**

What is in place against it:

* The `c_reach2` / `c_reach3` gate (§6.2) catches the extreme version, where the
  deep half of the table is never priced at all.
* `c_any[4]` and `c_lv14[4]` make the level-1 decision rate visible per track,
  so "the agent never faced this choice" and "the agent faced it and declined"
  are distinguishable in the log rather than after the fact.
* **The decomposition arms separate the two claims.** `nostep2` carries *no
  per-track content whatsoever* — it is a pure depth-shape change. If `nostep2`
  moves and `redist` does not, the per-track extrapolation is the suspect and
  not the idea.
* And the concrete follow-up, named now so the next session does not have to
  find it: **`rlab --grant R1` and `--grant T1` were never run.** That is one
  cheap `rlab` wave at ~1 s a game — the same driver, the same seeds, the same
  `--grant` arm that produced +4.37 for `C1` — and it would replace two
  extrapolated numbers with measured ones. If this experiment comes back null on
  `redist`, run that before believing the null.

### Runners-up, in order

1. **Under-power on score.** §2.2 measures the perturbation at 0.04–0.13 points
   on the positions the agent reaches. 400 blocks brackets a true +0.2 inside a
   ±0.38 interval and it would read as inert. This is why `shift` is a
   co-primary and why the landing gate is "does no harm" rather than "wins".
2. **The opponents keep HEAD's weights.** The armed seat is measured against
   three HEAD seats, so what is measured is the *advantage of holding the new
   weights against agents that do not*. Once it ships to everyone, contention
   for the buildings and Chichen spaces that Architecture and Theology exploit
   is tighter and the number is smaller. The measurement is an **upper bound**
   on the post-landing effect. (R17.4 makes the same point about the causal
   study.)
3. **`heuristic:full`'s wall-clock non-determinism does not apply here, and it
   is worth saying why.** `eval::FULL_BUDGET`'s 2,500 ms deadline lives in
   `rank_all_within`, which only `Candidates::All` uses. Both deciding agents
   are MCTS with a fixed simulation budget and no wall-clock dependence, so the
   arms are not exposed to load the way §3.1's false alarm was. The `head` null
   run measures whatever residual there is.
4. **The four games of a block share their agent instances**, so a searching
   agent's tree arena carries across them. `bin/arena --mode solo` does exactly
   the same thing, so the arms and every arena number in the repo share the
   property; it is noted rather than fixed, because changing it would make these
   blocks incomparable with every block already on disk.

## 9. What is on disk, and the cleanliness log

### 9.1 Files

Everything lives under
`<scratch> = /private/tmp/claude-501/-Users-simonchervenak-Documents-GitHub-tzgolk-in/c022ad4d-2e63-463b-bc70-9327f34a42b4/scratchpad`.

| path | what |
| --- | --- |
| `<scratch>/ts/run.sh` | **the runner. `run.sh all` starts the whole thing.** |
| `<scratch>/ts/gen.py` | generates every patch from the one table. Re-run it if HEAD moves. |
| `<scratch>/ts/tables.py` | the three-way constant cross-check |
| `<scratch>/ts/an.py` | **the pre-registered decision rule, executable** |
| `<scratch>/ts/patches/arm-{redist,nostep2,both,mirror,step2x}.patch` | the **landing** patches, unapplied |
| `<scratch>/ts/patches/harness.patch` | the **measurement** patch, unapplied — all six arms, per-seat |
| `<scratch>/ts/bin/{trackrace-harness,trackrace-head}` | the pinned binaries |
| `<scratch>/ts/bin/PROVENANCE` | rev, rustc, sha256 of every binary and every patch |
| `<scratch>/ts/tree/` | the `git archive` build tree — **not** the repo |
| `<scratch>/ts/out/`, `logs/` | empty. **Nothing has been run.** |
| `rs/src/bin/trackrace.rs` | the one source file this workstream owns. Compiles at HEAD. |
| `rs/docs/FINDINGS-track-shape.md` | this file |

### 9.2 The two patch layers, and why there are two

* **`arm-*.patch`** are minimal and shippable: twelve literals, or the deletion
  of one `if` block. They are what lands. `arm-nostep2` and `arm-both` also
  rename `engine_value`'s `horizon` parameter to `_horizon`, because deleting
  the block leaves it unused; a landing version should drop the parameter
  outright, which is a two-line follow-up and was left out here to keep the
  patch to one hunk and zero conflict surface against other in-flight `eval.rs`
  work.
* **`harness.patch`** is measurement-only and says so in its own comment. It
  adds the per-seat arm machinery (§1.1), a `[[[f32;3];4];6]` table, and one
  thread-local read inside `engine_value`. **Arm 0 returns through the original
  `research_step_value` and the original `0.4`, so the null is the unpatched
  file**, and `run.sh build` proves it by diffing the two binaries' `--verify`
  output seed by seed.

The thread-local read costs a few nanoseconds per `heuristic` call. It is paid
identically by every arm *and by the null*, so it cannot bias a comparison; and
the landing patches carry none of it.

### 9.3 Cleanliness — stated explicitly, as asked

```
$ git status --porcelain
 M rs/docs/FINDINGS-net.md      <- another agent's, untouched by this workstream
 M rs/src/bin/valprobe.rs       <- ditto (off limits)
 M rs/src/ffi.rs                <- ditto (off limits)
 M rs/train/{arch,features,model,selftest,train}.py   <- ditto (off limits)
?? rs/docs/FINDINGS-track-shape.md    <- this file
?? rs/src/bin/trackrace.rs            <- the one source file this workstream owns

$ git diff -- rs/src/eval.rs | wc -l
0

$ shasum -a 256 rs/src/eval.rs
13b70907bd393d07b77660a4acf3463519b308e77dea561dcebdb4ef6cb93320
$ git show HEAD:rs/src/eval.rs | shasum -a 256
13b70907bd393d07b77660a4acf3463519b308e77dea561dcebdb4ef6cb93320
```

**`src/eval.rs` is byte-identical to HEAD. It was never written to at any point
— the patches were produced by diffing two strings in memory, and the only
place any of them is ever applied is inside `<scratch>/ts/tree`, a `git
archive` copy.** `git apply --check` was run against the repo on all six and
`git diff rs/src/eval.rs` was checked empty before and after; `run.sh check`
re-runs both and refuses to continue if either fails.

Nothing under `src/ui.rs`, `src/bin/tui.rs`, `src/bin/uidump.rs`, `src/net.rs`,
`src/encode.rs`, `src/ffi.rs`, `src/bin/valprobe.rs` or `train/` was read for
anything but reference and none was modified.

### 9.4 No measurement run was started

`ps` at pin time: `selfplay` 536% CPU, `arena` 152% and 102%, `train.py` 96%,
load average 71/81/90 rising to 128 during the build. Nothing in this workstream
touched `arena`, `selfplay` or training.

What *was* run, and nothing else: `cargo build`, `cargo test --release` (into a
private `CARGO_TARGET_DIR`, so the shared `rs/target/` was not written),
`trackrace --table`, `trackrace --probe` (**no games**), `trackrace --verify 6
--agent heuristic:8` twice (12 games, ~30 s CPU, a driver-correctness check),
and **two single calibration blocks** (4 games each, `--concurrency 1`, 140 s
CPU total) whose only purpose was the cost table in §7.1 and whose output was
deleted. `<scratch>/ts/out/` is empty.

### 9.5 Tree state at handover

```
$ cargo test --release          (CARGO_TARGET_DIR=<scratch>/ts/target, so rs/target/ was not written)
TOTAL: 200 passed, 0 failed, 7 ignored

$ <scratch>/ts/run.sh check
  arm-both.patch         applies clean
  arm-mirror.patch       applies clean
  arm-nostep2.patch      applies clean
  arm-redist.patch       applies clean
  arm-step2x.patch       applies clean
  harness.patch          applies clean
  git diff rs/src/eval.rs after all checks : EMPTY
  table cross-check: OK

$ <scratch>/ts/run.sh build
  # table check: OK — identical to the table compiled into eval.rs
  --verify: 6 seeds, 0 disagreements
  arm 0 == HEAD: the two binaries produce identical scores on every seed.
```

**`<scratch>/ts/bin/PROVENANCE` is the authority on which binary produced which
file.** Re-run `run.sh build` if anything under `<scratch>/ts` or
`rs/src/bin/trackrace.rs` changes; it rewrites the sha256s.

## 10. If HEAD has moved before this is run

`run.sh check` prints a warning when the repo's HEAD differs from
`<scratch>/ts/PINNED_REV` and every patch is still checked against the repo as
it stands. If a patch stops applying:

1. `git -C <repo> rev-parse HEAD > <scratch>/ts/PINNED_REV`
2. `python3 <scratch>/ts/gen.py` — regenerates all six patches from the one
   table against the new HEAD
3. `<scratch>/ts/run.sh check && <scratch>/ts/run.sh build`

If `src/eval.rs`'s **research block itself** has moved — the twelve constants,
the `lvl == 2` step, or `RESEARCH_SCALE` — `gen.py` exits with the anchor it
could not find rather than producing a patch that applies to the wrong place.
That is deliberate: a silently-wrong patch here is exactly the failure this
whole file is built to avoid, and the right response is to re-read R16 against
whatever landed in the meantime, not to re-anchor the regex.

**And if any block has already been run, do not re-pin: the binary is the
experiment.** Start a new `<scratch>` directory and a new seed base instead, and
say in the log that the two waves are on different revs.
