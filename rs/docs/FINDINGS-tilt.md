# An agent that values research higher in the early game

> **Status: BUILT AND VERIFIED, NOT RACED.** The agent exists, it plays, and
> it demonstrably researches more than the champion. **No strength measurement
> was run** and none is claimed: the machine is committed to a 20,000-game
> self-play, two arena races and a training job. What is below is the design,
> the calibration, and the proof that the shipping evaluator did not move.
>
> **HEAD at the time of writing: `901257d`.** `cargo test --release`: **212
> passed, 0 failed, 8 ignored** (200 at HEAD, plus twelve added here).

## 0. The whole thing, in one screen

| | |
| --- | --- |
| **the agent** | `mcts:8192:heuristic:deeper,tilt=agri` — the champion's search, over an evaluator that prices research higher early, per track. |
| **the knob** | `tilt=agri\|causal` and `tiltk=K`, parsed beside `cp=` and `pmin=`, printed by `mcts_label`, so it works in `arena`, `selfplay` and `tui` alike. |
| **the shape** | the research block, re-priced by `1 + K * w[track] * early(horizon)`, where `early = min(1, 1.4 * horizon)` — flat through day 7, gone by day 27. §2. |
| **`agri`** (default) | Agriculture 1.00, Architecture 0.30, Theology 0.20, Extraction 0.10. Agriculture by far the most; the rest in R13's measured order. |
| **`causal`** | R13's measured causal value of a maxed track, normalised: Architecture 1.000, Theology 0.679, Agriculture 0.318, Extraction 0.192. |
| **`K`** | **12** by default, and the reason is §3: it is what makes a maxed Agriculture track cost roughly the +7.49 R13 says it is causally worth. A tilt of 4 is **not distinguishable from the champion** on either behavioural endpoint. |
| **the vehicle** | a **per-seat** tilt, read where `engine_value` is called. `mcts.rs` calls `eval::heuristic` directly for the prior and the edge ordering, so an injected `Evaluator` would move the leaf value and miss half the mechanism. §1. |
| **does it change play?** | Yes. 12 paired games of `mcts:256:heuristic:deeper`, seat 0 tilted against the same seeds plain: research levels on **day 14** go **0.92 -> 1.58**, final **1.42 -> 2.42**. §3. |
| **is `heuristic` unchanged?** | Yes, twice over: structurally (the untilted path does not execute the added term) and empirically (two binaries, 24 games, 96 final scores, byte-identical output). §4. |
| **watch it** | `tui SEED --agent CHAMP --seat 0 TILTED --seat 1 TILTED`. The Players panel gets a key letter per seat and a line decoding it. §5. |
| **not measured** | whether it is any *good*. §6. |

## 1. The vehicle: the tilt is a property of the seat

This is the load-bearing decision and it was settled by reading `mcts.rs`.

**`eval::heuristic` enters the shipping agent in three places and only one of
them goes through the `Evaluator` trait.**

| call site | `src/mcts.rs` | reached by a custom `Evaluator`? |
| --- | --- | --- |
| leaf value | `HeuristicEvaluator::evaluate` -> `eval::heuristic` | **yes** |
| prior over edges (`Priors::OnePly`, the default) | `one_ply`, line **2269** | **no** — hard-coded |
| edge ordering (`EdgeOrder::Gradient`, the default) | `Gradient::new`, lines **953/957** | **no** — hard-coded |

`MctsConfig::default()` sets both, and `deeper` (`cp=0.02,pmin=2`) changes
neither. The two hard-coded sites are where *which action the search even
considers* is decided — `Gradient` prices `Effect::AdvanceResearch(s)` by
probing `heuristic` — so an agent built by injecting a variant `Evaluator`
would be a leaf-value-only imitation of the one asked for, and systematically
weaker at exactly the thing under test.

`docs/FINDINGS-track-shape.md` §1.1 had already solved this for its arm
experiment: `heuristic(g, p)` takes the player, so make the variation a
property of the **seat**. Every one of the three sites passes the player whose
position is being estimated, so all three see it for free. That file proved its
null arm bit-identical to HEAD by diffing two binaries; this is the same
vehicle without a patch, and §4 repeats the same proof.

`tests/tilt.rs::the_tilt_reaches_the_edge_ordering` is the test that says this
agent is the intended one: it prices `AdvanceResearch(Agriculture)` through
`Gradient` and finds the tilt in it, to the expected 0.49 of a point.

### 1.1 Thread-local, not a global — a deliberate deviation

The brief specified `[AtomicU32; 4]`. This ships a **thread-local**
`[[f32; 4]; 4]` instead, and the reason is that a process-global table is wrong
wherever two games are in flight at once:

* `arena` and `selfplay` play many games per process, on many threads, and
  **rotate the seating between them** — seat 0 is the candidate in one game and
  the baseline in the next. Two threads writing one global table would corrupt
  each other's seating, silently, in the direction of "the result file cannot
  be attributed".
* `cargo test` runs its tests in parallel threads in one process.
* `bin/tui` runs every search on a **fresh worker thread**.

A thread-local is per *game*, because a game is one thread: `mcts.rs` spawns
none, and the heuristic backend is never batched (`Backend::batchable` is false
for it), so every `heuristic` call a seat's search makes runs on the thread
that installed the tilt. `record::tilted` installs the acting seat's row at the
top of every `Agent` method and drops a guard on the way out, which is what
makes the TUI's worker thread see it, and what keeps the viewer's own
`eval::margin` scoring — run between turns on the same thread — from being
quietly computed through somebody's opinion of research.

**Two consequences worth stating.**

1. **Only the acting seat's row is installed.** A tilted agent tilts its
   estimate of *its own* position and nothing else. It does not model its
   opponents as sharing its opinion, and an opponent's search never sees it —
   including a second tilted agent's, which installs its own row on its own
   turn. This is the honest reading of "an agent that values research more",
   and it is what makes 2-vs-2 one game rather than one evaluator with a mood.
2. **It does not reach `NetBatch::mix`**, which computes the heuristic half of
   a blended leaf on a batcher thread. A tilted spec over a *network*
   evaluator would tilt the prior and the ordering and not the blend. Nothing
   here uses one; `docs/FINDINGS-track-shape.md` §5 hit the same wall.

### 1.2 Why `engine_value` was not edited

The tilt is computed in a separate function and added to `c.engine` in
`components()`. It would read better inside `engine_value`'s research loop, and
it is not there for a specific reason: **all six unapplied patches in
`<scratch>/ts/patches/` have a hunk inside `engine_value`**, and that
experiment is pinned, prepared and unrun. Editing those lines would have
invalidated their context and forced a regeneration against a moving tree.

All six still `git apply --check` clean, before and after every edit here.

The cost is one duplicated line — the `uses = min(rounds_left / 3, 7)` ramp
exists in both places. `tests/tilt.rs::the_tilt_is_the_research_block_re_priced`
pins the arithmetic end to end so that changing one and not the other fails.

## 2. The shape

```text
    research block, per track, per held level
        research_step_value(s, l) * uses * RESEARCH_SCALE          (unchanged)
      + research_step_value(s, l) * uses * RESEARCH_SCALE * K * w[s] * early(h)
```

so a favoured track is priced at `1 + K * w[s] * early(h)` times what the
shipping evaluator says, and at `K * w = 0` it *is* what the shipping evaluator
says.

### 2.1 `early(horizon) = min(1, 1.4 * horizon)`

`docs/FINDINGS-research-value.md` R14 granted a free maxed track on days 0 / 7
/ 14 / 21 and measured what it was worth, relative to day 0:

| | day 0 | day 7 | day 14 | day 21 |
| --- | --- | --- | --- | --- |
| measured (R14) | 1.000 | **1.023** | 0.483 | 0.302 |
| `engine_value`'s `uses` ramp, already | 1.000 | 0.952 | 0.619 | 0.286 |
| `early` alone | 1.00 | 1.00 | 0.674 | 0.311 |
| the tilt's premium, `uses * early` | 1.00 | 0.95 | 0.42 | 0.09 |

Two things about that table.

**`early` is a second decay, and that is the point.** The block it multiplies
is already fading — R14 calls `uses` the best-fitting piece of the whole
evaluator. What `early` fades is the **premium**, because the brief is an agent
that values research more *in the early game*, not one that values it more
throughout. The bottom row is the thing that was designed: the tilt is spent in
the first half of the calendar, and the agent finishes the game as the champion
it started from.

**The plateau is real and is not a fudge.** R14 measures a track granted on day
7 as worth *at least* as much as one granted on day 0, and the knee is between
day 7 and day 14 because the temples score on `POINT_DAYS = [14, 27]`.
`min(1, 1.4h)` is flat exactly there and reaches zero on the last day. It
overshoots the knee at day 14 by 0.19; a steeper knee that fits day 14 misses
day 21 by as much in the other direction, and four measured points do not
support a curve with more than one parameter in it.

### 2.2 The two shapes

Normalised so the most-favoured track is 1.0, which is what makes `K` mean the
same thing — "the top track is worth `1 + K` times what the evaluator says" —
whichever shape is chosen.

| | Agriculture | Extraction | Architecture | Theology |
| --- | --- | --- | --- | --- |
| **`agri`** (default) | **1.00** | 0.10 | 0.30 | 0.20 |
| **`causal`** | 0.318 | 0.192 | **1.000** | 0.679 |
| R13's causal value of a maxed track | +7.49 | +4.51 | **+23.52** | +15.96 |
| `research_step_value` sums, today | 1.55 | 1.95 | 1.95 | 2.00 |

`agri` is what was asked for: Agriculture by far the most, the others much
smaller and descending. The *order* of the other three is R13's measured order
rather than a preference, because a preference there would be a number with no
evidence behind it. `causal` is R13 divided through by its largest — the shape
`docs/FINDINGS-track-shape.md`'s `redist` arm proposes for the shipping
evaluator, available here as an agent instead of as a patch.

## 3. The strength, and why 4 was not enough

**The anchor.** A maxed Agriculture track is priced by the research block at
`1.55 * 7 * 0.05 = 0.5425` points on day 0, and moves the whole estimate by
about 1.2. R13 measured what one is causally worth: **+7.49**. Solving
`(1 + K) * 0.5425 + 0.66 = 7.49` — the block re-priced, plus the ~0.66 of a
track's value that lives in terms the tilt does not touch — gives `K = 11.6`;
pricing the block alone at 7.49 gives 12.8. **Twelve is in the middle of that
band.** At level 1, where the agent actually decides, it prices Agriculture 1 at
1.6 points: one more corn on every green Palenque gather for the rest of the
game, which is about five corn, which is about 1.25 points plus what a spendable
corn is worth. The shipping evaluator says 0.12.

**The measurement.** `tests/tilt.rs::calibrate_the_strength`, kept and
runnable. 16 paired games of `mcts:256:heuristic:deeper` — seat 0 tilted, the
same seeds, the same opponents, so seat 0's own choices are the only thing that
differs — plus the share of 100 fixed early-game decisions answered with a move
that advances the mover's own research:

| K | shape | levels, day 14 | levels, final | research moves |
| --- | --- | --- | --- | --- |
| — | the champion | 0.94 | 1.38 | 8 / 100 |
| 8 | `agri` | 1.25 | 1.69 | 8 / 100 |
| **12** | **`agri`** | **1.56** | **2.25** | **12 / 100** |
| 16 | `agri` | 1.88 | 2.38 | 14 / 100 |
| 8 | `causal` | 1.44 | 2.25 | 13 / 100 |
| 12 | `causal` | 2.06 | 2.81 | 17 / 100 |
| 16 | `causal` | 3.00 | 3.69 | 15 / 100 |

An earlier sweep at `K` = 1, 2, 4 is the finding that matters most here:
**a tilt of 4 — "several-fold", and the first guess — is not distinguishable
from the champion on either endpoint** (1.25 day-14 levels against 1.00, and
*fewer* research moves, 7 against 8). Research is priced so far below what it
is worth that a several-fold correction still does not move a decision. That is
why the constant is documented on `ResearchTilt::DEFAULT_STRENGTH` rather than
picked.

**The shipped assertion**, `the_tilt_changes_what_the_agent_plays`: 12 paired
games, day-14 levels **0.92 -> 1.58** (1.7x), final **1.42 -> 2.42**. It is
deterministic — fixed seeds, fixed rng, fixed agent seeds — so it is a fact
about the default and not a sample that might not repeat.

**A note on the endpoint.** The first version counted research levels at the
end of unpaired games and read nothing: the levels a seat finishes with vary by
**3x between seats of the same agent**, and seat and draft effects swamped a
sixteen-game signal. Pairing on the seed and reading the level count at the end
of the tilt's own window is what made it legible at a dozen games.

## 4. `eval::heuristic` is unchanged with the tilt off

Proved twice, because the thing at risk is two live arena races, a
20,000-game self-play and a training job.

**Structurally.** `components()` reads:

```rust
c.engine = engine_value(g, p, rounds_left, horizon);
let tilt = research_tilt(g, p, rounds_left, horizon);
if tilt != 0.0 {
    c.engine += tilt;
}
```

`research_tilt` returns `0.0` on its first branch when the seat's row is all
zeros — the default, and what every existing caller gets — so with no tilt
installed the assignment is the *same expression* it has always been, not that
expression plus a float zero. Bit-identity does not rest on an argument about
signed zeroes.

**Empirically**, by the technique `docs/FINDINGS-track-shape.md` §3.1
prescribes: `bin/trackrace --verify N --agent SPEC` plays N seeds through two
drivers and prints all four final scores per seed. Run on
`<scratch>/ts/bin/trackrace-head` — the unpatched binary built from the pinned
tree, whose `src/eval.rs` is byte-identical to HEAD's — and on a binary built
from this tree, and the two outputs diffed:

| agent | seeds | result |
| --- | --- | --- |
| `heuristic:8` | 6 | **identical** |
| `heuristic:32` | 6 | **identical** |
| `mcts:128` | 6 | **identical** |
| `mcts:512:heuristic:deeper` (the champion's own config) | 6 | **identical** |

96 final scores, byte for byte. Logs in `<scratch>/tilt/f.*.txt` and
`<scratch>/tilt/FINAL-VERIFY.txt`.

**One agent was excluded and it is worth recording why.** `minimax:4:200`
differs between the two binaries — and the HEAD binary differs from **itself**
on the same seeds (`<scratch>/tilt/mm.head.selfdiff.txt`). A 200 ms wall-clock
deepening budget is not reproducible on a loaded machine; this is
`FINDINGS-research-value.md` R17.5's failure mode, met for the third time. Only
agents with no wall-clock deadline can carry this proof.

## 5. Watching it: two of each, on one screen

```
cargo run --release --bin tui -- 7 \
  --agent 'mcts:8192:heuristic:deeper' \
  --seat 0 'mcts:8192:heuristic:deeper,tilt=agri' \
  --seat 1 'mcts:8192:heuristic:deeper,tilt=agri'
```

`--agent` still means "every seat", which is what it has always meant;
`--seat N SPEC` overrides one, is repeatable, and takes the same specs as the
arena. Each seat gets its **own instance** — `AgentSpec` hands every instance a
distinct MCTS seed, and four seats sharing one would search identically and
contend on the one `Mutex` inside `SearchAgent` — and each seat now drafts its
own starting tiles, which matters because the tiles include a research level
and that is the first decision the tilt touches.

**What is on screen.** The Players panel, at 132x44:

```text
┌ Players ────────────────────────────────────────────────────────────┐
│   corn   W  S  G  sk   pts   wk  free  disc  bld mon  tiles         │
│>Ra   7   2  0  0   0      0  4+0    0     0     0   0  0c 0w        │
│ Ga   2   0  1  1   0      0  3+0    0     0     0   0  0c 0w        │
│ Bb  12   2  2  0   0      0  3+0    0     0     0   0  0c 0w        │
│ Yb   5   2  0  1   0      0  3+0    0     0     0   0  0c 0w        │
│ a RG tilt=agri:tiltk=12   b BY none                                 │
└─────────────────────────────────────────────────────────────────────┘
```

* **a key letter per seat**, one per *distinct* agent, in the column the row
  already spent on padding — so no column moves, no width is lost, and it is
  legible at every size the panel draws at.
* **a legend under the four rows** decoding the keys, printing only the part of
  each name that **differs**. Two of the champion against two of a variant of
  it share forty characters of prefix; a panel forty columns wide that spent
  them all on the shared half would say nothing. The shared half is in the top
  bar, which names the agent for the seat about to move in full.
* **squeezed panels keep the seating.** One spare row goes to the legend rather
  than to the `wk =` note; with no spare row at all the legend takes the panel
  *title*, because a key letter nothing decodes is worse than a column key
  whose meaning is guessable from four rows of numbers.
* **a table of one agent looks exactly as it did.** No keys, no legend, the
  column key back in its place. Pinned by
  `tests/rules.rs::a_table_of_one_agent_keeps_the_column_key`, with the mixed
  case pinned at eight terminal sizes by
  `the_players_panel_says_which_agent_is_in_which_seat`.

## 6. What this does **not** say

* **Nothing about strength.** No arena, no self-play, no race was run. The
  agent researches more; whether that wins more is unmeasured, and the two
  behavioural endpoints here (research levels, research decisions) were chosen
  *because* they measure the intended behaviour rather than standing in for
  strength.
* **The calibration is 12 to 16 games** of a 256-simulation agent. It is a
  behavioural check, not a measurement with an interval on it.
* **The `agri` shape is a weaker lever on play than `causal`** at equal `K`
  (day-14 levels 1.56 against 2.06 at K = 12), and the reason is structural
  rather than arithmetical: the premium per track is the same size, but
  Agriculture is one track and `causal` spreads over Architecture and Theology
  both. `agri` is the default because it is what was asked for.
* **A tilt over a network evaluator is only two-thirds installed.** §1.1.

## 7. The obvious next thing

`arena --candidate 'mcts:8192:heuristic:deeper,tilt=agri' --baseline
'mcts:8192:heuristic:deeper' --mode solo`, when the machine is free. Both sides
print distinct names — `mcts_label` appends the tilt last, so the champion's
name is a prefix of the tilted one's and the two sort together in a results
table — and `examples/nametest.rs` plus
`tests/tilt.rs::the_spec_and_its_instance_agree_on_the_name` are what stop the
race from reporting one name twice, which is a bug this project has already
paid for once.

If it loses, the interesting follow-up is not a smaller `K`: it is `tilt=causal`
at the same budget, which is the shape `FINDINGS-track-shape.md` proposes for
`eval.rs` itself and which this makes raceable without patching anything.
