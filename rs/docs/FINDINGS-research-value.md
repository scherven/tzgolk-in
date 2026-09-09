# What is research worth over a whole game? A causal measurement.

> Every earlier answer went through `src/eval.rs`. `docs/FINDINGS-eval.md`
> F50-F57 swept `RESEARCH_SCALE` on three agents and found no cell
> distinguishably positive; at `rs = 1.0` the agent researches exactly as the
> compounding theory predicts (+2.10 levels, 17x more maxed tracks, 81% of it
> before day 14) and loses **4.33** points [-4.86, -3.79].
>
> That is a statement about a *term*. Raising the global scale on a term whose
> *shape* is wrong makes the agent research at the wrong times, on the wrong
> tracks, in the wrong positions, and losing four points is what that predicts.
> It does not separate "research is worthless" from "the evaluator cannot
> express when research is worth something".
>
> This file removes the evaluator from the question. **One seat is compelled to
> pursue a stated goal on a stated schedule; the other three play the same base
> agent normally; the seat rotates through all four positions on a shared
> seed.** What comes back is the causal cost or benefit of the behaviour, with
> no term to be mis-priced.
>
> HEAD at start: `9bfcb6e`, 193 tests passing. **Nothing here is committed.**

## Platform

`<scratch>/rv/tree`, a pristine `git archive` of `9bfcb6e` plus one file this
workstream owns, `src/bin/rlab.rs`; built into a private `CARGO_TARGET_DIR` so
that no other agent's `cargo build` can replace the binary underneath a running
sweep (`OVERNIGHT.md`'s standing rule, and F13/F14's retraction).

Binary: `<scratch>/rv/target/release/rlab`. Rev pinned in
`<scratch>/rv/tree/PINNED_REV`.

### The design, in one paragraph

`bin/rlab` wraps any agent in a `Forced` shell that ranks the turn's moves
**lexicographically**: (1) progress toward the stated goal, capped at what the
goal asks for; (2) own workers standing on Tikal, capped, as a tie-break when no
move makes progress — research is cashed on *retrieval*, so a worker has to be
standing there first; (3) the base evaluator's own score. Key 3 is the whole
decision once the goal is met or the window closes, so the wrapper degenerates
to the base agent exactly.

Base agent is **`heuristic:full`** and that choice is load-bearing: at
`temperature = 0` a `Candidates::All` greedy agent draws no random numbers
(`draft` is `best_pair`, `extra_day` is a two-way probe, `choose` at temp 0 is
argmax), so a game is very nearly a pure function of its seed and the only
difference between arms is the candidate's own moves. `heuristic:64` samples
from the shared RNG, and worse, the forced turns would use an *exhaustive* walk
where the baseline seats use 64 draws — a strictly better search on exactly the
turns under test, which would bias the treatment upward.

### Driver

`rlab` uses its own copy of `record::play_game` because it needs the final
state, a day-14 snapshot, the dealt monument row and per-turn placement
accounting, none of which `GameResult` carries. `--verify` plays the same seeds
through both drivers and compares scores:

    ./rlab --verify 6 --seed 700000 --agent heuristic:full
    --verify: 6 seeds, 0 disagreements

### Cost, and the one source of noise

`/usr/bin/time` user CPU: **~1.0 s per game**, so ~4.2 s per rotation block, on
a machine at load 83/14 cores (a 20,000-game self-play run plus two other
workstreams). Fan-out held to 3.

**The null is not exactly zero, and here is why.** `eval::rank_all_within`
carries a 2,500 ms wall-clock deadline and falls back to sampled top-ups when it
trips. Under this load the trip point is not reproducible, so two runs of the
same seed can diverge on a very wide turn. `heuristic:full` against itself
therefore reads a few hundredths rather than the zero-width interval
`FINAL-LADDER.txt` records on an idle machine. It is noise, not bias — it hits
both arms — but it means **every arm below is read against a measured `--force
none` null at the same block count**, not against an assumed zero.

## Arms

| label | `--force` | what it compels |
| --- | --- | --- |
| null | `none` | nothing; the base agent |
| A3e | `res=A3@0-12` | max Agriculture, early |
| A3l | `res=A3@12-27` | max Agriculture, late |
| A1 | `res=A1@0-12` | one level of Agriculture, early |
| AR1 | `res=A1,R1@0-12` | breadth: two tracks at level 1 |
| C3e | `res=C3@0-12` | max Architecture, early |
| T6 | `temple=6@0-12` | **matched control**: 6 temple steps |
| B3 | `build=3@0-12` | **matched control**: 3 buildings |

The two controls are the point. Forcing *anything* on a greedy agent costs
points, so "forcing research costs N" is uninterpretable alone. Tikal 1/3 are
research, Tikal 2/4 are buildings, Tikal 5 is two temple steps: the same gear,
the same worker-turns, three different payoffs.

## Results

*(filled in as runs land)*

### R0. Pilot, and why the first forcing scheme was thrown away

The first scheme demanded the **whole** goal from day 0 and wanted **two** of
the seat's workers standing on Tikal until it was met. `res=A3@0-12`,
`heuristic:full`, 8 blocks, `<scratch>/rv/out/pilot_A3.jsonl`:

| | value |
| --- | --- |
| centred | **−20.06** [−23.08, −17.05] * |
| levels (cand / base) | 2.91 / 1.19 |
| tracks maxed | 0.56 |
| win (null 0.25) | 0.031 |

It reached the goal and lost twenty points. But a three-worker opening with two
workers pinned to one gear is not "commit to research" — it is a handcuff, and
that number is mostly the handcuff. **The scheme was replaced before any
conclusion was drawn from it** by one that (a) *paces* the goal across the
window — `res=A3@0-12` asks for level 1 by day 4, level 2 by day 8, level 3 by
day 12, and leaves the seat free in between — and (b) wants **one** Tikal worker,
not two. `--asap` and `--tikal N` keep the harsh reading available as a
comparison, because the gap between them is itself informative.

The rule this file follows: **give research its best shot.** If the most
competent forced-research policy that can be constructed still loses, the answer
is not an artefact of the forcing.

### R1. First reading of the matched control — and it is the whole answer in one line

Paced forcing, `heuristic:full`, **8 blocks each**, same seeds, same binary.
`<scratch>/rv/out/p2_*.jsonl`.

| arm | centred | 95% CI | levels | forced turns | window open |
| --- | --- | --- | --- | --- | --- |
| `res=A3@0-12` | **−11.91** | [−15.46, −8.35] * | 3.06 (base 1.21) | 1.78 | 3.59 |
| `temple=6@0-12` | **−0.70** | [−2.70, +1.31] | 1.19 (base 1.11) | 0.69 | 1.72 |

Forcing is not itself the handicap: compelling the same seat, through the same
gear, over the same window, to buy **temple steps** instead costs nothing
measurable. Compelling it to max a research track costs twelve points.

**But the two are not yet matched on how hard the constraint bites** — the
temple arm's window is open 1.7 turns a game against 3.6, because
`heuristic:full` already takes six temple steps by day 12 unprompted. The
controls are being re-calibrated to equal *forced turns*, which is the number of
Tikal actions actually diverted, before this is read as a like-for-like.

### R2. The gentlest possible forcing still costs ten points, and the money comes out of the temple track

`--tikal 0` removes the setup key entirely. The seat is never told to go to
Tikal; it is overridden **only on turns where a research advance is actually on
offer** — that is, it placed a worker on Tikal for its own reasons and is now
made to take the research rather than the building or the temple steps it would
have taken. There is no handcuff left in this arm at all.

`res=A3@0-12 --tikal 0`, `heuristic:full`, 8 blocks,
`<scratch>/rv/out/cal_A3t0.jsonl`:

| | candidate | baselines |
| --- | --- | --- |
| **centred** | **−10.55** [−13.92, −7.17] * | — |
| final score | 36.47 | 50.53 |
| research levels (of 12) | **2.97** | 1.22 |
| tracks maxed | 0.47 | — |
| **temple sum (3 tracks)** | **10.84** | **13.45** |
| workers placed | **21.03** | 19.49 |
| corn paid to place | **16.66** | 13.41 |
| turns that begged | 1.84 | 2.13 |
| forced turns / window open | 1.59 / 4.19 | — |

Three things at once, and they are the whole finding:

1. **It still gets the track.** Overriding ~1.6 turns a game is enough to reach
   2.97 levels and max a track half the time, because the seat is already
   standing on Tikal — it just spends the action differently.
2. **The user's mechanism is real and it is visible here.** The forced seat
   places **1.5 more workers** over the game and pays **3.3 more corn** to place
   them. Corn *did* convert into placement depth, exactly as the theory says.
3. **And it loses ten points anyway, out of the temple track.** The temple sum
   falls 13.45 → 10.84, about **2.6 steps**. Begging is *down*, not up, so this
   is not the forced seat scrounging for corn; it is Tikal 5 — two temple steps
   for one block — being spent on research instead.

The null at the same block count is **+0.33 [−0.45, +1.10]** (`cal_null`), and
most of its blocks are exactly 0.000; the spread is the wall-clock deadline in
`rank_all_within`, see the platform note.

### R3. Calibration: cost per **forced Tikal turn**, by what the turn buys

The controls have to be matched on how hard the constraint bites, not on the
size of the number in the goal. `hits` is the count of turns on which the
forcing actually changed the winning move — the Tikal actions genuinely
diverted. 8 blocks each, `heuristic:full`, `<scratch>/rv/out/cal_*.jsonl`,
binary `bin/rlab-r2`:

| arm | centred | forced turns | **per forced turn** | temple sum (cand / base) |
| --- | --- | --- | --- | --- |
| `none` (null) | +0.33 [−0.45, +1.10] | 0.00 | — | 13.28 / 13.20 |
| `temple=10@0-12` | −3.23 [−7.25, +0.78] | **3.03** | **−1.07** | 12.69 / 13.46 |
| `build=5@0-12` | −14.74 [−20.52, −8.96] * | 2.90 | −5.08 | 9.40 / 13.33 |
| **`res=A3@0-12 --tikal 0`** | **−10.55** [−13.92, −7.17] * | **1.59** | **−6.63** | **10.84 / 13.45** |

Read the temple row first. It is forced **twice as often** as the research arm
and costs a third as much — so the cost being measured is not "a greedy agent
under a constraint". Diverting a Tikal action into two temple steps is nearly
free; diverting one into a research advance costs about six points.

The building row is the useful surprise: **forcing buildings is nearly as
expensive as forcing research**, and its temple sum falls furthest of all
(9.40). The pattern across all four rows is one thing: whatever a Tikal action
is diverted *from*, the score follows the temple column. Temple steps are where
this game's points are, `temple` is 15.27 of the evaluator's 33.4-point estimate
(F53), and the causal experiment agrees with that ranking even though it never
consults the evaluator.

Dose-response arms (`res=A1` vs `res=A3`, `temple=6` vs `temple=10`, `build=3`
vs `build=5`) are in the main wave so the per-turn slope is measured rather than
divided out of one point.
