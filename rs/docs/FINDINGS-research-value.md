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
> HEAD at start: `9bfcb6e`, 193 tests passing; at finish `f9789b3`, **200 tests
> passing, build clean**. **Nothing here is committed.** All measurement binaries
> are pinned to `9bfcb6e`, the rev F50-F57 measured.

## The answer, in one table

Everything below is causal: one seat is compelled (or handed something free),
three play the base agent normally, the seat rotates through all four positions
on a shared seed, and the evaluator is never asked what it thinks. Read against
a measured null of **+0.079 [−0.108, +0.267]** at 48 blocks.

| | |
| --- | --- |
| **(a) Is early research good?** | **No — causally, clearly, and for a reason.** A maxed Agriculture track is genuinely worth **+7.49** points free, but buying it costs **18.43**, so forcing it loses **−10.94 [−12.16, −9.73]**. Forcing it *later* is **cheaper** (−3.15 / −2.71 / −1.45 / −1.51 across four windows at equal dose and equal levels gained — falling, then flat, never rising): the compounding premium has the **wrong sign**, −1.64 points early-minus-late. R5, R6, R9. |
| **(b) Does the monument row change it?** | **No, by +0.80 [−0.83, +2.43].** 60 blocks with #11 & #12 both up against 60 with neither. The row is theoretically worth 18 VP and the seat collects **0.02 monuments a game** — monuments #11/#12 are built at Tikal 4, the same gear that sells the research, so the payoff and its price compete for one worker. R7. |
| **(c) Does corn-as-placement-depth change the accounting?** | **The mechanism is real and the evaluator already prices it.** Corn is worth **+0.60** causally, 2.4× its 4:1 scoring rate — the user's reasoning is right, and free research does buy **+2.8 placements a game**. But `eval.rs` already values corn at **+0.53** and wood at **+1.00** against a causal **+1.05**. Nothing is missing here. R8, R9, R15. |
| **(d) Is `uses`'s shape defensible?** | **Yes — it is the best-fitting piece of the block.** Against the causal curve (1.000 / 1.023 / 0.483 / 0.302 at days 0/7/14/21) the term reads 1.000 / 0.952 / 0.619 / 0.286: right at three of four points, and its `min(7.0)` cap reproduces a **real plateau** — a track granted on day 7 is worth as much as one granted on day 0. R14. |
| **(e) The shape proposal** | **The error is per-track, and no scalar can reach it.** The four maxed tracks span **5.2×** causally (Extraction +4.51, Agriculture +7.49, Theology +15.96, **Architecture +23.52**) and **1.12×** in the evaluator (1.13 to 1.27). Proposal: a **scale-neutral redistribution** across tracks, plus **deleting the `lvl == 2 → +0.4` step**, which is 56–61% of the evaluator's opinion of depth and sits one level too low. R16, R18. |

**The one thing that might yet overturn "research is bad": Architecture.** It is
worth **+23.52** free and wins **90%** of games — the compounding engine the user
reasoned must exist does exist, and it is not on the track their example named.
It costs 28.90 to acquire under this file's deliberately crude forcing, so
benefit/price is **0.81**. A 19% cheaper acquisition policy would put it over the
line, and both known biases (a crude forcing policy, a granted seat that never
*plans* around the track) push that way. R12, R17.

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

## The main wave — `bin/rlab-r3`, `heuristic:full`, seeds 700000+

### R4. The null, at 48 blocks: **+0.079 [−0.108, +0.267]**

`--force none`, 48 rotation blocks (192 games), `<scratch>/rv/out/w_null.jsonl`.
Win rate 0.250 against a null of 0.250. Every arm below is read against this.

What a seat of `heuristic:full` does when nobody is forcing it, per game — this
is the baseline every treatment column moves away from:

| research levels (of 12) | tracks maxed | buildings | monuments | temple sum | workers placed | corn paid to place | corn held (integral) | turns begged | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.27 | 0.11 | 2.80 | 0.07 | 12.95 | 19.36 | 12.71 | 108.1 | 2.07 | 49.7 |

The monument column is the one to keep in view: **a seat takes 0.07 monuments a
game**, and monuments #11 and #12 — the two that pay for research — are
constructed at **Tikal 4**, the same gear whose spaces 1 and 3 sell the research
in the first place, off the same six-space ride. The payoff and its price
compete for one worker on one gear.

### R5. The main wave landed. **Early research is causally bad, and the cost rises with depth.**

`bin/rlab-r3`, `heuristic:full`, seeds 700000+, `<scratch>/rv/out/w_*.jsonl`.
Every arm read against the R4 null, `+0.079 [−0.108, +0.267]` at 48 blocks.

| arm | `--force` | centred | 95% CI | blocks | hits | **per hit** |
| --- | --- | --- | --- | --- | --- | --- |
| null | `none` | +0.079 | [−0.108, +0.267] | 48 | 0.00 | — |
| A1e | `res=A1@0-12` | **−3.14** | [−4.27, −2.00] * | 32 | 0.66 | −4.75 |
| A3e/t0 | `res=A3@0-12 --tikal 0` | **−10.94** | [−12.16, −9.73] * | 48 | 1.62 | −6.75 |
| A3e | `res=A3@0-12` | **−12.44** | [−13.70, −11.19] * | 48 | 1.97 | −6.31 |
| A3l | `res=A3@14-26` | −6.16 | [−7.12, −5.20] * | 32 | 1.84 | −3.35 |
| AR1e | `res=A1,R1@0-12` | −8.32 | [−9.42, −7.22] * | 32 | 1.24 | −6.71 |
| C3e | `res=C3@0-12` | −5.38 | [−7.29, −3.46] * | 32 | 1.93 | −2.79 |
| **T6** | `temple=6@0-12` | −1.03 | [−2.01, −0.04] * | 32 | 0.81 | **−1.27** |
| **T10** | `temple=10@0-12` | −3.61 | [−5.13, −2.08] * | 32 | 2.88 | **−1.25** |
| **B3** | `build=3@0-12` | −8.39 | [−10.05, −6.73] * | 32 | 1.80 | **−4.66** |
| **B5** | `build=5@0-12` | −13.43 | [−14.97, −11.88] * | 32 | 2.96 | **−4.54** |

`hits` is the count of turns on which the forcing actually changed the winning
move — the Tikal actions genuinely diverted. It is the dose, and the controls
say it is the right dose to divide by: **temple is −1.27 and −1.25 per hit
across a 3.5× range; buildings are −4.66 and −4.54 across a 1.6× range.** Both
matched controls are flat in the dose. The forcing machinery is linear, and the
per-hit number is therefore a real price and not an artefact of how hard the
constraint bites.

**Research is the one arm that is not flat.** Its dose-response slope, taken
between the two arms that differ only in depth:

    A1e -> A3e/t0 : (10.94 - 3.14) / (1.62 - 0.66) = -8.1 points per extra hit
    T6   -> T10   : ( 3.61 - 1.03) / (2.88 - 0.81) = -1.25  (flat)
    B3   -> B5    : (13.43 - 8.39) / (2.96 - 1.80) = -4.34  (flat)

The **marginal** forced research turn costs 8.1 points, against an average of
4.75 for the first one. **The cost of research is convex in depth.** That is
the precise opposite of the compounding hypothesis, which predicts concavity —
the second level should be cheaper than the first because the first has already
paid the setup.

Where the money goes is the same in every arm: the temple column.

| arm | temple sum (cand / base) | builds (cand / base) |
| --- | --- | --- |
| null | 12.95 / 12.96 | 2.80 / 2.80 |
| A3e/t0 | **10.60 / 13.20** | 2.21 / 2.85 |
| C3e | **8.58 / 13.43** | 4.66 / 2.88 |
| B5 | **9.66 / 13.23** | 5.50 / 2.92 |
| T10 | 13.02 / 13.11 | 2.80 / 2.90 |

Every arm that loses, loses temple steps; the arm that holds its temple sum is
the arm that costs nothing per hit. Tikal 5 is two temple steps for one block,
and it is the best thing on the gear by a factor of four.

### R6. The timing curve: **later is cheaper**, monotonically

One level of Agriculture, forced inside four disjoint four-day windows, 28
blocks each, `<scratch>/rv/out/w_A1w*.jsonl`. The dose is held constant by
construction and confirmed by `hits`.

| window | days | centred | 95% CI | hits | levels gained |
| --- | --- | --- | --- | --- | --- |
| w00 | 0–4 | **−3.15** | [−4.42, −1.88] * | 0.62 | +0.41 |
| w08 | 8–12 | **−2.71** | [−4.06, −1.36] * | 0.68 | +0.46 |
| w16 | 16–20 | **−1.45** | [−2.22, −0.68] * | 0.57 | +0.44 |
| w21 | 21–25 | **−1.51** | [−2.16, −0.86] * | 0.66 | +0.53 |

Same dose (0.57–0.68 hits), same level gain (+0.41 to +0.53), four different
prices, and the price **falls** the later the level is bought. Early minus late
is **−1.64 points** — the compounding premium has the wrong sign.

This is the cleanest refutation in the file of "early research is worth a lot
despite the delayed payoff", because it is the *same purchase* at four different
times, with the evaluator never consulted. A level bought on day 2 is held for
~25 days and one bought on day 23 for ~4, and the 21 extra days of holding it
are worth **less than nothing** net of what the early Tikal turn could have
bought instead. Days 0–12 are when the temple track is cheap and the corn to
climb it is plentiful; that is what the early research turn is spending.

Caveat, stated plainly: this identifies `cost(early) − cost(late)`, not the two
terms separately. It cannot by itself say whether the extra days of the level
are worth ~0 or whether the early Tikal turn is simply very expensive. **R7
separates them** by handing the levels over for free.

### R7. Monuments #11 and #12 do not rescue it — because the agent never builds them

Question (b), by selection: `--scan` reads the dealt row off the seed, 60 seeds
with both #11 and #12 face up and 60 with neither, the same `res=A3@0-12
--tikal 0` arm on each. `<scratch>/rv/out/mon_{both,neither}.jsonl`.

| row | centred | 95% CI | blocks | maxed | **monuments held** |
| --- | --- | --- | --- | --- | --- |
| both #11 & #12 up | −11.64 | [−12.86, −10.41] * | 60 | 0.58 | **0.02** |
| neither up | −12.43 | [−13.55, −11.31] * | 60 | 0.52 | **0.03** |

Difference **+0.80**, two-sample SE 0.83, so **+0.80 [−0.83, +2.43]** — not
distinguishable from zero, and two orders of magnitude short of the 18 VP the
row is theoretically worth.

The reason is in the last column and it is decisive: **the seat holds 0.02
monuments a game.** It maxes a track 58% of the time and collects the payoff
essentially never. Monuments #11 and #12 are constructed at **Tikal 4** — the
same six-space gear whose spaces 1 and 3 sell the research. The payoff and its
price compete for one worker on one gear, and the forced seat has already spent
that worker on the research. A term that prices maxed tracks by what the
monuments *would* pay is pricing a coupon the agent cannot redeem.

**Retraction of a subgroup read.** The within-run monument split on
`w_A3e_t0` (n=7 "both", −7.31; n=13 "neither", −11.60) suggested a +4.3 point
monument effect. The dedicated 60-block-per-cell experiment says +0.80 [−0.83,
+2.43]. The n=7 cell was noise. This is why the selection run was done.

### R8. Corn is worth **0.60 points**, not 0.25 — the user's mechanism is real, and it is priced here

Question (c). Two independent measurements, neither through the evaluator.

**The direct price, by subsidy.** `--gift RES:N/D` hands the candidate seat N
units every D of its own turns and changes nothing else. A game is ~26 own
turns, so `corn:1` is 26 corn and `corn:1/3` is 9. 28 blocks each,
`<scratch>/rv/out/w_g*.jsonl`.

| arm | gift | centred | 95% CI | units | **points per unit** |
| --- | --- | --- | --- | --- | --- |
| gc1 | `corn:1` | +15.48 | [+13.53, +17.42] * | 26 corn | **+0.595** |
| gc3 | `corn:1/3` | +5.83 | [+4.27, +7.38] * | 9 corn | **+0.647** |
| gw4 | `wood:1/4` | +7.36 | [+4.85, +9.87] * | 7 wood | **+1.05** |

Corn's scoring rate in the rules is 4 corn = 1 VP, i.e. **0.25 points**. Its
causal marginal value is **0.60**, mildly concave (0.65 at low dose, 0.60 at
high). **Corn is worth 2.4× its face value**, and the user's reasoning for why
is confirmed by the decomposition: with `corn:1` the seat begs 0.29 times a game
against 1.94, places +0.84 more workers, spends +9.25 more corn on placement —
and its **temple sum rises 13.13 → 15.89**. Corn converts into actions, and the
actions it buys are temple steps. Wood is worth 1.05 points and buys buildings
(2.75 → 4.37) and monuments (0.07 → 0.21).

**The mechanism, measured without pricing it.** `--census` records, for every
turn of 100 unforced games, workers in hand, corn held, how many workers that
corn could actually pay for, and how many more it could pay for with +1..+6
corn — the nth worker costs n plus the space index, so this converts corn into
*actions* directly. `<scratch>/rv/out/census.txt`, 10,332 turns:

| workers in hand | share of turns | +1 corn buys | +2 | +3 | +6 |
| --- | --- | --- | --- | --- | --- |
| 0 | 39.0% | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 38.8% | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 13.3% | +0.105 | +0.105 | +0.106 | +0.106 |
| 3 | 8.7% | +0.106 | +0.181 | +0.223 | +0.227 |
| 4 | 0.1% | +0.400 | +0.700 | +0.800 | +1.000 |

The user's prediction is **exactly right in shape**: corn buys placement depth
only for a player with workers in hand, and not at all otherwise. With 0 or 1
worker in hand — **77.8% of all turns** — an extra corn buys literally zero
extra actions, because the first placement is nearly always affordable. The
whole effect lives in the 22% of turns with 2+ workers up.

And it **saturates almost immediately**: at hand=3, the first corn buys 0.106
placements and the sixth buys a cumulative 0.227. Averaged over the real
distribution of hands, +1 corn buys **+0.024 placements per turn** and +3 corn
buys **+0.034** — the second and third corn are worth a seventh of the first.

So corn-as-placement-depth is real, is worth having in the accounting, and is
**strongly state-dependent and strongly concave** — which is exactly what a
linear `corn / CORN_PER_POINT` term cannot express. It is also not, by itself,
enough to save research: see R9.

### R9. The decisive split: **the levels are worth +7.5 points and cost 18.4 to buy**

Every arm above measures `benefit − price` in one number, which is why a
negative result could never distinguish "research is worthless" from "research
is fine and the board charges too much for it". `--grant` (added to
`bin/rlab-r7`, this workstream's file) hands the candidate seat the levels at
day 0 **for free** — no Tikal action, no corn, no worker-turn, `hits = 0` — and
changes nothing else. It measures the benefit alone.

| arm | what it is | centred | 95% CI | blocks | win |
| --- | --- | --- | --- | --- | --- |
| `--grant A3` | maxed Agriculture, free, day 0 | **+7.49** | [+5.81, +9.18] * | 48 | **0.49** |
| `--force res=A3@0-12 --tikal 0` | the same levels, bought | **−10.94** | [−12.16, −9.73] * | 48 | 0.05 |
| null | — | +0.079 | [−0.108, +0.267] | 48 | 0.25 |

    price = grant - force = 7.49 - (-10.94) = 18.43 points

**Both halves of the user's intuition are vindicated, and they still add up to
"don't research".** A maxed Agriculture track is genuinely worth **+7.5 points**
— it doubles the seat's win rate, 0.25 → 0.49. It is not a rounding error and it
is not worthless. But acquiring it on this board costs **18.4 points**, and no
amount of correctly valuing the asset fixes a price two and a half times the
asset.

**The mechanism the user predicted is exactly what the free track does.** Given
the levels for nothing, the seat:

| | granted | its baselines | Δ |
| --- | --- | --- | --- |
| workers placed | 22.21 | 19.39 | **+2.82** |
| corn paid to place | 18.66 | 13.03 | +5.63 |
| temple sum | 13.78 | 13.06 | **+0.72** |
| begs | 1.73 | 2.24 | −0.51 |
| buildings | 1.56 | 2.83 | −1.27 |

**Corn converts into actions, +2.8 placements a game, and the actions buy temple
steps.** That is the user's argument, measured, and it is right. It is *also*
why the price is what it is: the same +2.8 placements are what the forced seat
gives up during days 0–12 to stand on Tikal instead.

**What this says about the evaluator.** At `RESEARCH_SCALE = 0.05` the term
values a maxed Agriculture track at day 0 as

    (0.35 + 0.45 + 0.75) * uses(7) * 0.05 = 0.54 points

against a causal **+7.49**. The benefit term is **14× too small**. And yet
raising it loses 4.33 points (F50–F53), because the evaluator has **no term at
all** for the 18.4-point price. Raising the scale makes the agent buy a
7.5-point asset for 18.4 points more often, and losing four points is exactly
what that predicts. *Both* readings in the brief were half-right: the shape is
wrong, and the missing piece is not on the benefit side.

### R10. What the arms are trading, priced across all of them

A per-block regression of centred score on the columns the arms actually move,
pooled over all 18 wave arms (596 blocks), deltas taken against each block's own
three baselines, 95% intervals by bootstrap over blocks. `<scratch>/rv/reg.py`.

| term | points each | 95% CI |
| --- | --- | --- |
| temple step | **+2.62** | [+2.30, +2.92] |
| building | +1.58 | [+1.19, +1.96] |
| monument | **+11.49** | [+7.76, +15.06] |
| **research level** | **−0.33** | [−0.87, +0.21] |
| worker placed | −0.06 | [−0.55, +0.40] |
| | | R² = 0.61 |

Read this as an exchange rate, not as causation — it is a partial slope, so
"research level ≈ 0" says only that levels have no *direct* scoring route once
the things they help you buy are held fixed. That is nearly definitional, and it
is the point: **research has no direct scoring route in this game except
monuments**, its whole case must run through the temple and building columns,
and the monument column — the one place it pays directly — is worth 11.5 points
a monument and the agent holds 0.02 of them (R7).

The temple coefficient reconciles with the rules. `data/temples.rs` scores every
temple **twice**, at day 14 and day 27, with a majority prize each time (Brown
6/2, Yellow 2/6, Green 4/4). A step on Yellow from 4 to 5 is 4 → 6, twice, plus
a share of the prize: 2.6 points a step is what the table says it should be.
**That double-scoring is the real compounding quantity in this game, and its
clock runs out on day 14** — which is precisely the window R6 shows early
research is stealing from.

### R11. The benefit is **convex in depth**, and the price never stops exceeding it

`--grant` run at two depths against the matching `--force` arms. Same seeds,
same binary, `hits = 0` in every grant arm by construction.

| levels | benefit (`--grant`, free) | benefit − price (`--force`) | **price** | ratio |
| --- | --- | --- | --- | --- |
| `A1` | **+1.28** [−0.02, +2.58] | −3.14 [−4.27, −2.00] * | **4.42** | 3.5× |
| `A3` | **+7.49** [+5.81, +9.18] * | −10.94 [−12.16, −9.73] * | **18.43** | 2.5× |

Two things, and both matter for the shape.

**1. One level of Agriculture is worth nothing measurable.** `+1.28 [−0.02,
+2.58]`, win 0.28 against a null of 0.25 — the interval includes zero. Three
levels are worth `+7.49`, win 0.49. The value is **5.9× for 3× the levels**: a
maxed track is worth far more than three times one level. This is the game's
rule, not an artefact — `research.rs` makes Agriculture 3 *replace* Agriculture
1 (+3 corn on green, not +1), so the third level is where nearly all the money
is, and levels 1 and 2 are mostly the toll to reach it.

A term linear in level count is therefore **wrong in both branches**, exactly as
the brief predicted for the monument row: it over-prices the shallow track
almost everybody actually has (`lv = 1.27` in the null) and under-prices the
maxed track almost nobody reaches (1.9% of player-games, F52b).

**2. The price falls with depth too — but never below the benefit.** The ratio
improves, 3.5× at one level to 2.5× at three, so deep research is *relatively*
better than dabbling. It never crosses 1. There is no depth at which buying
Agriculture pays for itself on this board.

**The evaluator's per-step ladder is roughly the right shape and about 12× too
small.** At `RESEARCH_SCALE = 0.05`, day 0:

| | eval | causal | ratio |
| --- | --- | --- | --- |
| `A1` | 0.35 × 7 × 0.05 = **0.12** | **+1.28** | 10× |
| `A3` | 1.55 × 7 × 0.05 = **0.54** | **+7.49** | 14× |

The *ratio between* them (4.4× in the eval, 5.9× measured) is close. The
`research_step_value` ladder is not the broken part. The magnitude is 10–14×
low, and the acquisition price is absent.

### R12. The evaluator's real opinion, measured rather than reverse-engineered — and **the error is per-track, which no global scale can fix**

`--evalcurve` (added to `bin/rlab-r8`) evaluates every position twice — once as
played, once with the levels written into `state.research` — and differences
`eval::heuristic`. That is the evaluator's *whole* opinion of the track,
including everything the levels change indirectly through `space_value`,
`board_position` and the corn terms, not just the `research_step_value` block.
It is the exact analogue of the causal `--grant` arm, so the two are
commensurable. Pinned tree, `9bfcb6e`, the rev F50–F57 measured.

| track | what it is | **eval at day 0** | **causal (`--grant`)** | under-priced by |
| --- | --- | --- | --- | --- |
| `A1` | Agriculture 1 | +0.20 | +1.28 [−0.02, +2.58] | 6.2× |
| `A3` | Agriculture maxed | **+1.18** | **+7.49** [+5.81, +9.18] * | **6.4×** |
| `C3` | Architecture maxed | **+1.29** | **+23.52** [+20.84, +26.20] * | **18.2×** |
| `A1,R1` | breadth, two tracks | — | +1.14 [−0.49, +2.77] | — |

**The evaluator prices a maxed Architecture track and a maxed Agriculture track
within 9% of each other (1.29 against 1.18). Their causal values differ by
3.1× (23.52 against 7.49).** That is a *shape* error in the strict sense the
brief means, and it is the one error in this file that a global
`RESEARCH_SCALE` provably cannot repair: any scale that makes Architecture
right over-prices Agriculture by 2.8×, and any scale that makes Agriculture
right leaves Architecture 2.8× short. F50–F57 swept the only knob that exists
and it was the wrong knob — not because the magnitude is right, but because
one number cannot fix two tracks that are three times apart.

**Maxed Architecture is worth +23.5 points and wins 90% of games.** The
mechanism is in the buildings column and it is not subtle: the granted seat
builds **9.14** buildings a game against a baseline 2.71. `research.rs` makes
Architecture 3 a permanent one-block discount and Architecture 2 pay **+2 points
on every build**, so the track converts directly into repeated scoring. This is
the genuine compounding engine the user reasoned must exist. **It exists, and it
is not on the Agriculture track.**

And it is still not worth buying, by a hair:

| track | benefit | benefit − price | **price** | **benefit / price** |
| --- | --- | --- | --- | --- |
| `A1` | +1.28 | −3.14 | 4.42 | 0.29 |
| `A3` | +7.49 | −10.94 | 18.43 | 0.41 |
| `C3` | **+23.52** | −5.38 | **28.90** | **0.81** |

Architecture reaches **0.81 of break-even** against Agriculture's 0.41. It is
the only thing measured in this file that comes close, and the forcing policy
used to buy it is crude (R13). **This is where a second look belongs** — not at
Agriculture, which the user's example named and which is the weakest of the
tracks tested.

**Breadth is worthless.** `A1,R1` — one level on each of two tracks — is +1.14
[−0.49, +2.77], indistinguishable from zero and no better than `A1` alone
(+1.28). Depth is the entire story: the payoff lives at level 3 on a single
track. A term linear in `lv` — the total level count, which is what
`FINDINGS-eval.md`'s uptake columns report and what the sum over
`research_step_value` computes — cannot see the difference between the
worthless spread and the valuable spike.

**The shape of `uses`, read off the evaluator.** `--evalcurve` also gives the
evaluator's decay directly, which is `uses` plus every indirect term:

| day | 0 | 7 | 14 | 21 |
| --- | --- | --- | --- | --- |
| `A3` eval delta | +1.179 | +0.779 | +0.502 | +0.197 |
| relative | 1.000 | 0.661 | 0.426 | 0.167 |
| `(27 − d)/27` | 1.000 | 0.741 | 0.481 | 0.222 |

The `min(7)` cap does **not** produce a flat top in the evaluator's actual
output — days 0–6 fall from 1.000 to 0.692 — because the indirect terms decay
even while the capped one does not. The effective curve is slightly steeper than
linear in `rounds_left`. Whether that is the *right* curve is R14's question,
and it is answered by the causal grant-day sweep, not by inspecting constants.

### R13. All four tracks, free — and the evaluator cannot tell them apart

`--grant`, `heuristic:full`, seeds 700000+, `hits = 0` throughout. The eval
column is `--evalcurve`'s day-0 delta on the same pinned rev.

| track | what maxing it does | **causal (free)** | win | **eval** | under by |
| --- | --- | --- | --- | --- | --- |
| **Architecture** | +1 corn & +2 pts per build, then a block off the price | **+23.52** [+20.84, +26.20] * | **0.90** | +1.27 | **18.6×** |
| **Theology** | foresight, devout, a second skull from Yaxchilan 4 | **+15.96** [+14.48, +17.44] * | **0.81** | +1.26 | **12.7×** |
| **Agriculture** | +3 corn green, +1 blue, irrigation | **+7.49** [+5.81, +9.18] * | 0.49 | +1.13 | **6.6×** |
| **Extraction** | +1 block per gather, by type | **+4.51** [+3.02, +6.00] * | 0.34 | +1.26 | **3.6×** |

**The four maxed tracks span 5.2× causally (4.51 to 23.52) and 1.12× in the
evaluator (1.13 to 1.27).** The evaluator's `research_step_value` ladder sums to
1.55 / 1.95 / 1.95 / 2.00 for A / R / C / T — it ranks Extraction *equal to*
Architecture, and they differ by 5.2× on the board. No setting of
`RESEARCH_SCALE` repairs that, because it multiplies all four rows by the same
number.

Depth ladder on the two tracks measured at more than one level:

| | level 1 | level 3 | ratio |
| --- | --- | --- | --- |
| Agriculture, causal | +1.28 [−0.02, +2.58] | +7.49 | 5.9× |
| Agriculture, eval | +0.20 | +1.13 | 5.7× |
| **Architecture, causal** | **+4.37** [+2.90, +5.83] * | **+23.52** | 5.4× |
| **Architecture, eval** | +0.17 | +1.27 | 7.4× |

One level of Architecture is worth **+4.37 and a 0.42 win rate** — more than a
*maxed* Extraction track, and 3.4× a level of Agriculture. The evaluator prices
it at 0.17, below its price for a level of Agriculture. The *depth* ladder is
roughly right in both tracks (5.4–5.9× measured against 5.7–7.4× in the eval);
it is the **per-track** ladder that is wrong.

### R14. `uses` is the right shape and it is being cancelled by the terms around it

Question (d), answered without the acquisition cost in the way: the same levels
granted free on four different days. What a static evaluator should report on
day D is exactly the track's *remaining* value on day D, which is what this
measures.

| granted on day | 0 | 7 | 14 | 21 |
| --- | --- | --- | --- | --- |
| **causal** | +7.49 [+5.81, +9.18] * | **+7.67** [+6.17, +9.16] * | +3.62 [+2.28, +4.96] * | +2.26 [+1.62, +2.91] * |
| relative | 1.000 | **1.023** | 0.483 | 0.302 |
| `uses` term alone | 1.000 | 0.952 | 0.619 | 0.286 |
| **whole evaluator** | 1.000 | 0.661 | 0.426 | 0.167 |
| `(27 − d)/27` | 1.000 | 0.741 | 0.481 | 0.222 |

**The true curve is flat from day 0 to day 7 and then falls off a cliff.**
Granting a maxed track on day 7 is worth `+7.67` against `+7.49` on day 0 — the
first seven days of holding it are worth **nothing**, and the intervals overlap
almost exactly. By day 14 it is worth less than half.

`uses = (rounds_left / 3.0).min(7.0)` **has that plateau built in**: the cap
binds while `rounds_left >= 21`, i.e. days 0–6, which is the measured plateau to
within a day. Normalised, the term alone reads **1.000 / 0.952 / 0.619 / 0.286**
against a measured **1.000 / 1.023 / 0.483 / 0.302** — it matches at days 0, 7
and 21 and is 28% high at day 14, its worst point, and it beats a plain linear
`(27 − d)/27` at three of the four. **The `uses` shape is defensible. The cap is
not a guess that happened to be wrong; it is the best-fitting feature of the
curve.**

**Where the curve actually bends is day 14, and the rules say why.** Temples
score on `POINT_DAYS = [14, 27]`. A track granted on day 0 or day 7 is held
through **both** scorings and the two are worth the same (+7.49, +7.67); a track
granted on day 14 or 21 catches **one**, and the pair drops to +3.62 and +2.26.
The step between the regimes is 2.58×, close to the 2× that "two scorings
instead of one" predicts. The residual decay inside each regime is what `uses`
already models. If the curve is ever re-fitted, **the feature to add is a step
at each remaining `POINT_DAY`, not a different power of `rounds_left`.**

What is wrong is that it does not survive. The whole evaluator's delta falls to
0.661 by day 7 (R12) while the term it contains only falls to 0.952, because
every *other* term the levels touch — the corn bonus inside `space_value`, the
`held_premium` corn, `board_position` — decays linearly in `rounds_left` and
drags the total down through the plateau. **The cap is doing the right thing and
being outvoted.**

Note also `research_step_value`'s neighbour, the `lvl == 2 && horizon > 0.15`
step: `--evalcurve` shows `A2` decaying only 1.00 → 0.567 over days 1–22 against
`A3`'s 1.00 → 0.153. A flat +0.4 that switches off at a threshold is the one
piece of this block with a genuinely indefensible shape.

### R15. The evaluator's calibration table: **liquid currency is priced right, engines are priced 4–19× low**

`--pricecurve` (`bin/rlab-r9`) applies the double-evaluation trick to every
currency at once, on real positions, **days 0–6 only** — the window where the
forcing bites. Each row is `eval::heuristic` with the thing and without it, so
the rows are commensurable with each other *and* with the causal column beside
them, which comes from the `--gift` arms (corn, wood, temple), the `--grant`
arms (tracks) and the cross-arm regression (buildings). 30 games, 820
player-turns.

| thing | **eval** | **causal** | source of the causal number | eval / causal |
| --- | --- | --- | --- | --- |
| corn +1 | +0.53 | **+0.60** | `--gift corn:1`, 26 units | **0.88×** ✓ |
| corn +3 | +1.57 | +1.79 | `--gift corn:1/3` scaled | 0.88× ✓ |
| wood +1 | +1.00 | **+1.05** | `--gift wood:1/4`, 7 units | **0.95×** ✓ |
| temple step (best available) | +7.74 | **+5.19** | `--gift temple:1/6` and `1/3`, below | 1.5× high |
| building +1 | +0.46 | +1.58 | cross-arm regression | 0.29× low |
| Agriculture maxed | +0.97 | +7.49 | `--grant A3` | **0.13×** |
| Extraction maxed | +1.11 | +4.51 | `--grant R3` | **0.25×** |
| Architecture maxed | +1.12 | **+23.52** | `--grant C3` | **0.05×** |
| Theology maxed | +1.13 | +15.96 | `--grant T3` | **0.07×** |

The temple row is measured twice, and the two agree — `--gift temple:1/6` is
+17.85 [+15.51, +20.20] for **+3.47** realised steps (**+5.14** each) and
`--gift temple:1/3` is +32.16 [+29.87, +34.45] for **+6.14** (**+5.24** each),
28 and 23 blocks, a 1.8× dose apart. (The cross-arm regression's +2.62 in R10 is
the *average* step in the temple sum; `--gift` hands over the **best available**
step, and best being 2× average is what it should be.)

**The evaluator's resource pricing is excellent and its engine pricing is not.**
Corn is within 12% and wood within 5% of their causal marginal values — after a
whole file spent testing whether corn is mispriced, **it is not**. Buildings are
3.5× low and every research track is 4–19× low.

The pattern is one thing: **the evaluator is well-calibrated on what a turn
converts into now and badly calibrated on what compounds over many turns.**
Research is the most extreme compounding asset in the game and it is the most
under-priced row in the table. *That* is the user's intuition, located and
measured — it was right about the direction and about the reason.

The pivotal ratio for behaviour is the last two columns read against each other.
Per **Tikal action** in the early game:

| the action | causal | eval | |
| --- | --- | --- | --- |
| Tikal 5 — a block for **two temple steps** | ~+10.3 | ~+15.5 | eval 1.5× high |
| Tikal 1/3 — **one Architecture level** (marginal: +4.4 / +8.7 / +10.5) | +4.4 … +10.5 | +0.17 … +0.77 | eval **20×** low |

**The two actions are worth about the same and the evaluator rates one twenty to
sixty times above the other.** The champion is not being stubborn; it is reading
a table in which research has been rounded to zero.

### R16. What to do about it — a concrete shape, and what it is *not*

Owner of `src/eval.rs`: this is a proposal, not a patch. This workstream never
touched that file.

**Do not raise `RESEARCH_SCALE`.** F50–F57 already swept it and every cell at or
above `0.25` loses. R12/R13 say why, and it is not that the magnitude is right:
one scalar multiplies four tracks whose true values span **5.2×**, so any scale
that fixes Architecture over-prices Extraction by 5× and vice versa. **The knob
that exists cannot express the error that is there.**

**The change with the best evidence behind it is scale-neutral.** Keep the total
weight of the research block exactly where it is — so nothing about the agent's
overall appetite for research moves, and F54's plateau is not disturbed — and
**redistribute it across the four tracks** in the measured proportions.
`research_step_value`'s per-track sums are A 1.55 / R 1.95 / C 1.95 / T 2.00
(total 7.45). The causal maxed-track values are A 7.49 / R 4.51 / C 23.52 /
T 15.96 (total 51.48). Holding the total at 7.45:

| track | current sum | **proposed sum** | factor | current steps 1/2/3 | **proposed steps 1/2/3** |
| --- | --- | --- | --- | --- | --- |
| Agriculture | 1.55 | **1.08** | 0.70× | 0.35 / 0.45 / 0.75 | **0.25 / 0.31 / 0.52** |
| Extraction | 1.95 | **0.65** | 0.33× | 0.55 / 0.65 / 0.75 | **0.18 / 0.22 / 0.25** |
| **Architecture** | 1.95 | **3.40** | **1.75×** | 0.30 / 0.85 / 0.80 | **0.52 / 1.48 / 1.40** |
| **Theology** | 2.00 | **2.31** | 1.16× | 0.45 / 0.65 / 0.90 | **0.52 / 0.75 / 1.04** |

The within-track proportions are **kept as they are**, because they measure
right: the level-1-to-level-3 ratio is 5.9× measured against 5.7× in the table
for Agriculture, and 5.4× against 7.4× for Architecture; the Architecture
marginal ladder is 0.186 / 0.369 / 0.446 of the track measured against
0.154 / 0.436 / 0.410 in the table. **`research_step_value`'s shape *within* a
track is one of the better-calibrated things in the file.** Only the four
per-track magnitudes move.

Why this is the right first experiment: it is the one axis F55 never varied.
F55 rotated the *time* shape and found treatment and anti-control on top of each
other; it never rotated the *track* shape, and the track shape is where a 5.2×
error is. It costs one arena run, it cannot move overall research uptake much
because the total is fixed, and it has a directional prediction — the agent
should shift what little research it does from Extraction (benefit/price 0.29)
toward Architecture (0.81), which is the only track measured that comes near
paying for itself.

**Three smaller items, in order of confidence:**

1. **Delete the `lvl == 2 && horizon > 0.15 → +0.4` step.** *(Promoted — see
   R18, measured after this section was written.)* It is not a wart: it is
   **56–61% of the evaluator's whole opinion of a track's depth**, and it sits
   on level 2 when the causal ladder puts 45–65% of the value on level 3. Strip
   it and `research_step_value`'s own within-track shares land near the measured
   ones (Architecture 0.154/0.436/0.410 against a causal 0.186/0.369/0.445). It
   is also badly shaped in time — `--evalcurve` shows it decaying 1.00 → 0.57
   over the game against the neighbouring term's 1.00 → 0.15.
2. **Leave `uses` alone.** R14 measured it against the causal curve and it fits
   at days 0, 7 and 21 and is 28% high at day 14 — better than linear, and its
   `min(7.0)` cap reproduces a real plateau. If it is ever re-fitted, the missing
   feature is a **step as each `POINT_DAY` passes** (the day-14 knee is temples
   scoring twice versus once), not a different power of `rounds_left`.
3. **The plateau is being cancelled downstream.** The `uses` term alone decays
   1.00 → 0.95 over days 0–7; the evaluator's *total* opinion of the same track
   decays 1.00 → 0.66, because the corn and board terms the levels also touch
   decay linearly through the plateau. If the flat top is meant to survive, that
   is where to look — not in `uses`.

**And what none of this will do: make the agent research much.** Every track
measured costs more to acquire than it is worth (R12: benefit/price 0.29, 0.41,
0.81). A correctly-shaped term should still decline most of the time. What it
buys is that when the agent *does* research — or when it evaluates a position in
which someone already has — it will be looking at roughly the right number, and
it will pick Architecture instead of Extraction.

### R17. What would overturn this, and the one number that could

Stated so the next run does not have to rediscover them.

1. **The forcing policy is crude, and this is the real open question.**
   `--tikal 0` is the gentlest arm available — the seat is overridden only on
   turns where a research advance is already on offer — but it still takes the
   research whenever it is offered, no matter what else that turn could buy. A
   policy that researched only when the alternative was weak would pay less than
   the measured price. For Agriculture the price would have to fall **59%** to
   break even, which is implausible. **For Architecture it would have to fall
   19%** (28.90 → 23.52), which is not. Architecture at benefit/price **0.81**
   is the one cell in this file that a better acquisition policy could plausibly
   push over the line, and it is the thing to test next.
2. **The `--grant` numbers may be *low*.** The granted seat still plays greedily
   and never adapts its plan to the track it was handed; `--grant C3` builds 9.14
   buildings a game without ever having *decided* to be a builder. A seat that
   planned around the track would do at least as well. Both this and (1) push in
   the same direction for Architecture.
3. **One-ply base agent.** Everything here is `heuristic:full`, chosen because at
   temperature 0 it draws no random numbers and the arms differ only in the
   candidate's moves. `FINDINGS-eval.md` **F57a** is the check that this
   generalises: a six-turn `mcts:1024` search left alone takes **1.32** research
   levels of twelve against the one-ply agent's **1.24** — the lookahead does not
   discover a use for research that the shallow agent misses. The depth arms
   (`<scratch>/rv/wave2.sh`, written and not run) would price the *acquisition*
   under search; they were skipped deliberately with the machine at load 325 and
   memory the binding constraint.
4. **Opponents never research.** The benefit of a track is measured against three
   seats that mostly do not have one. In a table where everyone researches, the
   Chichen and building contention that Theology and Architecture exploit would
   be tighter and the numbers smaller.
5. **The null is not exactly zero.** `eval::rank_all_within`'s 2,500 ms
   wall-clock deadline is not reproducible under load, so two runs of one seed
   can diverge on a very wide turn. Measured null: **+0.079 [−0.108, +0.267]** at
   48 blocks; every arm above is read against it, not against an assumed zero.
   The same effect produced one spurious `--verify` disagreement on `rlab-r10`
   at load 325 which did not reproduce in three further runs (0/3 on `r9`, 0/3 on
   `r10`).
6. **Block counts are 28–60.** Intervals are quoted throughout; the block is the
   independent unit, four games a block, the candidate rotating through all four
   seats on a shared seed.

## Binaries, and what produced what

All built from `<scratch>/rv/tree`, a `git archive` of **`9bfcb6e`** — the rev
`FINDINGS-eval.md` F50–F57 measured — into a private `CARGO_TARGET_DIR`, so no
other agent's `cargo build` can swap the binary under a running sweep. Source
for each is beside it as `rlab-rN.rs`. `--verify` (this driver against
`record::play_game`) is clean on every one.

| binary | added | produced |
| --- | --- | --- |
| `rlab-r3` | the paced forcing scheme | `out/w_*.jsonl` — the main wave (R5, R6) |
| `rlab-r4` | `--scan`, `--gift`, `--census` | `out/mon_*` (R7), `out/census.*` (R8) |
| `rlab-r7` | **`--grant A3[,R1][@D]`** | `out/g_*` — R9, R11, R13, R14 |
| `rlab-r8` | **`--evalcurve`** | `out/evalcurve_*` (R12), `out/g_C1,C2,A2` |
| `rlab-r9` | **`--pricecurve`** | `out/pricecurve.txt` (R15) |
| `rlab-r10` | **`--gift temple:N/D`** | `out/w_gt*` (R15's temple row) |

`src/bin/rlab.rs` in the repo tracks `rlab-r10`. Scripts: `run.sh`, `wave.sh`,
`mon.sh`, `grant.sh`, `grant2.sh`, `grant3.sh`; analysis `an.py`, `reg.py`.

### R18. The full depth ladders — and the `lvl == 2` bonus is what breaks them

Both tracks measured at every level, free, day 0. `hits = 0` throughout.

| | level 1 | level 2 | level 3 | marginal 1 / 2 / 3 | shares |
| --- | --- | --- | --- | --- | --- |
| **Agriculture**, causal | +1.28 [−0.02, +2.58] | +2.59 [+1.28, +3.90] * | +7.49 [+5.81, +9.18] * | +1.28 / +1.31 / **+4.90** | 0.171 / 0.175 / **0.654** |
| Agriculture, evaluator | +0.20 | +0.83 | +1.13 | +0.20 / **+0.64** / +0.30 | 0.175 / **0.561** / 0.265 |
| **Architecture**, causal | +4.37 [+2.90, +5.83] * | +13.05 [+11.34, +14.75] * | +23.52 [+20.84, +26.20] * | +4.37 / +8.68 / **+10.48** | 0.186 / 0.369 / **0.445** |
| Architecture, evaluator | +0.17 | +0.94 | +1.27 | +0.17 / **+0.77** / +0.33 | 0.136 / **0.605** / 0.259 |

**The money is at level 3 on both tracks and the evaluator puts it at level 2 on
both.** Causally the third level carries 65% of Agriculture and 45% of
Architecture; in the evaluator it carries 27% and 26%, while level 2 carries 56%
and 61%. The ladder's peak is in the wrong place, in the same direction, on both
tracks measured.

**The cause is the `lvl == 2 && horizon > 0.15 → +0.4` step, and stripping it
fixes the ladder.** `research_step_value`'s own within-track shares, with that
bonus removed, are:

| | level 1 | level 2 | level 3 |
| --- | --- | --- | --- |
| Agriculture — `research_step_value` alone | 0.226 | 0.290 | 0.484 |
| Agriculture — **causal** | 0.171 | 0.175 | **0.654** |
| Architecture — `research_step_value` alone | 0.154 | 0.436 | 0.410 |
| Architecture — **causal** | 0.186 | 0.369 | **0.445** |

For Architecture that is a near-exact match (0.154/0.436/0.410 against
0.186/0.369/0.445). For Agriculture level 3 is still under-weighted, but the
peak is at least in the right place.

This **upgrades R16's item 1 from a tidy-up to the second most valuable change
in the file.** A flat +0.4 behind a threshold is not a small wart: it is 56–61%
of the evaluator's entire opinion of a track's depth, it sits on the wrong
level, and it is why a term whose underlying ladder is close to correct reads as
badly shaped. `FINDINGS-eval.md` F55 records that `rtop` — a convexity term
keyed on *finished* tracks — measured inert; that is consistent with this, since
the convexity was already being spent, at the wrong level, by the +0.4.
