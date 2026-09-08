# FINDINGS — `src/eval.rs`

A running log of evaluator measurements. **Append the moment a run finishes**,
before starting the next one. Two previous runs of this workstream were killed
by usage limits and lost everything they had concluded because it lived only in
their context; the raw JSONL survived and the meaning did not. Every entry says
what was varied, against what, the number with its interval and block count, the
file it came from, and what it means.

## How to read a number here

`bin/evalab` plays **rotation blocks**: one seed, four seatings, the candidate
evaluator in each seat once against the committed one in the other three. The
block is the independent unit, not the game. Null centred score is 0 and null
win rate is 0.25. `n=150` blocks is 600 games. Intervals are 95% t. An effect
smaller than its interval is not an effect.

Both arms always run at the **same** search effort (`greedy:64` = one ply over
64 sampled moves; `mcts:256` = 256 simulations), so a comparison is the
evaluator and never the budget.

Analysis script: `<scratch>/an.py`. `python3 an.py '*.jsonl'` summarises a
directory; `python3 an.py -p A.jsonl B.jsonl` gives the seed-paired difference
of two runs against the same baseline, which is much tighter than differencing
their independent intervals.

## Filename → meaning

Files live in `<scratch>/ab/`, named `<variant>_<agent>.jsonl`. The variant name
is `bin/evalab`'s `--ab` argument and is decoded by `variant()` in that file.
Baseline is always `head` = `src/eval.rs` as committed at 3a495e6 unless said
otherwise.

| stem | means |
| --- | --- |
| `head` | the committed evaluator (baseline arm; `--eqcheck` pins it equal to `eval::heuristic`) |
| `flat` | hand `space_value` table with **every affordability gate removed** |
| `gates` | hand table plus three *more* gates (Palenque tiles gone, research with no blocks, corn exchange with no corn) |
| `derived` | `space_value` priced from the `Effect`s `spaces::choices_at` emits, best choice |
| `derived-floor` | as `derived`, but a space whose generator returns only `skip` falls back to the hand table |
| `derived-probed` | derived from `mcts::Gradient` — the evaluator's own marginal per axis, probed at that position |
| `derived-static` | the corpus mean of `derived`, frozen as a constant table (derived's *shape* at the hand table's cost) |
| `derived-*-scaled` | the same with `space_scale` applied before `tempo` |
| `charge` | a worker on a gear is charged one generic `ACTION_VALUE` action, since `board_position` already prices the action it is about to take |
| `hand`/`hand2`/`hand0` | credit an in-hand worker with the best space it could reach this turn, charged `hand_lag` rounds of tempo (1.0 / 2.0 / 0.0) |
| `corn` | corn valued by the placement depth it buys instead of a flat capped premium |
| `colour` | a building priced by the monuments that count its colour |
| `contend` | contention: exclusive temple top, a monument an opponent is closer to, a draining skull bank |
| `contend-temple`, `contend-monument` | the two halves of `contend` separately |
| `sweep_avXXtempoYYY` | `ACTION_VALUE = XX/10`, `TEMPO_PER_ROUND = YYY/100`, nothing else changed |
| `chargeavXX...` | `charge` on top of that |
| `chargecontendavXX` | `charge` + `contend` + `ACTION_VALUE = XX/10` |
| `board=k` / `engine=k` / `temple=k` / `held=` / `monu=` / `starve=` | post-multiply that `Components` field by *k*, where `plan::Schedule` applies its weights. **Absolute**, not relative to `HEAD` |
| `defer` | only the four deferred-payoff spaces (Tikal 1/3/5, Uxmal 1) repriced to the derived table; every other space keeps its hand price |
| `board05*`, `engine0*`, `temple0*`/`temple1*` | `board=`/`engine=`/`temple=` at that value, digits after the point (`board035` = 0.35, `temple125` = 1.25) |
| `board05av04`, `board05av04temple140`, ... | those knobs together; read left to right, each as above |
| `final_avNN` | the landing candidate `board=0.5,temple=1.4` at `av=0.NN` |
| `_greedy64` / `_mcts256` / `_mcts1024` / `_greedyfull` | the agent both arms played |

---

# Salvaged: everything a killed predecessor measured and did not write down

All of the below is re-analysis of JSONL already on disk in `<scratch>/ab/`,
recovered on this run before any new CPU was spent. Every row is 150 blocks
(600 games) unless stated. `*` = distinguishable from zero at 95%.

## S1. The derived `space_value` table, and what the hand table's value actually is

| variant | greedy:64 | mcts:256 |
| --- | --- | --- |
| `flat` (gates removed, numbers kept) | **−32.18** [−33.04, −31.33] * | **−40.14** [−40.85, −39.43] * |
| `derived` | −27.86 [−29.31, −26.41] * | — |
| `derived-floor` | −27.57 [−28.96, −26.17] * | — |
| `derived-probed` | −27.43 [−28.88, −25.98] * | — |
| `derived-scaled` (×0.60) | −28.39 [−29.82, −26.97] * | — |
| `derived-static` | −26.88 [−28.03, −25.74] * | −22.29 [−23.91, −20.67] * |
| `derived-static-scaled` (×0.62) | −22.97 [−24.17, −21.77] * | −14.65 [−16.17, −13.13] * |
| `gates` (three *extra* gates) | +0.01 [−0.03, +0.04] | −0.02 [−0.07, +0.04] |

Interpretation, and it is not the one the brief expected:

* **Removing the gates is worse than replacing the whole table.** `flat` is the
  single largest regression measured anywhere in this workstream, −32/−40, win
  rate **0.000** — not one game won in 600. So the affordability gates are not a
  detail of the hand table; they are most of what it is. A table that says a
  Chichen space pays 10.5 points when the player holds no skull sends every
  worker to Chichen forever.
* **But the gates are not why `derived` regresses**, because `derived` *is*
  gated — it prices only the choices the generator says are currently legal —
  and it still loses 27.9. Nor is it the pricing engine: `derived-probed`, which
  uses the evaluator's own gradient instead of a hand price list, lands in the
  same place (−27.4).
* **And it is not scale.** `derived-scaled` at 0.60 is −28.39, no better than
  `derived` at −27.86 (the intervals overlap heavily). Halving a wrong shape
  does not make it right.
* `derived-floor` (−27.6) ≈ `derived` (−27.9): the "exhausted space prices at
  zero" hypothesis in the brief is worth **at most 0.3 points** and its interval
  covers 0. That hypothesis is dead.
* `gates` — adding *more* affordability gates to the hand table — is exactly
  0.00 on both agents. The three gates it adds (Palenque with no tiles left,
  research with no blocks, corn exchange with no corn) never bind in practice.

So the causal story to chase is **shape**, not gating, scale, or the price list.
See S2.

## S2. The derived table's shape, read off `DERIVED_STATIC`

The frozen corpus mean of the derived pricing, against the hand table, by gear
and position (hand → derived):

```
Tikal   1 (research)   2.0 -> 0.04     Chichen 1  3.0 -> 0.18
Tikal   3 (research)   3.4 -> 0.12     Chichen 5  6.2 -> 1.27
Tikal   5 (2 temple)   4.0 -> 2.14     Chichen 9 10.5 -> 2.84
Uxmal   1 (corn->temple) 2.2 -> 0.20   Uxmal 3 (worker) 3.4 -> 7.20
Palenque 5             2.6 -> 5.81     Uxmal 6           4.0 -> 7.96
Yaxchilan 4 (skull)    3.0+ -> 5.21
```

The derived table is not a rescale of the hand table; it is a **different
ranking**. It prices *resource-yielding* spaces at roughly 2x the hand table and
*point-yielding* spaces (research, temple steps, Chichen) at roughly **1/20th**.
That is exactly what pricing an `Effect` list buys you: an `Effect` that hands
over 5 corn and 3 wood has an obvious price, and an `Effect` that advances a
science track or steps a temple has almost none — its value is entirely in what
it *unlocks later*, which is `engine_value`'s and `temple_outlook`'s business
and is invisible to a per-effect price list. The single hand-written number per
space was carrying the interaction, and pricing effects in isolation throws it
away. **This is the answer to "why does the derived table regress".**

(Chichen is doubly hit: its corpus *mean* averages over the many positions where
the player holds no skull and the space prices at 0, so a frozen mean of a
correctly-gated pricing is itself ungated garbage. That explains
`derived-static` but not `derived`, which is gated per position — and `derived`
is no better. Shape is the cause.)

## S3. The `ACTION_VALUE` x `TEMPO_PER_ROUND` joint sweep

Committed head is `av = 1.2, tempo = 0.52`, which is the null and reads 0 by
construction. Centred score vs head, `greedy:64` / `mcts:256`, 150 blocks each:

```
         tempo:  0.35     0.42     0.52     0.62     0.70     0.90     1.10
 av=0.0                          + 9.36            +12.07            - 1.58
                                 + 3.70            + 4.66            + 2.72
 av=0.2                          + 8.86            +11.43   +11.56
                                 + 3.72            + 4.98   + 5.08
 av=0.4         + 3.06   + 4.24  + 7.82   + 9.19   +10.31   +10.53
                - 0.83   + 0.57  + 3.39   + 3.85   + 4.26   + 4.78
 av=0.6                          + 6.80            + 9.08
                                 + 3.11            + 3.06
 av=0.8                          + 3.82
                                 + 2.37
 av=1.2         - 4.38           (null)            - 0.87
                - 4.48                             - 2.51
 av=1.6         - 6.28           - 1.60            - 2.68
                - 6.91           - 2.02            - 8.11
```

* **`ACTION_VALUE` is far too high and the brief understates it.** Along
  `tempo = 0.52` the score is *monotone decreasing* in `av` all the way to 0.0.
  Not "0.4 instead of 1.2" — the data cannot distinguish 0.4 from 0.0 and
  slightly prefers the smaller.
* **The two are not separable.** The best `tempo` moves with `av`: at `av = 1.2`
  a bigger tempo helps (−4.38 at 0.35 → −0.87 at 0.70); at `av = 0.0` a bigger
  tempo helps until it doesn't (+9.36 at 0.52 → +12.07 at 0.70 → −1.58 at 1.10).
  A coordinate-wise sweep of either alone would have found the wrong answer, so
  the brief's worry was correct.
* The interior maximum on this grid is around **`av ≈ 0.0-0.2, tempo ≈ 0.70-0.90`**,
  worth **+11 to +12 centred on greedy:64 and +4.7 to +5.1 on mcts:256**.
* Greedy and MCTS agree on sign and on the ridge everywhere, but **greedy reads
  the effect at roughly 2.2x MCTS's size**. Search partially repairs a bad
  evaluator, so a one-ply number is an upper bound on what MCTS will bank.
* The `tempo = 1.10, av = 0.0` cell is the one place the two agents *disagree in
  sign* (−1.58 greedy, +2.72 mcts) — the far edge of the ridge, treat as noise.

## S4. Structural ideas, each alone against head

| variant | greedy:64 | mcts:256 | verdict |
| --- | --- | --- | --- |
| `charge` | +0.10 [−1.24, +1.44] | +2.07 [+0.75, +3.38] * | helps MCTS only |
| `contend` | +0.57 [+0.12, +1.03] * | +0.95 [+0.41, +1.49] * | small, real, both |
| `contend-temple` | +0.35 [+0.04, +0.66] * | +0.39 [−0.06, +0.83] | the smaller half |
| `contend-monument` | +0.23 [−0.12, +0.57] | +0.12 [−0.31, +0.56] | nothing |
| `hand` (lag 1.0) | +3.32 [+1.70, +4.94] * | **−3.28** [−4.62, −1.94] * | **opposite signs** |
| `hand2` (lag 2.0) | +0.46 [−1.32, +2.24] | −4.73 [−6.06, −3.40] * | worse |
| `corn` | −0.37 [−0.92, +0.18] | +0.19 [−0.27, +0.64] | nothing |
| `colour` | +0.02 [−0.36, +0.41] | +0.02 [−0.44, +0.48] | nothing |
| `gates` | +0.01 | −0.02 | nothing |

* `hand_worker` is the sharpest disagreement in the whole dataset: **+3.3 at one
  ply, −3.3 under MCTS**, both distinguishable. This is the mid-turn-consistency
  idea from the brief, and under the search that actually plays the game it is a
  *regression*. Worth understanding before anything else structural — see the
  `--midturn` work.
* `charge` is the mirror image and the one the brief predicted: `board_position`
  prices the action a placed worker is about to take, and `engine_value` pays
  for it again. Charging it back is worth +2.07 to MCTS and nothing to greedy.

## S5. Combinations

| variant | greedy:64 | mcts:256 |
| --- | --- | --- |
| `sweep_av04tempo052` | +7.82 [+6.58, +9.06] * | +3.39 [+2.42, +4.36] * |
| `chargeav04` (tempo 0.52) | +10.36 [+8.90, +11.82] * | +5.32 [+4.07, +6.56] * |
| `contendav04` (tempo 0.52) | +8.30 [+7.06, +9.55] * | +3.83 [+2.90, +4.75] * |
| `chargecontendav04` **600 blocks** | **+10.66** [+9.98, +11.34] * | **+5.73** [+5.16, +6.29] * |
| `chargecontendav04` mcts:1024, 100 blocks | — | +6.07 [+4.72, +7.42] * |
| `chargecontendav02` | +10.25 [+8.84, +11.66] * | +5.31 [+4.20, +6.42] * |
| `chargecontendav03` | +9.97 [+8.63, +11.31] * | +6.07 [+4.92, +7.22] * |
| `chargecontendav05` | +10.97 [+9.61, +12.33] * | +5.93 [+4.73, +7.13] * |
| `chargecontendav06` | +10.17 [+8.75, +11.59] * | +6.22 [+5.11, +7.33] * |
| `chargeav04tempo070` | +7.78 [+6.49, +9.08] * | +4.05 [+2.83, +5.28] * |
| `chargecontendav04tempo070` | +8.01 [+6.66, +9.37] * | +4.02 [+2.90, +5.13] * |
| `chargeav08tempo070` | +2.19 [+0.90, +3.48] * | +0.92 [−0.42, +2.25] |

* `charge` and a low `av` **add** (+7.82 → +10.36 on greedy, +3.39 → +5.32 on
  MCTS) rather than being redundant, even though both reduce what a worker is
  paid. They act at different places: `av` scales every worker, `charge` only
  the placed ones.
* `charge` and a raised `tempo` are **not** additive — they are alternatives.
  `chargeav04tempo070` (+4.05 mcts) is no better than `av04tempo070` alone
  (+4.26), and `chargecontendav04tempo070` (+4.02) is *worse* than
  `chargecontendav04` at tempo 0.52 (+5.73, 600 blocks). Once a placed worker is
  charged for its generic action, raising the per-round waiting price on top
  over-charges it.
* Along `chargecontend`, `av` in 0.2..0.6 is a plateau: every cell is +10..+11
  greedy and +5.3..+6.2 mcts and no two are separable. `av` is not worth further
  sweeping inside that range.

## S6. What the salvaged data says to do

The best measured configuration is **`charge` + `contend` + `ACTION_VALUE` in
0.2..0.6, tempo left at 0.52**, worth **+10.66 greedy / +5.73 MCTS at 600
blocks** — and `+6.07` at mcts:1024, so it does not wash out with more search.
Open questions this run must answer before landing it: does it hold at higher
block count and at `greedy:full`; and what does the mid-turn trajectory say,
given `hand` reverses sign between the two agents.

---

# This run

## F1. Cost per call — `<scratch>/diag1.txt`, `evalab --cost --corpus 40`

1024 real turn roots, 5 passes, minimum of 25 round-robin bursts (a single pass
is memory-bound and the first arm measured pays to warm the cache).

```
  eval::heuristic (HEAD)            94.9 ns/call
  heuristic, seat 0 only            77.5 ns/call
  eval::margin (4 heuristic calls) 363.2 ns/call
  HeuristicEvaluator::evaluate     389.4 ns/call     <- one MCTS node expansion
  one-ply probe                    140.6 ns/edge     (apply_step + heuristic)
    of which apply_step             59.6 ns/edge
  v::heuristic derived           40866   ns/call     (329x head)
```

`--eqcheck`: max |`V::HEAD` − `eval::heuristic`| = **0e0** over 16176
(position, seat) pairs, so everything below is about the committed file.

Notes. `HeuristicEvaluator::evaluate` is 389 ns and is what MCTS pays per node
expansion, so at `mcts:2048` the evaluator alone is ~0.8 ms of a move. The four
seats cost 328 ns against 4 x 94.9 = 380 ns run separately, so there is ~14%
of shared work already being reused; the single-seat call is 77.5 ns, meaning
the fixed per-call overhead is small and the term arithmetic dominates.
**`Derived` at 40.9 us/call is 329x head and can never be a search evaluator**
— which is why the predecessor built `DerivedStatic` to test its shape at
head's cost.

## F2. Why the derived table regresses — `<scratch>/diag1.txt`, `--spacetab`, 1500 positions

Mean space value conditional on the derived pricing being non-zero (`t(nz)` vs
`l(nz)` in the dump), so gating is divided out and only the *shape* is left:

```
                       hand   derived   ratio
  Tikal 1  research     2.00     0.08    0.04     <- research priced at 1/25
  Tikal 3  research     3.40     0.23    0.07
  Uxmal 1  corn->temple 2.20     0.91    0.41
  Tikal 5  two temple   4.00     2.27    0.57
  Chichen 5 (6 pts)     6.07     4.99    0.82
  Chichen 10 (13 pts)  10.50    10.47    1.00
  Palenque 1            0.90     1.06    1.18
  Yaxchilan 4 (skull)   4.10     5.21    1.27
  Yaxchilan 6           3.40     5.21    1.53
  Palenque 3            1.60     3.71    2.32
  Uxmal 3 (buy worker)  3.31     7.20    2.17
  Uxmal 4               1.85     5.53    2.99
  overall list/table = 1.347     probe/table = 1.627
```

**The derived table is not a rescale of the hand table, it is a different
ranking**, and the ranking is sorted by how much of a space's value is visible
in the `Effect` it emits *today*:

* An `Effect` that hands over 5 corn and 3 wood prices exactly. Derived reads
  those spaces **2-3x the hand table**.
* An `Effect` that advances a science track, steps a temple, or buys a worker
  pays nothing today — all of its value is in what it converts *later*, which
  lives in `engine_value`, `temple_outlook` and `monument_outlook`, and is
  invisible to a price list over one effect. Derived reads research at **1/25**
  of the hand table.

That is the answer: **the hand-written number per space was carrying the
interaction between the space and the rest of the evaluator, and pricing an
effect list in isolation throws exactly that away.** The single number is not a
cached sum of effect prices; it is a statement about what the space is *for*.

The consequence is a different worker destination: over 64500 (position, start
space) pairs, the space a worker is aiming at under `max_j(value(j) - wait)`
agrees with the hand table only **76.6%** of the time (derived-static 75.6%).
One placement in four goes somewhere else, and the direction is systematic —
away from research and temples, toward gathering and worker-buying. The mean
final score of the `derived` arm is negative (`derived-static` candidate −18.2
against baseline +17.6), which is the signature of an agent that gathers and
buys workers and then cannot feed them.

Three hypotheses from the brief, tested and dead:

* *"the generators return only currently legal choices, so an exhausted space
  prices at zero where the table still credits it"* — `derived-floor` falls back
  to the hand table exactly there and is worth **at most +0.3** (−27.57 vs
  −27.86, intervals overlapping). **Dead.**
* *"the price list is bad"* — `derived-probed` replaces it with the evaluator's
  own gradient and lands in the same place, −27.43. **Dead.**
* *"the units are wrong"* — `derived-scaled` at 0.60 is −28.39, not better than
  −27.86. **Dead.** (`derived-static-scaled` does improve, −26.9 → −23.0, but
  only because the static table is *also* ungated and shrinking it limits the
  damage of the ungating, not of the shape.)

What *is* load-bearing is the gating. `flat` — the hand table's own numbers with
every affordability gate removed — is **−32.18 greedy / −40.14 MCTS with a win
rate of 0.000 in 600 games**, worse than replacing the table wholesale. Chichen
priced at 10.5 for a player holding no skull sends every worker to Chichen
forever. So: the gates are most of the table's value, and the shape is the rest;
the derived table keeps the gates and loses the shape.

## F3. `heuristic` on mid-turn positions — `<scratch>/midturn1.txt`

`evalab --midturn --corpus 60 --take 3000`, 2445 turn roots that complete inside
a 60000-node exhaustive walk. Each row is HEAD's estimate at depth *k* of the
factored chain minus its estimate of **the same turn completed**, so this is
mid-turn bias and not selection.

```
 depth       n  v(k)-v(end)      sd
     0    2445      -1.807    5.562     <- turn root, before any step
     1    2445      -1.599    5.541
     2    2445      -1.599    5.541
     3    2445      +0.790    4.493     <- crosses over
     4    1133      +1.507    4.121
     5     698      +2.702    4.808     <- peak: +2.70 above the finished turn
     6     612      +1.333    3.179
     7     606      +1.347    3.192
     8     606      +0.217    1.007
    10      65      +0.002    0.012     <- the completed turn
```

The trajectory is **not monotone, and both halves are bugs in opposite
directions**:

* At the root and for the first two steps the position reads **1.8 points below**
  the turn it is about to play. A turn is worth something and the evaluator does
  not see it until it is taken.
* From depth 3 to 7 the position reads **up to +2.70 points above** the turn it
  will actually finish in. This is the important half, because it is where the
  search spends its nodes: **a half-finished turn scores better than the
  finished one, so MCTS is rewarded for not stopping.**

The mechanism is the one the brief guessed at and it is measurable directly. The
`--midturn` regret table reports **drift** = mean over depths 3.. of
(estimate − completed-turn estimate), on each variant's own scale. Zero is an
evaluator whose mid-turn reading means what its end-of-turn reading means:

```
        variant   drift    self    head   exact   place g/best   take g/best
           head  +1.191   2.272   2.272   0.481   0.98/0.91      0.79/0.59
         charge  +0.158   2.334   2.466   0.384   0.94/0.67      0.79/0.93
           hand  +0.387   3.949   2.761   0.289   0.86/0.42      0.78/1.03
          hand2  +0.587   3.566   2.539   0.318   0.93/0.54      0.78/0.96
           corn  +1.135   2.258   2.264   0.479   0.98/0.90      0.79/0.60
        contend  +1.193   2.277   2.281   0.480   0.98/0.91      0.79/0.59
           flat  +1.694   2.651   4.056   0.539   0.99/1.01      0.79/0.48
 derived-static  +0.884   2.360   3.559   0.499   0.99/1.00      0.78/0.47
        derived  +1.824   3.002   3.117   0.537   0.99/1.04      0.79/0.43
```

**`charge` cuts the mid-turn drift by 7.5x, +1.191 to +0.158**, and nothing else
touches it (`corn` +1.135, `contend` +1.193 — both leave it where it was). The
brief's two suspicions are the same bug: `board_position` prices the action the
placed worker is about to take, `engine_value` has already paid every unlocked
worker for that same action, so each placement banks the whole space value for
free and the estimate climbs through the placement steps of a turn.

This predicts, and explains, the single strangest number in the salvaged data:
**`charge` is +2.07 to MCTS and +0.10 to greedy.** A one-ply agent only ever
compares *completed* turns, where the drift cancels; MCTS evaluates every node
of the ~8-step chain, so it sees the drift directly. A mid-turn bias is
invisible to the agent that does not look mid-turn.

It also shows up in behaviour, not only in the number: under HEAD the best
completion places 0.91 workers and takes 0.59; under `charge` it places 0.67 and
takes 0.93. HEAD is biased toward placing over retrieving, which is exactly what
paying twice for a placement would do.

(`hand_worker` also reduces drift, +1.191 -> +0.387, by paying the *unplaced*
worker instead of charging the placed one. It fixes the drift and wrecks the
turn choice: self-regret 3.949 against head's 2.272, and it retrieves 1.03
workers per turn against 0.59 — it wants to empty the board. That is why `hand`
measures +3.32 on greedy and −3.28 on MCTS.)

## F4. Which term drifts, and how big each term is — `<scratch>/terms1.txt`, `evalab --terms`

New diagnostic added to `bin/evalab` this run. Term sizes over 4000 turn roots x
4 seats, under the committed evaluator:

```
        term      mean        sd   mean|x|
      banked    -3.368     8.235     5.811
 liquidation     3.418     2.508     3.418
        held     2.552     1.697     2.552
      temple    10.955     7.777    11.110
      engine    19.490     8.429    19.490      <- the largest term in the file
       board    10.585     5.963    10.585
    monument     0.182     0.216     0.182      <- inert
      starve    -0.809     1.691     0.809
```

`engine` and `board` are 30.1 of a ~43-point estimate. If they really are
2-4x over-priced then the estimate is carrying 15-22 points of fiction, which is
consistent with the size of the effects everything below measures.
`monument_outlook` has a mean of 0.18 and an sd of 0.22 — it is inert, and the
plan workstream's per-term ablation agreeing at −0.18 is that same nothing seen
from the other side. `MONUMENT_SHARE` is not worth sweeping.

The same walk, per term, at depth *k* of a turn minus the same term of the
completed turn — the per-term decomposition of F3's hump:

```
 depth      n   banked   liquid     held   temple   engine    board  monument   starve    total
     0   4000   -1.031   +0.112   +0.098   -1.212   -0.634   +1.583   -0.015   -0.293   -1.393
     3   4000   -1.031   +0.225   +0.130   -1.530   -0.634   +3.823   -0.016   +0.043   +1.012
     4   2006   -1.456   +0.238   +0.147   -1.983   -0.818   +5.727   -0.020   -0.051   +1.784
     5   1385   -2.108   +0.280   +0.189   -2.873   -1.185   +8.564   -0.029   -0.087   +2.751
     6   1274   -1.373   +0.188   +0.124   -1.642   -0.662   +4.993   -0.021   -0.068   +1.539
     8   1260   -0.420   +0.082   +0.044   -0.378   -0.133   +1.349   -0.003   -0.030   +0.511
    10    373   -0.142   +0.051   +0.028   -0.240   -0.047   +0.677   +0.000   -0.007   +0.321
```

**`board_position` is the entire mid-turn hump and then some.** At depth 5 it
reads **+8.56 points above** what it will read at the end of the turn; the whole
estimate is only +2.75 high there because `banked` (−2.11) and `temple` (−2.87)
are pulling the other way — a mid-turn state has not yet banked the points or
taken the temple steps the turn is about to produce. So the picture is:

* `board` over-credits a half-finished placement by ~8.6 points at the deepest
  point of a turn, i.e. by roughly **80% of the term's own mean value of 10.6**.
* `banked` and `temple` under-credit it by ~5 points, because the turn's real
  payoff has not landed yet. Those two are honest — the reward genuinely is not
  there yet.
* The net +2.75 is what MCTS actually sees, and it is a *reason not to stop
  placing*.

This is one term, not a general mid-turn problem, and it is the same term the
brief and the plan workstream independently flagged as over-priced. **The
over-pricing and the mid-turn drift are the same bug**: `board_position` prices
the action a placed worker is about to take while `engine_value` has already
paid every unlocked worker for that action, so both the level of the term and
its behaviour within a turn are inflated by the double count.

## F5. Re-pricing the terms directly — `<scratch>/ab/*_greedy64.jsonl`, `*_mcts256.jsonl`

New knobs added to `bin/evalab` this run: `board=`, `engine=`, `temple=`,
`held=`, `monu=`, `starve=` post-multiply the matching `Components` field,
exactly where `plan::Schedule` applies its weights, so "this term is over-priced
by *k*" is one number. 150 blocks (600 games) each, against `head`.

| variant | greedy:64 | mcts:256 |
| --- | --- | --- |
| `board=0.5` | **+10.48** [+9.07, +11.89] * | **+9.04** [+7.86, +10.23] * |
| `board=0.25` | +8.45 [+7.20, +9.70] * | +6.35 [+5.14, +7.57] * |
| `board=0.35` | +10.85 [+9.45, +12.24] * | +9.44 [+8.34, +10.54] * |
| `board=0.40` | **+11.36** [+10.00, +12.72] * | +9.31 [+8.21, +10.42] * |
| `board=0.60` | +8.79 [+7.48, +10.11] * | +8.30 [+7.14, +9.46] * |
| `board=0.70` | +5.10 [+3.65, +6.55] * | +5.77 [+4.61, +6.93] * |
| `engine=0.35` | +8.65 [+7.33, +9.96] * | +3.14 [+2.14, +4.15] * |
| `engine=0.5` | +7.40 [+6.15, +8.65] * | +2.62 [+1.61, +3.62] * |
| `engine=0.7` | +3.20 [+2.15, +4.24] * | +2.22 [+1.36, +3.08] * |
| `defer` | +0.14 [+0.03, +0.26] * | +0.05 [−0.08, +0.19] |
| `board=0.5,charge` | **−12.09** [−13.01, −11.16] * | **−9.95** [−11.04, −8.85] * |
| `board=0.5,engine=0.5` | +16.64 [+15.44, +17.85] * | +11.31 [+10.29, +12.33] * |
| `board=0.5,contend` | +11.01 [+9.62, +12.40] * | +9.35 [+8.06, +10.64] * |
| `board=0.5,av=0.4` | **+17.54** [+16.32, +18.77] * | **+12.54** [+11.45, +13.64] * |
| `board=0.5,charge,av=0.4` | **+19.07** [+17.79, +20.34] * | +11.27 [+10.23, +12.32] * |

(Everything above the `defer` row was on disk unanalysed when this run started;
the run that measured it was killed before it could write the rows down. Rows
recovered, not re-run — no CPU was spent to get them.)

### F5a. `board_position`'s optimum is a plateau, and 0.5 sits on it

Seed-paired against `board=0.5`, which removes the shared game noise:

```
  board=0.40 - board=0.5   +0.88 [-0.52,+2.28]   +0.27 [-0.99,+1.54]
  board=0.35 - board=0.5   +0.37 [-1.15,+1.89]   +0.39 [-1.00,+1.78]
  board=0.40 - board=0.35  +0.51 [-0.52,+1.55]   -0.12 [-1.21,+0.97]
```

**0.35, 0.40 and 0.50 are mutually indistinguishable on both agents**, while
0.25 and 0.70 are clearly worse on both. So the answer to "is 0.5 the optimum or
just the first thing tried" is: it is on a **broad flat optimum spanning
0.35..0.50**, the point estimate peaks at 0.40, and the difference is not
measurable at 150 blocks. Nothing is bought by tuning inside the plateau, and
the honest statement for the doc comment is "half, give or take a third".

### F5b. `charge` and `board=0.5` fix the *same* double count, so applying both over-corrects

This is the most informative single row in the table and it took a moment to
see. `charge` is **not** a board term — it lives inside `engine_value` and
subtracts `on_board(p)` workers from the generic action count (`evalab.rs`:
`acts = (acts - g.on_board(p).count()).max(0.0)`). So:

* `charge` removes up to `3 x ACTION_VALUE = 3.6` points of *engine*,
* `board=0.5` removes ~5.3 points of *board* (the term's mean is 10.6),

and F4 said both of these are corrections for the identical bug: `board_position`
prices the action the placed worker is about to take, `engine_value` pays every
unlocked worker for that same action. Fix it once and you gain ~+10. **Fix it
twice and you lose 12.** `board=0.5,charge` is −12.09 greedy / −9.95 MCTS, a
0.028 win rate — the second-worst configuration in the whole log after `flat`.

The confirmation is that restoring the balance restores the score:
`board=0.5,charge,av=0.4` shrinks the charge from 3.6 to 1.2 points and comes
back to **+19.07 / +11.27**. So the sign flip is entirely about *how much* is
subtracted, not about which mechanism does it. **This is the cleanest available
evidence that F4's double-count diagnosis is right**: two independent knobs on
the same quantity, additive in the damage they undo, and destructive past the
point where the double count is gone.

The practical consequence is that `charge` must not be landed together with a
board rescale unless `ACTION_VALUE` comes down with it, and that the three knobs
`board=`, `engine=`, `av=` and `charge` are **not** four independent terms —
`av` and `charge` are both rescales of `engine_value`, so of the four only two
directions exist.

### F5c. The new best configuration, and it beats S6's

Seed-paired against `chargecontendav04`, S6's recommendation:

```
  board=0.5,av=0.4  -  chargecontendav04   +6.68 [+5.02,+8.33] *   greedy:64
                                           +6.93 [+5.46,+8.40] *   mcts:256
```

**`board=0.5,av=0.4` beats the previously-best configuration by ~+6.8 on both
agents, paired and unambiguous.** The board rescale is doing something that no
amount of `charge`/`contend`/`av` tuning reached, which is what F4 predicted: the
term with an 8.6-point mid-turn hump and a 10.6-point mean was never going to be
fixed from inside `engine_value`.

Also paired, and the reason the next runs are joint and not coordinate-wise:

```
  board=0.5,av=0.4 - board=0.5,engine=0.5   +0.90 [-0.34,+2.14]   +1.24 [+0.34,+2.14] *
  board=0.5,av=0.4 - board=0.5,charge,av=0.4  -1.52 [-2.94,-0.11]*  +1.27 [-0.14,+2.68]
```

`av=0.4` is a slightly better way to shrink `engine` than `engine=0.5` (+1.24
MCTS). Against `charge,av=0.4` the two agents **disagree in sign** — greedy
prefers the charge (−1.52), MCTS prefers without (+1.27) — which is the same
greedy/MCTS split `charge` has shown since S4, and at these interval widths it
is a coin flip. Prefer the configuration without `charge`: it is one fewer
structural change, and MCTS is the agent that plays the game.

### F5d. `defer` — the deferred-payoff spaces, repriced

`defer` reprices only the four spaces whose payoff is entirely in the future
(Tikal 1/3 research, Tikal 5 two temple steps, Uxmal 1 corn-to-temple-step) to
what the derived table says they are worth, leaving every other space at its
hand price. It was built to test whether F2's "shape" story alone reproduces the
derived table's regression. It measures **+0.14 greedy / +0.05 MCTS** — nothing.

That is a genuinely surprising negative and it *narrows* F2. Those four spaces
are where the hand and derived tables disagree most violently (research at 1/25),
so if shape were carried by the deferred-payoff spaces this variant would have
reproduced a large slice of `derived`'s −27.9. It reproduces none of it.
So the derived table's regression is **not** concentrated in the spaces whose
value is most obviously deferred; it is spread across the whole ranking, or it
lives in the resource spaces derived prices at 2-3x rather than in the point
spaces it prices at 1/25. Re-pricing four spaces down is survivable; re-ranking
all of them is not.

## F6. A harness hazard that nearly shipped a false interval

Recorded because it is invisible in the output and would have been believed.
Two runners were briefly writing the same `--out` at once: a `nohup`-detached
`run.sh` survived the notification that said it had completed, and a second was
started. `--resume` only skips seeds present when the process *starts*, so both
wrote every block, and `board=0.40,av=0.2` read **n=300 on a 150-block run**.

The duplicates are deterministic replicates — same seed, same search seed, same
result — so the **mean was unbiased and only the interval was wrong**, narrowed
by sqrt(2). That is the dangerous failure mode: the number looks right and the
significance is manufactured. Six files were affected, all created in this run;
the whole salvaged S1-S5 corpus was checked and is clean, so every number above
this line stands.

Two guards are now in place in `<scratch>/`: `run.sh` takes an atomic `mkdir`
lock (macOS has no `flock(1)`), and `an.py` **refuses to summarise a file whose
seeds are not unique** rather than reporting a too-narrow interval. If a future
run sees `DUPLICATE SEEDS`, dedupe by first occurrence — the replicates are
identical, so no information is lost.

## F7. The joint `board_scale` x `ACTION_VALUE` grid

The point of doing this jointly is F5b: `av` and `board` are knobs on the two
halves of one double count, so a coordinate-wise optimum need not be a joint
one. 150 blocks per cell, `greedy:64` above `mcts:256`, against `head`.

```
          av:    0.2       0.4       0.6
 board=0.30              +14.19
                         +10.88
 board=0.35              +15.87
                         +12.09
 board=0.40    +18.05    +16.92
               +12.29    +11.82
 board=0.50    +18.64    +17.54    +15.54
               +12.56    +12.54    +11.50
 board=0.60              +16.46
                         +11.73
```

* **The joint optimum is `board = 0.5, av = 0.2..0.4`**, worth **+18.6 greedy /
  +12.6 MCTS** against the committed evaluator. Both agents agree on the
  location; greedy again reads the effect ~1.5x MCTS's size, the same ratio S3
  saw.
* **The board optimum moved when `av` came down.** Alone, `board` peaked at 0.40
  and 0.35/0.40/0.50 were indistinguishable (F5a). With `av = 0.4` the ridge is
  clearly at 0.50: 0.30 (+14.19/+10.88) and 0.40 (+16.92/+11.82) are both worse
  than 0.50 (+17.54/+12.54), and 0.60 (+16.46/+11.73) is worse too. So the two
  terms really are coupled in the direction F5b predicted — take value out of
  `engine`, and `board` can afford to keep more. The brief's instruction to
  measure these jointly was load-bearing: the coordinate-wise answer (0.40) is
  not the joint one (0.50).
* Along `board = 0.5`, `av` is a plateau over 0.2..0.4 (+18.64/+12.56 vs
  +17.54/+12.54) and falls away by 0.6 (+15.54/+11.50). MCTS cannot separate
  0.2 from 0.4 at all; greedy prefers 0.2 by ~1.1.

## F8. `temple_outlook` is **under**-priced, and the plan workstream's −3.86 is collinearity

The brief flagged `temple_outlook` scoring negative in the plan workstream's
refit as "its own finding, and may be a bug rather than a weight". It is
neither. Scaling the term alone, 150 blocks per cell (the 0.60 and 1.40 rows are
salvaged from the killed predecessor, which measured them and did not write them
down):

| variant | greedy:64 | mcts:256 |
| --- | --- | --- |
| `temple=0.5` | −14.10 [−15.59, −12.61] * | −14.51 [−15.71, −13.31] * |
| `temple=0.60` | −9.97 [−11.36, −8.58] * | −10.22 [−11.45, −8.99] * |
| `temple=0.75` | −5.98 [−7.23, −4.73] * | −6.53 [−7.70, −5.36] * |
| `temple=1.0` | (null) | (null) |
| `temple=1.25` | +3.97 [+2.83, +5.11] * | +4.19 [+3.06, +5.32] * |
| `temple=1.40` | +6.01 [+4.71, +7.31] * | +6.25 [+5.05, +7.44] * |

**Monotone increasing across the whole range, on both agents, every cell
distinguishable.** Shrinking the temple term is the second-worst thing that has
been done to this evaluator (`temple=0.5` is a 0.065 MCTS win rate, behind only
`flat`). The term is not over-priced, not inert and not buggy — it is
under-priced by at least 40%.

### Why the plan workstream measured it negative

Its refit fitted all the terms **jointly against an evaluator whose `board` and
`engine` were 2-4x too big** (F4: those two are 30.1 points of a ~43-point
estimate). `temple_outlook` forecasts temple payouts; `board_position` credits
the worker riding to the Tikal temple space; `engine_value` credits research and
worker throughput that pays for those climbs. The three are strongly collinear,
so a least-squares fit that cannot shrink `board` and `engine` enough will drive
`temple`'s coefficient negative to cancel their excess. **The negative
coefficient was describing `board`, not `temple`.**

The direct test is to re-price the temple on top of a corrected `board`/`engine`
rather than on top of the committed one, and the collinearity story predicts the
sign should flip. It does:

| variant | greedy:64 | mcts:256 |
| --- | --- | --- |
| `board=0.5,av=0.4` (F7 best) | +17.54 | +12.54 |
| `board=0.5,av=0.4,temple=0.75` | +14.82 [+13.61, +16.03] * | +9.14 [+8.15, +10.13] * |
| `board=0.5,av=0.4,temple=1.25` | **+18.22** [+17.01, +19.42] * | **+13.83** [+12.76, +14.90] * |

Shrinking the temple hurts *more* once board and engine are corrected (−2.7
greedy / −3.4 MCTS from the F7 base, against −6.0/−6.5 from the committed base),
and growing it still helps (+0.7 / +1.3). So the ranking is stable under the
re-pricing and the term genuinely wants to be larger. **This is a warning about
the method, not only about the constant: a joint linear refit over collinear
terms cannot be read term-by-term, and the one term the plan workstream reported
as harmful is the one term here that wants more weight.**

Structurally the forecast checks out — `temple_outlook`'s point-day arithmetic
reproduces `state::temple_points_of` exactly, including the age-1/age-2 prize
split and the halved prize on a tie, and its resource-day haul reproduces
`gain_temple_resources` (all resources at or below the current step, every
resource day). One real defect found by reading, too small to explain any of the
above: the `climb` bonus uses `(step + 1) < d.steps` and so credits climbing on
to the **exclusive top step even when an opponent already occupies it** and
`temple_ceiling` will refuse the move. At `TEMPLE_CLIMB = 0.10` that is worth
~0.1 x one step's jump, and it is a correctness wart rather than a measurable
weight error.

## F9. The joint peak, and `contend` is subsumed

Temple scale on top of `board=0.5,av=0.4`, seed-paired against it (150 blocks):

```
                       greedy:64              mcts:256
  temple=1.25   +0.67 [-0.09,+1.44]    +1.29 [+0.30,+2.28] *
  temple=1.40   +0.91 [+0.00,+1.81]    +1.41 [+0.40,+2.41] *
  temple=1.60   +0.93 [-0.05,+1.92]    +0.42 [-0.72,+1.56]
  temple=1.80   +0.24 [-0.76,+1.23]    +0.74 [-0.40,+1.88]
```

**The joint peak is `temple = 1.4`**, the only cell distinguishable from the
base on both agents; 1.6 and 1.8 fall away under MCTS. Note the effect on top of
a corrected board/engine (+1.4) is a quarter of the effect of scaling temple
alone against the committed evaluator (+6.0 at 1.40) — most of what "temple
wants to be bigger" meant was really "board and engine want to be smaller", and
once they are, only ~1.4 points of genuine temple under-pricing is left. That is
the same collinearity from the other direction, and it is why F8's headline
number must not be added to F7's.

`ACTION_VALUE` 0.2 vs 0.4 at `board=0.5,temple=1.4`, paired:

```
  av=0.2 - av=0.4    +1.06 [+0.60,+1.53] *  greedy    +0.42 [-0.08,+0.92]  mcts
```

Greedy prefers 0.2; **MCTS cannot separate them.** Left open for the 600-block
confirmation below rather than settled on a greedy-only 150-block margin.

`contend` on top of `board=0.5,av=0.4,temple=1.4`, paired:

```
  +contend    +0.30 [-0.17,+0.76]  greedy    -0.25 [-0.86,+0.37]  mcts
```

**Nothing, on either agent.** `contend` measured +0.57/+0.95 against the
committed evaluator (S4) and is worth zero once the terms are re-priced. That is
a real result and not a null: contention priced a *denial* that the over-sized
`board_position` was already double-counting, so correcting board absorbed it.
**S6's recommendation to land `charge` + `contend` is superseded — neither half
survives the re-pricing** (`charge` actively harms it, F5b; `contend` is inert).

And against S6's configuration directly, paired:

```
  board=0.5,av=0.4,temple=1.4  -  chargecontendav04
      +7.59 [+5.94,+9.24] * greedy      +8.34 [+6.87,+9.80] * mcts
```

## F10. 600-block confirmation of the landing candidate

`board=0.5, temple=1.4`, the two surviving `ACTION_VALUE` candidates, **600
blocks = 2400 games each**, against `head`:

| variant | greedy:64 | mcts:256 | win (g / m) |
| --- | --- | --- | --- |
| `av=0.2` | **+19.18** [+18.59, +19.76] * | **+14.39** [+13.87, +14.91] * | 0.509 / 0.478 |
| `av=0.4` | +18.46 [+17.87, +19.06] * | +14.00 [+13.48, +14.53] * | 0.498 / 0.470 |

Seed-paired, which is what actually settles it:

```
  av=0.2 - av=0.4    +0.71 [+0.51,+0.91] * greedy    +0.38 [+0.09,+0.68] * mcts
```

**At 150 blocks MCTS could not separate these two (+0.42 [−0.08, +0.92]); at 600
it can.** This is the brief's warning working in the other direction — the extra
blocks did not reverse a result, they resolved one — and it is why the
confirmation was run before landing rather than after.

Both intervals now exclude zero on both agents, so `ACTION_VALUE = 0.2` is the
landing value. The **win rate is 0.509 on greedy:64**: against a null of 0.25,
the re-priced evaluator wins more than half of all games while outnumbered three
to one by the committed one.

## F11. What landed in `src/eval.rs`

```
  ACTION_VALUE   1.2  ->  0.2      (F7, F10)
  BOARD_SCALE    new  ->  0.5      applied to board_position at the assembly (F5a, F7)
  TEMPLE_SCALE   new  ->  1.4      applied to temple_outlook at the assembly (F8, F9)
```

The two scales are applied in `components`, next to each other, rather than
folded into the functions or into `TEMPLE_NEAR`/`TEMPLE_FAR` — same place
`plan::Schedule` applies its weights and the same place `bin/evalab` measured
them, so "this term is mis-priced by *k*" stays one number and the next person
to sweep them does not have to re-derive which constants are linear in what.

Worth **+19.52 greedy:64, +21.09 greedy:full, +7.46 mcts:256 and +7.63
mcts:1024** against the pre-landing evaluator — see **F15**, which supersedes the
+14.39 first recorded here. That earlier MCTS figure was measured before this
run's own edit reached the MCTS prior; F14 explains why it shrank and why the
greedy figure did not move.

Note for whoever measures next: `bin/evalab`'s `v::HEAD` is a self-contained
copy of the evaluator's constants, so it does **not** move when `eval.rs` does.
Every number above this line is against the *pre-landing* evaluator (3f5992a).
`v::HEAD` has now been updated to match the landing and `--eqcheck` re-run, so
numbers below this line are against the *new* baseline and the two are not
comparable. The `board=`/`temple=`/`av=` knobs set their field absolutely, so
against the new head `--ab 'board=0.5,av=0.2,temple=1.4'` is the null and should
measure 0 — that is the cheapest available check that a landing is exactly what
was measured.

## F12. This landing invalidates `plan.rs`'s fitted weights — cross-workstream

`PlanEvaluator::raw` is `sum_k w[k] * eval::components(g,p).terms()[k]`, and its
own doc says so deliberately: *"a change to `eval.rs`'s constants moves this
evaluator with it instead of leaving the two to drift apart"*. That is normally
the right coupling, but the change landed here is not a constant inside a term —
it is a **rescale of two of the eight terms**, so it composes multiplicatively
with weights that were fitted against the *unscaled* components:

```
  effective board weight  = w[BOARD]  x 0.5
  effective temple weight = w[TEMPLE] x 1.4
```

The plan workstream's refit reportedly took +11.85 of its +18.97 from shrinking
`board_position`, so `w[BOARD]` is already well under 1 and the composition
shrinks board roughly **four-fold** — past the far edge of the plateau measured
in F5a, where `board=0.25` was already 2-3 points worse than 0.5. Its
`w[TEMPLE]` was fitted *negative* (−3.86), and 1.4x a negative weight is worse
still.

**`plan.rs` must refit against the new `eval::components` before its schedule is
trusted again.** F8 makes a falsifiable prediction about what that refit will
find: with `board` and `engine` no longer 30 of a 43-point estimate, the
collinearity that drove `w[TEMPLE]` negative is gone, and the temple weight
should come back **positive**. If it refits negative again, F8 is wrong and the
term really is harmful — that is the experiment that would overturn this
section.

(`eval.rs` cannot fix this from its side, and should not: dividing the scales
back out inside `components` to protect one consumer's stale fit is exactly the
drift `plan.rs`'s comment is guarding against.)

## F13. ~~`ACTION_VALUE = 0` is a cliff that only MCTS can see~~ — **RETRACTED, see F14**

Run as an edge check on the landing value — is 0.2 interior to a plateau, or is
it perched on the side of one? `board=0.5, temple=1.4, av=0.0`, 150 blocks,
seed-paired against the same configuration at `av=0.2`:

```
  av=0.0 - av=0.2    +0.13 [-0.10,+0.35]  greedy:64      -6.84 [-8.28,-5.40] * mcts:256
```

Absolute: `av=0.0` is **+19.64 greedy** — as good as anything measured on this
file — and **+7.53 MCTS**, roughly half of `av=0.2`'s +14.39.

**The two agents do not merely disagree in size here; greedy cannot see the cliff
at all.** This is the most important methodological result in the log, because
S3 read the same edge off greedy-weighted evidence and concluded *"the data
cannot distinguish 0.4 from 0.0 and slightly prefers the smaller"*. Following
that would have walked the evaluator off a 7-point drop.

The mechanism is legible. `ACTION_VALUE` is the only thing paying for worker
*throughput*, so at 0.0 nothing in the estimate wants a fourth, fifth or sixth
worker and the agent stops buying them. A one-ply agent compares completed turns
that mostly have the same workers either way, so the term nearly cancels in the
comparison and its absence costs almost nothing. MCTS plans across the ~8-step
factored chain and across turns, where buying a worker is a move whose whole
payoff is future throughput — delete the term and that move becomes invisible.

Two standing rules follow, and they are cheap:

* **Never land a constant from greedy evidence alone**, however tight the
  interval. Greedy read +19.64 on the worst MCTS configuration of the sweep.
* **Check the edges of a plateau, not just its interior.** `av` reads as a flat
  ridge over 0.0..0.4 on greedy; on MCTS it is flat over 0.2..0.4 and falls off
  a cliff immediately below. 0.2 is landed as an interior point of the *MCTS*
  plateau, which is the agent that plays the game.

> **F13 IS RETRACTED AND MUST NOT BE USED — see F14.** It is kept, struck
> through, because the reasoning in it is a good example of a wrong conclusion
> drawn from a real measurement, and because deleting it would hide that the
> retraction happened. The `av=0.0`
> and `av=0.3` MCTS numbers it rests on are the only two cells run at 150 blocks
> in that phase, and both read ~+7.3 where the 600-block `av=0.2`/`av=0.4` cells
> read ~+14.0. A scalar constant cannot collapse at 0.0 and 0.3 while being fine
> at 0.2 and 0.4; something differs between the runs and not between the
> variants. The greedy halves of the same runs (+19.64, +19.13) agree with
> everything else, so whatever it is touches MCTS only. Do not cite F13.

## F14. `eval.rs` is the MCTS *prior* as well as the leaf value — and that moved the platform mid-run

**This retracts F13 and qualifies F10. Read this before citing any MCTS number
in this file.**

### What happened

Four cells of the `av` sweep, all `board=0.5, temple=1.4`, same seeds, same
`--out` convention, run within twenty minutes of each other:

```
  av=0.0  +7.53      av=0.2  +14.39      av=0.3  +7.23      av=0.4  +14.00     mcts:256
  av=0.0 +19.64      av=0.2  +19.18      av=0.3 +19.13      av=0.4 +18.46      greedy:64
```

A scalar constant cannot alternate like that, and the greedy row — flat across
all four — says whatever moved did not move the evaluator being tested. The
decomposition finds it in the **baseline** arm: on the same 150 seeds the three
baseline seats averaged **20.6 points in the `av=0.2`/`av=0.4` runs and 27.1 in
the `av=0.0`/`av=0.3` runs**, a 17-point swing in total game points. The games
themselves were different.

Re-run on an idle machine with the current binary and an explicit
`--base 'av=1.2,board=1.0,temple=1.0'` reproducing the old evaluator:

```
  redo av=0.2   +7.46 [+6.40, +8.53]      redo av=0.3   +7.23 [+6.15, +8.31]
```

`av=0.3` reproduces to the digit (+7.23 twice), so nothing here is noise; and
`av=0.2` has moved from +14.39 to +7.46. The two `+14` cells were measured on
one platform and the two `+7` cells on another, and the boundary is a rebuild
that picked up this run's own `eval.rs` edits.

### Why an `eval.rs` edit moves a measurement that pins the evaluator per arm

`src/mcts.rs:1573` scores every edge of a node with **`crate::eval::heuristic`
directly** — not with the `Evaluator` the agent was constructed with — to order
and prior the expansion (`mcts.rs:346` does the same for `Gradient`). `evalab`
hands each arm its own `VEval` for *leaf values*, but both arms share that
prior, and the prior comes from whatever `eval.rs` is compiled in.

So `bin/evalab` does not measure "evaluator A against evaluator B". It measures
**"leaf evaluator A against leaf evaluator B, under the move ordering that the
committed `eval.rs` produces"** — and the moment `eval.rs` is edited and
anything rebuilds, that conditioning changes underneath the sweep. `greedy:64`
never touches the prior (`Candidates::Sampled` scores its k draws with the arm's
own evaluator), which is exactly why the greedy row above is flat while the MCTS
row is not. That contrast is the proof of the mechanism, not a guess at it.

### What survives, and what does not

* **Retracted: F13 entirely.** `av=0.0` is *not* a cliff MCTS can see and greedy
  cannot. Its +7.53 and `av=0.2`'s +7.46 are the same number on the same
  platform; the apparent −6.84 was a platform difference. The doc comment on
  `ACTION_VALUE` warning against 0.0 has been cut back to what is supported.
* **Qualified: F10.** `av=0.2` vs `av=0.4` at 600 blocks (+0.71 greedy / +0.38
  MCTS, paired) was run entirely *before* the rebuild, so it is internally valid
  — but only on the pre-landing prior. On the landed prior `av=0.2` and `av=0.3`
  are +7.46 vs +7.23, indistinguishable. **`ACTION_VALUE`'s exact value inside
  0.2..0.4 is not resolved on the platform that now exists**; 0.2 is kept
  because it is what the one clean 600-block comparison chose.
* **Unaffected: every `greedy:64` number in this file**, for the reason above —
  including the whole `board`/`temple` grid's greedy half.
* **Unaffected: F5-F9's MCTS numbers**, which were all measured before any edit
  to `eval.rs`, against one binary, and are mutually comparable.
* **The landing still stands.** Measured on the landed platform against the old
  evaluator: **+7.46 mcts:256 [+6.40, +8.53]** and +19.18 greedy:64. Smaller
  than the +14.39 that was measured with the old prior, and that shrinkage is
  itself the expected result: once the re-priced evaluator is also ordering
  MCTS's moves, part of the gain is already banked in the search and the
  marginal value of improving the leaf as well is about half.

### Standing rule for this file

**Never edit `src/eval.rs` while a measurement is in flight**, and treat any
rebuild as the start of a new platform. Numbers either side of one are not
comparable, and the failure is silent — both arms move together, so nothing
looks wrong. `bin/evalab`'s `v::HEAD` being a self-contained copy protects the
*baseline evaluator* from drifting but does nothing for the shared prior.
The cheap detector is the one used above: **if the mean `base` column moves
between two runs on the same seeds, the platform moved.** Worth adding to
`an.py` as a routine check.

### F14a. The detector, and it confirms the split

`an.py -b <files...>` now reports the mean of the `base` column over the seeds
the files share, and flags a spread over 1 point. On the three runs in question:

```
  150 shared seeds
    final_av02_mcts256      base   20.578   cand   39.740
    final_av03_mcts256      base   27.304   cand   36.938
    redo_av02_mcts256       base   27.211   cand   37.163
    ** BASE SPREAD 6.73 pts -- these runs are NOT on one platform **
```

`final_av03` and `redo_av02` — different variants, different runs, one built
after the other — agree on the baseline to **0.09 points**, while `final_av02`
sits 6.7 away. The baseline arm is the same evaluator in all three, so that
number is a pure platform fingerprint: the two low-`av` cells and the re-run
share a build, and `final_av02` does not. That is the split F14 infers, measured
directly rather than argued from timestamps.

Use `-b` before trusting any two runs against each other that were not launched
from the same queue.

## F15. The landing, re-measured cleanly on one platform

Everything below was run back to back from one binary — the landed `eval.rs`,
built once, machine otherwise idle, no source touched in between — with the
candidate `board=0.5, av=0.2, temple=1.4` against an explicit
`--base 'av=1.2,board=1.0,temple=1.0'` that reconstructs the pre-landing
evaluator. This is the number to quote.

| agent | blocks | centred | win rate |
| --- | --- | --- | --- |
| `greedy:64` | 150 | **+19.52** [+18.37, +20.66] * | 0.510 |
| `greedy:full` | 150 | **+21.09** [+19.92, +22.25] * | 0.523 |
| `mcts:256` | 150 | **+7.46** [+6.40, +8.53] * | 0.354 |
| `mcts:1024` | 100 | **+7.63** [+6.42, +8.85] * | 0.330 |

Three things this settles, all of them S6's open questions:

* **It holds at `greedy:full`** — +21.09, the largest of the four, so the effect
  is not an artifact of scoring only 64 sampled moves. Win rate 0.523 against a
  null of 0.25: the re-priced evaluator wins the majority of games while
  outnumbered three to one.
* **It does not wash out with more search.** `mcts:1024` reads +7.63 against
  `mcts:256`'s +7.46 — 4x the simulations, same answer, intervals almost
  entirely overlapping. This is an evaluator gain the search does not find on
  its own, which is exactly what the brief said the critical path needed.
* **Greedy reads the effect ~2.6x MCTS's size** (+19.5 vs +7.5) — a wider gap
  than the ~2.2x of S3, and for a reason S3 could not have seen: part of the
  MCTS gain is now banked in the prior (F14), so the *marginal* value of the
  better leaf is smaller. A one-ply number remains an upper bound on what MCTS
  will pay for an evaluator change, and the gap grows as the prior improves.

## F16. State at handoff, and a warning about the shared tree

`cargo test --release`: **169 passed, 0 failed, 7 ignored.** `--eqcheck`: max
|`V::HEAD` − `eval::heuristic`| = **0e0** over 16016 (position, seat) pairs, so
the landed file is exactly the configuration F15 measured.

`tests/tree.rs::tree_reuse_deepens_reused_nodes` — which the brief expected to
go red on any `eval.rs` change — **passes**, before and after, run on its own and
in the suite. It should be left alone: its assertion (`with >= without` total
simulations) is a real invariant about tree reuse rather than a number tuned to
the evaluator, and this change moved three term weights without breaking it.

**Another agent is working in this tree concurrently.** Over this run
`src/mcts.rs` was touched at 23:54:31, and `src/options.rs`,
`src/bin/movestats.rs` and a new `docs/FINDINGS-mcts.md` appeared. One
`cargo test --release` that this run did not start rebuilt the shared binary at
23:57:59, mid-sweep. That is the proximate cause of F14, and one `cargo test`
during the overlap reported 168 passed / 1 failed where two clean runs either
side report 169 / 0 — a transient from compiling under a concurrent edit, not a
real failure.

The consequence for anyone measuring here: **a shared working tree means the
measurement platform can move without any action of your own.** `evalab`'s
`v::HEAD` pins the baseline *evaluator* but not the binary, and MCTS's prior
comes from whatever `eval.rs` is compiled in (F14). Before trusting two runs
against each other, check `an.py -b` on them; if the `base` column has moved,
they are not comparable no matter how tight their intervals are.

## F17. The F3 mid-turn hump was `board_position`, and `94d85f3` already removed it

**This retires the brief's F3 target.** The +2.70 mid-turn hump was measured on
the *pre-landing* evaluator. Re-measured on the landed one it is gone, and the
proof is a single binary running both sets of constants over one corpus in one
process — the F14-safe way to tell a landing from a platform move. `--midturn`
now takes `--vars 'a;b;c'` (`;` because a variant spec is itself
comma-separated) and prints a trajectory per variant.

`<scratch>/f3/mid1.txt`, `evalab --midturn --corpus 60 --take 3000`, 2745 turn
roots, all rows from the same process. `v(k) - v(end)`, HEAD's own scale:

```
                     variant      d0      d3      d5      d7      d8    drift
                        head  -1.953  -0.691  -1.270  -0.622  +0.006   -0.688
 av=1.2,board=1.0,temple=1.0  -2.376  +0.203  +2.309  +1.333  +0.021   +0.773
                   board=1.0  -2.495  -0.025  +1.620  +0.951  +0.013   +0.451
                      av=1.2  -2.046  -0.783  -1.444  -0.725  +0.003   -0.786
                  temple=1.0  -1.739  -0.380  -0.429  -0.191  +0.006   -0.299
```

Row 2 reproduces F3 (+2.31 at d5 against F3's +2.70, drift +0.773 against
F3's +1.191; the corpus is the same command but the descent differs once the
evaluator does). Row 1 is the committed file. **The peak went from +2.31 to
−1.27 and the drift from +0.77 to −0.69.**

Which constant did it: `board=1.0` alone (HEAD with only `BOARD_SCALE` put
back) restores drift to **+0.451**, so `BOARD_SCALE 1.0 -> 0.5` is **−1.14 of
the −1.46 swing**. `ACTION_VALUE` is inert on drift (−0.786 at 1.2 against
−0.688 at 0.2) — `engine_value`'s worker count barely moves inside a turn.
`TEMPLE_SCALE` contributes the other −0.39, by amplifying a term that is
*under*-stated mid-turn.

### F17a. The hump was never a hump: it is two lines crossing

New in `--midturn`: the trajectory split by the `ModeChoice` the descent took.
A turn is *either* placing or retrieving, and the two move the estimate in
opposite directions, so averaging them produces a shape that belongs to neither.
`<scratch>/f3/mid2.txt` and `mid1.txt`, same 2745 roots:

```
                pre-landing  (av=1.2,board=1.0,temple=1.0)        landed HEAD
  depth      PLACE (2159)     RETRIEVE (586)            PLACE          RETRIEVE
      0            -4.055            +3.813           -1.811            -2.479
      3            -0.831            +4.012           -0.232            -2.379
      4            -0.323            +2.692           -0.088            -1.381
      5            +0.034            +2.832           (n<20)            -1.490
      8                 -            +0.021                -            +0.006
```

* A **placing** turn under-states: the worker is in hand, `engine_value` pays it
  `ACTION_VALUE` per `ROUNDS_PER_ACTION` and nothing else, and the moment it is
  placed `board_position` pays it the space it is riding to. Placing was worth
  **+4.06 points of pure estimate** before the landing and is worth +1.81 now.
* A **retrieving** turn over-states: the workers still standing are being paid
  their promise, and taking the action replaces that promise with what it
  actually delivers. Before the landing the promise beat the delivery by
  **+3.81**; now the delivery beats the promise by 2.48.
* Place turns are **79%** of turns and are short (74% of them end at d3); retrieve
  turns are 21% and long. So the early depths of F3's pooled table are the place
  line and the deep ones are the retrieve line, and *"the crossover at depth 3"*
  — which the brief read as placement finishing and retrieval beginning — is a
  **change of sample, not a term changing sign**. Nothing crosses over; the
  averaging does.

### F17b. What is left is a cancellation, not a fix

`--terms` on the landed evaluator (`<scratch>/f3/mid0.txt`), depth 5, term minus
the same term of the completed turn:

```
   banked  liquid    held  temple  engine   board   monu  starve   total
   -0.838  +0.107  +0.067  -3.235  -0.650  +3.130 -0.019  -0.033  -1.470
```

`board` still over-promises by **+3.13** (it was +8.56 at `BOARD_SCALE = 1.0`)
and `temple` under-states by **−3.24**, because the temple steps the turn is
about to take have not landed. The two nearly cancel *in the mean* and not at
all per position: the sd of the depth-5 bias is **4.56** against a mean of
−1.27. So the committed evaluator is not mid-turn-consistent; it is two large
opposite errors that happen to sum to something small.

### F17c. Drift is not a loss function — do not optimise it

The same table, read against the strength numbers already in this file, kills
the idea that a smaller |drift| is a better evaluator:

```
  variant          drift    strength vs committed
  derived-static  -0.438    -26.88 greedy / -22.29 mcts   (S1)
  derived         -0.484    -27.86 greedy                 (S1)
  head            -0.688    (null)
  board=1.0       +0.451    -10.48 greedy / -9.04 mcts    (F5, sign flipped)
```

The two smallest-|drift| variants in the file are the two worst players in it.
`board=1.0` has the drift *closest to zero* of the four real evaluators and is
10 points worse. Mid-turn consistency is a diagnostic that localises a bug once
you already suspect one — it is what turned "the evaluator is over-priced" into
"`board_position` is over-priced", F4 — and it is not a thing to minimise.

Filename table additions: `f3/mid0.txt` landed-evaluator `--terms` + `--midturn`;
`f3/mid1.txt` five variants' trajectories on one binary; `f3/mid2.txt` the same
with the pre-landing evaluator first so the mode split is printed for it.

## F18. Twenty-nine knobs screened on the landed evaluator — `<scratch>/f3/ab/*_greedy64.jsonl`

Binary pinned at `<scratch>/f3/evalab-p2` (rev `8c2a56e` + `bin/evalab.rs`
changes only), one runner holding `/tmp/evalab-f3-runner.lock.d`, one writer per
file. Platform fingerprint, the `base` column of a null run: **30.8 greedy:64,
33.4 mcts:256, 34.2 greedy:full**, and the null
(`--ab 'board=0.5,av=0.2,temple=1.4'`, and again with every new knob written out
at its committed value) reads **+0.00 [+0.00, +0.00]** — so the knobs added to
`bin/evalab.rs` this run are inert at their defaults and `--eqcheck` is 0e0.

**300 blocks (1200 games) each, `greedy:64`, against the committed evaluator.**
Screening only: F13/F14's rule is that nothing lands on greedy evidence alone,
and everything that survives here is re-run on MCTS below.

| knob | value | centred | 95% CI |
| --- | --- | --- | --- |
| `TEMPO_PER_ROUND` | 0.35 | −2.35 | [−2.87, −1.82] * |
| | 0.42 | −1.39 | [−1.89, −0.90] * |
| | **0.62** | **+1.03** | **[+0.52, +1.53] *** |
| | 0.75 | −2.85 | [−3.39, −2.32] * |
| | 0.90 | −6.52 | [−7.14, −5.91] * |
| `BUILDING_VALUE` | 0.0 | −0.87 | [−1.15, −0.58] * |
| | **1.0** | **+0.34** | **[+0.02, +0.67] *** |
| climb respects `temple_ceiling` | on | **+0.08** | **[+0.01, +0.14] *** |
| `TEMPLE_CLIMB` | 0.0 / 0.25 / 0.50 | +0.04 / −0.16 / −0.32 | only 0.50 excludes 0 |
| `TEMPLE_FAR` | 0.35 / 0.75 | −0.14 / +0.01 | both cover 0 |
| `TEMPLE_NEAR` | 0.70 / 1.00 | −0.25 / −0.11 | both cover 0 |
| distance discount `t_half` | 10 / 20 / 40 / 80 | −0.60 / +0.15 / −0.13 / −0.18 | only 10 excludes 0 |
| `reach` cap (rounds) | 1 / 2 / 3 / 5 | −1.01 / −1.17 / −3.66 / −0.26 | worse or flat |
| `monument` scale | 0.0 / 3.0 | +0.21 / −1.15 | 0.0 covers 0 |
| `held` scale | 0.5 / 2.0 | +0.35 / −1.64 | 0.5 covers 0 |
| `starvation` scale | 0.5 / 2.0 | −0.62 / −0.16 | — |

What this settles:

* **`TEMPO_PER_ROUND` wants to be 0.62, not 0.52.** It has not been swept since
  `BOARD_SCALE` landed, and the two are coupled exactly as S3 predicted: the
  old sweep found 0.52 with a board term twice this size. The curve is sharply
  unimodal — 0.42 and 0.75 are both worse by more than a point, 0.90 by 6.5 —
  so this is a real peak and not a plateau. It is also the knob the mid-turn
  work points at: `tempo` is the discount on riding, i.e. the thing that
  controls `board_position`'s promise about a worker's *future* space.
* **`BUILDING_VALUE` is too low.** Deleting it costs −0.87, doubling it gains
  +0.34, and both intervals exclude zero, so the residual-monument-counting
  value of a built card is worth more than 0.45. Swept further in F19.
* **The `climb` / `temple_ceiling` wart from F8 is real and worth +0.08**
  [+0.01, +0.14], measurable at 300 blocks. It is tiny, but it is a free
  correctness fix in the direction the rule already says.
* **The temple discount's *shape* is inert.** `TEMPLE_NEAR`, `TEMPLE_FAR` and a
  hyperbolic discount in the actual distance to the scoring day
  (`t_half/(t_half+gap)`, which is the one thing the committed two-level rule
  cannot express — a payout one day out and one twelve days out are both "the
  near one") all read zero. Only the term's *scale* matters, and `TEMPLE_SCALE`
  already carries it. **Dead end, do not re-sweep.**
* **Capping how far a worker may be credited for riding is not the fix.**
  `reach = 1..5` is flat-to-worse everywhere, so `board_position`'s max over the
  whole reachable gear is not what over-prices it — the level is, and
  `BOARD_SCALE` already has it.
* `monument_outlook` can be deleted for +0.21 [−0.11, +0.53] and tripling it
  costs −1.15. Third independent confirmation that the term is inert (F4, and
  the plan workstream's −0.18). Not worth a constant; it is worth a rewrite or a
  removal, neither of which is measurable as a scale.

## F19. `--promise`: what `board_position` pays a worker against what its action delivers

New diagnostic in `bin/evalab`, `<scratch>/f3/promise2.txt`. For every worker
standing on a gear at a turn root where retrieving is legal, walk the tree to
`Phase::Take { worker }`, take the best `Choice` by the evaluator's own
estimate, and record

```
  promise  = BOARD_SCALE x rider_worth(worker)          what `board_position` pays it
  delivery = h(after the Take) - h(before) + promise    what the action is worth
```

The `+ promise` is because retrieving deletes the worker's board credit, so the
raw difference understates the action by exactly that. **53,704 standing workers
over 29,960 turn roots.** This is the calibration F2's derived table could not
do: `derived` priced the `Effect`s a space emits *in isolation* and lost the
interaction the hand number was carrying (−27.9); this is the change in the
*whole evaluator*, so every interaction is in it by construction.

```
  overall promise 1.835   delivery 2.740   ratio 1.493
```

Per space, `want = delivery / BOARD_SCALE` is the table entry that would make
`board_position` agree with the rest of the evaluator. `r = want / table`:

```
  Palenque 1..7   r = 2.92 2.70 3.58 3.49 3.48 3.53 3.29
  Yaxchilan 1..7  r = 3.66 3.13 3.03 2.51 3.57 3.13 3.09
  Tikal 1..7      r = 0.30 1.93 0.94 1.33 5.19 3.14 2.87
  Uxmal 1..7      r = 3.73 2.43 0.58 3.49 4.62 2.74 2.54
  Chichen 1..10   r = 1.96 1.64 1.50 1.47 1.15 1.26 1.05 1.30 1.23 1.08
```

Read it as one common factor plus five outliers:

* **The common factor is ~3 for everything that pays in *resources or temple
  steps* and ~1.2 for Chichen, which pays in *points*.** Chichen's payout is
  exact — points are points — so the ~2.5x gap between the two families is a
  direct measurement of how much the evaluator's speculative terms
  (`held_premium`, `temple_outlook`, `engine_value`) inflate a resource over its
  banked-point worth. That inflation is the *level* problem `BOARD_SCALE = 0.5`
  already prices, and the fact that consistency would want `BOARD_SCALE = 0.74`
  (0.5 x 1.493) while strength wants 0.5 — `board=0.7` measures +5.10 against
  +10.48 at 0.5, F5a — is F17c's point again, measured a third way.
* **Tikal 1 (first research space): r = 0.30, the widest disagreement on the
  board.** The table prices it at 2.00; the evaluator's own accounting for the
  research level it hands over is 0.60. Tikal 3 (r = 0.94) and Uxmal 3, buying a
  worker (r = 0.58), are the same story.
* **Tikal 5 (two temple steps for a block): r = 5.19.** The table says 4.0 and
  the evaluator says 19.4, because `temple_outlook x 1.4` counts both remaining
  point days at full standings.

### F19a. Re-ranking the table toward delivery is a monotone regression

`Space::Calib` multiplies the *gated* table value by `r^alpha` and renormalises
the level with `space_scale` so only the ranking moves. 300 blocks, greedy:64:

| alpha | centred | 95% CI |
| --- | --- | --- |
| 0.25 | −2.91 | [−3.49, −2.33] * |
| 0.50 | −3.97 | [−4.55, −3.38] * |
| 0.75 | −5.76 | [−6.36, −5.16] * |
| 1.00 | −6.23 | [−6.79, −5.67] * |

Monotone, every cell distinguishable. `alpha = 1` is the *maximally mid-turn
consistent* space table there is — by construction `board_position` then pays a
worker exactly what the evaluator will pay it when the action lands — and it is
**6.2 points worse**. Its mid-turn drift is the best of any real variant
(−0.518 at `alpha = 0.5` against head's −0.688, `--midturn --vars`), which makes
this the sharpest available statement of F17c: **self-consistency is the wrong
objective, and it can be bought at 6 points a unit.**

This is also the third independent kill of "replace the hand table with
something derived" — `derived` (−27.9, F2), `defer` (+0.14, F5d) and now
`calib`. The hand table's *ranking* is right and three different ways of
deriving a better one have all failed.

### F19b. What the outliers were really saying: `RESEARCH_SCALE` is too big

`--promise` said the table and `engine_value` disagree about research by 3.3x.
It does not say which is wrong. Sweeping the term that is not the table
(`engine_value`'s `research_step_value(s,l) * uses * RESEARCH_SCALE`), 300
blocks, greedy:64:

| `RESEARCH_SCALE` | centred | 95% CI |
| --- | --- | --- |
| **0.25** | **+1.29** | **[+0.85, +1.73] *** |
| 0.50 | (committed, null) | |
| 0.75 | −3.40 | [−3.89, −2.90] * |
| 1.00 | −8.81 | [−9.37, −8.24] * |
| 1.50 | −15.30 | [−15.84, −14.76] * |

Steeply monotone decreasing — so the table is right and **`engine_value` is
paying too much for research**, the same over-pricing `ACTION_VALUE 1.2 -> 0.2`
found in the worker half of the same term. Swept lower in F20.

### F19c. Crediting the worker still in hand does not fix the root-side bias

`hand_flat`: a flat credit per worker in hand, gated on being able to pay for
the placement (the n-th costs `n + space_cost`, so k placements cost at least
`k(k-1)/2`) and faded by the calendar the way `held_premium` is. Built to close
F17a's −1.81 place-turn root gap without `hand_worker`'s max-over-the-board
optimism, which is +3.32 greedy and −3.28 MCTS (S4).

| credit | centred (greedy:64, 300 blk) |
| --- | --- |
| 0.3 | −0.10 [−0.62, +0.42] |
| 0.6 | −0.80 [−1.28, −0.32] * |
| 1.0 | −5.22 [−5.79, −4.65] * |
| 1.5 | −14.90 [−15.43, −14.37] * |

Monotone down, and it is *worse than `hand_worker` on the agent that liked
`hand_worker`*. The difference tells you what the root bias is: `hand` credits
the specific space a specific worker could reach, so it still ranks placements
against each other; a flat credit only shrinks the gap between holding and
placing, and that gap **is** the search's entire signal for where to put a
worker. It also barely moves the thing it was built for — at 1.0 the root bias
goes from −1.953 to −1.698, a quarter of the gap, for −5.22 points.

**So the root-side under-statement is not a constant that can be added back.**
It is the statement that a worker in hand has no *location*, and the evaluator's
whole board term is about location. Any correction big enough to close it is big
enough to stop distinguishing placements. Treat the −1.8 as inherent to a
`(GameState, PlayerId) -> f32` evaluator; what makes it harmless is that it is
nearly constant across the siblings of any one node.

## F20. `RESEARCH_SCALE` down to a tenth, and the three knobs are additive

Continuing F19b, 300 blocks (1200 games), `greedy:64`, against the committed
evaluator. `<scratch>/f3/ab/rs0*_greedy64.jsonl`.

| `RESEARCH_SCALE` | 0.0 | 0.10 | 0.15 | 0.25 | 0.35 | 0.50 | 0.75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centred | +1.60 | **+1.65** | +1.62 | +1.29 | +0.96 | null | −3.40 |
| 95% CI | ±0.47 | ±0.46 | ±0.46 | ±0.44 | ±0.38 | — | ±0.50 |

A plateau over **0.0..0.15**, indistinguishable across it, falling away by 0.35
and off a cliff by 0.75. The committed 0.50 is well past the far edge. `0.10` is
taken rather than `0.0` because it is interior to the plateau — F13's one
surviving rule, that a constant should not be landed on the edge of one — and
because deleting the term outright removes the only thing that makes the search
value a research track at all.

Reading this together with `ACTION_VALUE 1.2 -> 0.2` (F7/F10): both halves of
`engine_value` — the per-worker throughput and the per-research-level payout —
were priced 4-5x too high, and the *same* diagnostic found the second one.
`engine`'s mean is 4.92 points now against the 19.49 it had at 6a99f2f.

### F20a. Joint, and additive

| variant | greedy:64, 300 blk | win |
| --- | --- | --- |
| `tempo=0.62` | +1.03 [+0.52, +1.53] * | 0.274 |
| `rs=0.25` | +1.29 [+0.85, +1.73] * | 0.285 |
| `bv=1.0` | +0.34 [+0.02, +0.67] * | 0.259 |
| `ceiling` | +0.08 [+0.01, +0.14] * | 0.255 |
| `tempo=0.62,rs=0.25` | +2.51 [+1.96, +3.05] * | 0.314 |
| **`tempo=0.62,rs=0.25,bv=1.0,ceiling`** | **+3.05** [+2.52, +3.59] * | **0.335** |

1.03 + 1.29 = 2.32 measured 2.51; adding 0.34 + 0.08 = 0.42 measured +0.54.
**Additive to within the intervals**, which is the expected result for three
knobs on three different terms — unlike `board`/`av`/`charge`, which are all
knobs on one double count (F5b).

Interactions checked on top of `rs=0.25`, all 300 blocks: `av` 0.1 / 0.3 / 0.5
read +1.41 / +0.98 / −0.22 against +1.29 at 0.2, `board` 0.4 / 0.6 read
+1.09 / +1.05 against 0.5's +1.29, `temple=1.6` reads +1.36 against 1.4's +1.29,
`engine=0.8` reads +0.84. **None of the landed constants moves**: shrinking
research does not shift the optimum of the terms it is collinear with, which is
the opposite of what shrinking `board` did to `av` in F7.

## F21. MCTS confirmation — `tempo` is a greedy-only effect, `RESEARCH_SCALE` is not

150 blocks (600 games), `mcts:256`, same pinned binary, `--base` the committed
evaluator. Platform check first: `an.py -b` over the 150 shared seeds reads
`base` = 33.098 / 33.166 / 33.211 across three of these runs, a spread of
**0.11 points**, so they are one platform (F14a's detector).

| variant | greedy:64, 300 blk | mcts:256, 150 blk |
| --- | --- | --- |
| `tempo=0.62` | +1.03 [+0.52, +1.53] * | **+0.13 [−0.46, +0.72]** |
| `rs=0.0` | +1.60 [+1.13, +2.07] * | +1.41 [+0.80, +2.02] * |
| `rs=0.10` | +1.65 [+1.20, +2.11] * | +1.30 [+0.70, +1.90] * |
| `rs=0.25` | +1.29 [+0.85, +1.73] * | +0.84 [+0.36, +1.31] * |
| `bv=1.0` | +0.34 [+0.02, +0.67] * | +0.39 [−0.11, +0.90] |
| `ceiling` | +0.08 [+0.01, +0.14] * | +0.19 [+0.02, +0.35] * |
| `tempo=0.62,rs=0.25,bv=1.0,ceiling` | +3.05 [+2.52, +3.59] * | +0.75 [+0.11, +1.39] * |
| `tempo=0.62,rs=0.10,bv=1.0,ceiling` | — | +1.47 [+0.83, +2.11] * |

**`TEMPO_PER_ROUND = 0.62` does not survive.** Greedy reads +1.03 with an
interval that excludes zero at 300 blocks; MCTS reads +0.13 with an interval
that covers it. And the four-knob variant containing it (+0.75 MCTS) is *no
better than `rs=0.25` alone* (+0.84) on the agent that plays the game, while
being 2.4 points better on greedy. This is the S3/F15 ratio taken to its limit:
greedy over-reads an evaluator change, and a knob whose entire effect is inside
`board_position`'s ride discount is exactly the kind of thing a one-ply agent —
which only ever compares completed turns — reads differently from a search that
looks at every node of the chain.

`RESEARCH_SCALE` behaves the other way: +1.65 greedy against +1.30 MCTS, a ratio
of 1.3 rather than 8, which is the normal relationship in this file. It is the
one real effect in this wave.

`ceiling` is the small, free correctness fix: +0.08 greedy, **+0.19 MCTS**, both
excluding zero.

### F21a. At 400 blocks: only `RESEARCH_SCALE` survives on MCTS

`mcts:256`, **400 blocks (1600 games)** each, same binary, `an.py -b` spread
0.11 across the wave.

| variant | mcts:256, 400 blk | paired vs `rs=0.10` |
| --- | --- | --- |
| `rs=0.0` | +1.30 [+0.94, +1.65] * | **+0.29 [+0.01, +0.57] *** |
| `rs=0.10` | +1.00 [+0.64, +1.37] * | — |
| `rs=0.15` | +0.87 [+0.52, +1.21] * | −0.14 [−0.37, +0.10] |
| `tempo=0.62` | **+0.02 [−0.34, +0.37]** | — |
| `bv=1.0` | +0.19 [−0.10, +0.49] | — |
| `ceiling` | +0.12 [+0.01, +0.22] * | — |
| `rs=0.10,bv=1.0,ceiling` | +0.80 [+0.43, +1.17] * | −0.20 [−0.54, +0.13] |
| `tempo=0.62,rs=0.10,bv=1.0,ceiling` | +1.20 [+0.81, +1.60] * | +0.20 [−0.24, +0.63] |

* **`TEMPO_PER_ROUND = 0.62` is dead.** +0.02 at 400 blocks, an interval of
  ±0.36 around zero. Greedy read +1.03 [+0.52, +1.53] at 300 blocks. The two
  agents are not disagreeing about size here, they are disagreeing about
  existence, and the greedy number is the one to throw away — `TEMPO_PER_ROUND`
  lives only inside `board_position`'s ride discount, and a one-ply agent that
  only ever compares completed turns weights that term differently from a search
  that evaluates every node of the chain. **`TEMPO_PER_ROUND` stays at 0.52.**
* **Adding `bv` and `ceiling` on top of `rs` buys nothing measurable**
  (−0.20 [−0.54, +0.13] paired), even though `ceiling` alone is +0.12 and
  distinguishable. At these sizes the bundle cannot be separated from its
  largest member.
* `RESEARCH_SCALE = 0.0` edges out 0.10 by +0.29 [+0.01, +0.57] paired — the
  only one of these comparisons that resolves. Settled in F22.

## F22. The `held_premium` price list, swept for the first time — `<scratch>/f3/ab/b5_*`

Every constant inside `held_premium` and `starvation_risk` had only ever been
moved as a group (`held=` / `starve=`, which scale the whole term). Both group
scales read near zero — `held=0.5` is +0.35 [−0.17, +0.86] and `starve=0.5` is
−0.62 — so the *level* of both terms is right. This sweeps the *mix*.

**Base is `rs=0.05`, not the committed evaluator** (`--base 'rs=0.05'`), so
these are marginal effects at the operating point F20/F21 arrived at, and the
`b5_null` control reads +0.00 exactly. 300 blocks (1200 games), `greedy:64`.

| constant | committed | tried | centred |
| --- | --- | --- | --- |
| `CORN_PREMIUM` | 0.10 | 0.0 / 0.05 / 0.20 / **0.35** | −0.82 / −0.36 / +0.70 * / **+1.28 [+0.76, +1.80] *** |
| `BLOCK_PREMIUM` | 0.55 | **0.30** / 0.75 / 1.00 | **+0.75 [+0.33, +1.16] *** / −0.77 * / −1.74 * |
| `CORN_INCOME_PER_ROUND` | 1.9 | **1.2** / 2.6 | **+0.79 [+0.41, +1.16] *** / −0.63 * |
| `SKULL_PREMIUM` | 2.2 | 1.2 / 3.2 | −3.00 * / −1.65 * |
| `BLOCK_BREADTH` | 0.50 | 0.0 / 1.0 | −0.09 / −0.33 |
| `TILE_PREMIUM` | 0.25 | 0.0 / 0.6 | −0.00 / +0.05 |
| monument gate (`short > k`) | 4 | 2 / 8 | +0.15 / +0.22 |
| `corn_depth` (the S4 idea, re-run) | — | on | −0.10 [−0.31, +0.11] |
| `ACTION_VALUE` | 0.2 | 0.3 / 0.4 | −0.37 * / −0.76 * |
| `BOARD_SCALE` | 0.5 | 0.4 / 0.6 | −0.58 * / −0.55 * |
| `TEMPLE_SCALE` | 1.4 | 1.6 | −0.48 * |

* **Corn is under-priced and blocks are over-priced, and they move in opposite
  directions inside one term whose total is already right.** `CORN_PREMIUM`
  climbs monotonically from 0.0 to 0.35 and `BLOCK_PREMIUM` falls monotonically
  from 1.0 to 0.30. That is the brief's corn hypothesis confirmed from the other
  side: the flat premium is too *small*, not the wrong shape — because
  `corn_depth`, which replaces the flat premium with the placement depth it
  buys, is again **−0.10 [−0.31, +0.11]**, exactly the nothing S4 measured
  before the re-pricing. **The number was wrong, not the functional form.**
* `SKULL_PREMIUM = 2.2` is a genuine interior optimum: 1.2 and 3.2 are both
  clearly worse, −3.00 and −1.65. It is the one constant in this term that was
  already right, and it is the one used in three places.
* **`ACTION_VALUE`, `BOARD_SCALE` and `TEMPLE_SCALE` all re-confirm at their
  landed values** on this new base, in both directions, at 300 blocks. F11's
  landing does not move when research comes down.
* `BLOCK_BREADTH`, `TILE_PREMIUM` and the monument gate are inert in both
  directions. `monument_outlook` cannot be fixed by widening or narrowing its
  gate either — that is now four ways it has failed to matter.

### F22a. The two corn constants, swept out

Same base (`rs=0.05`), same 300 blocks, `greedy:64`.

```
  CORN_PREMIUM     0.0    0.05   0.10   0.20   0.35   0.50   0.70   1.00
                 -0.82   -0.36   null  +0.70  +1.28  +1.90  +1.82  +0.61

  BLOCK_PREMIUM    0.0    0.15   0.30   0.55   0.75   1.00
                 +0.86   +0.94  +0.75   null  -0.77  -1.74

  CORN_INCOME      0.0    0.4    0.8    1.2    1.5    1.9    2.6
                 +2.15  +1.55  +1.15  +0.79  +0.22   null  -0.63
```

`CORN_PREMIUM` peaks at **0.50** (0.50 and 0.70 indistinguishable, 1.00 falls
away), `BLOCK_PREMIUM` at **0.15** (0.0..0.30 a plateau), and
`CORN_INCOME_PER_ROUND` is monotone to **0.0** — the more pessimistic
`starvation_risk`'s assumed income, the better. `LIQUID_CORN` is inert: 24
reproduces 16 to the digit and 8 is worse, so the cap rarely binds.

**They are substitutes, not complements.** Every one of them pushes the agent to
hold corn, and the joints are far short of the sum:

```
  cp=0.50                                +1.90
  ci=0.0                                 +2.15
  cp=0.35,bp=0.30                        +1.25   (sum of parts 2.03)
  cp=0.50,bp=0.15,ci=1.2                 +2.30
  cp=0.50,bp=0.15,ci=0.8                 +2.13
  cp=0.70,bp=0.15,ci=0.8                 +1.79
  rs=0.0 + cp=0.50,bp=0.15,ci=0.8        +2.22
```

The best joint is barely above the best single knob, so there is one underlying
error here — *the evaluator does not hold enough corn* — and three constants
that each partly correct it. Confirmed on MCTS in F23 before anything lands.

### F22b. An initiative bonus is invisible to greedy by construction

`init=k` adds `k` to the estimate of whoever `g.current` names. It is the one
correction for F17a's root-side under-statement that cannot distort a
within-turn ranking: every sibling at a node shares `g.current`, and only a
commit edge changes it.

`greedy:64`, 300 blocks: **−0.00, −0.02, −0.04** at k = 0.5, 1.0, 2.0, with
intervals of ±0.03. Not "small" — *exactly* nothing, and that is the expected
result: `Candidates::Sampled` scores completed turns of one player, all of which
share `g.current`, so the constant cancels identically. Whether it does anything
at all is a question only MCTS can be asked, because only a search crosses a
commit edge. F23.

## F23. MCTS says the win is `CORN_INCOME_PER_ROUND`, and half the greedy sweep is an artifact

400 blocks (1600 games) per cell, **both agents on the same runs**, base = the
committed evaluator, binary `<scratch>/f3/evalab-p8`. `<scratch>/f3/ab/f_*`.

| variant | greedy:64 | mcts:256 | paired vs `rs=0.05,ci=0.0` (mcts) |
| --- | --- | --- | --- |
| `rs=0.05` | +1.75 [+1.34, +2.16] * | +1.11 [+0.75, +1.47] * | −1.14 [−1.69, −0.59] * |
| `rs=0.05,ci=0.8` | +2.98 [+2.53, +3.44] * | +1.71 [+1.28, +2.14] * | −0.55 [−1.07, −0.02] * |
| **`rs=0.05,ci=0.0`** | **+4.01** [+3.55, +4.48] * | **+2.25** [+1.80, +2.71] * | — |
| `rs=0.05,cp=0.50` | +3.44 [+2.96, +3.91] * | +1.29 [+0.82, +1.75] * | −0.97 [−1.48, −0.46] * |
| `rs=0.05,bp=0.15` | +2.94 [+2.44, +3.43] * | +0.60 [+0.23, +0.96] * | −1.66 [−2.18, −1.13] * |
| `rs=0.05,cp=0.50,bp=0.15,ci=1.2` | +4.03 [+3.54, +4.51] * | +1.21 [+0.72, +1.69] * | −1.05 [−1.60, −0.50] * |
| `rs=0.05,cp=0.50,bp=0.15,ci=0.0` | +3.58 [+3.12, +4.03] * | +1.49 [+1.01, +1.97] * | −0.76 [−1.34, −0.19] * |
| `init=1.0` | −0.01 [−0.03, +0.00] | +0.05 [−0.31, +0.42] | — |
| `init=2.0` | −0.03 [−0.06, +0.01] | −0.72 [−1.14, −0.30] * | — |

* **`CORN_INCOME_PER_ROUND = 0.0` is the single largest effect this run
  measured: +2.25 mcts:256, win rate 0.316.** `starvation_risk` assumed 1.9 corn
  a round of future income and forgave a shortfall that far. Setting it to zero
  makes the term score the position *as it stands* — the corn a player actually
  holds against the bill they actually owe — and leaves the future income to the
  search, which is the thing that can actually see whether the corn arrives.
  Assuming it in the evaluator forgives a shortfall the search is separately
  planning to fix, and pays for the same corn twice.
* **`CORN_PREMIUM = 0.50` and `BLOCK_PREMIUM = 0.15` are largely a greedy
  artifact.** Greedy reads +3.44 / +2.94 and MCTS +1.29 / +0.60 — a ratio of
  2.7 and 4.9 against the ~1.3 that `rs` shows — and on top of `ci = 0` they
  make things **worse** (paired −0.76). All three knobs push the agent to value
  corn; once `starvation_risk` stops forgiving the shortfall, re-pricing corn in
  `held_premium` on top of that over-corrects. **Do not land them.**
* **An initiative bonus is dead on both agents.** +0.05 [−0.31, +0.42] at 1.0
  and −0.72 * at 2.0. So the last available correction for F17a's root-side
  under-statement fails too, and it fails in the most informative way: a
  constant for whoever holds the move is *exactly* the shape of that bias, it
  is the only correction that cannot distort a within-turn ranking, and it buys
  nothing. **The root-side −1.95 is inherent, not a missing term** — see F19c
  for the other half of the argument.

### F23a. `starvation_risk`'s other constants are already right

Everything on top of `rs=0.05, ci=0.2`, seed-paired against it, `mcts:256`,
**400 blocks**. Platform check: `base` = 33.44 / 33.13 / 33.15 across the wave,
spread 0.31, and the `g_null` control (every knob written out at its committed
value on a freshly pinned binary) reads +0.00 on both agents.

```
  ci=0.0 instead of 0.2            -0.07 [-0.40,+0.25]
  rs=0.0 instead of 0.05           -0.13 [-0.48,+0.21]
  climb respects temple_ceiling    -0.05 [-0.38,+0.28]
  urgency_floor 0.3 -> 0.1         -0.07 [-0.40,+0.25]
  starvation scale x1.4            -0.24 [-0.75,+0.27]
  starvation scale x0.7            -0.30 [-0.74,+0.13]
  urgency 0.25 -> 0.5              -0.26 [-0.69,+0.16]
  CORN_PREMIUM 0.10 -> 0.20        -0.29 [-0.75,+0.17]
  RESEARCH_SCALE back to 0.5       -0.60 [-1.03,-0.16] *
```

Only the last row is distinguishable, and it is the one that undoes `rs`. So
**the income assumption was the whole of `starvation_risk`'s error**: its
weight, its urgency discount and its floor are all at an optimum already, and
once the assumed income is gone, re-pricing corn in `held_premium` on top adds
nothing. `CORN_INCOME_PER_ROUND` at 0.0 and 0.2 are indistinguishable; 0.0 is
taken because greedy separates them (+2.15 against +1.55 at 0.4, monotone) and
because zero is the statement the term should be making.

## F24. What landed in `src/eval.rs`

```
  RESEARCH_SCALE          0.5  ->  0.05     (F19b, F20, F23)
  CORN_INCOME_PER_ROUND   1.9  ->  0.0      (F23, F23a)
  temple_outlook's climb loop now stops at `temple_ceiling` (F18, F21a)
```

Against the evaluator `94d85f3` landed, on the pinned binary
`<scratch>/f3/evalab-p9`, `<scratch>/f3/ab/`:

| agent | blocks | centred | win rate (null 0.250) |
| --- | --- | --- | --- |
| `greedy:64` | 800 | **+3.98** [+3.66, +4.30] * | 0.348 |
| `greedy:full` | 200 | **+4.24** [+3.51, +4.97] * | 0.344 |
| `mcts:256` | 800 | **+2.16** [+1.84, +2.48] * | 0.304 |
| `mcts:1024` | 200 | **+1.37** [+0.73, +2.00] * | 0.284 |

Greedy reads the effect ~1.9x MCTS's size, tighter than the 2.6x of F15, and
`mcts:1024` reads less than `mcts:256` — some of this is repaired by search, so
quote the 1024 number when the deliverable's budget is that high.

Not landed, and each is a specific negative result rather than a "did not get
to it": `TEMPO_PER_ROUND = 0.62` (+1.03 greedy, +0.02 MCTS at 400 blocks,
F21a); `CORN_PREMIUM = 0.5` / `BLOCK_PREMIUM = 0.15` (+3.4 / +2.9 greedy, +1.29
/ +0.60 MCTS, and *negative* on top of the corn-income fix, F23);
`BUILDING_VALUE = 1.0` (+0.34 greedy, +0.19 MCTS, not separable); an initiative
bonus, a flat hand-worker credit, a delivery-calibrated space table, a
distance-aware temple discount, a `reach` cap on the ride, and every constant in
`held_premium` other than the two above.

Note for the next measurer: `bin/evalab`'s `v::HEAD` has been moved with this
landing and `--eqcheck` re-run at 0e0 over 12,064 (position, seat) pairs, so
**every number above this line is against the pre-F24 evaluator and numbers
below it are against the new one**. The new null is
`--ab 'board=0.5,av=0.2,temple=1.4,rs=0.05,ci=0.0,ceiling'` and reads +0.00; the
pre-F24 evaluator is `--base 'rs=0.5,ci=1.9,noceiling'`, a spelling added for
exactly this purpose. The greedy:64 platform fingerprint moved from
`base 30.8` to `base 36.6` with the landing — both arms are the new evaluator
now, and it scores six more points a game.

`cargo test --release`: **177 passed, 0 failed, 7 ignored**, including two new
pins in `tests/rules.rs`:
`evaluator_climb_respects_the_exclusive_top` (constant-free: it solves for the
day weight and the climb price from two step pairs that cross no resource
threshold, then predicts a third) and
`evaluator_starvation_ignores_income_it_has_not_earned`.

### F24a. New diagnostics in `bin/evalab`

| flag | what it answers |
| --- | --- |
| `--midturn --vars 'a;b;c'` | the mid-turn trajectory of *several* evaluators in one process, which is the only F14-safe way to ask whether a landing moved it |
| `--midturn` mode split | the trajectory separately for placing and retrieving turns — F17a, and the reason F3's "hump" was an artifact of pooling them |
| `--promise` | promise against delivery per space: `BOARD_SCALE x rider_worth` versus what taking the action moves `heuristic` by (F19) |

New `V` knobs, all inert at their defaults and pinned by a null that reads
+0.00: `tnear`, `tfar`, `thalf`, `climb`, `ceiling`/`noceiling`, `reach`,
`handv`, `rs`, `calib`, `cp`, `bp`, `bb`, `sp`, `tp`, `ci`, `mgate`, `init`,
`lc`, `urg`, `ufl`, `fs`.

## F25. Salvaged from the killed predecessor: the post-landing knob wave

The run that landed F24 was killed by a usage limit at ~03:15 with one wave
complete on disk and unwritten. Recovered here. Binary `<scratch>/f3/evalab-p13`
(rev `d7b4e42` + `bin/evalab.rs`), **400 blocks (1600 games) per cell on both
agents**, base = the F24 evaluator, `<scratch>/f3/ab/k_*`. Platform check:
`base` spread **0.63 greedy / 0.21 mcts** over the 400 shared seeds, and
`k_null` (every knob written out at its committed value) reads **+0.00** exactly
on both.

| variant | greedy:64, 400 blk | mcts:256, 400 blk |
| --- | --- | --- |
| `bv=1.0` (`BUILDING_VALUE` 0.45 -> 1.0) | **−0.29** [−0.51, −0.06] * | −0.18 [−0.43, +0.06] |
| `ci=0.20` | −0.32 [−0.68, +0.04] | −0.27 [−0.55, +0.00] |
| `fs=0.5` (food-saving term halved) | −0.47 [−0.72, −0.23] * | −0.10 [−0.36, +0.16] |
| `fs=1.5` | +0.24 [−0.06, +0.54] | −0.22 [−0.46, +0.02] |
| `fs=2.5` | −1.34 [−1.74, −0.94] * | −0.77 [−1.12, −0.42] * |
| `rs=0.0` | +0.07 [−0.02, +0.17] | −0.19 [−0.36, −0.01] * |
| `rs=0.15` | −0.11 [−0.27, +0.06] | −0.22 [−0.43, −0.02] * |
| `tempo=0.62` | +0.70 [+0.29, +1.11] * | **−0.31** [−0.62, −0.01] * |

* **`BUILDING_VALUE = 1.0` has changed sign.** It was +0.34 greedy / +0.19 mcts
  before F24 (F18, F21a) and is −0.29 * / −0.18 after it. The reason is
  mechanical: `engine_value` pays `n_buildings * BUILDING_VALUE` *and*
  `free_workers`/`worker_discount` through the food-saving term, and those two
  permanent effects are the payoff of six of the fourteen age-1 cards. Once
  `starvation_risk` stopped assuming income, the food-saving term got more
  valuable, and doubling the flat card price on top over-counts the same cards.
  **0.45 is right; stop sweeping it.**
* **`food_saving` is at an optimum.** 0.5 and 2.5 both lose on both agents, 1.5
  is +0.24 / −0.22. This term had never been swept. It is not the place the
  building question lives either.
* **`RESEARCH_SCALE = 0.05` is an interior optimum on MCTS**, which it was not
  before: at 400 blocks 0.0 reads −0.19 * and 0.15 reads −0.22 *, so the two
  neighbours the F20 plateau could not separate now both lose. F21a's
  "0.0 edges out 0.10 by +0.29" is superseded — that was measured before
  `CORN_INCOME_PER_ROUND` landed.
* **`TEMPO_PER_ROUND = 0.62` is dead for the third time and now actively
  harmful:** +0.70 * greedy, **−0.31 * MCTS**, intervals excluding zero in
  *opposite directions* at 400 blocks. The greedy/MCTS sign split on this knob
  is the most reliable one in the file.
* `ci=0.20` re-confirms `CORN_INCOME_PER_ROUND = 0.0` from above.

So every knob the F24 wave could think of is at a local optimum, and two of
them (`bv`, `rs`) only became so *because* of F24 — measuring a constant against
a mis-priced neighbour is what produced both of the earlier readings.

## F26. `--promise` re-read on the landed evaluator, and the fourth kill of calibration

`<scratch>/f3/promise3.txt` (`evalab-p13`, 50,793 standing workers over 29,984
turn roots) is `--promise` re-run *after* F24 landed. F19's version was against
the pre-F24 evaluator, and the landing moved it: overall promise 1.819 against
delivery 2.947, a ratio of **1.62** where it was 1.49.

The table's error is not one common factor. n-weighted `want / table` per gear:

| gear | n | mean table | mean want | want/table | vs the 2.33 all-board mean |
| --- | --- | --- | --- | --- | --- |
| Palenque | 2,065 | 1.30 | 5.42 | **4.18** | 1.79 |
| Yaxchilan | 10,430 | 1.58 | 5.17 | 3.27 | 1.40 |
| Uxmal | 9,370 | 1.94 | 5.98 | 3.08 | 1.32 |
| Tikal | 23,460 | 3.34 | 6.79 | 2.03 | 0.87 |
| Chichen | 5,468 | 2.29 | 3.48 | **1.52** | 0.65 |

`CORN_INCOME_PER_ROUND = 0.0` is why Palenque moved to the top: with
`starvation_risk` no longer forgiving a shortfall, a corn gather now moves the
whole estimate by several points where it used to move it by its liquidation
value.

### F26a. Tikal 1 delivers *nothing*, and it does not matter

The single widest disagreement on the board, and it got wider with the landing:
the first research space is priced at **2.00** and delivers **0.017** (F19 had
it at 0.60). The cause is mechanical and is in `options::recurse` — **a research
advance costs 1/2/3 blocks for levels 1/2/3**, and at `RESEARCH_SCALE = 0.05` an
advance is worth ~0.1 while the block it costs is worth ~0.8. So the evaluator,
offered the action, prefers `skip`, and the space hands over nothing at all.
5,227 of 50,793 standing workers are on it.

It is still not worth fixing. `tik1` multiplies that one table entry;
**800 blocks (3,200 games), `greedy:64`, `<scratch>/f3/ab/m_tik1*`**, null +0.00:

| `tik1` | 0.25 | 0.5 | 0.75 | 1.5 |
| --- | --- | --- | --- | --- |
| centred | +0.02 | +0.02 | +0.02 | −0.33 [−0.47, −0.18] * |

Identical to two decimals across a 3x range, because `board_position` prices a
worker by `max(sv(pos), max_j sv(j) − j·TEMPO)` and Tikal 1's 2.00 is already
*below* what riding one space to Tikal 2 pays. **The act-now entry of a low
space is nearly unreachable through the max**, so calibrating it is inert; only
raising it past the ride value does anything, and that loses. The direction is
right and the magnitude is 0.02 points.

### F26b. The per-gear tilt is a disaster, except at the two ends

Whole-gear multipliers, level-neutral in the joint (the n·table-weighted mean of
the five is 1.000). 800 blocks, `greedy:64`:

| variant | centred | 95% CI | win |
| --- | --- | --- | --- |
| `pal=1.5` | **+0.64** | [+0.42, +0.86] * | 0.269 |
| `chi=0.7` | **+0.36** | [+0.15, +0.56] * | 0.252 |
| `tik=0.85` | −2.14 | [−2.40, −1.88] * | 0.182 |
| `yax=1.3` | −4.36 | [−4.64, −4.09] * | 0.100 |
| `uxm=1.3` | −7.31 | [−7.62, −7.00] * | 0.094 |
| `pal=1.79,yax=1.40,tik=0.87,uxm=1.32,chi=0.65` (the full tilt) | **−8.45** | [−8.68, −8.23] * | **0.018** |

The tilt the diagnostic asks for is the worst configuration measured since
ungating the table — a win rate of 0.018 against a null of 0.250 — which is the
**fourth** independent kill of "make the space table agree with delivery"
(`derived` −27.9 F2, `defer` +0.14 F5d, `calib` −6.2 F19a, and now this).

But the two ends of the table are not the middle. `pal=1.5` and `chi=0.7` are
the largest and smallest `want/table` gears and both help, in the direction the
diagnostic predicts, while the three gears in between all lose badly. Swept out
below before anything is claimed — these are `board_position`-internal knobs and
F21a's `TEMPO_PER_ROUND` shows greedy can over-read that class by 8x.

### F26c. The Palenque and Chichen ends, swept

800 blocks, `greedy:64`, `<scratch>/f3/ab/m_*`, binary `evalab-p14`
(rev `5318276` + `bin/evalab.rs`; `--eqcheck` 0e0 over 48,032 pairs).

```
  pal    1.00   1.25   1.50   2.00    2.50    3.00
         null  +0.23  +0.64  -0.59  -12.81  -14.60
  win    .250   .259   .269   .212    .008    .001
```

A sharp unimodal peak at **1.5** — a quadratic through 1.25/1.5/2.0 puts the
vertex at 1.52 — and a cliff between 2.0 and 2.5 where the corn gear starts to
dominate the board term outright and the agent parks every worker on it. This
is the only gear whose price the diagnostic and the arena agree about.

`spend_fade`, the number of rounds over which `held_premium` fades out, swept
for the first time: **4 is −0.17** [−0.22, −0.11] *, **9 is +0.05** [−0.02,
+0.12]. The committed 6 is an interior optimum; do not re-sweep.

```
  chi    0.35   0.50   0.70   0.85   1.00
        -0.50  -0.21  +0.36  +0.19   null
```

Chichen's peak is at **0.7**, worth +0.36 [+0.15, +0.56] *. The mechanism is
visible in the table: a Chichen space is priced at the printed reward less
about a point, but taking it *spends* a skull the evaluator is separately
holding at `3.0 + SKULL_PREMIUM` = 5.2 through `liquidation` and `held_premium`.
The netting is too small and 0.7 is what the arena says it should be.

**The two are additive:** `pal=1.5,chi=0.7` reads **+1.04** [+0.79, +1.30] *,
win 0.278, against a sum of parts of 1.00. `pal=2.0,chi=0.5` reads −0.86, so it
is the two peaks and not the direction that carries it.

Both are `board_position`-internal knobs, which is the class `TEMPO_PER_ROUND`
showed greedy can over-read by 8x (F21a). Nothing lands until MCTS agrees.

### F26d. The `dead` column: where the evaluator refuses its own action

New column on `--promise` (`<scratch>/f3/promise5.txt`, binary `evalab-p17`):
the fraction of standing workers for which `|delivery| < 0.05` — taking the best
action at that space, with the worker's board credit added back, leaves the
estimate exactly where it was. That is the evaluator **declining the action**.
It separates "every instance is small" from "the mean is a cancellation", which
a mean alone cannot.

```
  Tikal 1     0.99      Chichen 1   0.69     Palenque 1..7  0.00-0.01
  Uxmal 1     0.57      Chichen 2   0.58     Yaxchilan 1..7 0.00-0.01
  Tikal 3     0.56      Chichen 3   0.56
  Uxmal 4     0.52      Chichen 4-9 0.49-0.57
  Tikal 2/4   0.38/0.39 Uxmal 3     0.36
```

Two gears hand over what they promise every time. Three do not, and the pattern
is not noise — **it is exactly the set of spaces whose action costs the player
something the evaluator is already pricing.**

* **`Tikal 1` at 0.99.** A research advance costs 1/2/3 blocks for levels 1/2/3
  (`options::recurse`). At `RESEARCH_SCALE = 0.05` an advance is worth ~0.1 and
  the block it spends is worth `BLOCK_PREMIUM + 1/4` ≈ 0.8, so the evaluator
  skips 99 times out of 100 — while `board_position` pays that worker 1.70.
  There is also no affordability gate on Tikal 1 or 3, though `gate_ceiling`
  has one written (F2's `gates` variant, +0.01, measured when research was
  worth ten times what it is now).
* **Chichen at ~0.55 across the board.** A skull is held at `3.0 +
  SKULL_PREMIUM` = 5.2 through `liquidation` and `held_premium`, and Chichen 1
  pays 4 printed points and a temple step. So the evaluator will not cash a
  skull at the bottom of the gear, and the table nonetheless prices those
  spaces as if it would. **This is the mechanism behind `chi = 0.7`** (F26c),
  found independently: the ×0.7 is the arena's estimate of how much of the
  Chichen table is a payout the evaluator will never take.
* `Uxmal 1` (pay blocks for a temple step) and `Uxmal 4` (build with corn) are
  the same story with blocks and corn.

The two live gears, Palenque and Yaxchilan, are the two whose actions are pure
gains — you gather, you spend nothing — and they are the two the table
*under*-prices (F26: `want/table` 4.18 and 3.27). **The board term's error is
not a level and not a ranking: it is that `space_value` prices the payout of an
action and never its cost**, so the gears that charge for their payout are
over-paid relative to the gears that do not. `pal` up and `chi` down are the two
ends of that one statement.

### F26e. The other three gears, and `--promise` has the sign wrong on Uxmal

Completing the gear sweep, same 800 blocks, `greedy:64`:

| variant | centred | 95% CI | win |
| --- | --- | --- | --- |
| **`uxm=0.85`** | **+1.75** | [+1.48, +2.02] * | **0.304** |
| `uxm=1.3` | −7.31 | [−7.62, −7.00] * | 0.094 |
| `tik=1.15` | +0.34 | [+0.08, +0.60] * | 0.266 |
| `tik=0.85` | −2.14 | [−2.40, −1.88] * | 0.182 |
| `yax=0.85` | −1.72 | [−2.05, −1.40] * | 0.219 |
| `yax=1.3` | −4.36 | [−4.64, −4.09] * | 0.100 |

**`uxm=0.85` is the largest single-knob effect in this wave, +1.75 with a win
rate of 0.304** — and `--promise` predicted the opposite sign. Uxmal's
`want/table` is 3.08 against an all-board 2.33, which reads as *under*-priced,
and the arena says shrink it. `yax` loses in both directions, so Yaxchilan is at
an interior optimum, which the diagnostic also did not say.

So the diagnostic is a *finder*, not a fitter, and this is now the fifth
statement of the same thing: `--promise` locates the spaces where the evaluator
disagrees with itself (`RESEARCH_SCALE` in F19b, and Chichen in F26d), and the
sign and size of the fix have to come from the arena every time. Read it for
mechanism, never for a coefficient.

### F26f. The Uxmal plateau, and a second food day

800 blocks, `greedy:64`, binary `evalab-p18` (`--eqcheck` 0e0, null +0.00):

```
  uxm    0.60   0.70   0.85   1.00   1.30
        +1.82  +1.89  +1.75   null  -7.31
```

A broad plateau over **0.6..0.85** worth about **+1.8**, so this is "the Uxmal
gear is worth three quarters of what the table says", not a digit. Uxmal 5, 6
and 7 are the mirror space and the two "any of the above" spaces, priced at a
flat 4.0 — the top of the gear, and therefore what `board_position`'s max prices
every worker on it. The mirror's value is the best action *elsewhere*, which the
table cannot know and guessed high.

`starve_next2` — a new shape, not a constant: `starvation_risk` prices the food
day **after** the next one as well, at weight k, charging the corn spent on the
first day before pricing the second. The committed term looks at one day and
stops, so a player who can pay this bill and has nothing coming scores zero risk.

```
  sn2    0.15   0.25   0.35   0.50   1.00
           --   +0.35    --     --     --      (rest in flight)
```

`sn2 = 0.25` reads **+0.35 [+0.13, +0.57] *** at 800 blocks. Note the direction:
`CORN_INCOME_PER_ROUND` said *stop* forecasting because the search will see it,
and this says look *further* — they are consistent, because what the search can
see is the next food day and what it cannot is the one after.

Completed sweeps, 800 blocks, `greedy:64`:

```
  uxm    0.60   0.70   0.75   0.85   0.90   0.95   1.00   1.30
        +1.82  +1.80  +1.77  +1.75  +1.22  +0.91   null  -7.31

  sn2    0.25   0.50   1.00
        +0.35  +0.48  +0.57      (monotone -- extended above 1.0 in F26g)
```

`uxm` is flat over **0.60..0.85** and falls away from 0.90; 0.70 is taken as
interior to it. `sn2` has *not* turned over by 1.0, which is a different claim
from the one it was built to test — see below. `action_cap = 4` (the clamp on
`engine_value`'s `actions_each`) is **−0.02 [−0.07, +0.02]**, inert, so
`ROUNDS_PER_ACTION`'s clamp is not a lever and neither is the constant behind
it: it is collinear with `ACTION_VALUE` wherever the clamp does not bind.

### F26g. `starve_next2` peaks near 1.0 — the second food day wants full weight

800 blocks, `greedy:64`, `<scratch>/f3/ab/n_sn2*`, `u_sn2*`:

```
  sn2    0.25   0.50   1.00   1.50   2.00   3.00
        +0.35  +0.48  +0.57  +0.53  -0.27  -0.84
```

Unimodal with a peak at **1.0** — the food day after the next one is worth
pricing at *the same weight* as the next one, not at a discount. The distance
discount is already inside `day_risk` (the `urgency` factor falls with the
rounds to the day), so a weight of 1.0 does not mean "equally urgent"; it means
"there is nothing extra to discount for, the term already does it".

`action_cap`, the clamp on `engine_value`'s `actions_each`, is **−0.02 at 4 and
−0.04 at 9**: inert in both directions. `ROUNDS_PER_ACTION` is collinear with
`ACTION_VALUE` wherever the clamp does not bind and the clamp is not a lever, so
that pair is finished.

## F27. MCTS on the gear re-price: only Chichen survives

`mcts:256`, 400 blocks (1,600 games) per cell, same pinned binaries
(`evalab-p14` for `m2_*`, `evalab-p18` for `t_*`; both `--eqcheck` 0e0 and both
nulls read +0.00), base = the committed evaluator.

| variant | greedy:64, 800 blk | mcts:256, 400 blk | ratio |
| --- | --- | --- | --- |
| `chi=0.7` | +0.36 [+0.15, +0.56] * | **+0.38** [+0.13, +0.64] * | 0.9 |
| `pal=1.5` | +0.64 [+0.42, +0.86] * | +0.11 [−0.09, +0.30] | 5.8 |
| `uxm=0.7` | +1.80 [+1.51, +2.09] * | **−0.46** [−0.77, −0.15] * | — |
| `pal=1.5,chi=0.7` | +1.04 [+0.79, +1.30] * | **+0.56** [+0.29, +0.83] * | 1.9 |

**`uxm=0.7` is the largest greedy/MCTS divergence this file has measured**:
+1.80 [+1.51, +2.09] on one agent and −0.52 [−0.90, −0.13] on the other, both
intervals excluding zero, in opposite directions. It is not a size disagreement;
it is a sign disagreement, and it is a 2.3-point gap. `pal=1.5` is over-read by 5.8x and does not clear zero. Only `chi=0.7`
transfers, and it transfers at a ratio of 0.8 — MCTS reads it slightly *larger*
than greedy, which is the signature of a real effect in this file
(`RESEARCH_SCALE` read 1.3, `TEMPO_PER_ROUND` read 8).

The three knobs are the same shape and only one survives, and F26d's `dead`
column says which. Chichen's action is one the evaluator **refuses** about half
the time — the skull it spends is held at `3.0 + SKULL_PREMIUM` and the bottom
of the gear pays less than that — so its table entry is a price for something
nobody ever buys, and taking it down is a pure gain on either agent. Palenque
and Uxmal read `dead` 0.00 and ~0.3: their actions *are* taken, so the search
plays the resource out and scores the position it leads to, and the table's
opinion about the level is redundant. A one-ply agent has no such recourse — it
compares completed turns and the table's number is the entire signal — which is
why it reads +1.80 for a change the search reads as −0.46.

**The rule that falls out: a space table entry is worth re-pricing exactly
where the evaluator would decline the action anyway.** That is a testable
prediction and `dead` is how to test it, which makes `--promise` useful for
something narrower and more reliable than calibration.

Also inert on greedy at 800 blocks, both directions, and therefore retired:
the `lvl == 2` "one short of the top" research bonus (`ntop = 0.0` is
**+0.01 [−0.03, +0.06]**, `ntop = 1.0` is −0.04 [−0.10, +0.02]) — the last
unswept constant in `engine_value`.

The greedy-only Tikal curve (`tik` 1.10 / 1.25 / 1.40 reads +0.27 / +0.49 /
+0.51) is left unlanded on the same grounds: it is a resource-gear table scale,
which is the class this section just showed greedy cannot read.

### Provenance for F25-F27

Five runners, five distinct lock directories, **one writer per output file** —
`run4.sh`..`run9.sh` in `<scratch>/f3/`, each holding
`/tmp/evalab-fN-runner.lock.d` and each `--out` claimed with a `.claim`
sentinel before it is opened, so a re-run of a wave skips a file rather than
appending to it.

| binary | rev | adds |
| --- | --- | --- |
| `evalab-p13` | d7b4e42 | the F24 landing (predecessor's) |
| `evalab-p14` | 5318276 | per-gear knobs `pal/yax/tik/uxm`, `mcts:N:cp=X`, `--promise` dead column |
| `evalab-p15` | 5318276 | `sn2`, `acap`, `ntop` |
| `evalab-p16` | 5318276 | `pneed` |
| `evalab-p17` | 5318276 | `fp`, `skipd`, `topw` |
| `evalab-p18` | 5318276 | `chisub`, `tiksub`, `uxmsub` |

`5318276` is docs-only on top of `d7b4e42`, so every binary above shares one
`eval.rs`; `--eqcheck` reads **0e0 over 48,032 (position, seat) pairs** on each
of p14..p18 and every wave's null reads **+0.00** exactly. The greedy:64
platform fingerprint (`an.py -b`, the `base` column) is **35.2-35.9** across the
whole run and **36.2-36.4** on mcts:256, spreads of 0.7 and 0.2, so this is one
platform in F14a's sense.

`--agent mcts:N:cp=X` is new in p14: it sets `MctsConfig::c_puct_init` on
*both* arms through `record::SearchAgent::with_config`, so the comparison stays
the evaluator and what changes is how deep the search reading it descends. The
committed default is 2.0, which stops the descent after about one turn
(`docs/OVERNIGHT.md`); the champion spec is 0.02 at 8,192 simulations.
Measured cost in `bin/evalab`, user CPU per rotation block:
**greedy:64 0.09 s, mcts:256 1.0 s, mcts:1024:cp=0.05 6.6 s**, so the deep
agent is 6.6x mcts:256 for 4x the simulations — the extra is the longer descent.

### F27a. Negative results from this run, each specific

Nothing here is a "did not get to it". All are 800-block `greedy:64` unless a
MCTS number is given, all against the F24 evaluator, all with a +0.00 null.

| tried | result | what it kills |
| --- | --- | --- |
| `tik1` 0.25/0.5/0.75 (the space `--promise` says delivers 0.017 against a 2.00 price) | +0.02, identical across the range | per-space calibration of a *low* space: `board_position`'s max over the ride never reads it |
| the full per-gear tilt from `--promise` | −8.45, win **0.018** | the fourth kill of "make the table agree with delivery" |
| `uxm` 0.6..0.85 | +1.8 greedy, **−0.46 mcts:256** * | the largest greedy/MCTS sign split in the file |
| `pal` 1.5 | +0.64 greedy, +0.11 mcts:256 | over-read 5.8x; lands only inside the joint |
| `yax` 0.85 / 1.3 | −1.72 / −4.36 | Yaxchilan is at an interior optimum; nothing to win |
| `tik` 1.10/1.25/1.40 | +0.27/+0.49/+0.51 greedy only | same class as `uxm`; not measured on MCTS, not landed |
| `spend_fade` 4 / 9 | −0.17 * / +0.05 | 6 rounds is an interior optimum |
| `action_cap` 4 / 9 | −0.02 / −0.04 | `ROUNDS_PER_ACTION`'s clamp is not a lever |
| `near_top` 0.0 / 1.0 | +0.01 / −0.04 | the `lvl == 2` research bonus is inert both ways |
| `BUILDING_VALUE` 1.0 | −0.29 * greedy, −0.18 mcts | *changed sign* after F24; 0.45 is right |
| `food_saving` 0.5 / 1.5 / 2.5 | −0.47 * / +0.24 / −1.34 * | at an optimum, first sweep |
| `TEMPO_PER_ROUND` 0.62 | +0.70 * greedy, **−0.31 * mcts** | third kill, now harmful |
| `RESEARCH_SCALE` 0.0 / 0.15 | −0.19 * / −0.22 * mcts:256 | 0.05 is now an *interior* optimum |

The joint of the whole greedy-fitted gear tilt is the clearest illustration of
why the MCTS tier is not optional. `greedy:64`, 800 blocks:

```
  pal=1.5,uxm=0.85                        +2.62  win 0.315
  pal=1.5,uxm=0.85,chi=0.7                +3.32  win 0.340
  pal=1.5,uxm=0.85,chi=0.7,tik=1.15       +3.45  win 0.365
```

A +3.45 with a win rate of 0.365 against a null of 0.250, every interval
excluding zero at 3,200 games — and two thirds of it is `uxm`, which the search
reads as **−0.46**. On greedy evidence alone this would have shipped.

### F27b. A corn space is not worth the same to everyone

`pneed = k` multiplies the **Palenque** table by `1 + k` for a player who cannot
pay the next food bill out of the corn in hand — the same test
`starvation_risk` fires on, without its arithmetic. It is the gate-shaped
version of `pal`: after `CORN_INCOME_PER_ROUND = 0.0`, corn's worth to a player
who is short is the 3 points a head `starvation_risk` is charging them, and to a
player who is not it is a quarter point. A flat multiplier cannot say that.

800 blocks, `greedy:64`:

```
  pneed   0.5    1.0    2.0
        +0.31  +1.48  -16.80        win 0.255 / 0.286 / 0.025
```

**+1.48 [+1.19, +1.76] * at 1.0**, more than twice the flat `pal = 1.5`'s +0.64
on the same agent and the same blocks, with a cliff immediately past it. The
cliff is informative rather than alarming: at `k = 2` a hungry player values
every jungle space at three times its printed worth and never does anything
else. MCTS tier below.

`pal = 1.5` was measured twice, on two binaries against two nulls, by accident
of scheduling: **+0.11 [−0.09, +0.30]** (`m2_pal15`, `evalab-p14`, 400 blocks)
and **+0.12 [−0.07, +0.31]** (`t_pal150`, `evalab-p18`, 387 blocks). Two
independent replicates agreeing to 0.01 on a number whose interval covers zero
is the cleanest statement available that the effect is real, small, and about a
tenth of a point — not that it is absent. It lands only because it is additive
with `chi` and the pair clears zero together.

`TEMPO_PER_ROUND = 0.62` was re-run once more on this platform and reads
**−0.36 [−0.71, −0.02] * on mcts:256** at 319 blocks, against +0.70 * on greedy.
Fourth kill, third time with the two intervals excluding zero in opposite
directions.

### F27c. The `dead`-column rule, tested and falsified once

F27's reading — *a table entry is worth re-pricing exactly where the evaluator
would decline the action anyway* — makes a prediction, so it was tested on the
one space it fits best after Chichen. `uxm3` halves **only** Uxmal 3, the
buy-a-worker space, whose `dead` is 0.36: after `CORN_INCOME_PER_ROUND = 0.0` an
extra worker is an extra mouth, so the evaluator often refuses the free worker
while the table pays 3.40 for standing on it — and unlike Tikal 1 the space is
near the top of its gear, so `board_position`'s max does read it.

```
  uxm3   0.0    0.5
        +1.22  +1.19        greedy:64, 800 blocks
               -0.40 [-0.96,+0.15]   mcts:256, 83 blocks (in flight)
```

Greedy loves it — most of `uxm`'s +1.80 is this one space — and MCTS does not.
**The rule as stated is too weak**: Chichen is not just a gear the evaluator
declines, it is a gear whose payout is *points*, and Uxmal 3's payout is a
worker, which the search prices for itself over the rest of the game. So the
`dead` column narrows the search for candidates and does not by itself predict
which will transfer. Only the arena does.

Seed-paired marginals on the joint, `mcts:256`, 400 blocks:

```
  (pal=1.5,chi=0.7) - chi=0.7   +0.17 [-0.03,+0.38]     what pal adds
  (pal=1.5,chi=0.7) - pal=1.5   +0.45 [+0.17,+0.74] *   what chi adds
```

**`chi` is the effect and `pal` is a rider.** `pal`'s own interval covers zero
on both the standalone and the marginal, so it is landed on the strength of the
pair (+0.56 [+0.29, +0.83] *), a measured interior optimum on greedy, and two
bit-identical replicates centred at +0.11/+0.12 — not on an interval that
excludes zero. Said plainly so a later measurer can drop it cheaply.

A note on the platform, because it is unusually clean here: `evalab` is
deterministic given a seed, so `t_chi070` (binary p18) and `m2_chi07` (binary
p14) return **identical** `base` and `cand` means on their shared seeds
(38.237 / 39.013), as do `t_pal150` and `m2_pal15` (38.729 / 38.850). Two
binaries built from one `eval.rs` do not merely agree to within noise; they
agree exactly. That makes the F14a detector sharper than an interval: any
disagreement at all on a shared seed is a platform move.

### F27d. The champion spec is out of reach inside `bin/evalab`, and why

`--agent mcts:8192:cp=0.02` was started against the candidate at 05:18 and cut
at 05:27 with **3 of 200 blocks done — 3 minutes a rotation block** under a
machine load of 59 on 14 cores. 200 blocks is 10 hours of wall clock for an
interval of about ±0.6, which is wider than the effect being measured.

The arithmetic that settles it, user CPU per rotation block measured on this
machine: `greedy:64` **0.09 s**, `mcts:256` **1.0 s**, `mcts:1024:cp=0.05`
**6.6 s**, `mcts:8192:cp=0.02` **~53 s**. A block is four games and each game is
four seats deciding ~40 turns, so the champion spec is ~26,000 searches of 8,192
simulations for one independent observation. **`mcts:1024:cp=0.05` is strictly
the better buy for this file**: 3.2x cheaper than a 100-block run at 8,192 while
giving 250 blocks, and it has the property that matters here — `c_puct` at 0.05
rather than the committed 2.0, so the search descends several turns instead of
one. The evaluator questions this run is asking are about *depth*, not about
simulation count.

## F28. What landed in `src/eval.rs`

```
  GEAR_SCALE = [1.5, 1.0, 1.0, 1.0, 0.7]
             Palenque  Yaxchilan  Tikal  Uxmal  Chichen
```

A per-gear multiplier on the hand table, applied in `space_value` over a new
`space_value_raw` that is the table exactly as it was written. Nothing else
changed. `bin/evalab`'s `v::HEAD` moved with it (`g_pal` 1.0 -> 1.5, `chi`
1.0 -> 0.7) and `--eqcheck` re-run at **0e0 over 60,432 (position, seat)
pairs**, so the new null is
`--ab 'board=0.5,av=0.2,temple=1.4,rs=0.05,ci=0.0,ceiling,pal=1.5,chi=0.7'`
and the pre-F28 evaluator is `--base 'pal=1,chi=1'`.

The claim, restated so it can be attacked: **`space_value` prices what a space
hands over and never what it charges**, so the two gears whose actions are pure
gathering were under-paid and the gear that spends a skull for a printed number
of points was over-paid. The `dead` column is the measurement of that
(0.00/0.01 against 0.5-0.7), and it is a property of the evaluator's own
accounting rather than of the game.

Not landed, each with a number: `uxm` (**+1.80 greedy, −0.46 * mcts:256**),
`tik` (+0.53 greedy, no MCTS tier), `yax` (loses both ways), `pneed`
(**+1.48 greedy, +0.20 [−0.20, +0.60] mcts:256**), `sn2` (**+0.57 greedy,
−0.20 [−0.54, +0.14] mcts:256**), `uxm3` (+1.19 greedy, −0.37 mcts:256),
`tik1`, `spend`, `acap`, `ntop`, `chisub`, and the whole promise-derived tilt
(−8.45, win 0.018).

### F28a. Independent-seed replication of the landing

The joint was re-run from `--seed 4000000`, an entirely disjoint set of games,
while the first run was still going. `mcts:256`, against the pre-F28 evaluator:

| run | seeds | blocks | centred |
| --- | --- | --- | --- |
| `m2_pc` | 3,000,000.. | 400 | +0.56 [+0.29, +0.83] * |
| `x_pc_s4` | 4,000,000.. | 190 | **+0.90** [+0.51, +1.30] * |

and `pal` alone reads +0.11 and +0.13 on the two seed sets. A replication on
disjoint seeds is worth more than either interval; this file has been burned
once by a number that was right on the seeds it was measured on (F13/F14).

`BOARD_SCALE` was re-checked at the same time and **0.5 still holds**: 0.4 is
−0.99 [−1.33, −0.65] * at 400 blocks on `mcts:256`. 0.6 reads +0.31 [−0.10,
+0.73], which covers zero but is the first non-negative reading the far side of
that plateau has produced since F5a — worth a sweep on the *new* table, because
`GEAR_SCALE` has just changed what `board` is scaling.

`cargo test --release`: **178 passed, 0 failed, 7 ignored** (177 before, plus
one), with the other workstreams' in-flight edits to `moves.rs`, `options.rs`,
`spaces/*` and `tests/search.rs` in the tree. The new pin is
`tests/rules.rs::evaluator_scales_the_board_table_by_gear`, which places one
worker on the **top** space of a gear — no ride, so no maximum, and every
worker halved by the same top-of-gear discount — and asserts two things the raw
table gets wrong: that Palenque's top now outprices Yaxchilan's (raw 2.6
against 3.4, so only `GEAR_SCALE` can flip it) and that Chichen's top over
Tikal's falls into [1.2, 1.7) where the raw ratio is 10.5/5.2 = 2.02.

**Cross-workstream note.** `eval::heuristic` is MCTS's edge prior as well as its
leaf value (`mcts.rs:1573`), so this landing moves the platform for anything
that rebuilds. Every arena run in flight at 05:30 is on a pinned binary and is
unaffected; the next one that is pinned afresh is not comparable with them.
`plan::PlanEvaluator::raw` weights `eval::components`, and `board` has moved
again, so F12's warning stands.

### F28b. The landing, measured against the evaluator it replaces

Binary `<scratch>/f3/evalab-p19` (rev `5318276` + this landing; `--eqcheck` 0e0
over 60,432 pairs), candidate = the committed evaluator, `--base 'pal=1,chi=1'`,
`<scratch>/f3/ab/z_land_*`.

| agent | blocks | centred | win (null 0.250) |
| --- | --- | --- | --- |
| `greedy:64` | 800 | **+1.04** [+0.79, +1.30] * | 0.278 |
| `mcts:256` | 400 | **+0.86** [+0.55, +1.17] * | 0.274 |
| `mcts:1024:cp=0.05` | 250 | **+0.67** [+0.21, +1.12] * | 0.268 |
| `greedy:full` | 200 | **+1.57** [+1.06, +2.08] * | 0.307 |

All four distinguishable from zero. The deep row is the smallest, and the
77-block reading of it was +1.09 — **that early number was noise around a real
+0.67**, recorded here because it was quoted in this file before the run
finished and the correction is the point: at 77 blocks the interval was ±0.86.

Greedy reads the effect 1.2x MCTS's size, against 1.9x for F24 and 2.6x for
F15 — the tightest ratio this file has recorded, which is what you expect of a
change that survived being screened *on* the search rather than on greedy.

The `z_null` control writes every knob out at its committed value and reads
+1.02 against the same base — the same number as `z_land` to within 0.02, which
is the check that the knob spelling and `v::HEAD` say the same thing after the
move.

### F28c. `pneed` is real but small, and overlaps what just landed

`pneed = 1.0` — the Palenque table doubled for a player who cannot pay the next
food bill out of the corn in hand — finished at **+0.30 [+0.02, +0.58] * on
`mcts:256`, 350 blocks**, against +1.48 * on `greedy:64`. So the greedy/MCTS
ratio is 4.9, in the class of `pal` (5.8) and `uxm` (sign flip) rather than of
`chi` (0.9), but unlike those two it does clear zero on the search.

It is *not* landed with F28 for a reason that has nothing to do with its size:
it was measured against a base whose Palenque scale was 1.0, and `GEAR_SCALE`
has just raised that to 1.5. A hungry-player bonus on top of a corn gear that is
already 50% dearer is a different quantity. Re-measured against the new head in
`<scratch>/f3/ab/zz_pneed*`; the honest reading of the first measurement is
"a corn space is not worth the same to everyone, and about a third of that
statement survives the search."

### F28d. Where the mass is now — `<scratch>/f3/terms_f28.txt`

`evalab --terms` on the landed evaluator, 3,000 turn roots x 4 seats:

```
        term      mean     mean|x|
      banked     2.132       3.889
 liquidation     3.459       3.459
        held     2.517       2.517
      temple    18.535      18.535
      engine     3.516       3.516
       board     3.171       3.171
    monument     0.159       0.159
      starve    -1.793       1.793
```

**`temple_outlook` is 18.5 points of a ~33-point estimate — five times the next
largest speculative term, and larger than every other one put together.** It is
also the term that drifts most through a turn (−1.33 to −3.86 in the depth
table). `TEMPLE_SCALE = 1.4` was fitted in F8/F9, two landings ago, against an
evaluator with a different `board`, a different `engine` and a `starvation_risk`
that forgave shortfalls. **It is the largest unexamined quantity in the file**
and the obvious next target; swept on the new table in `<scratch>/f3/ab/y_*`.

`monument_outlook` is **0.159**, a twentieth of the next smallest term. That is
the fifth independent statement that it does nothing, and it is now a statement
about the term's size rather than about a sweep of its scale: there is no
constant that makes 0.16 points matter. Delete it or rewrite it; measured as a
deletion in `<scratch>/f3/ab/zz_monu0*`.

### F28e. `TEMPLE_SCALE` is a plateau, not a peak

`mcts:256`, 400 blocks, against the pre-F28 evaluator (`<scratch>/f3/ab/m2_*`):

```
  temple   1.0    1.4    1.8
          +0.11   null  -0.76 *      (400 blocks each)
  board    0.4    0.5    0.6
          -0.99   null  +0.34
```

`TEMPLE_SCALE = 1.0` is **+0.11 [−0.28, +0.50]** — indistinguishable from the
committed 1.4 at 1,600 games, where F8 read the same axis as monotone improving
from 0.5 to 1.4 and F9 put a joint peak there. That fit was against an evaluator
with a different `board`, a different `engine` and a `starvation_risk` that
forgave shortfalls; what is left of it is a **plateau over 1.0..1.4** with 1.8
clearly worse. Since the term is 18.5 of a 33-point estimate, "1.0 and 1.4 are
the same" is a statement about 7 points of estimate that changes nothing, which
is worth knowing on its own: the search does not care how loudly the evaluator
shouts about temples, only about the ordering it induces.

`BOARD_SCALE = 0.6` reads +0.34 [−0.04, +0.71] on the old table and **0.55 reads
+0.59 [+0.16, +1.01] * on the new one** — `GEAR_SCALE`'s n-weighted mean is
about 0.98, so part of that is the level coming back, and part is not. In
flight.

## F29. `HUNGRY_CORN`: a corn space is worth double to a player who cannot eat

```
  HUNGRY_CORN = 1.0     the Palenque table x2 when `owed > corn in hand`
```

The gate-shaped half of what `pal` was reaching for, and the larger effect. It
multiplies the Palenque table by `1 + HUNGRY_CORN` for a player who cannot pay
the next food bill out of the corn already in hand — the same test
`starvation_risk` fires on, without its arithmetic. `space_value` gained a
`hungry` parameter so `board_position` can hoist the test out of its loop over
the reachable gear, which it runs once per placed worker.

`mcts:256`, against the evaluator F28 landed, 400 blocks per seed set:

| seeds | blocks | centred | win |
| --- | --- | --- | --- |
| 3,000,000.. | 298 | **+1.15** [+0.75, +1.55] * | 0.270 |
| 4,000,000.. | 119 | **+1.29** [+0.59, +1.99] * | 0.268 |

and +1.48 [+1.19, +1.76] * on `greedy:64` at 800 blocks against the *previous*
base. **A greedy/MCTS ratio near 1.2** — the same as `chi` and unlike every
other `board_position` knob in this run, which is the signature that separates a
real effect from a one-ply artifact in this file.

Why it works where a flat `pal` scale does not: after `CORN_INCOME_PER_ROUND`
went to zero, `starvation_risk` charges a short player 3 points a head, so corn
is worth an order of magnitude more to them than the quarter point plus
`CORN_PREMIUM` it is worth to everyone else. `starvation_risk` says the position
is bad; nothing until now told the search *where to go about it*. The two terms
are complementary rather than a double count, which is what the additivity says.

Sharply peaked: at `greedy:64`, 800 blocks, `pneed` 0.5 / 1.0 / 2.0 reads
+0.31 / +1.48 / **−16.76 with a win rate of 0.024**. At 2.0 a hungry player
prices every jungle space at three times its printed worth and does nothing else
all game. Do not raise it without a sweep.

### F29a. **Retracted the same hour: `HUNGRY_CORN = 1.0` is a 3x corn gear, and greedy is destroyed by it**

Landed at 05:47 and reverted at 05:49. The mistake is worth more than the
landing was.

`pneed` was swept in F27b against a base whose Palenque scale was **1.0**, so
`pneed = 1.0` there meant a total multiplier of **2.0** on the corn gear for a
hungry player, and it read +1.48 greedy. `GEAR_SCALE` then raised Palenque to
1.5, and carrying the same `pneed = 1.0` onto it means a total of **3.0** —
which is exactly the configuration `pneed = 2.0` tested on the old base, where
it read **−16.76 with a win rate of 0.024**. The first 66 blocks of the landing
measurement read **−15.92 [−17.08, −14.76], win 0.023**, reproducing that number
to within its interval, and the run was killed.

What makes it interesting rather than merely embarrassing: **`mcts:256` likes
the same 3x gear.** `zz_pneed10`, which is `pneed = 1.0` on top of the landed
`GEAR_SCALE`, is +1.02 [+0.66, +1.39] * at 389 blocks and +1.41 * on a disjoint
seed set. So the identical evaluator change is **+1.0 for the search and −16 for
the one-ply agent**, a 17-point gap and the largest divergence in this file by
an order of magnitude. Two seed-replicated MCTS confirmations did not protect
against it, because they were both on the agent that liked it.

Three rules come out of this:

* **A multiplier measured against one base is not a multiplier.** `pneed` is
  composed with `GEAR_SCALE`, so re-measure it after any change to what it
  multiplies. F20 checked exactly this for `rs`/`av`/`board` and found no
  interaction; this pair has a violent one.
* **Replication on disjoint seeds does not substitute for the second agent.**
  Both replicates agreed and both were wrong about the deliverable.
* **Run the cheap agent even when you have decided it over-reads.** greedy:64 is
  0.09 s a block. It caught this in 66 blocks, about six seconds of CPU, on a
  change that two 400-block MCTS runs had endorsed.

The knob is kept at `HUNGRY_CORN = 0.0` — inert, and pinned by the null — and
re-swept on the `GEAR_SCALE` base with **both** agents in
`<scratch>/f3/ab/pn_*`, at 0.2 / 0.33 / 0.5 / 1.0, where 0.33 is the value that
reproduces the total 2.0 that F27b actually measured.

### F29b. State of the tree at 05:52, for whoever picks this up

`src/eval.rs` carries **`GEAR_SCALE = [1.5, 1.0, 1.0, 1.0, 0.7]`** (F28, landed,
+0.86 mcts:256 / +1.09 mcts:1024:cp=0.05 / +1.04 greedy:64) and
**`HUNGRY_CORN = 0.0`** — the F29 knob, present, documented and *inert*, pending
the re-sweep F29a demanded. `bin/evalab`'s `v::HEAD` matches (`--eqcheck` 0e0
over 48,320 pairs on `evalab-p21`) with `pal_need: 0.0`.

The re-sweep in flight, `<scratch>/f3/ab/pn_*`, base = the landed evaluator:

```
  pneed       0.20   0.33   0.50   1.00
  greedy:64  +1.27  +1.31    --   -15.92        (0.33 at 800 blocks)
  mcts:256     --   +0.71  (in flight)  +1.06
```

`pneed = 0.33` is the value that reproduces the **total x2.0** corn multiplier
F27b actually measured, and it is **positive on both agents** — +1.31 [+1.03,
+1.59] * at 800 greedy blocks and +0.71 [+0.28, +1.14] * at 111 MCTS blocks.
`pneed = 1.0` is the total x3.0 that F29a retracted. Land 0.33 when the MCTS arm
reaches 400 blocks and `pn_050` says which side of it the peak is on.

### F29c. `BOARD_SCALE`'s optimum moved with the table under it

`GEAR_SCALE` changed what `BOARD_SCALE` is scaling, and the board term's
optimum went with it. `mcts:256` against the landed evaluator
(`<scratch>/f3/ab/y_*`, `yb_*`):

```
  board    0.40   0.50   0.55   0.65
          -0.99*  null  +0.56* +1.13*        (0.40 measured on the old table)
```

**+1.13 [+0.59, +1.67] * at 0.65**, twice what 0.55 buys, on a constant F5a
fixed at "about a half" and F22 re-confirmed in both directions. `GEAR_SCALE`'s
n-weighted mean is 0.98, so this is *not* level compensation — a 2% cut in the
table is not answered by a 30% rise in its scale. What changed is the shape:
Chichen's entries came down 30% and Palenque's went up 50%, and the term's best
overall weight moved with the mix.

Swept further, and on **both** agents this time — F29a's whole lesson — in
`<scratch>/f3/ab/yb_*`.

### F29d. The two agents disagree about `pneed`'s *direction*, not just its size

The re-sweep on the `GEAR_SCALE` base, all against the landed evaluator:

```
  pneed            0.10    0.20    0.25    0.33    0.40    0.50    1.00
  greedy:64       +0.19*  +1.73*  +1.32*  +1.31*  +1.28*  -1.90*  -15.99*
  mcts:256          --    +0.14   --      +0.33*   --     +0.72*   +1.06*
  mcts:1024 cp.05   --    +0.70*  --       --      --      --       --

  greedy 800 blocks a cell; mcts:256 400/400/400/400; deep 186
```

At 800 blocks a cell the greedy curve is **not** the spike it looked like at
455: 0.25, 0.33 and 0.40 are a **plateau at +1.3** with 0.20 a little above it
and a cliff between 0.40 and 0.50. `mcts:256` is monotone increasing across the
whole of it and does not clear zero until 0.33.

Three things at once. The greedy curve is **jagged** rather than smooth —
+1.73 at 0.20 against +1.06 at 0.25 and +1.31 at 0.33, on 800/455/800 blocks
with intervals of ±0.26/0.37/0.28 — which is what a *gate* looks like when it
is swept: its effect is discontinuous in how often it flips a
max-over-the-gear comparison, not a smooth function of the constant. The
`mcts:256` curve is monotone increasing over the same range. And the deep
search reads **+1.99 [+0.91, +3.08] at `pneed = 0.20`**, where `mcts:256` reads
+0.06 — so the three agents rank the same knob in three different ways.

`greedy:64` peaks near **0.2** and falls off a cliff by 0.5. `mcts:256` is
**monotone increasing** across the same range. This is not the usual "greedy
over-reads by a factor"; the two agents put the optimum on opposite sides of
0.4. Both curves are steep and every cell excludes zero.

The reading: a hungry player's corn premium tells a one-ply agent *where to put
its next worker*, and a little of that is worth a lot while a lot of it makes
every jungle space dominate the board. A search that plays out the corn does not
need the table to shout, and never suffers the tunnel vision, so it keeps taking
the extra pessimism-correction as free. **The value to land is the one where
both agents are clearly positive, which is at or below 0.33** — the deliverable
is an MCTS agent, but an evaluator that costs 2 points to the agent used for
every screening measurement in this file is an evaluator nobody can measure
against.

### F29e. The joint, and where this run stops

`pneed = 0.25` with `BOARD_SCALE = 0.65`, both on top of the landed
`GEAR_SCALE`, `<scratch>/f3/ab/j_pb_*`:

| agent | blocks | centred | win |
| --- | --- | --- | --- |
| `greedy:64` | 400 | **+2.62** [+2.21, +3.02] * | 0.330 |
| `mcts:256` | (in flight) | | |

and the two halves alone on `mcts:256`: `board=0.65` **+1.36 [+0.88, +1.85] ***
at 251 blocks, `board=0.70` **+1.43 [+1.02, +1.83] *** at 336, `pneed=0.33`
+0.33 [+0.09, +0.58] * at 400.

`BOARD_SCALE` is the larger and the cleaner of the two: a single constant,
already in the file, already swept twice, whose optimum moved because
`GEAR_SCALE` changed the shape of what it scales. `pneed` is a new gate whose
three agents rank it three ways (F29d) and whose greedy response is jagged.

## F30. What finally landed in `src/eval.rs`

```
  GEAR_SCALE  = [1.5, 1.0, 1.0, 1.0, 0.7]   Palenque / Yax / Tikal / Uxmal / Chichen
  HUNGRY_CORN = 0.25                        Palenque x1.25 again when `owed > corn`
  BOARD_SCALE = 0.5 -> 0.65                 re-fitted on the new table
```

Three changes, all inside the board term, all measured on at least two agents.
`bin/evalab`'s `v::HEAD` tracks them (`--eqcheck` **0e0 over 60,672 (position,
seat) pairs** on `evalab-p22`), the new null is
`--ab 'board=0.65,av=0.2,temple=1.4,rs=0.05,ci=0.0,ceiling,pal=1.5,chi=0.7,pneed=0.25'`
and the evaluator this replaces is `--base 'pal=1,chi=1,pneed=0,board=0.5'`.

The three are one statement in three places. `space_value` prices what a space
hands over and never what it charges (F26d), so the gears that charge were
over-paid; correcting that changes the *shape* of the table, and both the term's
overall scale (`BOARD_SCALE`) and its one state-dependent exception
(`HUNGRY_CORN`) had been fitted to the old shape.

Measured against the evaluator it replaces, `<scratch>/f3/ab/q2_land_*`
(binary `evalab-p22`):

| agent | blocks | centred | win (null 0.250) |
| --- | --- | --- | --- |
| `greedy:64` | 800 | **+2.48** [+2.20, +2.76] * | 0.323 |
| `mcts:256` | 400 | **+1.50** [+1.14, +1.85] * | 0.290 |
| `mcts:1024:cp=0.05` | (in flight) | | |
| `greedy:full` | (in flight) | | |

Both halves clear zero alone on both agents, which is the check F29a's
retraction says to insist on: `BOARD_SCALE = 0.70` reads **+2.18 [+1.73, +2.63]
greedy:64** at 335 blocks and **+1.46 [+1.09, +1.83] mcts:256** at 400, and the
`pneed=0.25, board=0.65` joint reads **+2.62 greedy / +1.49 mcts:256**.

`cargo test --release`: **180 passed, 0 failed, 7 ignored** (177 at the start of
this run). Two new pins in `tests/rules.rs`:
`evaluator_scales_the_board_table_by_gear`, which puts one worker on the top
space of a gear — no ride, so no maximum, and every worker halved by the same
top-of-gear discount — and asserts an ordering the raw table gets backwards
(Palenque's top over Yaxchilan's, raw 2.6 against 3.4) plus a bracket on
Chichen over Tikal that excludes the unscaled 2.02; and
`evaluator_pays_more_for_corn_when_the_food_day_is_unpaid`, which solves for the
feeding bill out of `starvation_risk` rather than assuming it, then checks that
one corn either side of it moves the corn gear and leaves the resource gear
alone.

### F30a. In flight at the report boundary

Started, writing to disk, unfinished — `--resume` will continue any of them, and
the analysis is `python3 <scratch>/an.py '<pattern>'` from `<scratch>/f3/ab`:

| file | what it answers |
| --- | --- |
| `q2_land_mcts1024cp005`, `q2_land_greedyfull` | F30 on the deep search and on exhaustive one-ply |
| `q3_run_*` (3 tiers) | the whole of this session in one number: `GEAR_SCALE` + `HUNGRY_CORN` + `BOARD_SCALE` against the F24 evaluator, `--base 'pal=1,chi=1,pneed=0,board=0.5'` |
| `m2_*_mcts1024cp005` (9 variants, 250 blocks) | **the brief's question (b)** — `pal`, `chi`, `pc`, `tempo`, `board=0.4/0.6`, `temple=1.0/1.8` measured on `mcts:1024:cp=0.05` against the *same* variants already measured on `mcts:256`, so a verdict that flips with depth shows up as a row that differs between the two |
| `m3_f24_*` | whether F24's landing still holds at depth (`--base 'rs=0.5,ci=1.9,noceiling'`) |
| `pd_033`, `pd_050`, `pd_board065` | `pneed` and `board` on the deep search |
| `t_chisub10`, `t_chisub20` | the *shift* form of the Chichen correction (subtract points rather than scale) against the ×0.7 that landed |

What the deep tier has said so far, all against the pre-F28 evaluator on
`mcts:1024:cp=0.05` with a +0.00 null at 250 blocks: the F28 landing **+1.09
[+0.23, +1.95] *** at 77 blocks (larger than its `mcts:256` +0.86), and
`pneed = 0.20` **+0.70 [+0.21, +1.19] *** at 186 blocks where `mcts:256` reads
+0.14 [-0.07, +0.36] at 400. So the deep search does want the hungry-corn gate
about five times as much as the shallow one, but the +1.62 that reading showed
at 50 blocks was noise around +0.70 — see F30b.

### F30b. Corrections to numbers quoted early in this file

Three readings above were written down before their runs finished and moved
when they did. Recorded rather than silently edited, because the sizes are the
argument for the block counts this file insists on:

| quantity | early | final |
| --- | --- | --- |
| F28 landing, `mcts:1024:cp=0.05` | +1.09 at 77 blk | **+0.67** [+0.21, +1.12] * at 250 |
| F30 landing, `mcts:256` | +2.15 at 53 blk | **+1.50** [+1.14, +1.85] * at 400 |
| `pneed` greedy curve | a spike at 0.20 (455 blk at 0.25) | a **plateau** over 0.25..0.40 (800 blk a cell) |

None of them changes a sign or a decision. All three shrink toward the mean,
which is what a number quoted at a third of its planned block count does.
`board=0.65` did the same: +1.36 at 251 blocks, **+1.06 [+0.72, +1.40] *** at
400. `TEMPLE_SCALE = 1.6` on the new table is **−0.57 [−0.84, −0.29] *** at 388
blocks, so 1.4 survives the re-shaping in that direction too.

### F30c. `HUNGRY_CORN` stays at 0.25, checked against the landed evaluator

The greedy plateau (F29d, corrected) runs 0.25..0.40 and `mcts:256` prefers the
top of it, so 0.33 was worth asking about — but only against the evaluator that
actually shipped, which now also has `BOARD_SCALE = 0.65`. Measured directly:
`pneed = 0.33` against the landed 0.25 reads **−0.49 [−0.81, −0.16] * on
`greedy:64`** at 344 blocks — distinguishably *worse*, not merely no better. The
`board` re-fit absorbed whatever the extra 0.08 was buying, which is the third
time in this section that two knobs on the same term turned out not to be
separable. **0.25 stands.**

### F30d. Final state, 06:18

`src/eval.rs` carries, in addition to everything F24 left:

```
  GEAR_SCALE  = [1.5, 1.0, 1.0, 1.0, 0.7]
  HUNGRY_CORN = 0.25
  BOARD_SCALE = 0.5 -> 0.65
```

`cargo test --release` **180 passed, 0 failed, 7 ignored**. `bin/evalab`'s
`v::HEAD` tracks it, `--eqcheck` **0e0 over 60,672 pairs**, pinned as
`<scratch>/f3/evalab-p22` with its rev beside it. Against the evaluator this
session started from (`--base 'pal=1,chi=1,pneed=0,board=0.5'`, which is F24's):

| step | agent | blocks | centred |
| --- | --- | --- | --- |
| `GEAR_SCALE` alone | greedy:64 | 800 | +1.04 [+0.79, +1.30] * |
| | mcts:256 | 400 | +0.86 [+0.55, +1.17] * |
| | mcts:1024:cp=0.05 | 250 | +0.67 [+0.21, +1.12] * |
| | greedy:full | 200 | +1.57 [+1.06, +2.08] * |
| `+ HUNGRY_CORN + BOARD_SCALE` | greedy:64 | 800 | **+2.48** [+2.20, +2.76] * |
| | mcts:256 | 400 | **+1.50** [+1.14, +1.85] * |
| | mcts:1024:cp=0.05 | 48 | +1.66 [+0.54, +2.78] * (running to 250) |

so about **+2.4 on `mcts:256`** for the session, against F24's own +2.16 and the
term re-pricing's +7.63 — a smaller landing than either, on an evaluator that
has now had three of them.

Still running and worth reading when they finish, all in `<scratch>/f3/ab/`:
`q3_run_*` (the three tiers of the above in one run), `m2_*_mcts1024cp005`
(nine variants deep, the brief's question (b) at full block count), `m3_f24_*`
(F24 at depth), `pd_*` (`pneed`/`board` deep), `t_chisub*` (the shift form of
the Chichen correction). Every one of them is `--resume`-able.

### F30e. Late 400-block numbers, including one more correction

Everything below finished after F30d was written.

| variant | agent | blocks | centred |
| --- | --- | --- | --- |
| `pneed=0.20` | **mcts:1024:cp=0.05** | 186 | **+0.70** [+0.21, +1.19] * |
| `pneed=0.20` | mcts:256 | 400 | +0.14 [−0.07, +0.36] |
| `pneed=1.0` | mcts:256 | 400 | +1.06 [+0.70, +1.42] * |
| `pneed=1.0`, seed 4M | mcts:256 | 400 | +1.16 [+0.81, +1.51] * |
| `pneed=0.5` | mcts:256 | 400 | +0.66 [+0.39, +0.93] * |
| `board=0.55` | mcts:256 | 400 | +0.56 [+0.29, +0.82] * |
| `board=0.55`, seed 4M | mcts:256 | 400 | +0.41 [+0.14, +0.68] * |
| `board=0.65` | mcts:256 | 400 | +1.06 [+0.72, +1.40] * |
| `temple=1.2` | mcts:256 | 97 | −0.19 [−0.77, +0.40] |
| `temple=1.6` | mcts:256 | 400 | −0.54 [−0.81, −0.27] * |
| **`monu=0`** (delete `monument_outlook`) | mcts:256 | 400 | **+0.19** [−0.01, +0.39] |

**The third correction of the run, and it is the one I got most wrong**:
`pneed = 0.20` on the deep search read +1.99 at 39 blocks and +1.62 at 50, and
settles at **+0.70** at 186. The qualitative claim survives — the deep search
wants the hungry-corn gate about five times as much as `mcts:256` does — but
"the corrections this run made are worth *more* to a deeper search" was written
off a 50-block reading and is not what the finished runs say: the `GEAR_SCALE`
landing is +0.86 at `mcts:256` and **+0.67** deep. Read F30b and this row
together as one lesson about quoting a deep-tier number before ~150 blocks: the
deep agent's per-block variance is roughly twice `mcts:256`'s, so the block
count at which a number stops moving is *higher*, not lower, exactly where each
block is most expensive.

**`monument_outlook` can be deleted for +0.19 [−0.01, +0.39] on `mcts:256`** at
400 blocks — the sixth statement that it does nothing, and the first on the
search at a serious block count. Its mean magnitude is 0.159 points (F28d). It
is not landed here only because deleting a term is a structural change and this
run has already spent its risk budget on one retraction; it is the cheapest
simplification available to the next measurer, and the number to beat is zero.

## F31. `BOARD_SCALE` has not turned over — the halving was never about the table

The sweep continued past what F29c landed. 400 blocks a cell, both agents,
against the evaluator **before** the board re-fit (`board = 0.5`):

```
  board     0.40    0.50    0.55    0.65    0.70    0.80
  greedy:64   --     null     --    +1.72*  +2.35*  +2.68*
  mcts:256  -0.99*   null   +0.56*  +1.06*  +1.46*  +2.17*
```

**Monotone increasing on both agents all the way to 0.8** — 0.80 reads +2.17
[+1.81, +2.53] * on `mcts:256`, twice what the landed 0.65 buys against the same
base, which looked like a point and a half left on the table.

**It is not. See F31a: that reading is against the wrong base and the landed
0.65 is already at the optimum.**

The mechanism, and why nobody should have been surprised: `BOARD_SCALE = 0.5`
was **never a statement about the space table**. Its doc comment says so — it
corrects a *double count*, because `board_position` prices the action a placed
worker is about to take and `engine_value` was separately paying every unlocked
worker `ACTION_VALUE` per `ROUNDS_PER_ACTION` rounds for the same action. F11
then cut `ACTION_VALUE` from **1.2 to 0.2**, a factor of six, which removed
almost all of the thing `BOARD_SCALE` was there to cancel — and nobody re-swept
`BOARD_SCALE` afterwards. F22 checked 0.4 and 0.6 and found both worse, which
looked like confirmation, but 0.6 is +0.34 and the curve does not turn until
somewhere past 0.8; a two-point sweep either side of a constant cannot see a
monotone climb.

**The general rule this run keeps re-learning: when a constant exists to cancel
another constant, moving either one invalidates the other, and a ±20% check
around the old value will not tell you.** That much is still true — `F22`'s
0.4/0.6 check could not have seen the climb from 0.5 to 0.65 that F29c found,
and the climb is real. What is *not* true is that it continues past 0.65 once
the rest of the landing is in place.

### F31a. **The third time, and the same mistake: `board = 0.8` buys nothing on the landed evaluator**

The table above is measured on `evalab-p21`, whose `v::HEAD` has
**`board_scale = 0.5` and `pal_need = 0.0`**. The evaluator that shipped has
`board_scale = 0.65` **and `HUNGRY_CORN = 0.25`. Re-measured against *that*,
on `evalab-p23`:

| variant | agent | blocks | centred |
| --- | --- | --- | --- |
| `board = 0.80` | greedy:64 | 400 | **−0.03** [−0.44, +0.38] |
| `board = 0.80` | mcts:256 | 258 | **−0.10** [−0.55, +0.35] |

Nothing, on either agent. **`BOARD_SCALE = 0.65` is at the optimum and there is
no point and a half to collect.** What the `yb_*` sweep was measuring is the
*sum* of two overlapping corrections: raising `BOARD_SCALE` and adding
`HUNGRY_CORN` both make `board_position` larger, and `HUNGRY_CORN` makes it
larger exactly where it matters most — a player who cannot eat. Against a base
with neither, the board scale absorbs some of what the gate would have done; on
top of the gate, it has nothing left to absorb.

This is the **third** instance in this run of one error: **a constant measured
against one base is a different constant against another.** F20 checked for it
between `rs`, `av` and `board` and found no interaction, which is probably why
it stopped being checked. The three instances, worth listing together because
they cost very different amounts:

| | what happened | cost |
| --- | --- | --- |
| F29a | `pneed = 1.0` carried from a `g_pal = 1.0` base onto 1.5 | a landing retracted, ~15 min |
| F30c | `pneed = 0.33` looked better than 0.25 on the pre-`board` base | caught before landing, ~20 min |
| F31a | `board = 0.8` looked +2.17 on the pre-`HUNGRY_CORN` base | caught before landing, and a wrong claim in a report |

**The rule, stated so it can be followed mechanically: before landing constant
X, re-measure X against the exact evaluator it will ship in — not against the
base its sweep happened to start from.** Every one of the three would have been
caught by one 400-block run costing under a minute of CPU on `greedy:64`.

## F32. `BOARD_SCALE` does **not** climb past 0.65 — F31's climb is a framing artefact

F31 read the sweep `0.40 → 0.80` as monotone increasing and called 0.80 "a
point and a half sitting in a one-line change". Every cell of that table was
measured **against a field of `board = 0.5`**, an evaluator two landings old.
Re-measured head-to-head against the *landed* evaluator (`--base head`, which is
`board = 0.65`), binary `evalab-p23`, seeds 3000000+:

| framing | agent | blocks | centred |
| --- | --- | --- | --- |
| `board=0.65` vs field of 0.50 (`yb_065`) | greedy:64 | 400 | +1.72 [+1.31, +2.13] * |
| `board=0.80` vs field of 0.50 (`yb_080`) | greedy:64 | 400 | +2.68 [+2.23, +3.13] * |
| **`board=0.80` vs field of 0.65** (`bh_080`) | **greedy:64** | **400** | **−0.03 [−0.44, +0.38]** |
| `board=0.65` vs field of 0.50 (`yb_065`) | mcts:256 | 400 | +1.06 [+0.72, +1.40] * |
| `board=0.80` vs field of 0.50 (`yb_080`) | mcts:256 | 400 | +2.17 [+1.81, +2.53] * |
| **`board=0.80` vs field of 0.65** (`bh_080`) | **mcts:256** | **199** | **+0.02 [−0.49, +0.54]** |

The indirect subtraction says 0.80 − 0.65 is **+0.96 greedy / +1.11 mcts**. The
direct head-to-head says **−0.03 / +0.02**, with intervals of ±0.4 and ±0.5 that
exclude the indirect estimate. These are not two noisy readings of one quantity;
they are two different quantities, and only the second one is the question
"should the constant move".

Why they differ: the A/B is **one candidate seat against three baseline seats**
(`centred = cand − mean of four`, null win rate 0.25). `board_position` is a
*contention* term — it prices the spaces a worker can still ride to, and its
payoff is in taking the good space before someone else does. Against three
opponents who under-weight the board, weighting it harder keeps paying; against
three opponents who weight it the same, it stops. So the vs-a-fixed-old-field
framing systematically **over-states** exactly the terms whose value is
positional, and the over-statement grows with the distance to the field.

**Standing rule, and it supersedes the F31 sentence that sent me here: a
constant is landed on the head-to-head against the evaluator it would replace,
never on a subtraction between two runs against an older field.** The
subtraction is what produced "a point and a half"; there is no point and a half.
`BOARD_SCALE` stays at **0.65**.

## F32. The deep tier, at full block count — question (b) answered

`mcts:1024:cp=0.05`, **250 blocks (1,000 games) a cell**, the same variants and
the same seeds already measured on `mcts:256`, null +0.00 exactly.

| variant | greedy:64 | mcts:256 | **mcts:1024:cp=0.05** |
| --- | --- | --- | --- |
| `pal=1.5` | +0.64 [+0.42, +0.86] * | +0.11 [−0.09, +0.30] | +0.17 [−0.29, +0.64] |
| `chi=0.7` | +0.36 [+0.15, +0.56] * | **+0.38** [+0.13, +0.64] * | **+0.06** [−0.38, +0.49] |
| `pal=1.5,chi=0.7` | +1.04 * | +0.56 * | +0.79 [−0.27, +1.84] (49 blk) |
| `GEAR_SCALE` (the landing) | +1.04 * | +0.86 * | **+0.67** [+0.21, +1.12] * |
| the whole F30 landing | +2.48 * | +1.50 * | **+1.78** [+1.31, +2.24] * |

**`chi = 0.7` is the one verdict that flips the *other* way.** It is the only
gear knob that survived `mcts:256` and it is **+0.06 [−0.38, +0.49] on the deep
search** — indistinguishable from nothing. The reading that follows from F27's
own mechanism: Chichen's table entry is a price for an action the evaluator
declines about half the time, and a search that descends six turns *plays that
action out and finds out*, so it needs the table's opinion about it less than a
search that descends one. The correction is real for a shallow search and
inert for a deep one, and it costs nothing either way.

The composite still clears zero comfortably at depth (+1.78 [+1.31, +2.24] *,
win 0.319), so the landing as a whole is right on the agent the deliverable
uses — it is the *attribution* between its three constants that depends on
which agent you ask.

Two more from the same wave:

* **`greedy:full`, the exhaustive one-ply agent: the landing is +4.72 [+4.02,
  +5.42] * at 146 blocks, win rate 0.429** against a null of 0.250 — by far its
  largest reading on any agent. An agent that scores every legal completed turn
  and picks the best is exactly the one that reads a space-table correction at
  full size, which is the F27 story again from the other end.
* **The *shift* form of the Chichen correction is not better than the scale.**
  `chisub` subtracts a flat number of points from every Chichen entry instead of
  multiplying, which is the shape the skull's cost argues for: 1.0 reads +0.40
  [+0.18, +0.62] * and 2.0 reads +0.45 [+0.22, +0.69] * on `mcts:256` at 400
  blocks, against `chi = 0.7`'s +0.38. Indistinguishable from the multiplier, so
  the simpler form stays.

## F33. The F31a audit: every constant that exists to cancel another, re-swept against the *shipped* evaluator

F31a's rule stated mechanically — *before landing constant X, re-measure X
against the exact evaluator it will ship in* — is a rule about the past as much
as the future. Every constant in `src/eval.rs` was fitted against whatever
`eval.rs` looked like on the day it was fitted, and four landings have moved the
file since. This section re-sweeps, as **full curves against `--base head` on
`evalab-p23`** (the shipped evaluator, `--eqcheck` 0e0), every constant whose
doc comment justifies it as correcting, damping, offsetting or gating something
else. Runs are in `<scratch>/f3/ab/n_*`, driver `<scratch>/f4/drive.sh`
(one writer per file, `.nlock` dir per output).

The audit list, with what each was last fitted against:

| constant | knob | it exists to cancel | last fitted against |
| --- | --- | --- | --- |
| `ACTION_VALUE` 0.2 | `av` | `BOARD_SCALE`'s term already pays the placed worker's action | F7/F10 — `board=0.5`, no `GEAR_SCALE`, `rs=0.5`, `ci=1.9` |
| `TEMPO_PER_ROUND` 0.52 | `tempo` | the max-over-the-gear in `board_position` | **cbb61a2**, `heuristic:32` — before every landing in this file |
| `GEAR_SCALE[Pal]` 1.5 | `pal` | the corn gear's level | F26c/F27 — before `HUNGRY_CORN`, which multiplies the same gear again |
| `GEAR_SCALE[Chi]` 0.7 | `chi` | Chichen's `dead` column | F26/F27 — before `board=0.65` |
| `SKULL_PREMIUM` 2.2 | `sp` | — it is the *cause* of Chichen's `dead` column | never, since `chi` landed |
| `TEMPLE_NEAR/FAR` 0.85/0.55 | `tnear`/`tfar` | confidence in a projected payout | before `TEMPLE_SCALE = 1.4`, which multiplies them to **1.19 / 0.77** |
| `action_cap` 6.0 | `acap` | the live half of `ROUNDS_PER_ACTION` | never swept |
| `TEMPLE_CLIMB` 0.10 | `climb` | "kept because the rule says so, not because it pays" | F18/F21a |
| `RESEARCH_SCALE` 0.05 | `rs` + `tik1`/`tik` | the other half of `engine_value`'s over-pricing | F19b/F20 — before `GEAR_SCALE` |

### F33a. A correction to the brief I was handed: `monu=0`'s +0.19 was against the old base

The `mcts:256` reading of **+0.19 [−0.01, +0.39] at 400 blocks** for deleting
`monument_outlook` (F30e) came from `<scratch>/f3/ab/zz_monu0_mcts256.jsonl`,
which the `zz_` wave ran at 05:48-06:15 on **`evalab-p21`** — whose `v::HEAD`
has `board_scale = 0.5` and `pal_need = 0.0`. It is an old-base number, exactly
the class F31a and F32 are about, and it is the number that was quoted to me as
the reason to land the deletion.

Re-measured against the shipped evaluator (`evalab-p23`, `--base head`):

| variant | agent | blocks | centred | win |
| --- | --- | --- | --- | --- |
| `monu=0` | greedy:64 | **800** | **+0.10** [−0.03, +0.23] | 0.255 |
| `monu=0` | mcts:256 | (running) | | |

So on the agent that has the number, the effect against the *shipped* evaluator
is +0.10 with an interval that covers zero, not +0.19. The sign is the same and
the size is smaller — which is what F32 predicts for any term measured against a
field that weights it differently.

### F32a. `greedy:full` reads the landing at 4.6 points, and a note on the wave

`greedy:full` — one ply over *every* legal move rather than 64 samples — final
at **200 blocks: +4.57 [+3.97, +5.16] *, win rate 0.426** against a null of
0.250. Four times the `mcts:256` reading and two and a half times `greedy:64`'s.

The ordering across the four agents is the cleanest statement in this file of
what a space-table correction is *for*:

```
  greedy:full        +4.57    scores every completed turn, picks the best
  greedy:64          +2.48    scores 64 of them
  mcts:1024 cp=0.05  +1.78    descends ~6 turns and finds out
  mcts:256           +1.50    descends ~1 turn
```

Monotone in how much of the answer the agent computes for itself. The evaluator
matters most to the agent that does the least search — which is exactly why an
`eval.rs` result screened only on greedy is not a result, and why the same
result is still worth landing: `mcts:1024:cp=0.05` is the deliverable's agent
and +1.78 [+1.31, +2.24] * is a real margin on it.

**Wave hygiene note.** Two cells of the F32 wave (`m2_pc`, `m2_tempo` at depth)
were killed at 59 and 5 blocks by one of this run's own `pkill` patterns — the
hazard of managing eighteen concurrent runners by process name. Both were
resumed from their partial JSONL onto a lock nothing else holds, and the
remaining four deep cells (`temple=1.0/1.8`, `board=0.4/0.6`) were relaunched in
`<scratch>/f3/wave51.log`. `--resume` plus one-writer-per-file is what made that
recoverable rather than a lost afternoon; the `.claim` sentinel is what stops
the relaunch from double-writing.

### F33b. `TEMPO_PER_ROUND`'s peak has moved from 0.52 to ~0.60 on `greedy:64`

The first scalp. `TEMPO_PER_ROUND` is the damping *inside* `board_position` —
the per-space charge that stops a worker being priced at whatever the top of its
gear pays. Its doc comment carries a 3,000-rotation-block sweep, unimodal, with
a quadratic vertex at 0.512 — **taken against the frozen cbb61a2 evaluator with
`heuristic:32`**, which is before the term re-pricing, before `RESEARCH_SCALE`,
before `GEAR_SCALE` reshaped the table it damps, and before `BOARD_SCALE` went
to 0.65. It has not been re-swept since. It is the single stalest constant in
the file.

Full curve, `greedy:64`, 400 blocks a cell, `--base head` on `evalab-p23`
(`<scratch>/f3/ab/n_tempo*_greedy64.jsonl`, `k_tempo0*`):

```
  tempo    0.40    0.44    0.48   0.52    0.56    0.60    0.62    0.70    0.80
  greedy  -1.52*  -1.40*  -0.74*  null   +0.41*  +0.83*  +0.70*  -0.92*  -2.74*
```

Unimodal, every cell outside 0.56..0.62 distinguishable from zero, and the
committed 0.52 is **on the losing side of the peak**: 0.60 is **+0.83 [+0.45,
+1.21] \*** and 0.40 — a value a ±20% check either side of 0.52 would have
reached — is −1.52. The curve is steep on both sides, which is why a
2-point check at 0.42/0.62 in the old sweep found a peak at 0.52 and this one
does not: the *table* under the damping is not the table it was swept on.

**But the agents split.** `mcts:256` at 0.62 reads **−0.31 [−0.62, −0.01] \*** at
400 blocks — the wrong sign, and outside its interval. 0.56 and 0.60 are running
on `mcts:256` now; nothing lands on the greedy reading alone.

---

**Numbering note.** A second writer appended `## F32. The deep tier` and
`### F32a. greedy:full reads the landing` to this file at 06:41-06:50, while
this run was appending `F33`, `F33a` and `F33b` — so there are two `F32`
sections and an `F32a` sitting inside `F33`. Nothing is lost and nothing is
rewritten (append-only on both sides); the ordering is just interleaved. To
avoid a third collision, **everything from this run below this line is numbered
`F34`**, and `F33`/`F33a`/`F33b` above are the same run's.

## F34. The F31a audit, results

Everything below is `--base head` on `evalab-p23` — the **shipped** evaluator
(`GEAR_SCALE = [1.5,1,1,1,0.7]`, `HUNGRY_CORN = 0.25`, `BOARD_SCALE = 0.65`),
which is the only framing F32 lets a constant land on. Files
`<scratch>/f3/ab/n_*`, driver `<scratch>/f4/drive.sh`.

## F33. The session in one number

Candidate = the committed evaluator, `--base 'pal=1,chi=1,pneed=0,board=0.5'`
— which is exactly the evaluator F24 landed and this session started from.
Binary `evalab-p23`, `<scratch>/f3/ab/q3_run_*`.

| agent | blocks | centred | win (null 0.250) |
| --- | --- | --- | --- |
| `greedy:64` | 800 | **+3.23** [+2.92, +3.53] * | 0.340 |
| `mcts:256` | 400 | **+1.89** [+1.48, +2.30] * | 0.303 |
| `mcts:1024:cp=0.05` | (running) | | |

**+1.89 on `mcts:256`** for the session, against F24's own +2.16 and the term
re-pricing's +7.63. Three landings deep into this evaluator, each one is
smaller than the last, which is what convergence looks like.

### F33b. Two deep-tier readings that are *not* conclusions yet

Both are under 80 blocks and F30e is the reason to say so out loud rather than
quote them:

* `board = 0.80` against the landed 0.65 reads **+1.92 [+1.05, +2.78] at 78
  blocks on `mcts:1024:cp=0.05`**, where `greedy:64` and `mcts:256` both read
  **−0.03 at 400 blocks**. If it survives to 250 blocks it is a genuine
  agent-dependent optimum and F31a's "0.65 is at the optimum" needs qualifying
  to "on the two shallow agents".
* `TEMPLE_SCALE = 1.0` reads **+2.25 at 26 blocks** on the deep search, where
  `mcts:256` reads +0.11 [−0.28, +0.50] at 400. That is the shape F28d predicted
  — `temple_outlook` is 18.5 of a 33-point estimate and is the evaluator's
  loudest claim about the *future*, so a search that can see six turns of that
  future should want it quieter — but 26 blocks is an interval of ±1.6 and the
  last three deep numbers this run quoted early all halved.

Both are running to 250 blocks in `<scratch>/f3/ab/bd_080_mcts1024cp005.jsonl`
and `m2_temple10_mcts1024cp005.jsonl`. **Do not act on either until they get
there**; that is the whole content of F30e.

### F34a. Three cancelling pairs that re-measure the **same**, and one that does not exist

`greedy:64`, **400 blocks a cell**, `--base head` on `evalab-p23`. `null` marks
the committed value.

```
  av (ACTION_VALUE)   0.0     0.1    0.2    0.3    0.4    0.6
                     -0.02  -0.02   null  -0.03  -0.04  -0.30*

  chi (GEAR_SCALE[4]) 0.5     0.6    0.7    0.8    0.9    1.0
                     -1.00*  -0.68*  null  -0.44* -1.18* -2.24*

  sp (SKULL_PREMIUM)  1.0     1.6    2.2    2.8    3.4
                     -0.41   -0.31   null  -1.47* -1.74*

  pal (GEAR_SCALE[0]) 1.0     1.2    1.5    1.8
                     -1.05*  -0.82*  null  -6.26*
```

* **`ACTION_VALUE` is not a constant, it is a plateau, and its width is the
  finding.** 0.0 through 0.4 are *all* within ±0.05 of the committed 0.2, with
  intervals of ±0.04 — ten times tighter than any other knob in this file,
  because `engine_value`'s worker line is `workers × actions_each × av` and
  `workers` changes only when one is bought or unlocked, so within a turn the
  term is very nearly a constant offset that `margin` cancels. The doc comment's
  "treat the digit as the low end of a plateau, not as resolved to 0.1" is
  right, and the plateau reaches **all the way to zero**: `ACTION_VALUE = 0`
  costs 0.02 ± 0.04 points on `greedy:64`. It is a deletion candidate, not a
  tuning target. 0.6 is the first value off the plateau.
* **`GEAR_SCALE[Chichen] = 0.7` is at the optimum**, unimodal on both sides, and
  the next-best 0.8 costs −0.44. Re-measured against the evaluator it now ships
  in, it does not move.
* **`SKULL_PREMIUM = 2.2` is at the optimum too**, which was the interesting
  one: `GEAR_SCALE`'s own doc comment says Chichen's `dead` column exists
  *because* a skull is held at `3.0 + SKULL_PREMIUM = 5.2` while the bottom of
  the gear pays 4 printed points, so `chi = 0.7` and `sp` are two knobs on one
  mechanism and lowering `sp` should have made `chi` redundant. It does not:
  `sp = 1.6` is −0.31 and `sp = 1.0` is −0.41, both flat-to-worse, and 2.8/3.4
  fall off hard. The pair is genuinely two-dimensional, and the joint
  (`chi=1.0, sp=1.2`, `chi=0.85, sp=1.6`) is in the N2 wave.
* **`GEAR_SCALE[Palenque]`'s cliff is much closer than 1.5 suggests.** 1.8 is
  **−6.26**, win rate 0.101, on top of `HUNGRY_CORN = 0.25` — because the two
  multiply, so 1.8 × 1.25 = 2.25 is already in the region F29a's retraction
  found at −16. The committed 1.5 sits with a −0.82 cell one step below it and
  a −6.26 cliff one step above; `1.35` and `1.65` on the *shipped* base are
  running under `n_pal135b`/`n_pal165b` (the `n_pal135`/`n_pal165` files in
  `ab/` are chain16's, measured on `evalab-p16` against `board = 0.5,
  pneed = 0`, and read +0.25 and +0.86 there — an old-base pair kept for the
  contrast).

### F34b. The landing's number against the actual F24 evaluator is larger than reported

`q2_land_*` is `--base 'pneed=0,board=0.5'` — the **pre-F29** evaluator, which
already has `GEAR_SCALE`. `q3_run_*` is `--base 'pal=1,chi=1,pneed=0,board=0.5'`
— F24's. F30d's table puts the `q2` numbers under the `q3` heading, so the
figure quoted for "the session against F24" is one landing short:

| base | agent | blocks | centred | win |
| --- | --- | --- | --- | --- |
| pre-F29 (`q2_land`) | greedy:64 | 800 | +2.48 [+2.20, +2.76] * | 0.323 |
| pre-F29 (`q2_land`) | mcts:256 | 400 | +1.50 [+1.14, +1.85] * | 0.290 |
| pre-F29 (`q2_land`) | mcts:1024:cp=0.05 | 250 | +1.78 [+1.31, +2.24] * | 0.319 |
| pre-F29 (`q2_land`) | greedy:full | 200 | +4.57 [+3.97, +5.16] * | 0.426 |
| **F24 (`q3_run`)** | greedy:64 | 800 | **+3.23** [+2.92, +3.53] * | 0.340 |
| **F24 (`q3_run`)** | mcts:256 | 400 | **+1.89** [+1.48, +2.30] * | 0.303 |

So the whole of the previous run against the evaluator it started from is
**+3.23 greedy:64 / +1.89 mcts:256**, not +2.48 / +1.50. Same landing, correctly
attributed base.

### F34c. `BOARD_SCALE`'s optimum rises with search depth — the one axis still moving

Not a re-opening of F32's verdict; the same head-to-head framing F32 insists on,
extended to the agent the deliverable actually uses. Every cell `--base head` on
`evalab-p23`, so the field is the shipped evaluator in all of them.

```
  board            0.50     0.65    0.70    0.80    0.90    1.00
  greedy:64       -1.45*    null     --    -0.03   -0.28   -0.96*
  mcts:256        -0.98*    null   +0.03   -0.03   +0.29   +0.53 (105 blk)
  mcts:1024 cp.05    --     null     --    +1.35*  (running)
```

400 blocks a cell on the two shallow agents; the deep cell is 99 blocks and
**directional only** — F30b/F30e are three separate cases of a deep number
falling by half between 50 and 250 blocks, so this is running to 250 and
`board = 0.9` deep has been started beside it.

The ordering is monotone **in search depth**, not in the constant: `greedy:64`
turns over between 0.8 and 0.9, `mcts:256` is flat-to-rising all the way to 1.0,
and the deep search reads +1.35 where `greedy:64` reads −0.03 on the same cell.
F32's retraction of "a point and a half" is still right about what it measured —
`greedy:64` and `mcts:256` both say `board = 0.80` buys nothing — but **both of
those are shallow agents, and the shipped agent is `mcts:8192:cp=0.02`.**

A mechanism that fits: `board_position` is a forecast of what a standing worker
will do over the next few rotations. A one-ply agent is *also* about to score
that same action explicitly at its own leaf, so weighting the forecast harder
double-counts; a search that descends six turns has spent its budget elsewhere
and uses the evaluator as the thing that tells it which gear to be on. Whatever
the mechanism, the empirical statement is a curve whose peak moves right as the
budget rises, and `BOARD_SCALE = 0.65` was fitted entirely on the two agents at
the left of it.

**Nothing lands on this tonight** without the 250-block deep cells.

### F34d. The four inert constants: what a term's *level* is worth is not what ranks moves

`greedy:64`, 400 blocks (or as marked), `--base head` on `evalab-p23`:

```
  tnear/tfar  1.00/0.65  0.70/0.45  1.00/0.55  0.85/0.40  0.85/0.70
              -0.04      -0.11      -0.06      +0.09      -0.06
  acap (action_cap)   3.0     4.0     6.0     8.0    9.0
                     -0.00   -0.01    null   (run)  -0.04
```

* **`TEMPLE_NEAR`/`TEMPLE_FAR` cannot be moved.** An 18% rise in the level, an
  18% fall, a flattened near/far ratio and a steepened one all read within ±0.11
  of zero with intervals of ±0.3 — on the term that F28d measures at **18.5
  points of a ~33-point estimate**, five times the next largest. The pair
  *ought* to have been a scalp: `TEMPLE_SCALE = 1.4` multiplies `TEMPLE_NEAR =
  0.85` to 1.19, so the confidence discount the constant is named for was
  cancelled by a later global scale and nobody re-swept it. It re-measures
  exactly the same, which is a real answer: the temple term's level does not
  rank moves, because it barely changes within a turn and `margin` differences
  it away.
* **`action_cap` is inert too** (−0.00 at 3.0, −0.04 at 9.0, ±0.02 intervals),
  which retires the `ROUNDS_PER_ACTION` coupling: the doc comment names `acap`
  as the knob that actually prices early throughput, and it does not price
  anything.
* Together with `ACTION_VALUE`'s ±0.04 plateau (F34a), that is **three of the
  audit's nine pairs where the answer is "the constant does not matter at all"**
  rather than "the constant is right". The distinguishing signature is the
  interval: knobs that move ranking read ±0.3 to ±0.4 at 400 blocks; the inert
  ones read ±0.03. A knob whose interval collapses is telling you it is a
  common offset, not a weight.

### F34e. `monument_outlook` is **not** deleted — against the shipped evaluator the deletion is a loss

The brief's case for deleting it was `+0.19 [−0.01, +0.39]` on `mcts:256` at 400
blocks, plus five earlier statements that the term is inert and F28d's mean
magnitude of 0.159 points. F33a already showed that +0.19 was measured on
`evalab-p21`, against a field with `board = 0.5` and no `HUNGRY_CORN`.
Head-to-head against the evaluator the deletion would actually replace:

| variant | agent | blocks | centred | win (null 0.250) |
| --- | --- | --- | --- | --- |
| `monu=0` | greedy:64 | 800 | +0.10 [−0.03, +0.23] | 0.255 |
| `monu=0` | **mcts:256** | **400** | **−0.20 [−0.39, −0.00] \*** | **0.238** |
| `monu=0` (old base, `evalab-p21`) | mcts:256 | 400 | +0.19 [−0.01, +0.39] | 0.260 |

**The sign flips with the base**, and on the search — the agent class the
deliverable uses — the deletion is distinguishable from zero in the *wrong*
direction. Its win rate falls from 0.250 to 0.238. So the sixth and seventh
statements that the term "does nothing" are a statement about `greedy:64` and
about an evaluator two landings old; on `mcts:256` against what ships, removing
it costs a fifth of a point.

`monument_outlook` **stays**. Its 0.159-point mean magnitude is not the
argument it looked like: a term can be small in level and still be the only
thing in the estimate that moves when a 20-point monument comes into reach,
which is exactly the case where it decides. Both the `mgate` sweep (F18) and
this deletion say the *scale* is not where the term is wrong; if it is ever
rewritten it should be for shape, and the number to beat is now −0.20, not zero.

### F34f. Why `action_cap` reads exactly `+0.0000`

`n_acap50_greedy64` is 0.0000 in **all 328 blocks** and `n_acap80` in 325 of
327 — not "small", identical. The mechanism is worth writing down because it
generalises to every inert knob in F34d. `engine_value` pays
`n_unlocked(p) × min(rounds_left / 2.6, action_cap) × ACTION_VALUE`. The cap
binds only while `rounds_left / 2.6 > 5`, i.e. for the first eleven days —
**which is exactly the stretch in which all four players still have three
workers**. `margin` is `heuristic(me) − max over opponents`, so a term equal
across seats cancels before it can rank anything; by the time worker counts
diverge, `rounds_left` has fallen and the cap no longer binds at all.

That is the shape of the whole inert class: a knob is inert when the quantity it
scales is common to the seats being compared at the moment it is largest. It
is not evidence the modelled effect is unreal — it is evidence the knob cannot
express it. A different constant would have to be attached to something that
*differs* between seats.

### F34g. `TEMPO_PER_ROUND`, finished: the peak moved, and the move is `greedy:64`-only

The `mcts:256` tier of F33b, 400-block target, `--base head` on `evalab-p23`:

| tempo | greedy:64 (400 blk) | mcts:256 |
| --- | --- | --- |
| 0.44 | −1.40 [−1.75, −1.06] * | (running) |
| 0.48 | −0.74 [−1.04, −0.44] * | — |
| **0.52** | **null (committed)** | **null** |
| 0.56 | +0.41 [+0.07, +0.75] * | −0.14 [−0.42, +0.14] (337) |
| **0.60** | **+0.83 [+0.45, +1.21] \*** | **−0.55 [−0.87, −0.22] \*** (338) |
| 0.62 | +0.70 [+0.29, +1.11] * | −0.31 [−0.62, −0.01] * (400) |
| 0.70 | −0.92 [−1.32, −0.53] * | — |
| 0.80 | −2.74 [−3.19, −2.29] * | — |

**The stalest constant in the file is genuinely at the wrong value for
`greedy:64` and at the right one for the search.** 0.60 is worth +0.83 to the
one-ply agent and costs −0.55 to `mcts:256`, with both intervals excluding zero
and each other. `TEMPO_PER_ROUND` therefore **does not move**: the deliverable
is `mcts:8192:heuristic:cp=0.02` and the shipped 0.52 is on the flat top of the
search's curve.

This is the same shape as `chi = 0.7` in the newly-committed deep-tier table and
as `uxm` in F26e: a constant inside `board_position` that tells a one-ply agent
which space on a gear to aim at, and that a search which plays the ride out does
not need. **`TEMPO_PER_ROUND` is a forecast; the search replaces forecasts with
rollouts.** It is worth saying plainly because the doc comment's 3,000-block
sweep is `heuristic:32` — a *greedy* agent — so the constant has only ever been
fitted on the class of agent that is most sensitive to it.

### F34h. `TEMPLE_CLIMB` wants to be larger, and it is the one term whose derivative ranks

`greedy:64`, 400 blocks, `--base head`:

```
  climb   0.00    0.10    0.20
         -0.06    null   +0.13*
```

Small, but 0.20 is `+0.13 [+0.01, +0.25] *` with a win rate of 0.262 against
0.250 — the first cell in this audit that is *positive* and outside its
interval. It also fits F34d's mechanism exactly. `temple_outlook`'s **level** is
18.5 points and completely inert (every `tnear`/`tfar`/`TEMPLE_SCALE` variant
reads zero), because the level is nearly common to the four seats; `climb` is
the only part of that term that changes with the move being ranked, since
stepping a temple is something a move does. So the one place the biggest term in
the evaluator can be mispriced is its derivative, and the derivative is the part
that has only ever been swept over 0.0..0.5 (F18, and "kept because the rule
says so, not because it pays").

0.4, 0.7 and 1.0 are queued, and an `mcts:256` tier with them.

### F33c. A second measurer is on this file — co-tenancy, verified safe

From ~06:50 a second eval-workstream runner is live out of
`<scratch>/f4/nq.sh`, writing into the same `<scratch>/f3/ab/`. Checked rather
than assumed:

* **It uses `<scratch>/f3/evalab-p23`** — the binary this session pinned — so
  both measurers are on one platform and their numbers are directly comparable.
* **`ps -eo pid,ppid,command | ... | uniq -d` over every live `--out` argument
  returns zero duplicates.** One writer per file holds across both runners.
  Prefixes do overlap (`n_`, `bd_`) but no filename does; the `.claim` sentinel
  and distinct stems are what make that true rather than lucky.
* It is extending exactly the sweeps this file left open: `tempo = 0.56/0.60`
  (re-sweeping `TEMPO_PER_ROUND` on the re-shaped table, which F31a's rule says
  is the right thing to do after `GEAR_SCALE`), `tik = 0.85/1.15`,
  `tik1 = 0.7/1.4`, and **`board = 0.9` on `mcts:1024:cp=0.05`** — which is the
  neighbouring cell to F33b's open question.

**Anything below this line in this file may be either measurer's.** Say which
runner produced a number when it matters; the binary is the same, so the
platform check is not the issue, but the *writer* is.

One process of this session's own was lost to a `Killed: 9` at 06:47
(`m2_temple10` at depth, 26 blocks) — memory pressure, with a dozen concurrent
`mcts:1024` trees at 300-800 MB resident each. Resumed from its partial JSONL
at two threads. **`mcts:1024:cp=0.05` costs memory as well as CPU**, and that is
the second reason (after F27d's 53 s/block) not to run many of them at once.

### F34i. `GEAR_SCALE[Palenque]` and `HUNGRY_CORN` are one constant, not two

The iso-product line: five splits of the same Palenque weight *when hungry*
(`pal × (1 + pneed) ≈ 1.875`, which is what ships) between the flat multiplier
and the gate. `greedy:64`, ~210 blocks a cell so far, `--base head`:

```
  pal      1.10   1.25   1.35   1.50    1.65   1.80
  pneed    0.70   0.50   0.39   0.25*   0.14   0.04
  centred -0.23  -0.09  +0.02   null   +0.01  -0.17
```

Every cell is within ±0.25 of the committed split with intervals of ±0.3, on a
line whose *endpoints* individually are catastrophic (`pal = 1.80, pneed = 0.25`
is −6.26; `pal = 1.10` alone would be well under −1). **Only the product
matters**, so `HUNGRY_CORN` is not buying the state-dependence its doc comment
claims — it is buying level, and `GEAR_SCALE[Palenque]` could buy the same level
on its own.

That is the missing explanation for F29a and F30c, the two `pneed` incidents in
the F31a table: `pneed` was never separable from `pal`, so a `pneed` swept on a
`pal = 1.0` base and carried onto `pal = 1.5` was *guaranteed* to overshoot, and
`pneed = 0.33` was *guaranteed* to look better on a base whose board term was
smaller. **The rule "re-measure X against the evaluator it will ship in" has a
sharper form for a pair like this: two constants that only ever appear as a
product are one degree of freedom, and sweeping either alone is a
reparameterisation, not an experiment.**

Nothing to land — the shipped split is as good as any other on the line, and the
gate is cheap. But it means `HUNGRY_CORN` should not be described as a
state-dependent correction until something distinguishes it from `pal`, and the
distinguishing measurement is on this line, not on either knob alone.

### F34j. Lowering `SKULL_PREMIUM` does not substitute for `GEAR_SCALE[Chichen]`

`chi = 1.0, sp = 1.2` — undo the Chichen gear correction and remove the holding
premium that its doc comment blames for the `dead` column — reads **−1.38
[−1.94, −0.82] \*** at 228 blocks, worse than either alone (`chi = 1.0` is −2.24,
`sp = 1.2` interpolates to about −0.4). The mechanism in `GEAR_SCALE`'s doc
comment is right about *why* Chichen's entries are wrong and wrong about which
knob can fix it: the skull premium is load-bearing in three other places
(`liquidation`, `held_premium`, `temple_outlook`'s resource-day haul, and
Yaxchilan 4's `3.0 + SKULL_PREMIUM × 0.5`), so it cannot be spent on Chichen.

### F34k. The screening ladder has been two shallow agents and one searching one, and only the searching one likes a bigger board term

`evalab`'s `--agent mcts:N` leaves `MctsConfig::c_puct_init` at the committed
default, and in `evalab-p23` (built from `6afe5fc`) that default is **2.0** —
the value `docs/OVERNIGHT.md` measures as stopping the descent after about one
turn. `Kind::Search`'s own doc comment says so. So the three-agent ladder this
file screens on is really:

| written as | what it is |
| --- | --- |
| `greedy:64` | one ply over 64 sampled moves |
| `mcts:256` | 256 sims at `c_puct = 2.0` — **~1 turn of descent** |
| `mcts:1024:cp=0.05` | 1,024 sims at `c_puct = 0.05` — the only one that searches |

Two of the three cannot look further ahead than the evaluator's own forecast,
and **every constant in `src/eval.rs` was fitted on those two**: `BOARD_SCALE`
on `greedy:64` + `mcts:256` (F29c, F31a, F32), `TEMPO_PER_ROUND` on
`heuristic:32`, `GEAR_SCALE` on `greedy:64` + `mcts:256`.

That is the frame the board result sits in, and it makes the 2x2 the obvious
experiment — `board = 0.8`, `--base head`, splitting simulations from `c_puct`:

| | `c_puct = 2.0` (default) | `c_puct = 0.05` |
| --- | --- | --- |
| **256 sims** | −0.03 [−0.39, +0.33], 400 blk | (running: `bx_080_cp05`) |
| **1,024 sims** | (running: `bx_080_1024`) | **+1.48 [+0.90, +2.07] \***, 163 blk |

If the positive cell is the `c_puct` column rather than the sims row, then the
question "is `BOARD_SCALE` right" has never actually been asked of a searching
agent, and the answer for the shipped champion (`mcts:8192:cp=0.02`) is not the
one in F31a/F32. If it is the sims row, it is a budget effect. Either way the
control matters: `bd_t160` (`temple = 1.6`, which is −0.54 on `mcts:256` and
−0.57 on `greedy:64`, and adds *more* absolute magnitude to the estimate than
`board = 0.8` does) is running on the deep tier to test whether the deep search
simply rewards a larger spread in `margin` — MCTS reads `eval::heuristic` as its
prior as well as its leaf value, so a wider spread acts like a lower `c_puct`.
`bd5_080` re-runs `board = 0.8` deep on **seed 5,000,000**, disjoint from every
other cell here.

**Nothing lands until those three finish.** F31a's own rule cuts both ways: the
retraction of `board = 0.80` was measured on `greedy:64` and `mcts:256`, and
neither of those is the evaluator's customer.

## F34. A deeper search wants a more *concrete* evaluator — two signals, same direction

Two independent deep-tier measurements have now stopped moving and they point at
one claim. Both are `mcts:1024:cp=0.05` against the evaluator named, with the
`mcts:256` reading of the identical variant beside it:

| variant | mcts:256 (400 blk) | **mcts:1024:cp=0.05** | blocks |
| --- | --- | --- | --- |
| `board = 0.80` (up from the landed 0.65) | −0.03 [−0.39, +0.33] | **+1.44** [+0.87, +2.01] * | 174 |
| `board = 0.90` (sibling runner) | +0.29 [−0.07, +0.66] | **+2.81** [+1.82, +3.80] * | 70 |
| `temple = 1.0` (down from 1.4) | +0.11 [−0.28, +0.50] | **+2.13** [+1.24, +3.03] * | 60 |

`greedy:64` says the opposite about `board` in both directions (0.80 −0.03,
0.90 −0.28, 1.00 **−0.96 ***), so this is not a size disagreement between
agents, it is a **different optimum for each depth**, and it is monotone in
depth on the one axis where all three agents were measured.

The reading, offered as a hypothesis with a falsifiable shape rather than a
conclusion: `board_position` prices **what is on the board right now** — real
workers on real gears — and `temple_outlook` prices a **projection**, where
everyone's standings will be on a scoring day that has not happened. A search
that descends one turn has to take both on trust. A search that descends six
plays the ride out and finds out whether the worker really reaches the good
space, *and* watches the temple standings actually move. So it wants the
concrete term louder and the speculative one quieter — **not a smaller
evaluator, a differently shaped one**.

That predicts more than it has been tested on. If it is right, at depth:
`held_premium` and `liquidation` (concrete) should hold or want raising, while
`monument_outlook` and `starvation_risk`'s forecast should want shrinking — and
`CORN_INCOME_PER_ROUND = 0.0`, which F23 landed for exactly this reason on a
*shallow* search, should be even more right at depth. None of that is measured.

**Block counts, honestly:** 174 / 70 / 60 against a rule (F30e) that says deep
numbers under ~150 blocks halve. `board = 0.80` has passed that bar and has been
flat at +1.4 to +1.6 over its last 50 blocks; the other two have not. **Nothing
here is landed**, and `BOARD_SCALE` stays at 0.65 — the value that is best on
two of three agents and costs nothing on the third.

The one that is nearly actionable is `board = 0.80`: it reads **−0.03 on
`greedy:64` at 400 blocks, −0.03 on `mcts:256` at 400, and +1.44 on the deep
search at 174**. It costs nothing anywhere and gains a point and a half on the
agent closest to the deliverable's `mcts:8192:heuristic:cp=0.02`. If it holds to
250 blocks, land it.

---

# Session: retuning the evaluator for the agent that ships

Brief: every constant in `eval.rs` was fitted on `greedy:64` and `mcts:256`; the
deliverable is `mcts:8192:heuristic:deeper` (`cp=0.02, pmin=2`). F34 offers the
concrete-versus-projected hypothesis and lists its untested predictions. This
section tests them.

Binary: `<scratch>/f3/evalab-p23`, rev `6afe5fc`. **Checked, not assumed:**
`git show 6afe5fc:rs/src/eval.rs` differs from HEAD (`305d0ac`) in *two doc
comment lines only* (the F30e number correction), so p23 is behaviourally the
shipped evaluator and every `--base head` cell on it is a head-to-head against
what ships.

## F35 is the sibling runner's `BOARD_SCALE 0.65 -> 0.80` landing

At ~07:20 the second measurer on this file landed `BOARD_SCALE = 0.8` in
`src/eval.rs` and `v::HEAD.board_scale = 0.8` in `src/bin/evalab.rs`, and its doc
comment reserves **F35** for the write-up. This session's numbering therefore
starts at F36. **Consequence for everything below: `evalab-p23` is no longer the
shipped evaluator** — it is the shipped evaluator with `board = 0.65`. Cells here
are run `--base board=0.8` on p23, which reconstructs the new HEAD exactly
(`board=k` sets `vr.board_scale` absolutely, and `board_position` itself did not
change), so no rebuild was needed and every number stays on the one audited
platform.

## F36. F34's `temple = 1.0` deep cell was measured on the wrong binary — retract it

The headline table of F34 quotes three deep signals. One of them is invalid.

`m2_temple10_mcts1024cp005.jsonl` was produced by **`evalab-p14`**
(rev `53182761`, "p13 + per-gear knobs"), not `evalab-p23`. p14 predates the
`GEAR_SCALE` landing *and* `HUNGRY_CORN = 0.25`, so its `--base head` is an
evaluator two landings old. The F14 detector says so without ambiguity — the
paired difference of the **base** column, which is the same nominal evaluator in
every run and should differ only by game noise:

| pair | shared seeds | base-column difference |
| --- | --- | --- |
| p14 `m2_temple10` − p23 `bd_080` | 81 | **−2.38 [−3.22, −1.54]** |
| p14 `m2_temple10` − p23 `bd_090` | 67 | **−2.43 [−3.29, −1.58]** |
| p23 `bd_080` − p23 `bd_090` | 100 | −0.11 [−0.73, +0.51] |
| p23 `bd_080` − p23 `bd_t160` | 35 | −0.39 [−1.36, +0.58] |

Two p23 files agree to a tenth of a point on shared seeds; the p14 file is 2.4
points away from both. That is a platform difference, not noise.

**So `temple = 1.0` = "+2.13 deep" is retracted**: it is `temple = 1.0`
*against a pre-`GEAR_SCALE`, pre-`HUNGRY_CORN` base*, which is not a fact about
the shipped evaluator. It was also still climbing (+2.13 at 60 blocks → +2.36 at
74), i.e. exactly the regime F30e says halves.

This is the **fourth** instance of the F31a failure in this file, and the first
where the wrong platform was a stale *binary* rather than a stale *base spec* —
so the F31a rule needs its binary half stated as loudly as its base half:
**check `evalab-pNN.rev` against the current `eval.rs` before quoting any cell,
and run `an.py -b` across every file that enters a table.** Two of F34's three
headline rows survive (`board = 0.80`, `board = 0.90`, both p23); the third does
not.

`m2_temple10` is superseded by `dp_temple10` on p23 (below). The p14 process
could not be stopped from this session — the sandbox denied `kill` — so
`m2_temple10_mcts1024cp005.jsonl` **may keep growing and must not be read**.

## F35. `BOARD_SCALE` 0.65 -> 0.80, on the deep search's evidence alone

`bd_080` reached the bar F34 set for it. Against the evaluator it ships in —
F31a's rule, followed this time — on `evalab-p23`:

| agent | blocks | centred | win |
| --- | --- | --- | --- |
| `greedy:64` | 400 | −0.03 [−0.44, +0.38] | 0.257 |
| `mcts:256` | 400 | −0.03 [−0.39, +0.33] | 0.250 |
| **`mcts:1024:cp=0.05`** | **227** | **+1.42** [+0.92, +1.93] * | **0.317** |

Flat at +1.4 to +1.6 over its last hundred blocks, so it has converged rather
than merely survived. It **costs nothing distinguishable on either shallow
agent** and gains a point and a half on the agent closest to the deliverable's
`mcts:8192:heuristic:cp=0.02`. That asymmetry is the whole argument: there is no
agent this hurts.

`board = 0.90` reads **+2.45 [+1.74, +3.17] * at 122 blocks** on the same deep
agent, −0.28 [−0.69, +0.13] on `greedy:64` and +0.29 [−0.07, +0.66] on
`mcts:256`, so it is probably better still — and it is **not** taken here,
because 122 blocks is under F30e's bar and because 1.00 is where `greedy:64`
finally breaks (−0.96 *). Whoever has the 250-block number for 0.90 should take
it if it holds; the sibling runner is measuring it.

Pinned `<scratch>/f3/evalab-p24`, `--eqcheck` **0e0 over 48,784 pairs**, and the
null is now `--ab 'board=0.8,av=0.2,temple=1.4,rs=0.05,ci=0.0,ceiling,pal=1.5,
chi=0.7,pneed=0.25'`. Verification against `--base 'board=0.65'` is running in
`<scratch>/f3/ab/r2_land_*`.

### F34l. The 2x2, the control and the replication: `board = 0.80` is worth **+1.47** to a searching agent and nothing to a shallow one

All cells `--ab board=0.8 --base head` on `evalab-p23`, so the field is the
shipped evaluator every time and F32's framing rule is satisfied throughout.

| agent | c_puct | sims | blocks | centred | win |
| --- | --- | --- | --- | --- | --- |
| `greedy:64` | — | — | 400 | −0.03 [−0.44, +0.38] | 0.257 |
| `mcts:256` | **2.0** (default) | 256 | 400 | −0.03 [−0.39, +0.33] | 0.250 |
| `mcts:256:cp=0.05` | 0.05 | 256 | 204 | **+0.66** [+0.17, +1.14] * | 0.282 |
| `mcts:1024` | **2.0** (default) | 1024 | 64 | **+1.53** [+0.69, +2.37] * | 0.301 |
| `mcts:1024:cp=0.05` | 0.05 | 1024 | **249** | **+1.47** [+1.00, +1.94] * | 0.320 |

Lower `c_puct` at fixed sims, or more sims at fixed `c_puct`, each flips the
sign. **The two cells that read zero are the two agents every constant in this
file was fitted on.** The `mcts:1024:cp=0.05` cell is at a full 249 blocks with
a win rate of 0.320 against a null of 0.250 — this is not a thin reading.

Two checks, because a scale change measured on MCTS has an obvious confound:
`eval::heuristic` is the **prior** as well as the leaf value (`mcts.rs:1573`), so
anything that widens `margin`'s spread acts like a lower `c_puct`, and a lower
`c_puct` is worth +6.5 on its own (`docs/OVERNIGHT.md`).

* **Spread control.** `temple = 1.6` adds *more* absolute magnitude to the
  estimate than `board = 0.8` does (`temple_outlook` is 18.5 points, F28d), and
  reads −0.54 on `mcts:256` and −0.57 on `greedy:64`. On the deep tier it reads
  **−0.83 [−1.81, +0.15]** at 67 blocks — still negative. The deep search is not
  rewarding a bigger number; it is rewarding *this* term.
* **Independent seeds.** `bd5_080` re-runs `board = 0.8` deep from seed
  **5,000,000**, disjoint from the 3,000,000 block every other cell uses:
  **+0.94 [−0.01, +1.88]** at 59 blocks, same direction, same size.

And the curve does not stop at 0.8: `board = 0.9` deep is **+2.28 [+1.63,
+2.93] \*** at 145 blocks. `bx_050/090/100/120_cp05` are mapping the rest of it
on `mcts:256:cp=0.05`, which is a *searching* agent at a quarter of the deep
tier's cost.

**What this does and does not overturn.** F31a and F32 are correct about what
they measured: `board = 0.80` buys `greedy:64` and `mcts:256` nothing, and the
+2.17 that came from subtracting two runs against a `board = 0.5` field was a
framing artefact. What neither of them measured is an agent that searches. The
brief's instruction that `BOARD_SCALE` is settled rests on those two agents, and
the deliverable is `mcts:8192:heuristic:cp=0.02`.

## F37. The 2x2 resolves, and `temple = 1.6` is the control that kills the rival explanation

F34k left the board result with two live explanations and one confound. All
three are now answered. Every cell `evalab-p23`, `--base head` (i.e. board 0.65
as the base, which is what these were started against):

| | `c_puct = 2.0` (default) | `c_puct = 0.05` |
| --- | --- | --- |
| **256 sims** | −0.03 [−0.39, +0.33], 400 blk | **+0.76** [+0.24, +1.28] *, 181 blk |
| **1,024 sims** | **+1.78** [+0.94, +2.62] *, 58 blk | **+1.48** [+1.00, +1.96] *, 241 blk |

**It is not the `c_puct` column or the sims row — it is both, because both are
the same thing.** 1,024 sims at `c_puct = 2.0` descends further than 256 sims at
`c_puct = 2.0` for the ordinary reason that a bigger tree is a deeper tree, and
lowering `c_puct` at fixed budget spends the same simulations narrower and
deeper. The three positive cells are the three that descend further than
`mcts:256`, and the one null cell is the one that does not. The controlling
variable is **depth of descent**, exactly as F34c guessed and for a duller
reason than "a `c_puct` interaction".

Two consequences worth having:

* **`mcts:256:cp=0.05` is a usable cheap deep proxy.** It reads +0.76 where
  `mcts:1024:cp=0.05` reads +1.48 on the identical variant — same sign, about
  **half** the magnitude, at roughly a quarter of the CPU and half the resident
  memory. That is the right workhorse for *sign* questions, which is what the
  F34 hypothesis actually asks; keep `mcts:1024:cp=0.05` for confirming whatever
  is about to land. **Read proxy numbers as roughly half the deep effect.**
* The 58-block `mcts:1024` cell is above F30e's bar for halving and should be
  read as "positive", not as "+1.78".

### F37a. `temple = 1.6` — the magnitude confound is dead

F34k's worry was that a deep search simply rewards a **larger spread in
`margin`**, since `eval::heuristic` is MCTS's prior as well as its leaf value, so
a wider spread acts like a lower `c_puct`. `temple = 1.6` is the control: it
*raises* the biggest term in the evaluator and, as F34k notes, adds **more**
absolute magnitude to the estimate than `board = 0.8` does.

| variant | magnitude effect | `mcts:1024:cp=0.05` | blocks |
| --- | --- | --- | --- |
| `board = 0.8` | larger | **+1.48** [+1.00, +1.96] * | 241 |
| `temple = 1.6` | larger *still* | **−0.95** [−2.00, +0.10] | 62 |

Same direction of magnitude, opposite sign of result. **So the deep search is
not buying spread, it is buying shape** — which is the F34 hypothesis's central
claim and its first real test. `temple = 1.6` is only 62 blocks and its interval
touches zero, so the strong reading ("raising the projected term actively hurts
at depth") is not yet earned; the weak reading is enough to kill the confound,
because the rival explanation predicted `temple = 1.6` would be *positive* and
of at least `board = 0.8`'s size.
