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
| `_greedy64` / `_mcts256` / `_mcts1024` | the agent both arms played |

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
