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
