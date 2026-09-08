# FINDINGS — the TUI

Phase E work: making the champion's play *legible*, not faster. Owner touches
`src/ui.rs`, `src/bin/tui.rs`, `src/bin/uidump.rs` and the UI tests in
`tests/rules.rs` only.

Baseline pinned at `07ec37a` as `<scratch>/bin/uidump.base`, per OVERNIGHT's
rule that a shared `target/release/` binary moves underneath you.

## T1 — three different units arrive in one score column

`Ranking` has no field saying what its scores mean, and four producers put
three different scales into `Ranking::moves`:

| producer | scale |
| --- | --- |
| `eval::rank_all*` (the `full` / `sampled` views) | `eval::margin` points |
| `search::Search::search` (minimax, via `MinimaxAgent`) | backed-up `eval::margin` points |
| `GreedyAgent::ranked_moves` (`heuristic:full`, `greedy:*`) | the **evaluator's** value, squashed to (-1, 1) — *not* points |
| `SearchAgent::ranked_moves` (mcts) | **visit share in percent** |

The panel printed `{score:>7.1}` with no unit for all four. That is why the
champion's shortlist read `55.7` then `5.5` and looked like a 10x scoring gap:
73.2 and 4.3 are percentages of the finished simulations, and they were
*correct*. Nothing was wrong with the number; the column never said what it was.

Each producer does state its scale — in `Ranking::note`, in prose. The panel put
that note in the status line, where it was truncated at the right edge, so the
one place the units were written down was the one place not on screen.

**Wanted from `eval.rs`: a `unit` field on `Ranking`.** Working around it by
reading the note back (see `ScoreUnit::infer`) and keeping the note on screen
next to the number, so a wrong inference is visible rather than believed.

## T2 — eight of ten rows were one retrieval, reordered

`retrieve w0[+3 corn] w1[-3 corn, G+1]` also appeared with `w2[skip]` inserted
before, between and after — and again with `w3[skip]` as well. `Move::same_effect`
is right to keep them: picking a worker up with `skip` still returns it to hand,
so the *state* differs. But the three placements of one `skip` in the sequence
do not differ from each other.

`apply_move` walks a retrieval as `choice.apply(); retrieve_worker()`, and a
`skip` choice has no effects, and choices are fully-resolved data that never
read the board (`effect.rs` module note). So moving a `skip` entry past an
effectful one cannot change the state reached. That is the fold, and it is a
display fold only — generation and `same_effect` are untouched.
