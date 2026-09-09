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

## T3 — what was actually on disk at `48c39bd`, and what was not

Re-read before resuming, because the hand-off note and the tree disagreed. T1
and T2 are **landed in `ui.rs`**: `ScoreUnit::infer` + the `units` line, and
`same_outcome` / `row_text` / `fold_rows` behind `App::rows()`. What is *not*
landed, verified by `uidump 7 9 132 44 --agent mcts:8192:heuristic:cp=0.02,pmin=2`
and by reading `bin/tui.rs`:

* **No worker thread.** `agent_step` searches on the draw loop and nothing ever
  writes `App::thinking`, so `ui::thinking` is dead code. `refresh` is the
  worse offender of the two: `MoveSource::Agent` calls `ranked_moves`, which is
  a *second* full-budget search, and `advance` calls it after every turn. One
  `n` at the champion's budget is two searches, both blocking.
* **`uidump --thinking` parses its argument and drops it** — `let thinking =`
  is built and then `thinking: None` is passed to `App`. The compiler says so
  (`unused variable: thinking`). So the one screen that cannot be reviewed
  after the fact had no way to be reviewed at all.
* **`j`/`k` index the wrong list.** They wrap on `App::candidates().len()`
  (raw `Ranking::moves`) while `selected_move` indexes `App::rows()`. After the
  fold those differ — 10 raw rows folded to 3 — so the selection walks off the
  end and the Why pane goes blank for seven of ten presses.
* Two rendering faults visible in that dump: the row text truncates on the
  **right**, which is exactly where the fold puts the tail that makes rows
  different (`... w1[-3 corn, G+1] +..`), and the units line loses its last
  clause off the right edge (`... NOT the whole move space` never appears).

## T4 — the search's own numbers, and which of them are reachable

`TurnLine` (mcts.rs) carries `visits` **and** `value` — the backed-up value of
that whole turn. `SearchAgent::ranked_moves` reads both and puts only `visits`
into `Ranking::moves`, because `Ranking` has one score slot. So the backed-up
value per candidate exists, is computed, and is discarded one frame before the
display. **That is the single most useful thing to expose** (see the report).

Reachable without touching an off-limits file, via `TurnOutcome::nodes`:

| field | what it says |
| --- | --- |
| `Node::root_value[turn]` | backed-up value at that sub-decision, `z_rel` scale |
| `Node::visits` | the **whole** distribution, sorted, not just the argmax |
| `Node::total_visits` | the true denominator (the vec is capped at `MAX_VISITS`) |
| `Node::n_edges` | legal edges, so 97% reads differently at 2 options than 40 |

`z_rel` is `tanh((score - table mean) / 25)`, so `atanh(v) * 25` is points above
the field — reported with a `~` because a mean of `tanh` is not `tanh` of a
mean, and the sign and the ordering are what a reader is using it for.

## T5 — the search runs on a worker thread, and the screen says what it knows

`bin/tui.rs` now spawns one detached thread per search and keeps drawing. Two
things made this more than a `thread::spawn`:

* **A turn is two searches, not one.** `play_turn` picks the move; `ranked_moves`
  — a whole second full-budget `ranked_turns` — builds the shortlist for the
  *next* player. The second was the one freezing the terminal after every move,
  including moves the human played by hand, and it was never on the key that
  looked expensive. They are now stages 1 and 2 of one job, and the pane says
  which is running.
* **The opening list is a search too.** `refresh` used to run before
  `EnterAlternateScreen`, so the first thing a `--agent mcts:…` run did was sit
  on a blank terminal. It is now the first background job, and the board is up
  with a clock on it from frame one.

**No progress fraction, deliberately.** Nothing outside `mcts.rs` can see a
simulation counter, so a filling bar would be inventing the only number on the
screen. What is drawn is measured: which search of two, the elapsed clock, what
stage 1 returned (the move, named), and elapsed **against what the last search
of the same kind cost** — labelled "a scale, not a deadline", because it runs
past the end of the bar and that overrun is information rather than a stuck
widget. With no prior, a sweep and a line saying it means nothing more.

### Measured, through a pty

`mcts:65536:heuristic:cp=0.02,pmin=2`, 132x44, bytes of terminal output in
250 ms buckets after pressing `n`:

    [1406, 487, 523, 549, 544, 493, 549, 508, 547, 531, 520, 455,
      641, 547, 490, 490, 548, 544, 490, 570, 502, 544, 534, 538]

Zero silent buckets across six seconds, all eight spinner frames observed, no
panic. Under autoplay at the champion's own budget, 24 complete stage-1/stage-2
cycles in 20 s, no refusals and no panic. (Counting *semantic* state off a pty
does not work — ratatui writes only changed cells, so `day 11/27` never appears
twice in the stream. Bytes-per-bucket is the measurement that survives that.)

## T6 — three width bugs, all of them cutting the load-bearing end

1. `moves`'s gutter constant was `2 + 7 + 8` where the prefix it describes is
   `2 + 6 + 2 + 8`. One column short, so the `xN` marker that the fold budgets
   room for was itself clipped — the one mark saying a row stands for several.
2. Row text truncated on the **right**, which after the fold is exactly where
   the difference lives: `… w1[-3 corn, G+1] +..` for two different outcomes.
   `fit` elides the middle instead; `fit_tail` keeps three quarters at the end,
   for the producer's note, where right-truncation had turned
   `— NOT the whole move space` into ` move space`.
3. The right column's three `Length` constraints over-subscribed below ~44 rows
   and the solver starved the panel with the weakest constraint — the shortlist,
   the one panel the viewer exists for. At 80x30 it was **one row**. Allocated
   explicitly now, shortlist first; it keeps all three outcomes down to 60x20.

Also: below 100 columns the preview refused to split and showed only the search
pane, so under an agent the move diff was **unreachable** and `j`/`k` moved a
selection nothing described. They stack vertically now.

## T7 — what a reader can now see about the reasoning

* **Two columns and the gap between them.** Visit share is the search's opinion;
  `1-ply` is `eval::margin` of the successor, computed by the viewer, so its
  unit is never in doubt. Where they disagree the gap *is* the lookahead, and
  `lookahead_line` says so in points, under the top row only.
* **Whose turn it is.** The search pane sits beside a shortlist belonging to the
  *next* player in the position *after* the move. `App::last_played` labels it
  (`Y played retrieve w18[…]`) — without that the two panes read as one account
  of one turn, which is what they did.
* **What it declined.** Each choice point shows the chosen step, its share, the
  edge count (`96% /24` — 97% out of two edges is a coin, out of forty it is a
  conclusion) and the best step it turned down. That last column is the only
  thing on the row not already somewhere else on screen, so the bar is cut
  before it is.
* **What it thinks the game is worth.** `Node::root_value[turn]` through
  `Decision::points`, printed `~+12 pts vs field`. The `~` is load-bearing: the
  backed-up number is a mean of `tanh`, so `atanh` pulls the magnitude towards
  zero. Sign and ordering are exact, being monotone.

## T8 — `plan.rs` is not surfaced, and that is the finding

The champion is `mcts:8192:heuristic:cp=0.02,pmin=2`. Its evaluator is
`eval::heuristic`; `plan::PlanEvaluator` is not in the spec, `Line` is not
consulted, and `mcts.rs:1573` orders edges with `eval::heuristic` too. So the
agent the user is watching **has no plan** — no line it is on, no temple it is
targeting. Rendering `line_progress` beside it would be the viewer's own read of
the board wearing the agent's face, and OVERNIGHT's rule against a number that
looks like a measurement applies to a panel just as much as to a log.

What would have to change first: a spec that actually carries `PlanEvaluator`,
refitted after `94d85f3` (its board weight is shrunk ~4x, past the measured
plateau), beating the champion. Then the hook is one field on `App` and a line
in the search pane, and the plan on screen would be the one being played.

## T9 — state at the end of this run

`cargo build --release` clean (no warnings from `ui.rs`, `bin/tui.rs`,
`bin/uidump.rs`); `cargo test --release` **185 passed, 0 failed** — 181 plus
four inserted into `tests/rules.rs`:
`ui_renders_while_thinking_and_after_a_search` (the two screens the size sweep
could not reach, at the same five sizes), `selection_stays_on_a_real_row`
(pins the `j`/`k` fix), `the_shortlist_folds_restated_retrievals` (the fold
sums shares and names the returned workers in a fixed order) and
`elision_keeps_both_ends`. Nothing committed; the tree is left dirty for the
report boundary.

### The one thing wanted from a file this workstream does not own

**`eval::Ranking` has one score slot, and MCTS produces two numbers per turn.**
`Mcts::ranked_turns` returns `TurnLine { steps, visits, value }` — `value` is
the backed-up value of that whole turn, `e.w[seat] / e.n`. `SearchAgent::
ranked_moves` reads it, sums only `visits` into `Ranking::moves`, and drops
`value` one frame before the panel that wants it.

The viewer's substitute is a *one-ply* `eval::margin` of each successor, which
is a genuinely different quantity — it is the thing the search disagrees with,
not a stand-in for what the search believes. So the shortlist can say "the
search likes this and one ply does not" but cannot say "the search likes this
**by** this much", and the ordering below the top row is visits only.

Smallest change that fixes it: a second optional score on `Ranking` (say
`values: Vec<f32>`, parallel to `moves`, empty where a producer has none), plus
the `unit` field T1 already asked for. `ui.rs` would render it as a third
column beside `visits` and `1-ply` and needs nothing else.
