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

## T10 — what the Board panel was actually drawing, read off dumps at four sizes

Resumed after a kill. `App::focus`, `Marks`, `ring`, `space_cell`, `space_info`
and `board` are all on disk and working; the rings render, the space labels come
out of `spaces::choices_at`, and the `+`/`-` markers land on the right cells.
What the dumps say is wrong is **where the panel's rows go**, not what it knows.

`uidump 7 9 132 44`, `7 9 100 30`, `7 9 80 24`, and `7 14/20/24 132 44`:

1. **At 80x24 the space list is not on screen at all.** The left column gets ten
   rows; `bh` takes seven, the rings need five of its five interior rows, and
   `spare` is then zero. So at the default terminal size the panel that exists
   to say what a space does says nothing about any space. Every larger size
   shows it, which is why this survived.
2. **The three rows under the Board at 80x24 are a Players box with a header in
   it and not one player**, and the right column spends four more on a Temples
   box and a Research box with nothing but borders. Seven rows drawing zero
   facts, next to a panel that was cut for want of one.
3. **`rides` — the "moving in circles" line — is drawn only at `spare >= 2`,**
   which at 100x30 and 80x24 never happens. It reached the screen at 132x44 and
   nowhere else, and it reads `rides  nobody on this gear(1 space a day, then
   off)` with no space when the gear is empty.
4. **Temples truncates from the bottom, and the bottom is step 0.** At 132x44
   the panel is one row short of the tallest track, so the row it drops is the
   one every player starts on: at seed 7 round 9 the dump shows steps 8..1 and
   two players standing on step 0 are simply not on the board. A `Paragraph`
   cuts its tail, and this panel's tail is its most-occupied row.
5. `rides` prints days-left (`R@1 7d`) and nothing about *where*. The brief's own
   framing — "this worker reaches Tikal 4 in two days" — is not on screen; the
   arithmetic is `pos + k` on day `day + k`, off at `Gear::size()`, and none of
   it needs a search.

Not bugs, checked and left alone: the ring winding is genuinely clockwise from
the bottom-centre entry (`SMALL` slots go bottom-centre → bottom-left → up the
left → across the top → down the right, which is 6 → 7 → 9 → 12 → 3 o'clock),
so the `↻` hub is honest; the `+Y+` / `-B-` markers are correct on both a place
and a retrieve, verified at rounds 9 and 14; and the `spare`/`room` arithmetic
that looked off-by-one cannot actually clip the table, because `spare` is
recomputed from the table that was already sized.

**Decisions taken from this.** (a) Rows a panel cannot use go to a panel that
can — a box with only a header is worse than no box, and Players, Temples and
Research each get a floor below which they are skipped. (b) The Board's floor
rises to eight rows so the move line always has somewhere to go. (c) The extras
under the rings are ranked *move, rides, key*, and a whole table outranks all
three; the flowed list outranks none, because a sentence ending in `…` says
less than the move line does. (d) Temples anchors its window on the highest
occupied step instead of on the top of the ladder.

## T11 — the lap is arithmetic, and it fits where an animation would not

The user's ask was "workers moving in circles". The gear advances one space a
day, so a worker on `pos` is on `pos + k` on day `day + k` and rides off when
that would reach `Gear::size()`. **`ui::lap` is that sum and nothing else** — no
search, no projection, no evaluator — which is why it can be printed as a fact
next to a still picture instead of being animated into one.

Three renderings of it, at three widths, because the panel is a different size
in each place it appears:

| where | form | e.g. |
| --- | --- | --- |
| Board, `rides` | every worker on the focused gear | `R1→7 d18   Y2→7 d17` |
| Board, `move`  | the highlighted move's workers  | `+Y Tik 0→7 d19` / `-B Chi 6→hand (5d early)` |
| Why pane       | words, sized to the pane        | `giving up 5 more days of the lap, which ends at 10` |

**Cut at day 27, deliberately.** A worker put on Tikal on day 25 does not reach
the top of it, and `→7` there would be the panel stating a plan the game has no
time for. `lap` clamps the last space to `pos + (LAST_DAY - day)` and says
`end` instead of a day. Pinned in `a_lap_stops_at_the_end_of_the_calendar`,
including a sweep asserting the ride never runs off the end of a gear and never
goes backwards, for every gear, space and day.

**What a retrieval costs is on the wheel and nowhere else.** The state diff can
say the worker is in hand and what its space paid; only the gear knows it had
five more days of ride to give up. `move_line` prints `(5d early)` and the Why
pane spells it out.

### The row budget, which is what this actually was

Rings, three lines under them and a whole space table do not fit together below
about 44 rows, and the ranking between them had never been written down. It is
now, in `board`:

1. **A whole table outranks everything.** A partial one looks complete — there
   is nothing on a truncated column to say four more spaces exist — so it gets
   first refusal on the rows it needs and the extras take the remainder.
2. **Where the table cannot fit at all, the extras win** and the flowed
   sentence takes what is left, because a list ending in `…` says less than the
   move line does. One row is always kept back for the list *unless one row is
   all there is*, in which case the move line is the better use of it.
3. Extras rank `move`, `rides`, `key`. The key was first before and is a legend
   for marks that the move line now spells out in words.

### Rows a panel cannot use go to a panel that can

Three boxes were drawing borders with nothing inside them at 80x24 — Players
with a heading and **not one player**, Temples and Research with nothing at all
— seven rows stating nothing, next to a Board that had been cut for want of one
and a shortlist that had three rows. Each panel now has a floor and is skipped
below it: Players 6 (and `players` drops its heading before it drops a player),
Research 7 (four tracks or none — at six it drew three sciences with nothing
saying a fourth existed), Temples 6. The Board's floor rose 7→8 and its ceiling
13→14, the height at which everything is on screen at once.

Measured at 80x24, which is the size all of this was invisible at:

| | before | after |
| --- | --- | --- |
| Board | 5 ring rows, no space named | rings, `move`, `rides`, and the list |
| Players | a heading, no players | not drawn |
| Temples / Research | two empty boxes | not drawn |
| shortlist | **3 rows** | **7 rows** |

The right column gained four rows by this and gave up none.

### T10.4, fixed: Temples was hiding the players

`Paragraph` truncates its tail and this panel's tail is step 0, where everybody
starts. At 132x44 it was one row short of the tallest track, so the row it
dropped was the most-occupied one on the board: seed 7 round 9 drew steps 8..1
while `RGY / GBY / RGBY` stood on a step that was not on screen. The window is
anchored on the highest step anyone has reached instead, with `↑` and `↓` in the
gutter where the ladder carries on past the panel. Pinned by
`the_temple_panel_never_truncates_away_an_occupied_step`, which reads *only the
panel's own columns* out of the buffer — a colour letter elsewhere on screen
must not be able to stand in for one missing here — and requires each player's
letter three times, once per track.

### Also landed

* **The entry space is marked** `→0` on the rings at three columns a cell. Eight
  numbers in a circle do not say where a lap starts; with it the `↻` in the hub
  has something to turn from. The number stays, because the list below is
  indexed by it.
* **`clip`**, a span-wise truncation that ends in `…`. Lines built out of spans
  carry the colour that tells one player's worker from another's and so cannot
  be round-tripped through `fit`; left to ratatui the overflow was clipped
  silently at the right edge, which is how a second placed worker would go
  missing from the move line with nothing on screen saying so.
* **`lap_words` is sized, not wrapped.** The Why pane wraps, and the first cut of
  this cost it a *second row per marked space*: two retrievals turned four lines
  into six and pushed the state diff — the pane's own subject — off the bottom.
  It picks the longest of a long form, a short form and nothing that fits.
* `rides  nobody on this gear(1 space a day…)` had no space in it, and the line
  only ever reached the screen at 132x44 in the first place.

### Tests

Four inserted into `tests/rules.rs` (78 → 82; suite 199 passed, 0 failed):
`a_lap_stops_at_the_end_of_the_calendar`,
`the_board_marks_placed_and_retrieved_workers_without_colour` (reads the buffer
as text at 60x20 / 80x24 / 132x44 across three positions and every candidate
that touches the board — the glyph is the only one of the three markings that
survives a dump, and `Modifier::SLOW_BLINK` is widely ignored),
`the_board_always_says_what_at_least_one_space_does` (five sizes x three
positions x three focus settings; this is the 80x24 regression),
`the_temple_panel_never_truncates_away_an_occupied_step`.

A fifth assertion was written and thrown away: "no doubled marker glyph
anywhere on screen" failed on the **calendar strip** in the header,
`-------R--#--P------R-----P`. The distinction is pinned through the Why pane's
words (`+Y onto` / `-B off`) instead, at the one size that draws it whole.

### Not fitted, and why

* **Five rings and eight space labels do not both fit at 80x24.** 41 columns of
  Tikal's labels is six flowed rows; the panel has three after the rings and the
  two lines that answer the hovered question. It shows the rings, the move, the
  rides and a list that ends in `…`. There is no arrangement of that panel that
  holds all of it, and the honest cut is to answer what the cursor is on.
* **Temples is not drawn at 100x30** — Research is fixed at seven rows and
  all-or-nothing, Temples scales, so Research takes its seven first and Temples
  takes a remainder that is only three. Reversing the order costs the shortlist
  three rows. Both orderings are defensible; this one favours the deliverable.
* **`Gear::ALL` order is fixed and Chichen's ring is 4x4** with the bottom-right
  slot empty, so the five rings need 56 columns at three per cell. Below 32 the
  panel drops to flat rows and there is no `↻` and no ring at all; the `rides`
  line is the only thing carrying the lap there.
* **No trail on the ring.** A worker entering at 0 passes every other space, so
  a trail marker from the entry space is the whole wheel and says nothing. The
  `rides` line carries the same fact in a form that is legible in a dump.
