# tzolkin (Rust)

Base game only. Port of the Go engine in `../src`, which stays as the reference
spec until this reaches parity.

```bash
cargo test                                    # 44 rules, regression and UI tests
cargo run --release --bin tui                 # watch a game  (>=120 cols)
cargo run --release --bin tui -- 42           # ...from a given seed
cargo run --release --bin fuzz -- --games 10000
cargo run --release --bin bench               # playout throughput
cargo run --release --bin stats               # branching + timing distribution
cargo run --release --bin uidump -- 7 12 132 44 [--ranked]   # render the TUI to stdout
```

## Watching a game

`bin/tui` draws the board, all four players, both card rows, the gears, the
temples and the research tracks, plus a move list and a **preview pane** showing
exactly what the highlighted move would change. The preview is a state diff:
apply the candidate to a copy and compare — cheap only because `GameState` is
`Copy`.

    j/k    move the selection          n    let the current player act
    enter  play the highlighted move   r    redraw the candidate list
    t      sampled <-> ranked          a    autoplay        q  quit

Two move sources: ten distinct draws from the rollout policy, or the ten best by
`eval::heuristic`. The heuristic is a placeholder with the same signature the
value head will have. It barely discriminates between *placements* — a worker's
value only appears when you retrieve it, so a one-ply evaluation of "put a worker
on gear X" is close to blind. That is a fair illustration of why this needs a
learned value function rather than a hand-written one.

There is no human input yet; it is a viewer. Adding play is a keystroke handler,
since the move list and `apply_move` are already there.

## Design

Board spaces are **generators**: `fn(&GameState, PlayerId) -> Vec<Choice>`,
answering "what can this player do here, right now?" against live state. That is
the Go design and it survives intact.

What they generate is **data**: a closed 13-variant `Effect` enum, applied by one
`match`. Values are resolved at generation time, so `Effect::Corn(7)` means seven
corn — not "recompute the agriculture bonus whenever this runs". A `Choice`'s
label is *derived* from its effects, so the log cannot describe something other
than what executed.

That buys:

- `GameState` is `Copy`, 312 bytes, zero heap. Snapshotting for search is
  `let saved = *state;`. This replaces `Freeze`/`Save`/`Load` and its
  `10000 * ply` / `20000 * ply` keys, which collide.
- `Move` is `Clone + Eq + Hash`, so moves can be compared in tests, deduped, and
  used as transposition-table keys.
- Move generation cannot alias. Sibling moves own their contents.

`Effect` is 3 bytes; `Choice` is 32 (a `SmallVec<[Effect; 8]>`, no allocation for
anything the base game produces).

## Status

Phases 0–2 of the port, plus move generation.

- [x] Effect vocabulary and `apply`
- [x] `Copy` state, gears, temples, research, decks
- [x] Data tables: 32 buildings, 13 monuments, 21 tiles, 3 temple tracks
- [x] All five gears including mirrors
- [x] Round flow, food days, ages, end-game scoring
- [x] Legal move generation
- [x] Every legal retrieval *ordering*
- [x] Pay corn to retrieve from a lower space
- [x] Invariant fuzz over 10,000 seeded games
- [x] Sampling rollout policy (`sample_legal_move`)
- [ ] Lazy move generation for tree nodes
- [ ] Search
- [ ] TUI

**10,000 seeded games to completion, 0 failures, 269s** (37/s), validating the
whole state after every turn and independently re-checking every generated move.

Branching: mean 271 on the opening turn, but a peak of **250,680** at one
late-game turn with six workers on the board. Those are distinct outcomes, not
redundant orderings — the memo already collapses those. At 264 bytes per `Move`
that is ~66 MB for a single node, so search will need lazy generation or move
ordering with a cutoff rather than materialising every child.

## Rules audit

`RULES-AUDIT.md` records an audit against the official CGE rulebook: 17 rule
violations, 8 missing rules and 5 edge cases, **all now fixed**, each with a
named regression test. Highlights: the calendar was a round short and every food
day fired early; the 13-crystal-skull supply limit did not exist at all;
"do nothing" was not offered, so players were forced into harmful actions; the
market could only buy, never sell; and the extra calendar day was an RNG coin
flip rather than a decision.

A second audit verified the fixes and found twelve more issues, including three
introduced by the first round of fixes. All are now fixed; see `RULES-AUDIT.md`.

Two of them reversed earlier changes of mine that were wrong: temple ties pay
half each (not `prize / n_tied`), and a starting player who claims the space
again passes the marker left. Both were right in the Go original.

## Go bugs fixed

Each has a named regression test in `tests/rules.rs`.

| # | Bug | Where it was |
|---|-----|--------------|
| 1 | Move generation aliased its backing array; siblings overwrote each other from the 4th worker on | `model/move.go:70,81` |
| 2 | Maxed agriculture paid 1 corn instead of 3 (unreachable branch) | `model/research.go:117` |
| 3 | Worker count tested `Wheel_id > 0`, missing all of Palenque | `model/player.go:67` |
| 4 | Food day missed the worker on the first player space | `model/game.go:355` |
| 5 | Food day spent corn before applying free workers | `model/game.go:355` |
| 6 | Food days fired one rotation early (`CheckDay` ran before `Day++`) | `model/game.go` |
| 7 | Foresight options discarded exactly when the space was full | `impl/wheels/chichen.go:87` |
| 8 | Devout was mandatory, removing the plain skull placement | `impl/wheels/chichen.go` |
| 9 | Tikal top: duplicate ordered temple pairs, no same-temple option | `impl/wheels/tikal.go:57` |
| 10 | Jungle drove Palenque corn tiles negative, inverting `corn_showing` | `impl/wheels/palenque.go` |
| 11 | Age-1 and age-2 building ids collided (both 1..14) | `impl/buildings/*.go` |
| 12 | Age-2 building 16 was commented out: face up but unbuildable, jamming a slot | `age2.go:279` |
| 13 | Age-2 building 3 promised 4 points and awarded none | `age2.go:44` |
| 14 | Built buildings were never removed from the display | `model/option.go:69` |
| 15 | Monument 10 indexed `[0,6,5,4]` by player count — **panicked in every 4-player game** | `monument.go` |
| 16 | Monument 6 indexed a 6-entry table by up to 6 workers | `monument.go` |
| 17 | Monument 13 dereferenced nil `CData` on the Chichen mirror space | `monument.go` |
| 18 | Two research advances could spend the same block twice | `model/option.go` |
| 19 | Corn exchange mixed budget with total corn and dropped its base case | `impl/wheels/uxmal.go` |
| 20 | Workers on a gear's entry space could never be retrieved | `impl/wheels/*.go` |
| 21 | `ComputeMove` returned a zero `Move`, which begs on temple 0 as Red | `model/randimax.go:76` |
| 22 | Screen rows were allocated at 2× width, masking an unbounded `Put` | `disp/screen.go:22` |

22 isn't reproduced here because there is no renderer yet; it is listed so the
next TUI pass bounds `Put` before fixing the allocation.

## Why sampling, not faster generation

Measured over 18,512 turns:

| | per turn |
|---|---|
| `legal_moves()` | 0.254 ms |
| `apply_move()` | 0.0001 ms |

Applying is **3,680x cheaper than generating**. The state representation is done;
there is nothing to win by making it faster. All the cost is in enumerating moves
that a rollout throws away.

Branching is also far less alarming than the peak suggests — p50 41, p90 239,
p99 1,261. It is a long tail, not a uniformly huge space.

So `sample_legal_move` walks the same recursion and calls the same space
generators, but decides at each step instead of branching:

| | playouts/s | 10,000 rollouts |
|---|---|---|
| enumerate then pick | 39 | 254 s |
| sample directly | 7,559 | **1.3 s** |

**196x.** It is a rollout *policy*, not a uniform sample — see the doc comment on
`sample_legal_move` for where it deliberately departs and why. Sanity check over
200 games: mean final score -12.6 (range -63..36) versus -24.9 (-70..19) for
uniform-over-enumeration, so it is if anything slightly less bad. Both play
badly; random play starves.

## Retrieval ordering

Order matters — a contested building, a Palenque tile, corn a later action needs
— so every ordering is explored. Most orderings commute, though, and two that
land in the same position are the same move for every purpose.

`retrieve_rec` therefore keys on the **resulting `GameState`**, which doubles as
a memo: a position already reached has had its whole subtree explored and is
never expanded again. Cost is proportional to the number of distinct reachable
positions rather than to the number of orderings — only possible because the
state is `Copy + Hash`.

Move lists are sorted before returning, since `HashMap` iteration order varies
between instances and a seeded game has to replay identically.

## Rules calls that need your check

Places where the Go source was ambiguous or self-contradictory and I had to pick.

1. **Temple prize ties.** Split `prize / n_tied`. Go paid `prize / 2` on any tie
   regardless of how many players were level — your README's "fix multitie".
2. **Entry spaces.** A worker on position 0 can now be retrieved for no effect.
   Go generated no options there, so the worker was stuck until it rode off.
3. **Monument 6** extended to 24 points at six workers, continuing Go's
   `0,0,0,6,12,18`.
4. **Monument 10** pays 4 per monument at four players, reading Go's `[_,6,5,4]`
   as indexed by player count.
5. **Age-2 building 3** now awards the 4 points its own description claimed.
6. **Begging** still requires corn < 3, as in Go.
7. **The Uxmal mirror** copies Yaxchilan, Tikal and Uxmal 1–4 only, not Palenque
   or Chichen — matching Go, and matching the space definitions rather than what
   is showing on the board.
8. **Paying down to an entry space** is not offered: spending corn to reach a
   space that does nothing is strictly dominated by doing nothing for free.

Settled by you, and now implemented:

- The first player marker **does not move** when nobody claims the space. It goes
  to the claimer otherwise. (Go advanced *past* the claimer if they already held
  it, so taking the space could cost you the seat you paid for.)
- The first player tile is **per-player**, and **flips back when its owner
  reaches the top of a temple**. Go only flipped it when a step would have
  overshot the top, so arriving exactly on the top step did nothing.
