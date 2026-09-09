# Status — 2026-09-09 16:50 EDT

A single page that says what the strongest agent is, how to watch it, and what
is running. Written because eleven agent runs here have been killed mid-flight
by usage limits and only what was on disk survived.

## The strongest agent, and how to see it

**Best measured overall** — `docs/FINAL-LADDER.txt`, 400 games:

```
mcts:32768:heuristic:deeper   vs heuristic:full   +11.83  [+11.02, +12.64]
mcts:8192:heuristic:deeper    vs heuristic:full    +9.21  [ +8.60,  +9.83]  (800 games)
```

**But the leaf evaluator has since been beaten by a trained blend**
(`FINDINGS-net.md` N21/N22). Same search, same simulations, *only* the `EVAL`
field of the spec differs:

```
mcts:2048:ft-champ@0.5:cp=0.06  vs  mcts:2048:heuristic:cp=0.06   +5.07  [+4.30, +5.84]  100 blocks
mcts:8192:ft-champ@0.5:deeper   vs  mcts:8192:heuristic:deeper    +1.76  [ ±1.22       ]   28 blocks
```

So the strongest thing that has actually been raced at the champion's budget is

```bash
cargo run --release --bin tui -- --agent 'mcts:8192:data/net/ckpt/ft-champ.safetensors@0.5:deeper'
```

Smoke-tested end to end at 16:45 (spec resolves, net loads, batching runs,
games complete). Press `t` for the ranking, `a` for autoplay.

**Untested and the obvious next race:** `mcts:32768:...ft-champ@0.5:deeper`.
The blend's edge *shrinks* with budget (+5.07 at 2,048, +1.76 at 8,192) because
a deeper search substitutes for its leaf evaluator, so do not assume it holds
at 32,768 — measure it.

The `deeper` preset is sugar for `cp=0.02,pmin=2` (`record.rs:3090`) and is only
right at a large budget.

## Data on disk

| what | where | state |
| --- | --- | --- |
| champion self-play | `data/replay-champ/` | 8,984 / 20,000 games, 835k records, 420 MB, mean score +33.7, 0.58 games/s, ETA ~5 h |
| warm-start buffer | `data/replay/` | 10 GB, mean score −14.7, 93 % `Mode` phase, **0 % policy targets** — warm start only |
| nets | `data/net/ckpt/` | `warm-s2053.safetensors` (buffer only, costs the champion 2.71), `ft-champ.safetensors` (+400 champion games, gains 1.76 at 0.5 blend) |

400 champion games beat 200,000 warm-start ones. The buffer is the warm start
and *only* the warm start.

## Live workstreams

Each owns its own files and its own findings log; none may touch another's.

| workstream | owns | log | open question |
| --- | --- | --- | --- |
| board legibility | `src/ui.rs`, `src/bin/tui.rs`, `src/bin/uidump.rs` | `FINDINGS-tui.md` | what each space does; gears as rings; blink the placed/retrieved workers of a hovered move, with a non-blink fallback |
| network training | `src/net.rs`, `src/encode.rs`, `train/*` | `FINDINGS-net.md` | retrain on 8,500+ champion games; does the policy head pay now it has `TREE_EDGE` targets; beat `eval::heuristic` at 8,192 and 32,768 |
| research value | `src/bin/rlab.rs` | `FINDINGS-research-value.md` | causally, is early research good; does it depend on the monument row; corn-as-placement-depth; is `uses = (rounds_left/3).min(7)` the right shape |

`src/eval.rs` is currently unowned. The evaluator workstream died at "measure
one cell on the champion's own spec" and has not been relaunched.

## Standing rules that were learned expensively

- Race on a **private pinned binary**, never `target/release/` — `mcts.rs`
  changed 830 lines between two pins and `--agent mcts:...` meant two different
  searches (0 of 44 shared-seed rows matched).
- **One writer per output file.** Two runners on one file narrow the interval
  by √2; a summariser once played fresh random games *into the file it was
  reading*.
- The **rotation block** is the independent unit, not the game. Null win rate is
  **25 %**, not 50 %. Always say how many blocks.
- Re-measure a constant against the evaluator it will ship in: a better
  evaluator *shrinks* the search's measured edge (+13.32 → +7.45 → +9.21).
- `--promise` is circular for delayed-payoff terms — proven, `FINDINGS-eval.md`.
