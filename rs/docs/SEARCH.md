# Search

The search half of an AlphaZero-style agent for this engine. The network half is
`LEARNING.md`; the boundary between them is one call, specified in §6.

Everything here rests on one decision, so it comes first: **the tree has one node
per *sub-decision*, not one per turn.** Sections 2 and 3 argue for that and lay
out the node types; the rest follows from it.

---

## 1. The numbers this design rests on

Measured on a copy of the working tree taken mid-audit (one line patched in the
copy — `skulls_remaining` in the `GameState` initialiser — so it would compile;
nothing in `rs/` was touched). Release build, seeded games, driven by
`take_turn_sampled` unless stated.

The snapshot contains A2 (27-round calendar, food days at 8/14/21/27) and B1
(the 13-skull supply). It does **not** yet contain A5, A6, A8 (Tikal "1 or 2"),
A9 ("do nothing" everywhere), B5 (two-way market) or C1 (pity). Every one of
those *widens* nodes. **These numbers are a floor.**

It also predates the gear resize (`Gear::size` 8/11 -> **7/10**, `ids.rs:216`),
which cuts the two mirror spaces at the top of each small gear to one and drops
Chichen's. That is the one in-flight change that makes nodes *narrower*: the
mirror spaces are the widest on the board (§1.3), so expect the `Take` tail to
come down somewhat from what is tabulated below. It does not change any
structural conclusion.

### 1.1 Flat branching, and why the README understates it

The README quotes p50 41, p90 239, p99 1,261, max 250,680. I reproduce that
almost exactly — but only when I drive the game the way `stats.rs` does, picking
`moves[seed % moves.len()]`. Driving it with the sampling rollout policy instead,
which the README itself measures as playing *less badly* (mean final score −12.6
vs −24.9), the same engine gives:

| driver (60 games, ~5,810 turns each) | p50 | p90 | p99 | max |
|---|---|---|---|---|
| `stats.rs` modulo pick | 42 | 241 | 1,484 | 204,373 |
| `sample_legal_move` | **129** | **520** | **11,925** | **978,885** |

Same code, same seeds. The entire difference is play quality: a player who is not
throwing corn away has more workers on more gears with more resources, and every
one of those multiplies. p99 rises 8x, the max rises 5x.

**The branching factor is a function of how well the agent plays.** A trained
agent plays better than `sample_legal_move`, so it will face wider nodes than any
number in this document. Any design sized against the README's 250,680 is sized
against the branching of a player who is losing.

Over a longer run (150 games, 14,564 turns, sampled driver):

| | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| flat `legal_moves` per turn | 1,286 | 128 | 519 | 10,884 | **1,916,438** |
| `legal_moves` wall time | **8.08 ms** | | | | |

At 264 bytes per `Move`, that worst node is 480 MB if you materialise it.

### 1.2 Factored node widths

Same runs, 400 games / 38,844 turns for the factored figures:

| node | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| widest sub-decision in a turn | 17.4 | 7 | 20 | 225 | 8,959 |
| `Beg` | — | 1 | — | — | 4 |
| `Mode` | — | 2 | — | — | 3 |
| `PlaceTarget` (incl. stop) | 6.8 | 7 | 7 | 7 | 7 |
| `PickWorker` (incl. stop) | 2.4 | 2 | 4 | 5 | 7 |
| `Take` — `choices_for_worker` | 11.0 | 1 | 15 | 192 | 8,959 |
| sub-decisions walked per turn | 7.3 | 8 | 8 | 12 | 14 |

Cost of generating every `choices_for_worker` a turn could need: **0.0026 ms**,
against 8.08 ms to enumerate the turn flat. Roughly 3,100x.

Widest node: **1,916,438 → 8,959**, a 214x reduction, and 10,884 → 225 at p99.
The price is 8 evaluations per turn instead of 1 (§3.6).

### 1.3 Where the residual tail lives

Splitting `choices_for_worker` into two nodes — *which space do I step down to*
(`Pos`) then *which choice there* — barely helps:

| | p50 | p90 | p99 | max |
|---|---|---|---|---|
| one-level (`choices_for_worker`) | 1 | 15 | 192 | 8,959 |
| two-level: which space | 2 | 4 | 7 | 11 |
| two-level: `choices_at(j)` | 1 | 6 | 98 | **7,355** |

The tail does not come from the pay-down dimension. It comes from inside a single
space's generator. Per-space widths on mid-game positions:

| space | p50 | p99 | max |
|---|---|---|---|
| Tikal 6/7 (mirror) | 31 | 299 | 325 |
| Uxmal 6/7 (mirror) | 25 | 313 | 334 |
| Uxmal 5 (mirror) | 17 | 306 | 314 |
| Tikal 3 (research ×2) | 15 | 227 | 242 |
| Tikal 4 (build two) | 1 | 92 | 100 |
| Chichen 10 (mirror) | 1 | 16 | 16 |

> **Correction (2026-09-07).** This paragraph was written during a window when
> `Gear::size()` had been wrongly changed to 7/10. That change was a regression
> and has been reverted: the small gears have **8** spaces (0-7) with mirrors at
> **both** 6 and 7, and Chichen Itza has **11**, its mirror at 10. So these
> measurements are against the live layout and nothing here is stale for that
> reason. Anyone citing "one mirror per gear instead of two" is repeating the
> reverted state.

`Tikal::build_two` (`spaces/tikal.rs:37`) and `uxmal::mirror_choices`
(`spaces/uxmal.rs:79`) are products of two generators; that is the whole tail.
Factoring *those* would need each space generator to expose its own sub-decision
structure, which none of them do — they return a finished `Vec<Choice>`. I do not
recommend restructuring them (§2.5). The tail is rare enough to cap instead:

| threshold | % of `Take` nodes above it | % of `choices_at` nodes above it |
|---|---|---|
| > 32 | 5.6% | 2.6% |
| > 64 | 3.4% | 1.5% |
| > 128 | 1.8% | 0.7% |
| > 256 | 0.6% | 0.14% |
| > 1024 | 0.02% | 0.002% |

A top-K cap at K = 32 leaves 94% of nodes untouched.

---

## 2. The action-space problem

### 2.1 A flat policy head is not on the table

Not merely too wide — **undefinable**. A flat head needs a fixed, finite action
vocabulary with a stable index. A `Move` is a `SmallVec` of `(WorkerId, Choice)`
pairs where `Choice` is itself a `SmallVec<[Effect; 8]>` of *value-resolved*
effects (`effect.rs`): `Effect::Corn(7)` is a distinct action from
`Effect::Corn(5)`, and which one exists depends on the player's agriculture
level. There is no enumeration to index into. Dismissed.

### 2.2 Progressive widening over `legal_moves_capped` — rejected

Two independent reasons, both from the code rather than from theory.

**It returns a biased prefix, not a sample.** `visit_legal_moves`
(`moves.rs:109`) loops beg options; inside each it runs `visit_placements` to
completion and *then* `visit_retrievals` (`moves.rs:153-154`). So every placement
move precedes every retrieval move. `legal_moves_capped(g, p, 64)` in a position
with more than 64 placements returns **zero retrieval moves** — the agent would
be structurally unable to consider picking a worker up. Within placements it is
worse: `place_rec` emits preorder, so the prefix is "worker 0 on Palenque's
lowest space", then that plus worker 1, and so on — one corner of the space,
repeated. The doc comment on `legal_moves_capped` (`moves.rs:214-217`) already
says it is not a canonical subset. It is right.

**The cost model is wrong.** Widening from k to k+1 children re-walks the
generator from scratch, so reaching k children costs O(k) full walks. And the
walk is the expensive thing: 8.08 ms per turn against 0.0026 ms for every
`choices_for_worker` the same turn could want. Progressive widening is the right
tool when children are cheap and numerous; here they are numerous *because*
enumeration is expensive.

Progressive widening does have a place — but at the leaf choice node, over a list
we can afford to generate whole. See §2.6.

### 2.3 Action abstraction — rejected as the primary mechanism

The suggested example, paying wood vs stone for the same research advance, is not
actually an equivalence. Wood is worth 2 corn at scoring and stone 3
(`ids.rs:116`), and buildings demand specific blocks (`data/buildings.rs`). The
two moves leave materially different positions. Collapsing them is lossy in a way
that bites exactly in the tight endgames search exists to win.

`options::dedup` (`options.rs:372`) already collapses everything that is
*genuinely* identical — same effect vector, different derivation. What is left is
different by construction.

And it attacks the wrong term. The explosion is a **product** over workers and
over orderings; abstraction shrinks a factor, not the exponent. Halving the
payment variants takes 1.9M to 950k.

The decisive objection is maintenance. An abstraction is a hand-written claim
about which positions are strategically equivalent. The rules are being rewritten
right now — seventeen violations, eight missing rules — and every one of those
edits can silently invalidate such a claim, with no test that fails. The design
principle for the rest of this document is that **search asks the engine what is
legal and never encodes a rule itself.** Abstraction is the one mechanism that
would break it.

### 2.4 Recommendation: factor the turn into the chain `sample_legal_move` walks

`sample_legal_move` (`moves.rs:618`) already decomposes a turn into a chain of
small decisions: beg? → place or retrieve → which worker → which space/choice →
stop or continue. That decomposition is not an artefact of the sampler. It is the
structure of the game: Tzolk'in turns are a sequence of per-worker actions, and
the flat `Move` type is the *product* of that sequence flattened out.

Build the tree over the chain instead of the product. A turn becomes a path of
~8 nodes of width ≤ 32 instead of one node of width up to 1.9M.

Three properties make this nearly free here:

1. **Intermediate states are real states.** `apply_move` (`moves.rs:459`) applies
   retrievals one at a time — `choice.apply(g, p); g.retrieve_worker(w)` — and
   placements one at a time. A half-executed turn is a well-formed `GameState`
   that passes `invariants::validate`. Applying the sub-actions in sequence
   yields bit-identically what `apply_move` yields for the assembled `Move`, so
   **no new state type is needed**: a node is a `GameState` plus a small phase
   tag.
2. **The engine's ordering memo becomes the transposition table.** `retrieve_rec`
   keys on the resulting `GameState` to collapse commuting orderings
   (`moves.rs`, and the README's "Retrieval ordering" section). In a factored
   tree those orderings arrive at the same `(GameState, Phase)` node and the
   transposition table merges them — the same collapse, at search time, sharing
   statistics rather than discarding branches. The `HashSet<GameState>` walk
   memo drops out of the hot path entirely, along with the nondeterministic hash
   iteration order it needs sorting to paper over.
3. **The whole thing is generator-driven.** Every node's edge list comes from
   `choices_at`, `lowest_free`, `available`, `on_board`, `beg_options`. The
   search encodes no rule.

### 2.5 One level, not two

The two-level split (§1.3) moves the worst node from 8,959 to 7,355 — 18% — and
costs 2 extra evaluations per turn (p50 8 → 10, max 14 → 20). That is a bad
trade: ~25% more network calls for almost nothing, because the tail lives inside
one space's generator, not in the pay-down.

**Recommend one-level factoring**, with `choices_for_worker` (`moves.rs:355`) as
the edge source for the `Take` node. It already fuses the pay-down and the
choice, and it is the function the rules track.

If the tail ever needs to shrink further, the place to cut is `build_two` and
`mirror_choices`, not the pay-down. I would not do it: it means giving those two
generators a second, sub-decision-shaped API that has to stay in sync with the
`Vec<Choice>` one, in exactly the two spaces whose rules are most intricate. The
cap in §2.6 handles them for a fraction of the risk.

### 2.6 Where progressive widening *does* belong

At the `Take` node only, and in a form that is cheap because the list is cheap:

- Generate the full `choices_for_worker` list (sub-microsecond in the ordinary
  case; the whole turn's worth is 0.0026 ms).
- Score every candidate with the network's per-choice head (§6.2).
- Keep the top **K = 32** by prior as edges. 94% of nodes are unaffected.
- Beyond K, widen: admit the m-th child once `N(node) ≥ C · m^α`, with
  C = 2, α = 0.5 — so a node has to be visited ~2,000 times before it looks at a
  1,024th option. Cap absolutely at 128.

This is widening over a *ranked* list, which is what makes it sound. The
`legal_moves_capped` version fails precisely because its list is not ranked.

This K is not the same knob as `LEARNING.md` §2's two-stage candidate scorer,
which fast-ranks every candidate and deep-scores the top 64. They compose: the
head ranks all, deep-scores 64, the tree opens 32. Keep **tree K <= head K**, or
the tree will open edges carrying only a fast-stage score.

**Where I am uncertain:** K = 32 and the widening constants are guesses. They
should be tuned by playing fixed-budget matches at K ∈ {8, 16, 32, 64}. My prior
is that the cap barely matters and K = 16 is fine, because a 300-wide Tikal
mirror node is 300 variations on about five ideas — but I have not measured how
much prior mass concentrates, and the LEARNING side cannot tell you either until
there is a trained head.

### 2.7 The node types, precisely

`P` is the player whose turn it is. All nodes below except `ExtraDay` and
`Draft` belong to `P`.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    /// Start of P's turn. Edges: `Beg(None)`, plus `Beg(Some(t))` for each
    /// temple P can step down, when corn < 3.  Source: `beg_options`.
    /// Width 1..=4.
    Beg,

    /// Edges: `Mode(Place)` if P has an available worker and can afford a
    /// space; `Mode(Retrieve)` if P has a worker on a gear; `Mode(Pity)` only
    /// when neither holds — so the node is degenerate whenever pity is legal.
    /// Width 1..=2.
    Mode,

    /// n workers already placed this turn. Edges: `Place(Gear(g, lowest_free))`
    /// for each gear where `cost_so_far + n + pos <= budget`;
    /// `Place(FirstPlayer)` if unclaimed; `StopPlacing` if n > 0.
    /// Source: `lowest_free`, `Placement::space_cost`.  Width <= 7.
    Placing { n: u8 },

    /// Edges: `PickWorker(w)` for each w still on a gear;
    /// `StopRetrieving` if at least one has been taken.
    /// Source: `GameState::on_board`.  Width <= 7.
    PickWorker,

    /// A worker has been picked and not yet resolved. Edges: `Take(choice)` for
    /// each element of `choices_for_worker(g, P, gear, pos)`, capped per §2.6.
    /// Applying the edge runs `choice.apply` then `retrieve_worker(w)`.
    /// Width <= K.
    Take { worker: WorkerId },

    /// Not part of any `Move`. Reached after the last player's turn, once
    /// `resolve_first_player` names a claimer who `may_take_extra_day`.
    /// Two edges. **The mover here is the claimer, not P.**
    ExtraDay { claimer: PlayerId },

    /// Reached only through `Mode(Pity)`. Edges: `Pity(spot)` for each
    /// equal-cheapest space.  Source: `pity_moves`.  Width <= 7.
    PityPlace,

    /// Setup only: deal 4, keep 2, taken as **two sequential picks** (4 edges
    /// then 3) rather than one 6-way choice. `LEARNING.md` §3b asks for this
    /// shape so the pick reuses the same per-choice head as `Take`; it is also
    /// the more consistent factoring.
    DraftTile { dealt: [u8; 4], kept: u8 },
}
```

Transitions. `StopPlacing`, `StopRetrieving`, and `Pity(spot)` out of
`PityPlace` each commit the turn: apply, then `refill_buildings`, then hand off to the next seat's `Beg`, or
to round end. Round end applies `resolve_first_player`, then `ExtraDay` if it is
offered, then the day advance with its food day and scoring. All deterministic.

Two invariants the implementation must hold, both mirroring
`visit_legal_moves`:

- `StopPlacing` is illegal at `n = 0`, and `StopRetrieving` is illegal before a
  worker has been resolved. `visit_legal_moves` never emits an empty `Move`
  (`check_move` rejects one explicitly, `moves.rs:509`).
- `Mode(Pity)` is offered only when nothing else is — the same condition
  `visit_legal_moves` uses to decide whether to fall through to `pity_moves`.

### 2.8 The test that makes this safe

The factored tree is a reimplementation of legality, so it needs to be pinned to
the existing one. For any position with fewer than ~5,000 legal moves:

> the set of `GameState`s reachable by walking every path from `Beg` to a commit
> edge equals the set of `GameState`s reachable by `apply_move(s, p, m)` over
> every `m` in `legal_moves(s, p)`.

Sets of states, not sets of moves — the engine's ordering memo means one state
can correspond to several move objects. This is cheap, it is a proptest over
seeded positions, and it will catch every divergence that matters. Additionally,
every `Move` the search reconstructs should pass `check_move` before being
played, at least while `debug_assertions` are on.

---

## 3. MCTS core

### 3.1 Memory: states in nodes, not replay from root

**Store the `GameState` in the node.** `GameState` is 312 bytes with no heap and
no interior pointers, so a node is a memcpy.

Replay-from-root would save that 312 bytes and cost almost nothing in `apply`
(0.0001 ms). But it is the wrong trade, because the thing you would have to redo
on the way down is not the apply — it is the *generation*. To know what edge
index 3 means at a `Take` node you must regenerate `choices_for_worker` at every
ancestor. That is the expensive half. Storing the state means you generate a
node's edges exactly once.

And you need the state anyway: it is the transposition key.

The budget is not tight. Nodes are created only on expansion, so node count ≈
simulation count. At 800 simulations per move, a node is 312 bytes plus ≤32
edges at ~48 bytes, so ~1.8 KB — about 1.4 MB per move. Even 100k simulations is
under 200 MB. Use an arena and 32-bit indices, not `Box`/`Arc`:

```rust
pub struct Arena { nodes: Vec<Node>, index: HashMap<(GameState, Phase), u32> }

pub struct Node {
    state: GameState,          // 312 B
    phase: Phase,
    to_move: PlayerId,
    edges: Box<[Edge]>,        // one allocation, sized at expansion
    visits: AtomicU32,
    virtual_loss: AtomicU16,
}

pub struct Edge {
    action: SubAction,         // <= 32 B; only `Take(Choice)` is large
    prior: f32,
    child: AtomicU32,          // 0 = unexpanded
    n: AtomicU32,
    w: [AtomicI32; N_PLAYERS], // signed fixed-point; see 3.4 and 4.5
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum SubAction {
    Beg(Option<Temple>),
    Mode(Mode),
    Place(Placement),      // 2 B
    StopPlacing,
    PickWorker(WorkerId),
    Take(Choice),          // 32 B
    StopRetrieving,
    Pity(Placement),
    ExtraDay(bool),
    DraftTile(u8),        // one of the four dealt; taken twice
}
```

Reconstructing the `Move` to hand back to the engine is a walk over the
`SubAction`s on the played path — that is the only thing the payloads are needed
for during descent, since the child node already holds the resulting state.

### 3.2 Transpositions

Index nodes by `(GameState, Phase)`, not by tree position. `GameState: Eq + Hash`
makes this free, and it is worth more here than in a normal MCTS because
factoring *creates* transpositions that the flat move generator was collapsing by
hand: retrieval orderings that commute, and beg variants that converge (begging
brown then stepping brown up lands where begging yellow does — which is why
`Walk::reached` is created once outside the beg loop, `moves.rs:138-143`).

Back up **only along the path actually traversed**, single-parent. A node reached
through several parents accumulates a visit count larger than any one parent
attributes to it, so PUCT's `sqrt(ΣN)` term is inflated there. That is the known
cost of graph search and it is the trade KataGo makes. The alternative — exact
multi-parent accounting — is a large amount of machinery for a second-order
effect.

**If you want to de-risk it:** ship the transposition table first as a pure
*evaluation cache* keyed on `GameState` (reuse priors and value, but give each
tree position its own node), measure, and only then share nodes. The cache alone
captures most of the win, because the network call is the expensive part.

### 3.3 PUCT

```
                                     sqrt( Σ_b N(s,b) )
  a* = argmax_a   Q(s,a)  +  c(s) · P(s,a) · ------------------
                                        1 + N(s,a)

  c(s) = c_init + log( (1 + N(s) + c_base) / c_base )
```

Use the **log-scaled** `c_puct` (c_init ≈ 2.0, c_base ≈ 19652), not a constant.
(c_init is set for the `z_rel` value range agreed in §4.5, not AlphaZero's 1.25.)
The specific reason is factoring: node visit counts in this tree span orders of
magnitude *within a single turn*. A `Mode` node receives every simulation that
enters the turn; a `Take` node three levels down a rarely-chosen branch may see a
handful. One constant cannot be right for both — tuned for the deep nodes it
under-explores the shallow ones, and vice versa. The log form makes the
exploration term grow with visits, so it self-adjusts along the chain.

`Q(s,a) = W(s,a)[to_move(s)] / N(s,a)` — each node maximises its own player's
component. See §4.

**Unvisited edges** take first-play urgency, not zero:
`Q_fpu = Q(parent)[to_move] − 0.2 · sqrt(Σ P of already-expanded children)`.
Zero would be a specific claim about the value scale — that "no information" and
"a draw-ish position" coincide — which is only true if the value head is centred,
and in a four-player game with a `[f32; 4]` that sums to 1 the natural centre is
0.25, not 0. FPU sidesteps the question by anchoring to the parent.

### 3.4 Backup

A simulation reaching a leaf gets `value: [f32; 4]`. Walk back up the traversed
path; at each edge:

```rust
edge.n += 1;
for i in 0..N_PLAYERS { edge.w[i] += value[i]; }
```

The whole 4-vector propagates, unchanged, at every level. No negation, no
player-relative flip. The player identity enters only at *selection* time, via
`to_move(s)`. This is the single most important structural difference from
two-player AlphaZero and it is what makes §4 work.

Note that along a sub-decision chain the mover does not change until a commit
edge, so the argmax component is constant for ~8 consecutive levels. `ExtraDay`
is the one mid-round node where it changes.

Store `w` as **signed** fixed-point `AtomicI32` (value × 2^16) so backup is
lock-free. Signed, because the value backed up is centred and runs in
(-1, 1) — see §4.5.

### 3.5 Virtual loss and parallelism

Recommend **tree parallelism with virtual loss and batched leaf evaluation**:
N threads descend concurrently, each adding a virtual loss on the way down,
queueing its leaf; one thread runs the batch through the network; all back up and
remove their virtual losses.

Virtual loss matters *more* here than in a normal AlphaZero, and that is a direct
consequence of factoring. A turn is a chain of ~8 nodes, several of which have
width 2–7; threads entering the same turn have very little room to diverge in the
first few levels and will collide on the same `Beg` and `Mode` edges. Without
virtual loss, N threads do N copies of one simulation.

Apply it to the **mover's component only**:

```rust
// descending
edge.n += VL;  edge.w[to_move] -= VL;   // in fixed point
// on backup
edge.n -= VL;  edge.w[to_move] += VL;
```

with `VL = 1`. Higher values are usually recommended for wide shallow trees; this
tree is narrow and deep, where an aggressive virtual loss pushes threads into
genuinely bad branches.

The engine cooperates: `GameState` is `Copy` with no heap and no interior
mutability, so it is `Send + Sync` for free and the arena can be shared behind a
`RwLock` on the node vector (or a chunked arena to avoid reallocation) with all
per-edge statistics atomic.

**Batching is the real reason for parallelism.** Sizing: the network call will be
~0.1–1 ms; generating a node's edges is ~0.001 ms. The search is
evaluation-bound by two to three orders of magnitude, so target a batch of 32–128
leaves and expect throughput to track GPU utilisation, not CPU.

### 3.6 The cost of factoring, stated plainly

A turn is p50 8 sub-decisions (p99 12, max 14). If the evaluation budget is fixed
at S network calls, factoring buys **S/8 turns of lookahead instead of S**.

That is the real price and it should not be glossed. Three things make it a good
trade anyway:

1. The alternative is not "S turns of lookahead" — it is a flat node of width up
   to 1.9M that no policy head can address and no widening scheme can sample
   fairly (§2.2). There is no cheaper correct option on the table.
2. **Every visited sub-decision node yields a training example.** One searched
   turn produces ~8 policy targets instead of 1, so the same self-play compute
   fills the replay buffer roughly 8x faster. The cost in search depth is partly
   refunded in sample efficiency. Hand this to the LEARNING side; it changes
   their buffer sizing.
3. Intermediate nodes are shallow and highly transposed — the `Beg` and `Mode`
   nodes of a turn are visited by essentially every simulation entering it, so
   after the first they are cache hits, and with §3.2's evaluation cache they
   cost no network call at all.

**An optimisation I considered and am not recommending yet:** skip the network at
`Beg` and `Mode` (widths 1–4, often 1) and use a uniform prior. Nodes with a
single edge should certainly be collapsed without any evaluation — that is free
and obviously right. Beyond that, measure first.

### 3.7 Root noise, and what "the root" means when a turn is a chain

Dirichlet noise mixed into the root priors is what stops self-play collapsing to
one line. With factoring, applying it only at the literal root node — the `Beg`
node — would be nearly useless: the `Beg` node's edges are "beg or don't", and
noise there cannot diversify *which building you construct*.

**Recommendation: apply Dirichlet noise at every node belonging to the root
player's current turn** — every node from the root down to (not including) the
first commit edge. That is the set of decisions the agent is about to make, and
it is the exact analogue of "the root" in an unfactored tree.

Use `α = 10 / mean_edges ≈ 0.6` and `ε = 0.25`. Because node widths here vary
from 2 to 32, α should be set per node from that node's edge count rather than
fixed, or the noise will be negligible at wide nodes and overwhelming at narrow
ones.

**This is an inference, not something I can point at prior work for.** It follows
from the factoring, and I am fairly confident it is right, but it is the item in
this document most worth an ablation: run self-play with noise at the `Beg` node
only vs the whole turn chain, and compare policy entropy after a few thousand
games.

### 3.8 Choosing the move

The played action is a *path*, not a single edge. Sample it by descending from
the root, at each node drawing an edge with probability `N(s,a)^(1/τ) / Σ`.
Sampling each node independently from its own visit distribution samples the path
in proportion to the product of the conditionals, which is exactly the tree's
visit distribution over complete turns. No extra machinery needed.

`τ = 1` for the first ~10 rounds of self-play, then `τ → 0`. Evaluation games use
`τ = 0` throughout. Emit one `PolicyTarget` per node on the sampled path.

---

## 4. Four-player backup

### 4.1 Recommendation

**A 4-vector value with per-player backup, selecting on the component of the
player to move.** This *is* maxⁿ (Luckhardt & Irani 1986), implemented inside
MCTS rather than in a depth-limited minimax. §3.4 is the whole implementation:
propagate the vector unchanged, take the mover's component at selection.

### 4.2 Why not paranoid

Paranoid reduces the game to two-player zero-sum by assuming the other three
coordinate against you. Its appeal is that it restores alpha-beta pruning — which
MCTS does not use, so the appeal does not apply here. Its cost is that the value
function it induces is a lie: it would learn that a five-point temple lead is
worth less than it is because it prices in three opponents spending their turns
attacking. In a race game where the dominant interaction is *competition for
scarce board spaces* rather than direct attack, that distortion is large.

There is no mechanism in Tzolk'in for three players to coordinate against a
fourth. There is no attack action at all. Paranoid is modelling a game that isn't
being played.

### 4.3 Why the game genuinely is not zero-sum, so a scalar cannot work

This is not a technicality:

- **Temple ties share.** `gain_temple_points` splits the prize among everyone
  level with the leader (`game.rs`). Two players tied at the top both score.
- **Starvation destroys points.** An unfed worker costs 3 points that go to
  nobody (`food_day`, `game.rs:343`). Total points are not conserved.
- **Monuments and buildings score independently.** Most of the endgame is each
  player converting their own position, not taking from others.
- **The 13-skull supply (B1) is the one genuinely conserved resource** — and it
  was just added. That makes Chichen Itza the sharpest zero-sum sub-game on the
  board and raises the value of search depth there specifically.

A scalar value head would have to compress all of that into one number and would
lose the distinction between "I gain 5" and "I gain 5 and the leader loses 5".

### 4.4 What AlphaZero-for-N-players actually does

Honestly: DeepMind's AlphaZero is two-player zero-sum only, and there is no
canonical N-player version from them. The extension that people actually use is
the obvious one — a value head with one output per player, and selection on the
mover's component. I believe the clearest published statement of it is Petosa &
Balch's *Multiplayer AlphaZero* (2019), and OpenSpiel's MCTS supports vector
returns for the same reason. **I am recalling both from memory and cannot check
them here**, so treat the citations as pointers to look up rather than as
established support. The design does not depend on them; it depends on §4.3.

### 4.5 The choice that actually matters: what the four numbers mean

More consequential than max^n-vs-paranoid, and the one genuine negotiation with
the network side. **`LEARNING.md` §3.2 and this section were written
independently and reached different answers; what follows is the reconciliation,
and I concede the substance.**

What search *requires* of the value, and will not compromise on, is two
properties:

1. **Bounded**, in a fixed range. PUCT's exploration constant is calibrated
   against the spread of Q (§3.3), and FPU anchors to it (§3.3). An unbounded
   target makes `c_puct` meaningless.
2. **Centred / approximately constant-sum.** This is the one that bites. If
   `value[i]` is player i's *absolute* expected score, per-player backup yields
   an agent that maximises its own score and is **completely indifferent to who
   wins**. Offered "I score 42 either way, but line A gives the leader 60 and
   line B gives them 30", it chooses arbitrarily. It will lose games it could
   win, and the pathology is invisible in training loss — the value head is
   perfectly accurate and the agent is still not playing the game. Tzolk'in is
   won by rank, not by score.

My initial answer was **win share** — 1 for an outright win, split among tied
winners, summing to 1 — because it satisfies both trivially, and the engine
already computes the tiebreak (`Game::winners()`, `game.rs:434`).

**That was wrong, and `LEARNING.md` §3.2 has the argument I did not weigh:** the
data budget. A one-hot winner label carries about 2 bits per game, and this
project will see ~10^5 self-play games, not AlphaZero's 10^7-10^8. For the first
two-thirds of a 27-round game the label is nearly independent of the position,
so the value head learns the prior and hands the search a flat landscape with no
gradient to climb. A sparse target is affordable at AlphaZero's scale and is not
affordable here.

**Adopted: search backs up `z_rel`.**

```
z_rel[i] = tanh( (score[i] - mean(score)) / 25 )
```

This satisfies both of my requirements and is dense: bounded in (-1, 1) by
construction, and centred by subtracting the mean — which is exactly the
property that makes denying the leader raise your own component. It is *not* the
absolute-score target I was arguing against; the mean subtraction is doing the
work. Margin is preserved through the bulk of the distribution and saturates
only on blowouts, so there is signal from round 1.

Consequences for §3:

- `Q(s,a) = W(s,a)[to_move] / N(s,a)` lands in (-1, 1). Use `c_puct_init` in
  1.5-2.5 rather than the 1.25 quoted in §3.3, per `LEARNING.md`.
- FPU (§3.3) anchors to parent Q, so it is unaffected by the change of scale.
- Fixed-point backup (§3.4) must be *signed* — `AtomicI32` with values scaled by
  2^16, not `AtomicU32`. This is the one concrete correction to §3.4.
- Re-centre at inference: `rel <- rel - mean(rel)` before backing up, so the
  constant-sum invariant holds exactly rather than approximately.

`LEARNING.md` also carries `rank`, absolute `score` and a six-way score
`decomp` as jointly-trained auxiliary heads. Search does not consume them and
should not: they exist to shape the trunk. The one search-visible use is
reporting — `P(win) = rank[i][0]` is the number to quote in evaluation matches,
not mean `z_rel`.

**The residual worry, recorded rather than resolved.** Centring *asserts*
constant-sum, and §4.3 shows the game is not quite: starvation destroys points
that go to nobody (`food_day`, `game.rs:343`), and tied temple leaders both
collect. So a position where everyone starves and one where nobody does can
share a `z_rel` vector while being very different games. I expect this to be
second-order — the mean subtraction absorbs the common component and starvation
is mostly a *relative* failure — but if the trained agent turns out to be
strangely relaxed about feeding, this is the first place to look, and the
absolute `score` auxiliary head is the diagnostic.

## 5. Hidden deck order

### 5.1 What is actually hidden

`Deck<N>` stores the entire shuffled order inside `GameState`
(`state.rs:187`: `ids: [u8; N], next: u8`). So the state a search node holds
contains information no player has. Three decks:

- `age1: Deck<14>` and `age2: Deck<18>` — 6 face up at a time, refilled one at a
  time by `refill_buildings`, called once at the end of each turn
  (`game.rs:243`). **Live leak.**
- `monument_deck: Deck<13>` — drawn from six times at setup and **never again**;
  `refill_buildings` does not touch monuments, and the README confirms monuments
  are never replenished. After setup its remaining order is dead information.
  Not a leak.
- The two discarded starting tiles per player, now that the draft (B7) is
  implemented. A one-off leak at setup.

So the real leak is: which building card fills the next empty slot.

### 5.2 Recommendation: ignore it, and prove it is small before doing anything else

Ignore it. Search the true order.

The reason is what ignoring it buys. With the deck order in the state, **the game
after setup is a fully deterministic perfect-information game**: `apply_move`,
`refill_buildings`, `rotate`, food days and scoring are all pure functions of
`GameState`, and the extra-day coin flip is being promoted to a decision node
(B8). No chance nodes, no belief states, `GameState: Eq + Hash` stays a sound
transposition key, and everything in §3 works as written. That is a large amount
of structure to give up for a small correction.

The leak itself is bounded and short-horizon. Refill happens once per turn, so
knowing the deck buys you at most "don't build the mediocre card, the slot
refills with a better one next turn" — a fraction of one building over a game.

**The one thing you must not do is let the network see it.** If the featurisation
includes `age1.ids[next..]`, the value head will learn to be confident about
futures the deployed agent cannot see, and the policy will encode plans that
depend on it. Search-cheats-but-network-doesn't is stable and standard; both
cheating is not. This is a LEARNING-side constraint but it originates here:
**exclude the unseen tail of every deck from the input features.** The face-up
rows (`buildings_up`, `monuments_up`), `next`, and `remaining()` are all fair
game and should be included.

### 5.3 Determinization, and why not yet

Per-simulation determinization (PIMC / IS-MCTS) means reshuffling the unseen tail
for each rollout. It is cheap here — shuffling ≤14 bytes — but it costs the
transposition table (two determinizations of the same position hash differently)
and it brings PIMC's known pathologies, strategy fusion in particular: the search
assumes it can react differently to each shuffle, which the real agent cannot.
Paying that for a leak this small is a bad trade.

**Measure before building.** The experiment is cheap and unambiguous: take a
trained agent, play it against a copy of itself whose search reshuffles the
unseen deck tail at the root before each move, over a few hundred games. If the
cheating side wins by less than about a point of score, the leak is noise and
this section is closed forever.

If it does matter, the escape hatch is **root determinization**, not
per-simulation: reshuffle the unseen tail once per search, run a handful of
independent searches over different determinizations, average their root visit
counts. Each individual search stays deterministic, so §3 is untouched — the only
change is an outer loop. That is the version I would build.

Modelling the deck properly — a belief state over orders — is not worth
considering until the measurement says the leak is worth several points, which I
would be surprised by.

---

## 6. Interface

### 6.1 What search needs from the engine

Available today and sufficient:

```rust
GameState: Copy + Eq + Hash + Send + Sync        // 312 B, no heap
Move:      Clone + Eq + Hash + Ord

spaces::choices_at(&GameState, PlayerId, Gear, Pos) -> Vec<Choice>   // spaces/mod.rs:24
moves::choices_for_worker(&GameState, PlayerId, Gear, Pos) -> Vec<Choice> // moves.rs:355
moves::apply_move(&mut GameState, PlayerId, &Move)                   // moves.rs:459
moves::check_move(&GameState, PlayerId, &Move) -> Result<(), String> // moves.rs:509
moves::legal_moves(&GameState, PlayerId) -> Vec<Move>                // testing only
Choice::apply(&self, &mut GameState, PlayerId)
Choice::affordable(&self, &GameState, PlayerId) -> bool
GameState::{lowest_free, available, on_board, loc, can_temple_step,
            first_player_space, refill_buildings, worker_on_last_space}
invariants::validate(&GameState) -> Result<(), String>
```

### 6.2 Gaps — things the engine does not currently provide

**G1. Round flow on `GameState`, not on `Game`. This is the blocker; nothing
works without it.** The between-turn and between-round logic lives on `Game`
(`game.rs`), which owns a `StdRng` and a `Vec<String>` log. A search node cannot
hold a `Game` — that is a heap allocation per node and it discards the entire
point of the `Copy` state. And `Game::rotate` is `pub(crate)` (`game.rs:302`),
`end_game` is private (`game.rs:411`), so a search cannot advance the calendar at
all from outside the crate.

The logic is already RNG-free apart from the extra-day coin flip, which is
becoming a decision. So it can move wholesale:

```rust
impl GameState {
    /// Advance the calendar one day: rotate gears, accumulate corn, resolve any
    /// food day, temple payout, age change and end of game. Deterministic.
    pub fn advance_day(&mut self);
    /// Hand the first player space to its claimer; returns the claimer.
    pub fn resolve_first_player(&mut self) -> Option<PlayerId>;
    pub fn may_take_extra_day(&self, p: PlayerId) -> bool;
    pub fn take_extra_day(&mut self, p: PlayerId);
    pub fn end_game(&mut self);
    /// Final scores. Valid only after `end_game`; search calls it on its own copy.
    pub fn scores(&self) -> [i16; N_PLAYERS];
    pub fn winners(&self) -> Vec<PlayerId>;
}
```

`Game` then keeps the RNG, the log and the driver loop, and becomes a thin
wrapper that calls these. `Game::may_take_extra_day` / `take_extra_day` /
`resolve_first_player -> Option<PlayerId>` already exist in that shape
(`game.rs:252,286,291`) — they just need to move down onto the state.

**G2. Three private functions the factored tree needs.** All are the *sole*
statement of a rule the search must not restate:

- `moves::beg_options` (`moves.rs:251`) — the `Beg` node's edges.
- `Placement::space_cost` (`moves.rs:25`) — the budget check at `Placing`.
- `moves::pity_moves` (`moves.rs:160`) — the `Pity` edge.

Make them `pub`. Reimplementing any of them in the search is exactly the coupling
this design is trying to avoid.

**G3. Allocation-free choice enumeration — flagged, not urgent.** `choices_at`
returns a `Vec<Choice>` and the builders under it (`building_choices`,
`research_choices`, `corn_exchange`) allocate freely. At ~8 expansions per turn
and 800 simulations that is a few thousand `Vec`s per move. Measured cost is
0.0026 ms per turn's worth, so this is nowhere near the network call and should
**not** be optimised now. If profiling later says otherwise, the shape is the one
`visit_legal_moves` already uses:

```rust
pub fn visit_choices_at<F>(g: &GameState, p: PlayerId, gear: Gear, pos: Pos, f: F)
    -> ControlFlow<()>
where F: FnMut(&Choice) -> ControlFlow<()>;
```

**G4. A stable tag on `Effect` for featurisation.** The vocabulary just grew from
13 to 14 variants (`BurnPalenqueWood`). The network embeds effects, so it needs a
dense index, and it needs to break loudly when a variant is added:

```rust
impl Effect {
    pub const COUNT: usize = 14;
    pub fn tag(self) -> u8;   // exhaustive match, no wildcard arm
}
```

A wildcard arm here would silently mis-index the embedding table on the next
rules change. This is small and worth doing before the network exists.

### 6.3 What search exposes to the training loop

```rust
pub struct Eval {
    /// One prior per legal edge, in the order the search enumerated them.
    pub priors: SmallVec<[f32; 32]>,
    /// Per-player value. See §4.5 for the semantics contract.
    pub value: [f32; N_PLAYERS],
}

/// What the network is being asked about. Needed because a prior at `PickWorker`
/// ranges over workers and at `Take` over choices — see 6.4.
pub struct DecisionCtx<'a> {
    pub phase: Phase,
    pub edges: &'a [SubAction],
}

pub trait Evaluator: Sync {
    fn evaluate(&self, g: &GameState, p: PlayerId, ctx: &DecisionCtx) -> Eval;

    /// The form search actually calls. Default-implements over `evaluate`, but a
    /// GPU-backed implementation must override it: search is evaluation-bound by
    /// 2-3 orders of magnitude and needs batches of 32-128.
    fn evaluate_batch(&self, items: &[(GameState, PlayerId, DecisionCtx)]) -> Vec<Eval>;
}

pub struct SearchConfig {
    pub simulations: u32,
    pub c_puct_init: f32,      // 2.0, see 3.3/4.5
    pub c_puct_base: f32,      // 19652
    pub fpu_reduction: f32,    // 0.2
    pub dirichlet_eps: f32,    // 0.25
    pub max_edges: usize,      // 32, see 2.6
    pub widen_c: f32,          // 2.0
    pub widen_alpha: f32,      // 0.5
    pub temperature_moves: u32,
    pub threads: usize,
    pub batch_size: usize,
    pub virtual_loss: u32,     // 1
}

pub struct PolicyTarget {
    pub state: GameState,
    pub phase: Phase,
    pub to_move: PlayerId,
    /// Visit share over this node's legal edges. Sums to 1.
    pub visits: Vec<(SubAction, f32)>,
}

pub struct SearchResult {
    /// The assembled turn, ready for `apply_move`. Passes `check_move`.
    pub mv: Move,
    /// One target per sub-decision on the played path — ~8 per turn, not 1.
    pub targets: Vec<PolicyTarget>,
    pub root_value: [f32; N_PLAYERS],
}

pub fn search(root: &GameState, p: PlayerId, ev: &dyn Evaluator,
              cfg: &SearchConfig, rng: &mut impl Rng) -> SearchResult;

pub struct Trajectory {
    pub targets: Vec<PolicyTarget>,
    pub final_scores: [i16; N_PLAYERS],
    /// What search backs up: tanh((score[i] - mean(score)) / 25). See 4.5.
    pub z_rel: [f32; N_PLAYERS],
    /// Reporting and the auxiliary heads only; 1/|winners| each.
    pub win_share: [f32; N_PLAYERS],
}

pub fn self_play(seed: u64, ev: &dyn Evaluator, cfg: &SearchConfig) -> Trajectory;
```

**Self-play driver durability.** Self-play runs for hours. The driver should
append each finished `Trajectory` to disk as it completes rather than
accumulating in memory, and install a Ctrl-C handler that finishes the current
game and flushes rather than aborting — so an interrupted run loses at most one
game. Same discipline as the long-running scripts elsewhere in this project.

### 6.4 The one change I need from the network side

The stated signature is `evaluate(&GameState, PlayerId) -> (policy_priors, value)`.
Under factoring that is under-specified in two ways, and both need agreeing:

**(a) The priors need a context.** The same `GameState` is asked different
questions at different phases: at `PickWorker` the priors range over workers, at
`Take` over choices, at `Beg` over temples. Passing `Phase` and the edge list
lets one trunk feed per-phase heads. Concretely: add `ctx: &DecisionCtx`.

**(b) Priors must score edges, not index a fixed vocabulary.** A `Take` node can
have thousands of candidates, and they are value-resolved `Choice`s that do not
enumerate (§2.1). So the head must consume a featurisation of each candidate
`Choice` and emit a score.

This is easier than it sounds, and it is a gift from the `Effect` design: a
`Choice` is a sequence of at most 8 elements from a **closed 14-variant
vocabulary** (`effect.rs`), each 3 bytes, and the field is public
(`pub struct Choice(pub Effects)`). A small embedding per `Effect` variant, its
scalar argument as a feature, summed or mean-pooled over the choice, then a
2-layer MLP against the trunk's state embedding, gives a score per candidate.
Softmax over candidates gives the priors.

That head is also the most rules-robust piece of the whole system: adding an
`Effect` variant adds one embedding row (and G4 makes the compiler say so).
Adding a *building* changes nothing at all, because buildings are expressed as
effects.

The narrow nodes (`Beg`, `Mode`, `Placing`, `PickWorker`) have small fixed edge
vocabularies — temples, two modes, 6 placements, 6 workers — so they can use
ordinary fixed heads.

---

## 7. Rules coupling

The rules are being rewritten while this is being designed, so: here is exactly
which parts of the design survive that and which do not.

### 7.1 Rules-agnostic

These touch `GameState` only as an opaque `Copy + Eq + Hash` blob and would
survive a port to a different game:

- The arena, node and edge layout (§3.1); the transposition table (§3.2).
- PUCT, FPU, backup, virtual loss, batching, Dirichlet, temperature (§3.3–3.8).
- The 4-vector backup and per-player selection (§4).
- The self-play loop and the `SearchResult` / `Trajectory` shapes (§6.3).
- **Every edge list**, because they come from generators. `Take`'s edges are
  whatever `choices_for_worker` returns; `Placing`'s are whatever `lowest_free`
  says; `Beg`'s are whatever `beg_options` says. **The search states no rule.**
  This is the property that makes the rest of this section short.

### 7.2 Rules-coupled

Exactly two things:

- **The list of `Phase` variants** (§2.7). A new *kind* of decision — not a new
  option within an existing one — adds a variant.
- **The featurisation**, which is LEARNING's, but which this document constrains
  in §5.2 (hide the deck tails) and §6.4 (the `Effect` vocabulary).

### 7.3 The five in-flight changes, one by one

**13-crystal-skull global supply.** *Already landed* — `N_SKULLS`
(`state.rs:214`), `skulls_remaining`, `take_skulls` (`state.rs:376`), and a
conservation invariant. **Search impact: none.** It is a state field; the
generators already return the narrowed choice set (Yaxchilan 4 returns
`Choice::skip()` at an empty bank). Two second-order notes: it is the game's only
strictly conserved resource, which sharpens the zero-sum content of Chichen Itza
and raises the payoff to deep search there; and the featurisation must include
`skulls_remaining` or the value head will misprice every skull position.

**"Do nothing" legal at every space.** **Search impact: exactly one extra edge
per `Take` node, absorbed without comment.** Under a flat design it is a
catastrophe — it multiplies the per-worker choice count by (n+1)/n at every
worker simultaneously, and the flat branching factor is already 1.9M before it
lands. This change is on its own sufficient reason to factor; any design that
enumerates the flat move list is dead as of it.

Two knock-ons: it invalidates the dominance argument currently in
`choices_for_worker` ("paying corn to reach a space that does nothing is
dominated by doing nothing for free") — the search does not care, it reads the
generator. And `Choice::skip()` becomes a common, often-correct action, so the
network's per-choice head must score the *empty* effect vector distinctly rather
than defaulting it to zero. Flag to LEARNING: give skip an explicit token.

**Two-way market.** `corn_exchange` (`options.rs:303`) enumerates corn → blocks
over a 3-dimensional budget recursion. Symmetric exchange roughly squares that:
Uxmal 2 currently maxes at 34 choices, and could reach the hundreds. **Search
impact: one wider node, handled by the K = 32 cap.** No structural change. If it
turns out to be the widest node on the board, it is the best candidate for a
sub-split ("sell what" then "buy what") because unlike `build_two` its structure
is a clean product with no rules subtlety — but do not do it pre-emptively.

**The extra calendar day becoming a decision node.** This is the only one of the
five that changes my node set, and the design already carries it: `Phase::
ExtraDay { claimer }`, two edges, inserted between round end and the next round.
It is half-landed — `resolve_first_player` already returns the claimer and
`may_take_extra_day` / `take_extra_day` are already separate methods
(`game.rs:252,286,291`), which is precisely the shape search needs. Two things to
get right: **the mover at that node is the claimer, not the player whose turn
just ended**, so backup switches components there; and the decision is genuinely
hard (it can skip a food day, and per B4 it must not push any player's worker off
a gear), so it deserves a real evaluation rather than a heuristic.

**26 → 27 rounds.** *Already landed* — `LAST_DAY = 27`, `RESOURCE_DAYS = [8,21]`,
`POINT_DAYS = [14,27]` (`state.rs:256-260`). **Search impact: none structurally.**
The horizon grows by ~4 turns, so a full-game path grows by ~40 nodes, and final
scores drift up slightly, which matters only if the value head is normalised
against a fixed score range. Flag to LEARNING: **do not hard-code 26 or 27 in the
featurisation** — no fixed-width one-hot over days, no constant in a
normalisation. Derive from `LAST_DAY`.

**Gear sizes 8/11 -> 7/10** (landed while this was being written,
`ids.rs:216`). Not on the original list, but it is a rules change and it lands in
the same window. **Search impact: none structurally, and mildly favourable.** It
removes one mirror space from each small gear and one from Chichen, and the
mirror spaces are the widest `Take` nodes on the board (§1.3), so the tail
shrinks. The pay-down node narrows from <=11 edges to <=10. Nothing in §2.7
changes, because every one of those widths is read from a generator rather than
written down. The one thing to watch is that workers now ride one round less,
which shortens the effective planning horizon per placement and slightly raises
the value of the first-player space.

### 7.4 What would actually force a redesign

Worth naming so the boundary is visible. None of the five in-flight changes come
near any of these:

- **Simultaneous or interleaved turns.** The design assumes strictly sequential
  play (`play_round` iterates the four seats). Simultaneity would require
  information sets and a different solution concept entirely.
- **Concealed information beyond deck order** — a hand of cards, a hidden
  objective. Would force IS-MCTS and cost the transposition table (§5.3).
- **Genuine randomness in move resolution** — a die, a random draw mid-turn.
  Would require chance nodes. The engine is currently RNG-free after setup, and
  that is worth protecting deliberately.
- **A player's legal moves depending on another player's concealed choice.**
  Breaks the "a node is a `GameState`" identity outright.

---

## 8. Build order

1. **G1 — move round flow onto `GameState`.** Nothing else can start.
2. Arena, `Phase`, `SubAction`, factored expansion driven by `choices_at`, with a
   uniform-prior evaluator. No PUCT yet.
3. **The equivalence proptest (§2.8).** Set of states reachable through the
   factored tree == set reachable through `legal_moves` + `apply_move`, on
   positions under ~5,000 moves. This is the highest-value test in the project
   and it is cheap because the engine already has the reference implementation.
4. PUCT, FPU, 4-vector backup, transposition table as an evaluation cache.
5. Virtual loss, threads, batching — once there is a real evaluator to batch for.
6. Swap in the network. Ablate §3.7 (noise placement) and §2.6 (K).

One caveat on step 2: the uniform-prior version is a **correctness** harness, not
a playing agent. `rollout_quality.rs` exists to answer whether random rollouts
carry enough signal to rank moves, and the README's own numbers — random play
starves, mean final score −12.6 — strongly suggest the answer is no. Expect the
uniform-prior search to play badly and do not read that as a bug. The value
network is load-bearing here, not an optimisation.

`sample_legal_move` has no role in the final agent — AlphaZero has no rollouts —
but it stays useful for two things: bootstrapping before a network exists, and as
a cross-check that the factored tree reaches the same distribution of turns.
