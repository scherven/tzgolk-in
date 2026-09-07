# The network half

Design for the evaluator behind an AlphaZero-style Tzolk'in agent. The search
half lives in `docs/SEARCH.md`; this document owns everything from
`&GameState` to `(priors, value)` and everything from self-play records back to
weights.

Read `README.md` first for the measurements this is built on: `legal_moves()`
costs 0.254 ms, `apply_move()` costs 0.0001 ms, the sampling rollout policy runs
7,559 full playouts/s single-threaded, and branching is p50 41 / p99 1,261 /
max 250,680.

Everything below assumes the target machine: **M4 Pro, 10 performance cores,
4 efficiency cores, 24 GB unified memory, no cluster.**

---

## 0. The three things that decide this design

**(a) Move generation is the bottleneck, not the network.** Generating a full
move list costs 0.254 ms; a forward pass of the network proposed here costs
~20 µs. The engine can only enumerate ~4,000 turns/s/core. Any design that puts
a net eval on the critical path of an enumerated node is optimising the wrong
term by two orders of magnitude. This is why the network must be small, why it
must be batched, and why the factored action space matters as much to me as it
does to the search half.

**(b) Training compute is free; training data is not.** A 4.2 M-parameter MLP at
batch 1024 costs ~26 GFLOP per optimiser step. The M4 Pro will do that in under
100 ms on the CPU alone. In one hour of pure training you could present 50 M
samples — far more than a week of self-play will produce. **The entire budget is
self-play throughput.** Every decision below trades network capacity and
training sophistication for self-play games, and where the two conflict, games
win.

**(c) `GameState` contains hidden information.** `age1.ids[next..]` is the
undrawn deck in shuffled order (`state.rs:235`). A network fed the raw struct
learns to read the future. Search over the raw struct is a cheating agent. The
encoder below **never touches `Deck::ids`** — it feeds the cursor and the
publicly-derivable *unseen* set instead. See §1.6. This is not optional and it
is cheap.

---

## 1. State encoding

### 1.1 Shape

One flat `[f32; 3072]` per query, built by a hand-written Rust encoder. No
convolutions, no image, no history stack.

```rust
pub const D_IN: usize = 3072;   // 2,684 live + 388 reserved

/// Perspective-relative encoding of `g` as seen by `p`.
///
/// `p` need not be `g.current`. The seat offset of `g.current` from `p` is one
/// of the features, which is what lets one trunk pass answer for all four
/// players at once (see §3) and what makes the 4x perspective augmentation in
/// §6.4 legitimate rather than a lie.
pub fn encode(g: &GameState, p: PlayerId, phase: Phase, out: &mut [f32; D_IN]);
```

Blocks, in order:

| Block | Offset | Width | What |
|---|---|---|---|
| A. Board | 0 | 600 | 50 gear cells x 12 channels |
| B. Players | 600 | 1,128 | 4 seats x 282, seat-relative |
| C. Global | 1,728 | 956 | calendar, displays, supply, phase |
| R. Reserved | 2,684 | 388 | zeros; see §8 |

### 1.2 Block A — the gears: flat, not spatial

**The gears are not a board and must not be treated as one.** A convolution
shares weights across positions, which is correct exactly when positions are
exchangeable. Tikal 3 (research x2) and Tikal 4 (build two, or take a monument)
are unrelated actions; Palenque and Chichen Itza are unrelated gears. There is
no translation invariance in either axis, so a conv layer buys nothing and
spends its capacity pretending otherwise.

The one genuine sequence structure — rotation moves every worker up one space,
and a worker on the last space falls off (`state.rs:rotate_gears`) — is a
*nonlinear* function of `(pos, gear.size())`, so it is handed over explicitly as
a feature rather than left for the net to rediscover.

50 cells, indexed `cell = gear.idx() * 10 + pos.0`. Chichen has 10 spaces, the
other four have 7 (`ids.rs:216`), so 12 cells never exist and channel 0 says so.

(That geometry changed under me mid-document: `Gear::size()` was 11 and 8 and is
now 10 and 7, the extra mirror spaces having been removed. The block is sized
from `Gear::size()` at compile time, so it tracks; but it is exactly the kind of
change §8.2 is about, and the reserved tail is why it costs a recompile rather
than a discarded checkpoint.)

| ch | meaning |
|---|---|
| 0 | cell exists (`pos < gear.size()`) |
| 1 | empty |
| 2 | holds one of **my** workers |
| 3 | holds a worker of the opponent 1 seat clockwise of me |
| 4 | ... 2 seats |
| 5 | ... 3 seats |
| 6 | is `lowest_free(gear)` — the unique legal placement target on this gear |
| 7 | `(gear.size() - 1 - pos) / 10` — rounds before the occupant falls off |
| 8 | `pos / 10` — corn surcharge to place here (`Placement::space_cost`) |
| 9 | Chichen space already spent (`chichen_filled`), 0 elsewhere |
| 10 | `palenque[pos].corn / 4`, 0 off Palenque |
| 11 | `palenque[pos].wood / 4`, 0 off Palenque |

Channels 2-5 are the perspective rotation applied to worker *ownership*:
`WorkerId::owner()` is mapped through `(owner - p) mod 4`. Worker identity
within a player is never encoded — `place_rec` consumes available workers in id
order precisely because they are interchangeable (`moves.rs:~275`), so worker
identity carries no information.

Note what is deliberately **absent**: any encoding of what each space *does*.
A dense layer over the flattened block already has per-cell weights, so "Tikal 4
is the double build" is learned, not transcribed. That is the whole argument for
flat over convolutional here, and it is also what keeps §8 short.

### 1.3 Block B — the players: seat-relative, weight-shared

Four repeats of a 282-wide record, ordered by seat offset from `p`. Slot 0 is
always the querying player; slot 1 is always the player who acts next. Because
turn order in Tzolk'in is fixed clockwise from `first_player`, seat offset is
semantically stable, so this ordering is meaningful and not merely a
canonicalisation.

| feature | dims | encoding |
|---|---|---|
| corn | 17 | `min(corn,60)/60` + thermometer at `{1,2,3,4,5,6,7,8,10,12,15,18,21,25,30,40}` |
| resources | 32 | 4 x (`min(r,12)/12` + thermometer at `{1,2,3,4,5,7,9}`) |
| points | 14 | `(points-60)/60` + thermometer at `{-20,-10,0,10,20,30,45,60,75,90,105,120,140}` |
| corn / wood tiles | 14 | 2 x (scalar + thermometer `1..=6`) |
| free_workers | 7 | scalar + thermometer `1..=6` |
| worker_discount | 3 | one-hot (clamped 0..=2, `effect.rs:82`) |
| may_skip_day | 1 | flag |
| buildings | 40 | 32-bit bitset, padded to 40 |
| monuments | 16 | 13-bit bitset, padded to 16 |
| research | 20 | 4 x one-hot(4) + 4 x `level/3` |
| temples | 45 | 3 x one-hot(9) + 3 x `step/8` + 3 x rank one-hot(4) + 3 x gap-to-leader |
| workers | 47 | available / on-board / locked one-hot(7) each, on-first-player flag, per-gear count 5 x one-hot(5) |
| is first player | 1 | flag |
| feeding | 17 | mouths one-hot(7), corn bill/12, slack/12, starve flag, shortfall one-hot(7) |
| projections | 5 | temple points due at next point-day; temple resources due at next resource-day |
| holdings value | 3 | `total_corn()/60`, owned-monument score at current state /40, skull VP |
| **total** | **282** | |

Three notes on why this shape:

**Thermometers, not raw scalars.** A thermometer code (`x >= t` for a ladder of
thresholds) makes every threshold in the rules linearly separable at the first
layer. `corn < 3` gates begging (`moves.rs:191`); `corn >= 3` gates the Uxmal
temple purchase (`uxmal.rs:32`); `corn >= price` gates each build-with-corn
option. A small net finds those instantly from a thermometer and slowly, if at
all, from a single normalised magnitude. The extra width is free.

**Temple rank, not just temple position.** `gain_temple_points` pays a prize to
everyone level with the highest player and splits it (`game.rs:~400`), so the
value of a temple step is a *relative* quantity. Encoding "alone at the top /
tied at the top / mid / bottom" per track hands that over directly instead of
asking a 512-wide trunk to compute three argmaxes. Under a one-week data budget,
this class of relational feature is worth more than an extra residual block.

**The stem that reads this block is shared across all four players.** One
`Linear(282 -> 96)` applied four times, weights tied. "How to read a player's
holdings" is then learned from 4x the data. This is the domain's real analogue
of convolutional weight sharing: share over the axis that *is* exchangeable
(players, card slots) and not over the axis that isn't (board positions).

### 1.4 Block C — global

| feature | dims |
|---|---|
| day one-hot(32), `day/27`, rounds-left/27 | 34 |
| age one-hot(2) | 2 |
| rounds to next food day one-hot(9); its type one-hot(3) | 12 |
| is final round | 1 |
| accumulated_corn scalar + thermometer(0..=12) | 14 |
| first-player space: empty / mine / opp1 / opp2 / opp3 | 5 |
| `first_player` seat offset one-hot(4) | 4 |
| **`current` seat offset from `p`** one-hot(4) | 4 |
| turn index within the round one-hot(4) | 4 |
| `skulls_remaining` scalar + thermometer(0..=13) | 15 |
| `chichen_filled` bits(11) + popcount | 12 |
| Palenque totals: corn left, wood left, `corn_showing` per jungle space | 8 |
| `buildings_up`: 6 slots x 73 | 438 |
| `monuments_up`: 6 slots x 45 | 270 |
| unseen-building mask(40) + count | 41 |
| unseen-monument mask(16) + count | 17 |
| deck cursors: `age1.next/14`, `age2.next/18`, `monument_deck.next/13` | 3 |
| phase one-hot(6) | 6 |
| phase detail: `placed` one-hot(7), `corn_spent/12`, `retrieved` one-hot(7), target cell one-hot(51) | 66 |
| **total** | **956** |

A building display slot (73 dims): id one-hot(40), cost bundle /3 (4), cost
thermometer (12), colour one-hot(4), payoff-kind one-hot(8) from
`buildings.rs:Payoff`, "player *k* can afford it" for each of the 4 seats,
is-empty flag. A monument slot (45 dims): id one-hot(16), cost /4, cost
thermometer, colour, 4x affordable, **and its `score(&GameState, p)` evaluated
right now for each of the 4 players, /20** — the monument scoring functions are
pure reads (`monuments.rs:21`) and calling six of them four times is a few
hundred nanoseconds. Handing the net "monument 11 is currently worth 20 to me
and 9 to the leader" is far cheaper than teaching it `TABLE[maxed]`.

Both display blocks go through their own **shared** slot encoder, so a card in
slot 4 is read the same way as in slot 1.

### 1.5 Perspective relativity, in one place

Everything that is per-player is rotated by `(owner - p) mod 4`: block A's
occupancy channels, block B's slot order, block C's first-player and current
offsets, the value vector, and the score-decomposition auxiliary. Nothing else
changes. `Color` is never encoded — it is a display attribute and encoding it
would break the rotation (`ids.rs:13` already separates `PlayerId` from
`Color` for exactly this reason).

### 1.6 Hidden information

**Do not encode `Deck::ids`.** What is legitimately public is:

- the cursors `next`, which say how many cards have left each deck;
- the **unseen set**, `ALL_IDS \ (face_up ∪ ⋃ players.buildings)`, which any
  player at the table can compute, and which is what actually matters (it tells
  you the chance that the double-build card is still coming).

`buildings_up`, `monuments_up` and every player's bitset are already in the
encoding, so the unseen mask is a 40- and 16-bit derived feature. Total cost:
57 features. Total benefit: the value head learns an expectation over deck
orders instead of a lookup, and the agent transfers to play against humans.

The search half then owns the corresponding decision — whether to determinise
(resample the undrawn deck at the root) or to search the true state and simply
keep the evaluator blind. Either works with this encoder; the second is much
cheaper and is what I would do first.

### 1.7 History planes: none

AlphaZero stacks history for repetition detection (chess) and superko (Go).
Tzolk'in has neither: the public projection above is Markov for the rules, and
`day` already orders time. The only thing history could buy is opponent
modelling, and in self-play against copies of yourself that is worth close to
nothing relative to a 4x input and the overfitting it invites.

**Decision: no history. Zero planes.** If opponent modelling ever becomes the
bottleneck, add the previous round's `MoveKind` per player (4 x ~10 features),
not a state stack.

---

## 2. Policy head against the factored action space

### 2.1 The factorisation, read off `moves.rs`

A turn decomposes into a chain of small decisions. This is not a modelling
convenience — it is the literal shape of `visit_legal_moves`:

| node | domain | source |
|---|---|---|
| **Beg** | `{none, Brown, Yellow, Green}`, non-`none` only when `corn < 3` | `moves.rs:200` |
| **Mode** | `{place, retrieve, pity}` | `moves.rs:153-154` |
| **Place step** | `{Palenque, Yaxchilan, Tikal, Uxmal, Chichen, FirstPlayer, STOP}` — 7, repeated up to 6 times | `moves.rs:~300` |
| **Retrieve who** | one of the 50 gear cells holding my worker, or `STOP` — 51 | `moves.rs:370` |
| **Retrieve what** | a variable, state-dependent list of `Choice` | `moves.rs:355` |
| **Extra day** | `{decline, take}` | `game.rs:286` (`may_take_extra_day`; the driver still coin-flips it) |

Two things fall out that matter enormously:

**Placement position is not a decision.** A worker always enters on
`lowest_free(gear)` (`state.rs:~390`), so choosing a gear chooses the space. The
placement branch is 7-way, not 51-way.

**Worker identity is not a decision for placement.** `place_rec` walks `avail`
in id order because permutations of the same placement are the same move
(`moves.rs:~275`). It *is* a decision for retrieval, because order matters there
(`moves.rs:~400`) — but retrieval identity is fully determined by the cell, so
the 51-way cell softmax covers it.

So four of the six node types have small **fixed** domains and get plain
softmaxes. Only `RetrieveWhat` is open-ended, and it gets a pointer head.

### 2.2 The heads

```rust
pub enum Priors {
    Beg([f32; 4]),            // [none, Brown, Yellow, Green]
    Mode([f32; 3]),           // [place, retrieve, pity]
    Place([f32; 7]),          // [Pal, Yax, Tik, Uxm, Chi, FirstPlayer, STOP]
    RetrieveWho([f32; 51]),   // gear*10 + pos, index 50 = STOP
    RetrieveWhat(SmallVec<[f32; 32]>),  // parallel to the candidate slice
    ExtraDay([f32; 2]),       // [decline, take]
    DraftTile([f32; 4]),      // parallel to the four dealt tiles
}
```

All fixed heads are a single `Linear(512 -> n)` off the trunk, ~35 k parameters
in total. The `Place` and `RetrieveWho` heads additionally see the phase detail
(`placed`, `corn_spent`, `retrieved`) — but that is already in block C of the
input, so no separate conditioning path is needed. Re-encoding the state at each
sub-decision, rather than carrying a recurrent context, is what makes this work:
each sub-decision state is a genuine `GameState` (`place_rec` and `retrieve_rec`
each mutate a probe `GameState` and restore it), so the network stays a pure function
of `(GameState, PlayerId, Phase)` and transpositions still key correctly.

### 2.3 `RetrieveWhat`: a two-stage pointer head over `Choice`

`choices_for_worker` can return hundreds of choices — Tikal 4's double build
crossed with six face-up buildings and every payment split
(`tikal.rs:43`, `options.rs:165`), Uxmal 2's corn exchange combinatorics
(`options.rs:276`), the Uxmal mirror recursing into Tikal (`uxmal.rs:78`). The
list is variable-length and its indices are meaningless across positions. A
fixed softmax is not merely awkward here, it is undefined.

Score each candidate by **what it does to the state**:

```rust
pub const D_CHOICE: usize = 96;

/// Delta features for one candidate at `(g, p)`. Computed as
/// `probe = *g; c.apply(&mut probe, p);` then differencing a fixed set of
/// summaries. ~70 ns per candidate.
pub fn encode_choice(g: &GameState, p: PlayerId, c: &Choice, out: &mut [f32; D_CHOICE]);
```

| idx | feature |
|---|---|
| 0 | bias = 1.0 |
| 1 | `is_skip` (empty effect list) |
| 2 | `n_effects / 8` |
| 3-9 | Δcorn /10 and thermometer at `{-8,-4,-1,+1,+4,+8}` |
| 10-25 | Δres[W,S,G,Skull] /3 and thermometers |
| 26-32 | Δpoints /10 and thermometer at `{-4,-1,+1,+4,+8,+13}` |
| 33-38 | Δtemple[B,Y,G] /2, plus per-track "the step was clamped and wasted" |
| 39-42 | Δresearch per track |
| 43-47 | Δfree_workers, Δworker_discount, Δavailable workers, Δcorn_tiles, Δwood_tiles |
| 48-49 | Δ(Palenque corn left), Δ(Palenque wood left) — separates a burn from a take |
| 50-61 | Δ`chichen_filled` popcount and which position |
| 62-76 | built a building: cost, colour, payoff-kind |
| 77-82 | took a monument: cost, and its `score()` at the resulting state /20 |
| 83 | Δ`skulls_remaining` — how much of the global supply this drew |
| 84 | corn-equivalent delta: `Δcorn + 2ΔW + 3ΔS + 4ΔG`, /10 |
| 85 | immediate VP-equivalent: `Δpoints + corn_equiv/4 + 3Δskulls`, /10 |
| 86-87 | projected Δ(temple points at next point-day), Δ(temple resources), corn-equivalent |
| 88-89 | does this leave me unable to feed at the next food day; corn after /20 |
| 90-95 | reserved |

**This is the single most rules-robust decision in the document.** Because the
features are measured off the resulting state rather than parsed out of the
`Effect` enum, adding an `Effect` variant, changing what a space grants, or
making the market two-way costs the policy head nothing. §8 leans on this.

Scoring is two-stage, to bound cost when the list is large:

```
h        = trunk(x)                                   R^512
q_deep   = W_q [h ; cell_emb(gear,pos) ; step_ctx]    R^128
q_fast   = W_f h                                      R^96

stage 1: s_i = q_fast · f_i                           96 MACs per candidate
stage 2: keep the top K = 64 by s_i;
         k_i = MLP_key(f_i)  (96 -> 128 -> 128)       ~28 k MACs per kept candidate
         logit_i = q_deep · k_i
         the discarded tail shares a residual prior mass of 0.03, split evenly
```

Cost at 500 candidates: 48 k MACs for stage 1 (free) plus 1.8 M for stage 2
(~20% of a trunk pass). Both stages are trained with the same cross-entropy,
so stage 1 learns to be a cheap approximation of stage 2 rather than a
hand-written heuristic that will rot when the rules move.

### 2.4 Normalising over the legal subset

Masked softmax, with the mask applied **inside the loss**, not just at
inference. Illegal logits are set to `-inf` before the softmax, so no gradient
ever reaches them and the net never spends capacity ranking moves it cannot
make. The mask has to be reconstructible at training time — see §6.2, where the
replay buffer stores raw `GameState`s and regenerates it.

For `RetrieveWhat` there is no mask: the candidate list *is* the legal set.

`STOP` deserves a note. In `place_rec`, a partial placement is emitted at every
depth (`moves.rs:~300`), so "stop after two workers" is a real, always-legal
action at every step past the first. It is therefore a normal member of the
7-way and 51-way softmaxes and not a special case — but it is the single most
frequently correct action in the game and the head should not be prevented from
saying so.

---

## 3. Value head for four players

### 3.1 A 4-vector, in perspective order

```rust
#[derive(Clone, Copy, Debug)]
pub struct Value {
    /// Centred, bounded. Index 0 is the querying player; index k is the player
    /// k seats clockwise. Sums to ~0 after `center()`. Back this up.
    pub rel:   [f32; 4],
    /// Absolute final-score estimate, in units of (score - 75) / 50.
    pub score: [f32; 4],
    /// rank[i][r] = P(player i finishes in place r). Rows sum to 1.
    pub rank:  [[f32; 4]; 4],
}
```

Not a scalar. A max^n / Sturtevant backup needs the *mover's* value at every
node, and the mover changes every turn; a scalar-from-the-current-player's-view
head would require a second forward pass with a rotated input at every node
where the mover differs from the querier. The vector costs 4 extra output
weights per hidden unit and removes that entirely.

### 3.2 The target: centred, bounded score — not win probability

Three candidates, and the choice is not close.

**Win probability is wrong here.** With four players, a one-hot winner label
carries 2 bits per game and treats "second by one point" identically to "fourth
by sixty". AlphaZero could afford a binary outcome because it had 10^7-10^8
games. This project will have ~10^5. Under that budget a binary target is a
credit-assignment disaster: for the first two-thirds of a 27-round game the
label is essentially independent of the position, so the value head learns the
prior and the search gets a flat landscape to climb. **Do not use it as the
primary target.**

**Raw score is also wrong.** Scores run 50-150 in real play and go negative
under bad play (`README.md`: mean -12.6 for the sampling policy, -24.9 for
uniform). Most of that variance is "how far into the game are we", which the
`day` feature already answers, and MSE on a wide heteroscedastic target
conditions badly.

**Primary target: `z_rel`.**

```
z_rel[i] = tanh( (score[i] - mean(score)) / 25 )
```

- **Centred**, so it is (approximately) zero-sum and max^n backups have a single
  currency: taking a point from the leader and gaining one yourself are the same
  move. Tzolk'in is close to constant-sum in practice — temples, buildings and
  monuments are all contested — so centring loses very little.
- **Bounded** in (-1, 1), so PUCT's `c_puct` calibrates the way it does in the
  literature (use 1.5-2.5).
- **Margin-preserving.** `T = 25` is roughly the observed spread of final scores
  around the mean in a real game, so the transform is near-linear through the
  bulk of the distribution and saturates only on blowouts. That gives the search
  a smooth gradient to climb from round 1, which the rank and win-probability
  targets do not.

At inference, subtract the mean: `rel ← rel - mean(rel)`. Cheap, and it enforces
the invariant the search relies on without constraining the optimisation.

**Secondary heads, trained jointly, all cheap and all worth their weight:**

- `score[i] = (final_score[i] - 75) / 50`, clipped to [-2, 2]. The dense
  absolute signal. Huber loss, not MSE — early self-play produces wild outliers.
- `rank[i][r]`, cross-entropy against the one-hot finishing place (ties split
  evenly). Sharpens endgame discrimination and gives `P(win) = rank[i][0]` for
  free, which §6.5 uses for reporting.
- `decomp[i][c]`, the end-of-game decomposition of each player's score into six
  components: temple points, building points, monument points, corn conversion,
  skull VP, starvation penalty. All six are computable from `end_game`
  (`game.rs:~430`) and `gain_temple_points`. **This is the highest-value-per-byte
  head in the design.** It is dense, it is available on every terminal position
  for free, and it forces the trunk to represent *why* a position is worth what
  it is worth rather than only *how much*. This is the KataGo lesson — their
  auxiliary ownership and score heads are credited with a large multiple in
  training efficiency — applied to the one auxiliary structure this game offers.

**What the search should back up: `rel`.** Say so in `SEARCH.md`.

---

## 3b. Two decision nodes the engine grew while this was being written

`src/` moved under me. Both additions fit the design without changing it, but
they need heads, so specify them now rather than bolting them on later.

**Pity** (`moves.rs:40`, `MoveKind::Pity`). Available only when nothing else is,
and its variants are the equal-cheapest spaces. It is the third slot of the
`Mode` head; when it is legal, nothing else is, so the head is degenerate there
and the only real decision is *which* cheapest space — reuse the `Place` head's
7-way softmax masked to the tied-cheapest set. **No new head.**

**Tile drafting** (`game.rs:83`, deal four keep two, currently drawn at random).
This is a genuine decision, it happens once per player per game, and it is worth
several points. It needs its own head because its action is "pick 2 of 4 dealt
tiles" over a 21-entry table (`tiles.rs:16`), which nothing else in the design
covers.

```
Phase::DraftTile { dealt: [u8; 4], kept: u8 }   // one pick at a time, twice
head: pointer over the 4 dealt tiles, keyed by the same `encode_choice`
      delta features applied to the tile's `&[Effect]`
```

Reusing `encode_choice` here is the payoff for having made it a state-delta
function: a starting tile is just a `Choice` that happens not to come from a
board space. **No new feature code, ~1 k parameters for the query projection.**

Note that at draft time the state is nearly empty, so this head is learning
almost a context-free ranking of 21 tiles. That is a small, very learnable
function and it will converge in the first few generations.

---

## 4. Architecture and size

### 4.1 The shape

A factored stem into a plain pre-norm residual MLP trunk. No convolutions
(§1.2), and **no attention**: the only relational computations worth having are
over four players and twelve card slots, and those are cheaper and more reliably
learned as explicit features (§1.3) than as a 4-token attention block. One week
is not the budget in which to debug attention.

```
                     +-- board 600 ---------------> Linear -> 192 ------+
                     |                                                  |
  encode(g,p,phase)  +-- player 282 x4 --> SHARED Linear -> 96 -> 384 --+
      [f32; 3072] ---+                                                  +--> concat 1184
                     +-- bldg slot 73 x6 -> SHARED Linear -> 48 -> 288 -+        |
                     |   mnmt slot 45 x6 -> SHARED Linear -> 32 -> 192 -+        |
                     +-- global rest 248 ---------> Linear -> 128 ------+        |
                                                                                 v
                                        Linear(1184 -> 512) + LayerNorm + GELU
                                                                                 |
                          6 x [ LN -> Linear(512,512) -> GELU -> Linear(512,512) ] + skip
                                                                                 |
                                                                          h : R^512
                                                                                 |
     +--------+--------+---------+-----------+--------------+-------------+------+
     v        v        v         v           v              v             v
   value     beg     mode     place     retrieve_who   retrieve_what   extra_day
  (+ score, rank, decomposition)                       (2-stage pointer)
```

### 4.2 Parameter count

| Component | Params |
|---|---|
| board stem `600 -> 192` | 115,392 |
| player stem `282 -> 96`, shared x4 | 27,168 |
| building-slot stem `73 -> 48`, shared x6 | 3,552 |
| monument-slot stem `45 -> 32`, shared x6 | 1,472 |
| global stem `248 -> 128` | 31,872 |
| fusion `1184 -> 512` + LN | 606,720 |
| **stem total** | **786,176** |
| 6 x residual block (2 x 512x512 + 2 LN) | 3,158,016 |
| value / score / rank / decomposition heads | ~145,000 |
| beg + mode + place + extra_day | ~9,000 |
| retrieve_who `512 -> 51` | 26,163 |
| retrieve_what: `q_deep`, `q_fast`, `MLP_key` | ~100,000 |
| **total** | **~4.22 M** |

**8.4 MFLOP per forward pass.** That is the number that matters, not the
parameter count.

### 4.3 Why this size, and a ladder

Four million parameters against maybe 10^6 self-play games (§7) is
over-parameterised by the usual supervised-learning heuristic, but AlphaZero
targets are far richer than scalar labels: each position carries a full visit
distribution over its legal set plus four value targets plus 24 decomposition
targets. KataGo's workhorse nets sit at 3-6 M and are trained on comparable data
volumes.

The real risk is not overfitting, it is **wasting the first two days training a
big net on a tiny buffer**. So use a ladder:

| stage | width | blocks | params | when |
|---|---|---|---|---|
| `small` | 384 | 4 | ~1.9 M | warm-start + generations 1-30 |
| `main` | 512 | 6 | ~4.2 M | generation 30 onward |

Transfer by re-initialising: keep nothing, retrain `main` from scratch on the
accumulated buffer for ~4,000 steps (about 6 minutes) and swap. Net-surgery
transfer is not worth the bugs at this scale.

### 4.4 Node rate inside the search — measured, not guessed

Benchmarked on this machine (M4 Pro, macOS 15.3.1), fp32 SGEMM through
Accelerate:

| shape | latency | GFLOP/s |
|---|---|---|
| 8 x 256 x 256 | **3.9 µs** | 271 |
| 64 x 256 x 256 | 6.8 µs | 1,226 |
| 256 x 256 x 256 | 30.3 µs | 1,109 |
| 1024^3 | 1.17 ms | 1,829 |
| 2048^3 | 7.02 ms | 2,447 |

**~1.1-2.4 TFLOP/s with a ~4 µs call latency.** At 8.4 MFLOP per evaluation and
the ~1.1 TFLOP/s the 512-ish shapes actually reach, that is **~130,000 evals/s**
of pure GEMM, or **~65,000 evals/s** after derating 50% for LayerNorm, GELU, the
heads, im2col-free but still real memory traffic, and contention with the search
threads.

**This is the number that vindicates §1.2.** A conv trunk of the same parameter
count over 50 board cells would cost `2 x params x cells`, not `2 x params` —
a 50x multiplier, putting a 2 M-parameter conv net at roughly 0.2 GFLOP/eval and
**~3,000 evals/s**. The flat MLP is 20-40x cheaper *for the same capacity*, on
top of being the better inductive bias. Choosing flat over convolutional here is
not a stylistic preference; it is the difference between a usable evaluator and
an unusable one.

The batching contract matters more than any of the above, so state it as an
interface requirement:

- **The search must accumulate at least 64 pending leaves before calling.** Use
  virtual loss and parallel descents. The 3.9 µs call latency is fixed, so at
  batch 8 you spend it on 8 evaluations and at batch 256 on 256.
- Target batch **128-256**, with a **200 µs** maximum queue wait so a search
  thread never stalls behind a half-full batch.
- Run **one or two dedicated inference threads**. Apple's AMX blocks are shared
  per performance-core cluster, so two servers can occupy both clusters while N
  search threads stay free for move generation. Set `VECLIB_MAXIMUM_THREADS=1`
  so Accelerate does not spawn its own pool and fight the search threads.

---

## 5. Rust ML stack

### 5.1 The recommendation

**Split the work, and do not use a Rust training framework.**

| stage | what | why |
|---|---|---|
| inference in search | **hand-written Rust over Accelerate `cblas_sgemm`** | ~20 GEMM calls for the whole net; 3.9 µs call latency, 1.1-2.4 TFLOP/s, zero dependencies, deterministic latency, trivially thread-safe |
| training | **PyTorch, in Python, on the CPU** | ~26 GFLOP per step at batch 1024 — under 100 ms. The model is far too small to need a GPU, and every GPU path on this machine carries a correctness or maturity risk |
| batching for training | **a Rust `batcher` binary** reading replay shards, emitting pre-encoded shuffled batches | reuses the *exact* inference encoder, which is the correctness property that matters most; no PyO3, no GIL |
| weight interchange | **safetensors**, both directions | PyTorch writes it natively; the `safetensors` crate reads it with no C dependency |

The forward pass is six residual blocks of two 512x512 matmuls plus a factored
stem and small heads. That is roughly 20 `cblas_sgemm` calls, three elementwise
kernels (LayerNorm, GELU, masked softmax), and a bias add. Call it 250 lines.
Against that, every framework here is a net *addition* of risk.

### 5.2 Why not each of the alternatives

**candle 0.11.0 — rejected, on two specific grounds.** Its Metal GEMM was 1.9x
to 11.3x slower than MLX/PyTorch-MPS until [PR #3313](https://github.com/huggingface/candle/pull/3313)
(merged Jan 2026) tuned the tile config; that fix still early-returns the
untuned fallback tile for `m < 16`, which is small-batch, which is us. More
seriously, [issue #2659](https://github.com/huggingface/candle/issues/2659)
reports repeated Metal forward passes degrading from ~2 ms to 300-500 ms after
about 17 iterations — open, unanswered, and describing *exactly* an MCTS access
pattern. And `candle-nn`'s entire optimiser module is SGD (documented as *not*
supporting momentum) and AdamW, with **no LR schedulers at all**. That is too
bare for this training loop.

**burn 0.21 — the real all-Rust alternative, and the one to pick if you insist
on no Python.** It is the only Rust framework with a genuine training story: ten
optimisers (including AdamW, LAMB, Muon), eight LR schedulers, a `Learner` with
metrics and checkpointing, a native-MSL `metal` backend, and an early `burn-rl`
crate. The costs are real: no 1.0, a README that still promises breaking
changes, and **no published Apple Silicon benchmarks anywhere** — you would be
the one measuring it. Take this trade if the Python boundary is what you most
want to avoid; otherwise the split above is lower-risk.

**tch-rs 0.26 — no.** `Device::Mps` exists but is a thin passthrough with no
`is_available()` helper, and whether the official `libtorch-macos-arm64-2.13.0`
build (85 MiB compressed, several hundred MB expanded) ships MPS compiled in
could not be confirmed. Enormous dependency for a 4 M-parameter MLP.

**ort 2.0.0-rc.13 / ONNX Runtime + CoreML — the best *inference-only* option, and
still not worth it here.** It is the most mature crate in this survey by a
distance (17.6 M downloads, two open issues), its CoreML EP is properly wrapped
(`MLProgram` format, `CPUAndNeuralEngine`, static input shapes, a model cache
directory), and if the net were convolutional I would take it. But the ANE has a
**~0.2-0.4 ms floor per CoreML call**, of which most is XPC/IOKit dispatch, and
it is fp16-only with no way to force placement. At our batch sizes and with a
dense MLP — the ANE's weakest shape — that floor is a wash against Accelerate's
3.9 µs. Revisit only if the net ever becomes convolutional.

**PyTorch MPS — actively avoid.** Not for speed but for correctness: there is a
standing pattern of small nets that train fine on CPU and **silently fail to
learn on MPS with no error** ([#137964](https://github.com/pytorch/pytorch/issues/137964)),
plus worse loss/perplexity than CPU ([#92615](https://github.com/pytorch/pytorch/issues/92615),
[#109457](https://github.com/pytorch/pytorch/issues/109457)). A 26 GFLOP training
step does not need a GPU, so there is nothing to buy with that risk. Train on
CPU. If you ever do enable MPS, diff its loss curve against CPU for the first
few hundred steps before trusting it.

**MLX — no usable Rust binding.** MLX itself is thriving (v0.32.2, very active),
but `mlx-rs` last published **0.25.3 in December 2025**, nine months stale and
seven minor versions behind upstream. Not a foundation for a one-week project.

### 5.3 One operational warning

**Do not upgrade macOS mid-project.** This machine is on 15.3.1. There are open
reports of `torch.mps.is_available()` returning false on macOS 26 across PyTorch
2.9-2.12 ([#167679](https://github.com/pytorch/pytorch/issues/167679),
[#177819](https://github.com/pytorch/pytorch/issues/177819)), still unresolved.
The recommended stack does not use MPS, so this is a low risk — but a mid-run OS
upgrade is a needless variable in a week that has none to spare.

### 5.4 When to reach for the GPU

One condition, and it is a real one: **if the search can reliably deliver
batches of 256 or more.** At batch 8-64 a Metal round trip (~1 ms, measured on
comparable hardware) costs more than the entire trunk does on the CPU
(0.7-1.4 ms for the whole thing), so the GPU cannot win. At batch 256 the
dispatch amortises and Metal should give 2-3x, which would take self-play from
~5,000 to ~12,000 games/hour.

Keep the forward pass behind the `Evaluator` trait so this is a contained swap,
and only make it if a profile shows the network above 40% of self-play wall
time. It probably will not be.

---

## 5b. The interface with the search half

```rust
pub const D_IN:     usize = 3072;
pub const D_CHOICE: usize = 96;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    DraftTile { dealt: [u8; 4], kept: u8 },
    Beg,
    Mode,
    Place        { placed: u8, corn_spent: u8 },
    RetrieveWho  { retrieved: u8 },
    RetrieveWhat { gear: Gear, pos: Pos, retrieved: u8 },
    ExtraDay,
}

pub struct Request<'a> {
    pub g: GameState,
    pub p: PlayerId,
    pub phase: Phase,
    /// Legality for the fixed-arity heads. Ignored for RetrieveWhat/DraftTile.
    pub legal: LegalMask,
    /// The caller's candidate list. Required for RetrieveWhat and DraftTile,
    /// ignored otherwise. Must be in `options::dedup` order.
    pub candidates: &'a [Choice],
}

pub struct Eval {
    pub value: Value,     // see §3.1
    pub priors: Priors,   // see §2.2; already masked and normalised
}

pub trait Evaluator: Send + Sync {
    /// The signature you asked for. Equivalent to `evaluate_at` at a turn
    /// boundary with an all-legal mask and no candidates.
    fn evaluate(&self, g: &GameState, p: PlayerId) -> Eval;

    /// The general form. `p` need not be `g.current`.
    fn evaluate_at(&self, r: &Request<'_>) -> Eval;

    /// **Use this one.** One trunk pass over `reqs.len()` positions, results in
    /// the same order. A search that calls `evaluate` one leaf at a time runs
    /// at roughly a twentieth of the batched rate — see §4.4.
    fn evaluate_batch(&self, reqs: &[Request<'_>], out: &mut Vec<Eval>);

    /// Value only, no policy heads. ~15% cheaper, and it is what a terminal or
    /// near-terminal node wants.
    fn value_only(&self, g: &GameState, p: PlayerId) -> Value;
}
```

**Three things the search half needs to know:**

1. **`evaluate(&GameState, PlayerId)` alone is not sufficient**, and this is the
   one place I have to amend the agreed signature. A mid-turn state — two
   workers placed, deciding the third — is a perfectly good `GameState`, but
   nothing in it says whether we are choosing a gear, choosing a worker to
   retrieve, or choosing that worker's action. `Phase` supplies that. It is a
   `Copy` enum; threading it through costs nothing.
2. **`value.rel` is what you back up.** Index 0 is the querying player, index k
   the player k seats clockwise. It is centred and bounded in (-1, 1), so
   `c_puct` in 1.5-2.5 is calibrated. One forward pass gives you all four
   players' values, which is what a max^n backup needs.
3. **Candidate order must be stable.** The replay format stores visit counts as
   indices into a regenerated candidate list (§6.2), so `choices_for_worker`'s
   order — currently guaranteed by `options::dedup` sorting — is load-bearing
   for training, not just for reproducibility. Please do not reorder it in the
   search layer.

---

## 6. The training loop

### 6.1 Shape: synchronous generations

```
  for gen in 0.. :
      1. self-play    ~30 min, all cores, current weights            -> replay/gen_NNNN.bin
      2. train         ~5 min, ~3,000 steps over the sampling window -> ckpt/gen_NNNN.safetensors
      3. every 10 gens: anchor panel (~10 min)                       -> eval/gen_NNNN.json
      4. atomically rewrite manifest.json
```

Not asynchronous. Continuous training as KataGo does it is more sample-efficient,
but synchronous generations are far easier to reason about, to checkpoint, and to
resume after a crash — and with a ~35-minute cycle you still get **~200
generations in a week**, which is more than enough policy-improvement steps. The
sample-efficiency gap is small compared to the risk of losing a day to an async
staleness bug.

### 6.2 What a self-play worker records

**Raw `GameState`s, not encoded tensors.** Each recorded decision is:

```rust
#[repr(C)]
pub struct Record {
    pub state: GameState,        // ~340 bytes, Copy, no heap
    pub p: PlayerId,
    pub phase: Phase,
    pub visits: [(u16, u16); 24], // (action index, visit count), top 24, 0-terminated
    pub total_visits: u32,
    pub gen: u16,
}
// game-level trailer, backfilled at game end:
//   final_scores: [i16; 4]
//   decomposition: [[i16; 6]; 4]
```

~400 bytes per record. This costs one design decision and buys three:

- **You can change the encoder and retrain on old games.** Given that `src/`
  gained a skull supply, a pity rule, a tile draft and a 27th round *while this
  document was being written*, that is not a hypothetical. Storing encoded
  tensors would mean throwing away every game whenever a feature is added.
- **The legality mask is regenerated, not stored.** Masked cross-entropy (§2.4)
  needs the mask; re-deriving it costs one `lowest_free` sweep or one
  `choices_for_worker` call per sample, in the data-loader threads.
- **The `RetrieveWhat` candidate list is regenerated too.** Storing it would be
  ~500 candidates x 96 floats per sample; regenerating costs ~35 µs. There is no
  contest.

Storing `visits` as (index, count) pairs rather than a dense distribution keeps
the record fixed-size even for the pointer head, where the index is a position in
the regenerated candidate list. That requires candidate generation to be
**deterministic**, which it is — `choices_for_worker` ends in
`options::dedup`, which sorts (`options.rs:372`), and `visit_legal_moves`
documents its order as deterministic (`moves.rs:107`). Add a regression test that
pins this; the whole replay format depends on it.

### 6.3 Which nodes get written

Not all of them. A turn-start node may have 800 visits; the "which choice for the
third retrieved worker" node three levels down may have 4, and a 4-visit
distribution is noise.

- Write a record only when `total_visits >= 16`.
- Weight its policy loss by `log(1 + total_visits) / log(1 + 800)`.
- Always write the root of each turn regardless.

### 6.4 Exploration and the policy target

- **Dirichlet noise** at every decision-node root, `alpha = 10 / |A|` clamped to
  `[0.03, 1.0]`, mixed at `epsilon = 0.25`. `|A|` varies from 2 (extra day) to
  500+ (`RetrieveWhat`), so a fixed alpha is wrong here in a way it is not in
  chess.
- **Temperature** on move selection: 1.0 for rounds 1-9, 0.5 for 10-18, 0.1 for
  19-27. (Not "the first 30 moves" — this game is 27 rounds x 4 turns x ~3
  sub-decisions, so move counts do not transfer from chess.)
- **Playout cap randomisation** (KataGo). 25% of turns get the full budget and
  produce training records; 75% get ~1/8 the budget and produce none. Roughly
  doubles games per hour at constant target count. **Take this; it is the single
  cheapest throughput win available.**
- **Forced playouts and policy-target pruning** (KataGo). Guarantee each child
  `sqrt(k * P_i * N)` visits, then subtract those forced visits back out of the
  target before writing it. Without this, Dirichlet noise leaks directly into the
  policy target and the policy slowly flattens.
- **4x perspective augmentation on the value heads.** Every recorded position is
  encoded four times, once per querying player, and each gives a full value
  target. This is legitimate precisely because `encode` takes `p` independently
  of `g.current` (§1.1) — and it trains exactly the off-turn query that a max^n
  backup makes. It is this domain's substitute for Go's 8-fold dihedral
  augmentation, and it is the only one available.

### 6.5 Loss

```
L =   1.0 * CE_masked(pi_beg)            * w(visits)
    + 1.0 * CE_masked(pi_mode)           * w(visits)
    + 1.0 * CE_masked(pi_place)          * w(visits)
    + 1.0 * CE_masked(pi_retrieve_who)   * w(visits)
    + 1.5 * CE(pi_retrieve_what, stage 2) * w(visits)
    + 0.5 * CE(pi_retrieve_what, stage 1) * w(visits)     # cheap scorer tracks the deep one
    + 0.5 * CE_masked(pi_extra_day)      * w(visits)
    + 0.5 * CE(pi_draft)                 * w(visits)
    + 1.5 * Huber(v_rel,   z_rel)
    + 0.5 * Huber(v_score, z_score)
    + 0.3 * CE(v_rank,     one-hot rank)
    + 0.2 * Huber(v_decomp, component scores)
```

Only the head matching the record's `phase` contributes a policy term; the value
terms contribute on every record. Huber (smooth L1) rather than MSE throughout —
early self-play produces scores in the -70..+40 range (`README.md`), and MSE lets
those outliers dominate the gradient.

### 6.6 Optimiser and schedule

- **AdamW**, not SGD+Nesterov. AlphaZero used the latter, but it needs LR tuning
  this project has no time for; AdamW is forgiving and converges faster on small
  nets.
- LR **constant at 2e-3** for generations 1-60, **1e-3** for 60-150, **3e-4**
  after. Constant-with-drops rather than cosine, because cosine assumes a known
  horizon and the run will be stopped whenever it is stopped.
- Weight decay **1e-4**, excluding LayerNorm parameters and biases.
- Gradient clip at global norm **1.0**.
- 200-step linear warmup after every LR change and after the net-size swap.
- Batch **1024**. Sample uniformly from a window of the **last 20 generations**,
  capped at **1.5 M records** (~600 MB resident; the machine has 24 GB).
- ~3,000 steps per generation. That presents ~3 M samples against a buffer of
  ~1.5 M, i.e. each record is seen roughly twice per generation and ~40 times
  over its 20-generation lifetime. That is the standard ratio and it is a knob
  worth watching: if training loss falls while anchor-panel strength does not,
  cut the steps.

### 6.7 Warm start — do this before generation 1

This is the highest-leverage two hours in the whole plan, and it is available
only because the engine is already fast.

**Value warm-start (~1 hour).** `run_sampled` does 7,559 playouts/s
single-threaded; on 10 cores that is ~270 M playouts/hour. Sample 20 positions
along each of 500,000 playouts and regress the value heads on the resulting
`(GameState, final scores)` pairs. The result predicts "final score under random
continuation", which is biased but strongly informative — it captures that corn,
resources, temple position and workers-in-play are worth something. A
randomly-initialised value head captures nothing.

**Policy warm-start (~30 min).** Run plain MCTS with random rollouts and no net,
~200 simulations, over ~20,000 positions drawn from those playouts. Train the
policy heads on the visit distributions.

**Then gate on it.** If the warm-started net does not beat `sample_legal_move`
by a wide margin in a 200-game match, something is wrong with the encoder, the
target, or the loss. Finding that on day 1 rather than day 5 is the entire point
of doing this first.

### 6.8 Evaluating whether a new net is stronger

**Do not gate.** Accept every generation and keep training; gating burns games,
and games are the scarce resource. Keep a rolling "best by anchor score" pointer
so a bad stretch can be rolled back.

**Do measure, against a fixed panel, every 10 generations.** Four-player games
are markedly less transitive than two-player ones: A beats B beats C beats A is
common, and a loop that only ever compares to the previous champion can walk in
a circle while every comparison says "improved". The fixed panel is the defence.

| anchor | what it catches |
|---|---|
| `sample_legal_move` rollout policy | absolute floor; must never be lost to |
| plain MCTS, random rollouts, 400 sims, no net | is the net beating a search that ignores it? |
| generation-0 warm-started net | monotone progress from the supervised prior |
| best net so far by this metric | the actual champion comparison |

**Metric: mean centred score**, `mean(score_i - mean(score))`, not win rate.
With four players and a few hundred games, win counts are far noisier than
margins — the same argument as §3.2, applied to measurement. Report win rate
too, from the `rank` head's own prediction alongside the observed one; a growing
gap between them is a calibration warning.

**Seat balancing.** For a 2-vs-2 match there are six distinct seatings of
`{A,A,B,B}` around a fixed turn order. Play equal numbers of each. Turn order in
Tzolk'in confers a real advantage (the first-player marker, placement costs), so
an unbalanced match measures seats, not agents. Also run 1-vs-3 configurations:
they are the ones that expose an agent that has only learned to play against
copies of itself.

**Sample size.** Centred score has a per-game standard deviation of roughly 25.
Detecting a 3-point edge at p<0.05 needs on the order of 550 games. Budget
~600 games per panel run, ~10 minutes at self-play rates with search budget
reduced for evaluation.

### 6.9 Checkpointing and interruption

The run will be interrupted. Design for it:

- Self-play workers append length-prefixed records to `replay/gen_NNNN.bin.part`
  and `fsync` every 32 completed games. A `SIGINT` handler finishes the in-flight
  game, flushes, and exits; worst case loses 32 games.
- After training, write `ckpt/gen_NNNN.safetensors` and
  `ckpt/gen_NNNN.opt.safetensors` (Adam moments — without these a resume
  silently restarts the optimiser and loses a generation's progress).
- Update `manifest.json` by write-to-temp-then-`rename`, so it is never observed
  half-written.
- A resume reads the manifest, discards any `.part` shard newer than the last
  completed generation, and continues. Total worst-case loss from a crash or a
  laptop lid: **one generation, ~35 minutes.**

---

## 7. A realistic one-week schedule

### 7.1 The arithmetic

Per game, with the factorisation of §2.1:

| quantity | value |
|---|---|
| rounds x players | 27 x 4 = 108 turns |
| decision nodes per turn | ~3 (mode, then 1-3 place-steps or who/what pairs) |
| decision nodes per game | ~350 |
| simulations per turn, full budget | 800 |
| with playout cap randomisation (25% full, 75% at 100) | ~275 average |
| **network evaluations per game** | **~30,000** |

The measured ceiling from §4.4 is ~65,000 evals/s for the whole machine, so:

| | games/hour |
|---|---|
| NN-bound ceiling, machine does nothing else | 7,800 |
| NN gets ~60% of the machine, generation the rest | **~4,700** |
| plus a 2-3x Metal win at batch >= 256 (§5.4) | ~12,000 |

**Plan on 3,000-6,000 games/hour**, with the honest caveat that this is a **±2x
estimate**. The dominant uncertainty is not the network — that is now measured —
but the cost of expanding a `RetrieveWhat` node, which ranges from a fraction of
a microsecond on a Palenque entry space to several hundred on a Tikal 4 double
build with six face-up cards (`tikal.rs:43` calling `building_choices` twice).
Node expansion happens once per *new* tree node, not once per simulation, which
is what keeps it affordable; measure it before trusting anything here.

The second lever is `sims/turn`. Dropping the full budget from 800 to 600 takes
evaluations per game from 30,000 to ~22,000 and games/hour from ~4,700 to
~6,400. That trade — a third more games for a quarter less search — is probably
worth taking early, when the net is weak and the search is doing most of the
work, and worth reversing late.

### 7.2 The week

| day | what | games |
|---|---|---|
| 0-1 | encoder, `Evaluator`, replay format, `evaluate_batch`, plumbing into the search half | 0 |
| 1 (pm) | warm start (§6.7) and the day-1 gate | supervised only |
| 2-6 | ~110 hours of self-play generations | 330k-660k |
| 7 (am) | final anchor panel, checkpoint selection, write-up | ~5k |

**Total: 300,000-600,000 self-play games**, and that assumes days 0-1 suffice to
build and debug the pipeline. Realistically they will not, so the honest range is
**200,000-600,000**.

That is roughly twice what I estimated before measuring Accelerate on this
machine. The measurement is the reason: 1.1-2.4 TFLOP/s through AMX is a great
deal more than a laptop CPU has any right to, and a flat MLP is the one
architecture that can spend it on evaluations rather than on spatial positions.

### 7.3 Is that strong play? No.

Asked directly, and answering directly: **this will not produce the strongest
play achievable, and one week on this machine is not close to enough for that.**
It will produce an agent that crushes random play, comfortably beats a plain
rollout MCTS, and is plausibly competitive with a decent club player. It will
not be near-optimal, and it will have identifiable holes in long-horizon
planning.

The reasons are specific, not general pessimism:

1. **Reference scale.** AlphaGo Zero used 4.9 M games; AlphaZero chess 44 M.
   Half a million is an order of magnitude short of the smaller of those, on a
   game that is *harder* along every axis that matters.
2. **The successful laptop-scale replications are not this game.** Connect Four
   and 6x6 Othello reach near-optimal on 10k-100k games — with branching under
   100, games under 50 moves, two players, zero sum, perfect information.
   Tzolk'in has p99 branching of 1,261, 108 turns, four players, non-zero-sum
   scoring, and a hidden deck.
3. **Credit assignment over a very long horizon.** A worker placed on Tikal's
   entry space pays off five rounds later; a temple investment pays at rounds 14
   and 27. That is the hardest thing for a value function to learn and it is
   exactly what half a million games does not buy.
4. **Four-player non-transitivity.** Self-play in a non-zero-sum four-player
   game can cycle rather than climb. §6.8's anchor panel detects it; nothing in
   the budget fixes it.
5. **No symmetry augmentation.** Go gets 8x free from the dihedral group. This
   game gets 4x on the value head only (§6.4) and nothing on the policy.
6. **The action space is the hard part.** The `RetrieveWhat` head has to learn
   which of 32 buildings, at which payment split, in which board context, is
   worth taking — and each of those appears in a small fraction of games.

A concrete, defensible target for the week: **beats the rollout-MCTS anchor in
better than 80% of 4-player matches, and beats the warm-started supervised net
by more than 10 points of mean centred score.** That is achievable. "Strongest
play achievable" is not, and no amount of cleverness in this document changes
that — only more games do.

### 7.4 What would actually get to strong play

In descending order of leverage per dollar and per hour:

1. **Buy self-play compute.** This workload is CPU-bound and does not GPU-
   accelerate: the network is a dense MLP that runs happily on AMX, and the
   other half is move generation, which is branchy pointer-chasing. Sixteen
   32-vCPU spot instances for three days is ~1,150 core-days against this
   machine's ~60 — **roughly 20x the games for a few hundred dollars.** Nothing
   else on this list is close. The self-play binary is already the right shape
   for it: weights in, replay shards out, no shared state.
2. **Freeze the rules first.** See §8. This costs nothing and is currently the
   largest live risk to the week.
3. **Run for a month, not a week.** Same machine, ~200 generations/week; 800
   generations is a materially different agent.
4. **Push harder on auxiliary supervision.** The score-decomposition head (§3.2)
   is the cheap version. The expensive version — predicting each opponent's next
   `MoveKind` and gear choice, and predicting temple standings eight rounds out
   — is more of the same medicine, and it is what makes small-data AlphaZero
   work at all.
5. **Blend rollouts into the leaf value while the net is weak.** See §7.5.

### 7.5 The hybrid that makes the first three days usable

While the value head is bad, an unbiased noisy estimate beats a confidently
wrong one. AlphaGo Lee mixed the two and so should this:

```
V_leaf = (1 - lambda) * v_rel  +  lambda * centred_score(truncated_playout)
```

A truncated playout — run `take_turn_sampled` for 5 rounds, then evaluate the
resulting position with the net — costs ~20 sampled turns at 1.2 µs each plus
one network evaluation, so roughly **2.5x a bare evaluation**, not the 9x a full
playout would cost. Schedule `lambda` from 0.4 at generation 1 to 0 by
generation 60.

Before committing to this, run `src/bin/rollout_quality.rs`. It exists to
measure exactly the thing that decides the question: the spread of per-move
rollout means against the standard deviation of a single rollout. If the
signal-to-noise it reports is very low, rollouts cannot rank moves and `lambda`
should start small; if it is respectable, start it high.

---

## 8. Rules coupling

The user asked whether this work is rules-agnostic. Plainly:

### 8.1 Survives any rules change

- The **value head design** entirely. It depends on "four players, integer final
  scores" and nothing else.
- The **training loop**: replay format, buffer window, schedule, optimiser,
  gating policy, anchor panel, checkpointing. All of it.
- The **`RetrieveWhat` pointer head**, because `encode_choice` measures state
  deltas rather than parsing `Effect` variants (§2.3). New effects, changed
  yields and changed prices all flow through unchanged.
- The **perspective rotation** and the **player-shared stem**.
- The **ML stack** and the batching contract.
- The **warm-start procedure**, which calls the engine rather than encoding
  knowledge about it.

### 8.2 Does not survive

- **`D_IN` is pinned to the state struct.** A new field means new features means
  a new input width means a re-initialised first layer and a discarded
  checkpoint. Mitigation: **388 reserved zero dimensions** in the encoder
  (§1.1), plus bitset padding (32 buildings -> 40, 13 monuments -> 16, day
  one-hot sized 32 for a 27-round calendar). New scalars go in the reserved tail
  and the checkpoint still loads. This is cheap insurance and given the rate
  `src/` is moving, it will get used.
- **Block A's 55 cells** break if the number of gears or their sizes change.
  They will not.
- **The fixed heads' arities** break if the action structure changes — a sixth
  gear would change `Place` from 7-way to 8-way. Also will not happen.
- **The value function itself** is invalidated by any change to tempo or
  scoring, even when every tensor shape survives. This is the important one and
  §8.4 is about it.

### 8.3 The five changes named as in-flight — all of which landed mid-writing

| change | cost to this design |
|---|---|
| **13-skull supply limit** (`state.rs:247`, `skulls_remaining`) | 15 features in block C, from the reserved tail. Also makes `Effect::Res(Skull, +n)` non-deterministic in magnitude, since `take_skulls` clamps (`state.rs:355`) — which is a second argument for delta-based choice features, because they measure what actually happened rather than what was requested. **No architecture change.** |
| **"Do nothing" legal everywhere** (`moves.rs:361`) | One extra candidate, whose delta vector is all zeros. The only requirement is that the choice feature vector can *represent* nothing distinguishably: an all-zero key would score zero regardless of context. Feature 0 is a constant `1.0` bias and feature 1 is an explicit `is_skip` flag for exactly this reason (§2.3). **No code change; the design already accounted for it.** |
| **Two-way market** (`options.rs:303`) | Signed deltas already. **Zero change.** It does inflate the Uxmal-2 candidate count considerably, which is what the two-stage pointer head in §2.3 is there to bound. |
| **Extra day as an explicit node** (`game.rs:286`, `may_take_extra_day` / `take_extra_day`) | `Phase::ExtraDay`, a 2-way head, ~2 k parameters. Specified from the start, so it costs nothing. |
| **26 -> 27 rounds** (`state.rs:259`) | Day one-hot sized 32, so the tensor survives. **The learned function does not** — see below. |

Two more that landed while this was being written and were not on the list:
**pity moves** (`moves.rs:40`) and **starting-tile drafting** (`game.rs:83`).
Both are handled in §3b, both at negligible cost, and the fact that they were
absorbable is the argument that the factorisation in §2 is the right one.

### 8.4 The thing to actually worry about

Tensor shapes are the easy part. **A rules change invalidates the learned value
function even when every shape survives.**

The calendar change is the clearest case. Going from 26 to 27 rounds with food
days moved from `{7,13,20,26}` to `{8,14,21,27}` (`state.rs:259-263`) changes
the tempo of the entire game: how many rotations a worker gets, when to commit
to a temple, when the age-1 prize lands. A value head trained on the old
calendar has learned a different game, and no amount of reserved input
dimensions helps. The same is true of the exclusive temple top
(`state.rs:277`, `temple_ceiling`), the skull supply, and accumulated corn arriving
once per round rather than once per rotation (`game.rs:256`).

**So: do not start the real run until the rules are frozen.** `RULES-AUDIT.md`
lists 17 violations, 8 missing rules and 5 edge cases; the README now says all
are fixed, and the code confirms the ones I checked. Good. But `src/` changed
under me four times in the course of writing this document, and a week of
compute spent against a moving target is a week wasted.

Concretely:

- **Build the pipeline now, against whatever the rules currently are, and treat
  that run's weights as disposable.** The point of days 0-1 is to find the
  encoder bugs, the batching stalls and the replay-format mistakes, not to
  produce a strong net.
- **Tag the commit the real run starts from** and record its hash in
  `manifest.json`. When the rules move again — and they will — you want to know
  exactly which games were played under which rules, so the buffer can be
  truncated rather than thrown away.
- **Add a `RULES_VERSION` constant** bumped by hand on any change that alters
  play, stamped into every `Record`, and have the data loader drop records whose
  version predates the current one. Ten lines, and it turns "the whole buffer is
  suspect" into "the last 40 generations are still good."

That last item is the single cheapest piece of insurance in this document and
I would write it before writing the encoder.
