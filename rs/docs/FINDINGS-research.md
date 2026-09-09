# Research: are the four tracks implemented correctly?

> Audit of `src/research.rs`, its consumption sites in `src/spaces/*` and
> `src/options.rs`, against the CGE rulebook. HEAD at start: `941ca8b`,
> 186 tests passing.
>
> Prompted by `docs/FINDINGS-eval.md` F50-F53: raising `RESEARCH_SCALE` makes
> the agent research exactly as theory predicts and it **loses 4.33 points**
> [-4.86, -3.79]. The evaluator prices research correctly *for the rules as
> implemented*. This file asks whether the rules as implemented are right.

## Source of truth

The audit trail in `RULES-AUDIT.md` cites the CGE English rulebook (© Oct 2012)
but the repository has no copy. One was fetched during this pass and the
technology pages extracted; the quotations below are from p.7 (advance costs),
p.9 (Tikal / Uxmal actions) and p.12 (the four tracks). A scratch copy of the
extracted text lives outside the repo and is not committed.

The Go original (`src/model/research.go`, `src/impl/wheels/*.go`) was read as a
second reference. It agrees with the Rust port everywhere the port does not
already document a Go bug it fixed.

## R1. The queries in `research.rs` are all correct

Rulebook p.12, verbatim, against `src/research.rs`:

| Track | Rulebook | Code |
|---|---|---|
| Agriculture 1 | "Take 1 more corn any time you harvest corn from the jungle (Palenque action 2, 3, 4, or 5; but not action 1)." | `corn_bonus(_, Green) == 1` |
| Agriculture 2 | irrigation, and "Take 1 more corn every time you fish (Palenque action 1)." | `irrigation()`, `corn_bonus(_, Blue) == 1` |
| Agriculture 3 | "Take 2 more corn any time you harvest corn from the jungle" — and "because bonuses are cumulative, a player with the 3rd level ... will get 3 extra corn" | `corn_bonus(_, Green) == 3` |
| Extraction 1/2/3 | +1 wood / +1 stone / +1 gold, "only ... Yaxchilan and Palenque" | `resource_bonus` keyed 1/2/3 |
| Extraction, skulls | not a resource ("Only wood, stone and gold are resources") | `Skull => return 0` |
| Architecture 1/2 | 1 corn / 2 VP per building constructed | `build_bonus` |
| Architecture 3 | 1 block off at Tikal, 2 corn off at Uxmal 4 | `builder()` |
| Theology 1 | perform the Chichen action 1 space ahead, no extra corn | `foresight()` |
| Theology 2 | after a Chichen action, spend 1 block for a temple step | `devout()` |
| Theology 3 | +1 skull from Yaxchilan action 4 only | inlined in `yaxchilan.rs:29` |
| Past level 3 | temple step / 2 blocks / 3 VP / 1 skull | `options::top_payoffs` |

The Go bug the port's comment records is real and the fix is right: Go's
`CornBonus` tested `HasLevel(Agriculture, 1)` before `HasLevel(Agriculture, 3)`
with a `>=` predicate, so the level-3 arm was dead. **No sibling of that bug
survives** — every other query tests one predicate, and the two-armed one
(`corn_bonus`) now tests the highest tier first.

Monument #11's table `[0, 9, 20, 33, 33]` matches the appendix verbatim
("9 for one, 20 for two, 33 for three or all four"), and #12's "3 VP per level"
likewise. Neither is a bug.

## R2. The advance cost is right

Rulebook p.7: advancing from the start to level 1 costs **1 block**, level 1 to
2 costs **2 blocks**, level 2 to 3 costs **3 blocks**, and "when you 'advance'
from level 3, you must pay 1 block" for the one-off bonus. That is exactly
`options::recurse`'s `if free { 0 } else if lvl >= 3 { 1 } else { lvl + 1 }`.
`RULES-AUDIT.md` A6 was correctly diagnosed and correctly fixed.

Tikal 3's "(The 2 may be either once on two tracks or twice on one track.)" is
also honoured: `recurse` passes `floor = s.idx()`, not `s.idx() + 1`, so the
same track can be taken twice.

## R3. Every bonus reaches every space it should — with two exceptions

Traced by hand, and the coverage is otherwise complete: the free-choice spaces
(`palenque/yaxchilan` `_ => mirror(.., 6)`, `tikal/uxmal` `_ => for i in 0..6`,
`chichen` `_ => for i in 0..=9`) and Uxmal 5's mirror all re-enter the same
`at()` generators, so every bonus flows through them; `moves::choices_for_worker`
re-generates per step-down fee against the same generators; and
`options::dominated_dedup` keys `AdvanceResearch` into the *structural* group,
so no research option can be pruned by a wealthier sibling.

The two exceptions are logged as B1 and B2 below.

## B1. Devout could not spend the block the Chichen action had just handed over

**Rule.** Theology level 2, p.12, in full: "After performing an action on the
Chichen Itza gear, you may immediately spend 1 resource block to move up 1 step
on the temple of your choice. **(If you gained a resource block from your
Chichen Itza action, it is available for you to spend in this way.)**"

The parenthetical is not decoration. Chichen spaces 6, 8 and 9 — the three
dearest, worth 8, 11 and 13 points — each hand out a block of the player's
choice, and the sentence exists to say that block pays for the step.

**Bug.** `chichen.rs` filtered the block to spend against the holding *before*
the action:

```rust
for b in Resource::BLOCKS {
    if g.players[p.idx()].get(b) == 0 { continue; }   // <- pre-action `g`
```

so a devout player arriving at Chichen with an empty block stock — the ordinary
case, since blocks are what research and buildings consume — was offered no
temple step at all, on exactly the three spaces the rule was written for. The
temple test one line below already probed the *post*-core state, which is what
makes the miss hard to see: the loop looks like it is reading the right state.

**Fix.** Probe once per core, then filter both the block and the temple against
that probe. `Choice::affordable` walks the effect list in order and the core's
`Res(b, +1)` precedes the step's `Res(b, -1)`, so the resulting choice is
affordable without any further change.

**Test.** `devout_may_spend_the_block_the_chichen_action_granted` in
`tests/rules.rs`. Fails before, passes after.

**Width.** The pass only *adds* options, and only where the rule was being
denied. At a block-granting spot a devout player holding `k` block types used to
get up to `3k` cores x temples; they now get one temple fan per (core, held
type) pair computed after the grant. Holding all three, the count is unchanged;
holding none it goes 0 -> 9. Chichen has three such spots, so the worst case is
27 choices added to one gear for one player in one position.

## B2. A double build could not use the architecture level the first card gave

**Rule.** Tikal action 4, p.9: "If you construct two buildings, you can apply
your architecture technologies to either the first or the second, but one of
them must be built without architecture technologies. **If the first building
gives you a new architecture technology, you may apply that effect (and any
others) to the second building, as long as you applied no architecture effects
to the first one.**"

**Bug.** `tikal::build_two` gave the bonus to the first card and nothing to the
second, and covered "either the first or the second" by enumerating both orders
— which is sound for a level the player *already holds*, because the two orders
differ only in which card gets the discount. It cannot cover the sentence in
bold: building #6 is one gold for a free architecture advance, and the level it
hands over exists only in the probe, never in `g`. So the line "build #6, then
spend the level it just gave you on the card beside it" was ungeneratable.

Six cards can do this — #6 and #27 name architecture outright, and #18, #20,
#26, #31 and #32 are free-choice advances — and the payoff is the whole of the
track: 1 corn at level 1, 2 VP at level 2, a block off at level 3.

**Fix.** A second pass in `build_two`, guarded on the architecture level
actually moving, that builds the first card plain and generates the second
against the post-first state with the bonus on. The guard is what keeps this
from tripling the width of every ordinary double build: without it, the pass
would re-spell the mirror the first loop already produces.

**Test.** `architecture_won_by_the_first_build_pays_for_the_second`. Fails
before, passes after.

## B3. Nothing else is missing — pinned by four coverage tests

The remaining risk is the one that does not show up in a code reading: a bonus
computed correctly and never read. Four table-driven tests now measure each
track's payoff as a *delta* in the best a space offers, at the space the
rulebook names and at every route that repeats it, and pin it at zero
everywhere the rulebook excludes:

* `agriculture_reaches_every_palenque_space_and_nothing_else` — +1/+3 on
  Palenque 2-5, +1 on Palenque 1 at level 2 and **not** at level 1, through
  Palenque 6 and 7 and through Uxmal 5, 6 and 7; zero on Yaxchilan's corn.
* `extraction_reaches_every_harvest_and_nothing_else` — wood on Yaxchilan 1 and
  Palenque 3-5, stone on Yaxchilan 2 and 5, gold on Yaxchilan 3 and 5, through
  every repeat; zero on Chichen's block and zero on the market.
* `architecture_reaches_every_construction_and_no_monument` — 1 corn and 2 VP at
  Tikal 2, Tikal 4 and Uxmal 4 and every repeat; a monument pays its listed
  price to a maxed builder and scores nothing.
* `theology_three_reaches_only_yaxchilan_four` — +1 skull at Yaxchilan 4 and its
  repeats, and not on the track's own top-of-track skull.

Writing the architecture one turned up a payout worth naming, since it is not
a bug and reads like one: from Tikal 3 (and so from Tikal 6 and 7) a player at
architecture 2 can push to level 3 for 3 blocks and then cash the top-of-track
bonus for 3 VP for 1 more, and "this option remains open to you every time you
use a technology advancement action" (p.7). One block for three points,
repeatable, is a real line the rules grant.

## B4. Adjacent, not research: an unusable mirror made card #30 unbuildable

Found while checking that the architecture bonus reaches every construction.
`Payoff::Mirror` expanded to `mirror_choices`, which returns an empty list when
the player has no corn (the mirror costs one) or when the chain is at its depth
bound. `building_choices` reads an empty payoff as "no way to build this", so
building #30 — 2 wood 1 stone 1 gold for the mirror and 2 VP — fell out of the
game for a player holding no corn.

Measured directly before the fix, with `research` zeroed and 4/4/4 blocks in
hand: **0 ways to construct #30 at 0 corn, 300 at 1 corn.** Architecture level 1
masked it — the corn the build itself pays lands in the probe before the payoff
is expanded, and one corn is exactly the mirror's price, so at architecture 1
the card was buildable at 0 corn and at architecture 0 it was not.

Fixed by falling back to the bare tail when the mirror yields nothing, which is
what the rules describe: an unusable privilege is wasted, not withheld — the
same principle already applied to Chichen's temple step (A5) and agriculture's
top-of-track step (C5). Pinned by
`a_card_whose_payoff_is_unusable_is_still_buildable`.

**Still open, same family, not fixed here:** `mirror_choices` never offers
"decline the mirror" when the player *can* pay, so building #30 and Uxmal 5
force the corn once they are chosen. That is `RULES-AUDIT.md` A9's shape and
belongs with A9's owner, not with a research pass.

## Answers

**(a) Is research implemented correctly?** Almost. The queries are exactly
right, the costs are exactly right, and the coverage is complete except for the
two holes above, both now closed and pinned:

| | What was wrong | Test |
|---|---|---|
| B1 | Devout read the block stock from before the Chichen action, so the block the action granted could not pay for the temple step | `devout_may_spend_the_block_the_chichen_action_granted` |
| B2 | A double build could not spend an architecture level the first card had just handed over | `architecture_won_by_the_first_build_pays_for_the_second` |
| B4 | (adjacent) an unusable mirror payoff made card #30 unbuildable | `a_card_whose_payoff_is_unusable_is_still_buildable` |

No sibling of the Go dead-arm bug survives, and no bonus is computed and never
read: that is the claim four coverage tests now hold, space by space, in both
directions.

**(b) Is research genuinely weak here?** Yes — **the game is as written and
research is genuinely weak in this engine's play**. Nothing found in this pass
is remotely large enough to move a 4.33-point verdict. B1 fires only for a
theology-2 player reaching Chichen 6, 8 or 9 with an empty block stock; B2 only
for a double build whose first card is one of six that grant architecture; B4
only at literally zero corn. The three together are worth a fraction of a point
per game, and none of them touches the *price* of research, which is where the
decision is made.

**(c) Is the advance cost right?** Yes, verbatim. Rulebook p.7: start to level 1
costs 1 block, 1 to 2 costs 2 blocks, 2 to 3 costs 3 blocks, and "when you
'advance' from level 3, you must pay 1 block". `options::recurse` charges
`if free { 0 } else if lvl >= 3 { 1 } else { lvl + 1 }` — an exact match, and
A6's earlier diagnosis and fix were both correct. Six blocks and three
advancement actions to max a track is the real, printed price.

**(d) What this changes about the evaluator's verdict.** Nothing that needs a
re-measurement. F50-F53 established two things: that `--promise` is circular for
research, and that raising `RESEARCH_SCALE` makes the agent research hard and
lose 4.33 points [-4.86, -3.79] with no reversal out to `rs = 0.35`. The first
is a diagnostic defect and is unaffected by anything here. The second is a
statement about a payoff curve that these fixes do not bend: the cost side is
untouched (it was already right), and the benefit side gains three conditional
crumbs. Re-measuring `RESEARCH_SCALE` against an effect this small would be
spending a ladder to resolve something far inside its own interval — the
project's own rule about never calling an effect smaller than its interval an
improvement cuts both ways.

What *would* justify a re-measure is a rules change that moves the price or the
per-use payout. None was found.

## The monument question, and where the compounding actually lives

The brief asked whether monuments #11 and #12 are a second-order symptom. They
check out: #11's table `[0, 9, 20, 33, 33]` and #12's "3 VP for each level of
technology you have" are verbatim from the appendix, both are reachable only at
Tikal 4 as the rules require, and `N_DISPLAY = 6` matches "deal 6 monuments face
up ... put the rest in the box".

Which makes the arithmetic worth stating, because it is the strongest case for
research in the box and it is *already implemented*: with both #11 and #12 on the
table, maxing a single track pays 9 + 9 = 18 VP on top of whatever the track
itself does, for 6 blocks and three advancement actions. That is the
compounding the theory expects. It is also conditional on two specific tiles
out of thirteen both being among the six dealt — jointly about one game in five
— and on holding them, which costs 5 and 6 blocks more.

So the payoff for research is not linear in levels; it is a step function with
most of its mass behind a monument that is absent four games in five. A term
scaled on level count cannot represent that shape, which is a plausible reason
the `RESEARCH_SCALE` sweep found flat-then-negative rather than a threshold. It
is a hypothesis about the evaluator's *shape*, not its scale, and it belongs to
`src/eval.rs`'s owner; this pass only establishes that the rules put the money
there.

## Two judgement calls left standing

* **A free advance from a maxed track is free here.** p.7 says "when you
  'advance' from level 3, you must pay 1 block"; the building and tile symbol
  says "advance 1 level ... without paying any resources". `free_track` and
  `research_choices(.., free = true)` waive the block. Go did the same. Either
  reading is defensible and the difference is one block on a rare card.
* **Yaxchilan 4 at theology 3 with one skull left in the bank** emits
  `Res(Skull, 2)`; `take_skulls` caps the delivery at 1, correctly. The effect
  list therefore overstates by one in that position. `dominated_dedup`
  deliberately keeps skulls out of its wealth axes for exactly this reason, so
  generation is sound; only a consumer that reads the effect list as a promise
  would be misled.

## State at the end of this pass

`cargo build --release` clean, `cargo test --release` **193 passing** (was 186;
seven added: three regressions for B1, B2 and B4, and four coverage tests).
Files touched: `src/spaces/chichen.rs`, `src/spaces/tikal.rs`, `src/options.rs`,
`tests/rules.rs` (inserted, never rewritten), and this file. Nothing committed.
