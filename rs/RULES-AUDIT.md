# Rules audit vs. the official CGE rulebook

> **Status: sections A, B and C are all fixed**, each with a named regression
> test in `tests/rules.rs`. Section D is untouched — those need checking against
> the physical components. A re-audit is in progress.
>
> Two of the fixes reverse earlier "fixes" of mine that were wrong: temple ties
> (A1) and the starting-player self-pass (A4) were both correct in the original
> Go code, and I broke them.
>
> **Gear sizes: 8 on the small gears, 11 on Chichen — the original numbers.**
> A physical count gives 7 and 10, and briefly went into the code as such. That
> count is right about the *printed numbers*: they run 1..7 and 1..10, because
> space 0 is the unnumbered entry space. Total spaces a worker can stand on is
> one more. The rulebook settles it in four places, decisively at p.10, where a
> worker on space 6 blocks the extra day and one on space 7 does not — so 7
> exists and is distinct from the blocking space. The gears also have more
> *holes* than spaces (10 and 13); two fall in the dead arc where the gear meshes
> with the central calendar and no action is printed.
>
> Consequently "action space 6" and "action space 9" are the **second-to-last**
> spaces — the ones the *extra* day would push a worker off. The last spaces are
> pushed off by the ordinary advance regardless.

## Second audit

A re-audit verified the fixes and found more. Everything below is now fixed.

| Issue | What was wrong |
|---|---|
| Gear sizes | Changed to 7/10 mid-audit. Reverted to 8/11; Chichen's free-choice space at 10 restored. |
| Tikal 5 | An earlier fix over-corrected and allowed both steps on one temple. The rules require two *different* temples — and a test had been written asserting the violation. |
| Temple-top flip | Fired on any upward step that *landed* on the top, including one that went nowhere because the player was already there — so a player camped on a temple top could refresh the extra day indefinitely. Now only on actually arriving. |
| Extra day (B4) | Blocked on the last space; the rules block on the second-to-last. Also gated twice, once spuriously before rotation. |
| Extra day (C2) | Advancing two days resolved the food day on the round left behind. It now resolves on the round landed on, as one `advance_days(2)`. |
| Extra day | Could run past the end of the calendar. |
| Passing (A17) | Still silently legal in three drivers. |
| Pity (C1) | Unreachable from `sample_legal_move`, so self-play silently skipped the turn in exactly the position the rule exists for. |
| Pity test | Vacuous — it claimed to fill the entry spaces and did not, so an ordinary move always existed and `Pity` was never generated. |
| Corn deltas | `Effect::Corn` was `i8`; a large market exchange wrapped silently. Widened to `i16`. |
| Irrigation | Offered even with a corn tile showing. Strictly dominated, but it inflated branching. |
| End of game | The gear turns once more after final scoring, which can change the tiebreaker. |

**B7 was a false positive.** The audit read the page-16 symbol glossary as the
tile vocabulary and concluded that tiles were missing "may construct a
building", "may exchange at the market", "pay 1 corn and perform any action",
2 VP, the feeding discount and the choose-your-own symbols. That glossary is
headed "Starting Wealth Tiles **and Building Effects**" — it is shared between
the two, and those symbols belong to the buildings. Checked tile by tile
against a physical copy: all 21 compositions matched what was already in the
code, exactly, as a multiset. The table has been renumbered to match the
components and pinned by a regression test, and no starting tile carries a
decision.

Still open:

- **B8** — the extra day is a decision the *search* can see (`Phase::ExtraDay`)
  but `legal_moves` still cannot enumerate it. That is the right place for it to
  live, so this closes when the factored tree lands.

Source: CGE English rulebook PDF (© Oct 2012), corroborated against Board Game
Arena's `Gamehelptzolkin` and UltraBoardGames. BGG's FAQ/errata was unreachable
(403), so anything not sourced is marked unverified.

**Verified correct** and not listed below: placement cost table, all three temple
tracks (steps, per-step VP, resource payouts, all six age bonuses), all nine
Chichen spots, agriculture/extraction/architecture levels, gear space numbering,
blank entry spaces, `corn_showing`, tile supply, 4:1 corn conversion, market
prices 2/3/4, monuments never replenished, begging needs corn < 3, the
per-player two-sided first-player board and its temple-top reset, and the
4-player monument multiplier of 4.

## A. Rule violations

| # | Issue | Where |
|---|---|---|
| A1 | **Temple ties pay half, not `prize / n_tied`.** Rulebook's own example has three players tied each scoring half. The Go behaviour was right; my "multitie fix" was wrong. | `game.rs:334` |
| A2 | **Calendar is one round short and every food day fires a round early.** Should be 27 rounds with food days at the ends of rounds 8, 14, 21, 27. Age 1 is 14 rounds, not 13. | `state.rs:246-250` |
| A3 | **Feeding can never cost 0.** Two farm buildings should make feeding free; `.max(1)` prevents it. | `game.rs:286` |
| A4 | **No starting-player self-pass.** If the claimer already holds the marker it passes to their left. Go was right here too. | `game.rs:214` |
| A5 | **Chichen spaces unusable when the temple step would be wasted.** The privilege is wasted but the action is still legal — skull, VP and block all still happen. | `chichen.rs:68` |
| A6 | **Advancing past level 3 costs 3 blocks; should cost 1.** | `options.rs:99` |
| A7 | **Yaxchilan action 3 pays 1 corn; should pay 2.** | `yaxchilan.rs:18` |
| A8 | **Tikal action 3 forces exactly two research advances.** Should be "1 or 2". | `tikal.rs:18` |
| A9 | **"Do nothing" is not offered** except when a space is otherwise empty. Forces harmful moves: buying an unwanted worker on Uxmal 3, spending a block on Tikal 5, building on Tikal 2/4. | `moves.rs`, `tikal.rs`, `uxmal.rs` |
| A10 | **The Uxmal mirror omits Palenque.** Rulebook: Palenque, Yaxchilan, Tikal or Uxmal. | `uxmal.rs:88` |
| A11 | **Monument "3 VP per step above start" uses the absolute index.** Pays 6 at the starting position instead of 0. | `monuments.rs:52` |
| A12 | **Monument "workers in play" counts only workers in hand and caps at 24.** "In play" means not in the bank — includes workers on gears — and the cap is 18. | `monuments.rs:100` |
| A13 | **Burned wood tiles count for the wood-tile monument.** They go back to the box. | `palenque.rs:56`, `effect.rs:86` |
| A14 | **Second building of a double build still gets the architecture block discount.** It should cost full price. | `options.rs:189` |
| A15 | **The extra calendar day is forbidden before a food day.** It is legal; the new round just becomes the skipped food day. | `game.rs:218` |
| A16 | **Accumulated corn is added every rotation.** Should be once per round, and only when the space is unoccupied. | `game.rs:239` |
| A17 | **Passing is treated as legal.** "You cannot skip your turn." | `game.rs:175` |

## B. Missing rules

- **B1. The 13-crystal-skull supply limit is entirely absent.** No global counter
  exists. Skulls spent at Chichen never return, so this binds in real games.
  Affects Yaxchilan 4, theology's bonus, green temple step 5, building 26, tiles.
- **B2. Theology level 3** (+1 skull whenever you take one from Yaxchilan 4).
- **B3. The top step of each temple is exclusive** — once occupied, no one else
  may reach it.
- **B4. The extra day may not push any worker off a gear** (space 6 of the small
  gears, space 9 of Chichen) — any player's worker, not just your own.
- **B5. The market cannot sell.** Only corn -> blocks is implemented; the
  exchange is symmetric.
- **B6. End-of-game tiebreaker** (most workers left on gears, then a shared win).
- **B7. Starting tiles are not drafted** — deal 4, keep 2. Several tile effects
  from the appendix are also unrepresented.
- **B8. The extra-day decision is not a `Move`** — it is an RNG coin flip inside
  `resolve_first_player`, so a search cannot reason about it.

## C. Edge cases

- **C1. "The gods take pity"**: with no workers on gears, no affordable
  placement, and no temple to beg from, a player may place exactly 1 worker on
  the cheapest space and give all their corn to the bank. Currently passes.
- **C2. Phase ordering**: first-player resolution runs before feeding. Net effect
  is currently right but fragile, and the extra-day rotation resolves a food day
  between the two advances.
- **C3. Chichen foresight from the entry space** is routed away before it can
  look ahead.
- **C4. `Payoff::BuildAnother` forces a temple step** and yields nothing when all
  three temples are maxed.
- **C5. Agriculture's bonus payoff vanishes** at max temples rather than being
  wasted.

## D. Unverified

- **Monument 4's formula** is internally inconsistent with its siblings, and the
  monument colour distribution (8 of 13 red) looks like a Go default rather than
  real data. Worth checking against the physical tiles.
- **Monument and building costs**, and the 14/18 age split, are printed only on
  the components and could not be verified from any source.
- **Building definitions** likewise, except the three farm buildings, which match.
