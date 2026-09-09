//! Regression tests for every bug carried over from the Go implementation.
//!
//! Each test names the original defect so a future refactor that reintroduces it
//! fails loudly rather than being found by playing a game and hand-diffing a
//! 64KB log.

use tzolkin::data::buildings::{BUILDINGS, N_BUILDINGS};
use tzolkin::data::monuments::{def as mdef, MONUMENTS};
use tzolkin::effect::{Choice, Effect};
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::invariants::validate;
use tzolkin::moves::{check_move, legal_moves, MoveKind};
use tzolkin::spaces::choices_at;
use tzolkin::state::{GameState, WorkerLoc};

fn fresh() -> Game {
    Game::new(42)
}

// ---- move generation ---------------------------------------------------

/// Go's `Move.Place` did `append(m.Workers, worker)` on a shared backing array,
/// so from the fourth worker on, sibling branches overwrote each other and the
/// engine played a move other than the one it logged.
#[test]
fn sibling_moves_are_independent() {
    let mut g = fresh();
    // Unlock everything so placement reaches the depth where Go aliased.
    for p in PlayerId::ALL {
        for _ in 0..3 {
            g.state.unlock_worker(p);
        }
        g.state.players[p.idx()].corn = 60;
    }

    let moves = legal_moves(&g.state, PlayerId(0));
    let deep: Vec<_> = moves
        .iter()
        .filter(|m| m.n_workers() >= 4)
        .collect();
    assert!(!deep.is_empty(), "expected moves placing 4+ workers");

    // Every generated move must be distinct, and no two may share a suffix that
    // differs only because one clobbered the other.
    let mut seen = std::collections::HashSet::new();
    for m in &moves {
        assert!(seen.insert((*m).clone()), "duplicate move generated: {m}");
    }
}

/// Every move the generator emits must survive an independent legality check.
#[test]
fn generated_moves_are_legal() {
    for seed in 0..40u64 {
        let mut g = Game::new(seed);
        for _ in 0..6 {
            for p in PlayerId::ALL {
                g.state.current = p;
                for m in legal_moves(&g.state, p) {
                    check_move(&g.state, p, &m).unwrap_or_else(|e| {
                        panic!("seed {seed}: illegal move generated: {m} -- {e}")
                    });
                }
                g.take_turn();
            }
            g.play_round();
        }
    }
}

/// Placement cost is the space index plus one per worker already placed.
#[test]
fn placement_cost_matches_placements() {
    let g = fresh();
    for m in legal_moves(&g.state, PlayerId(0)) {
        if let MoveKind::Place(v) = &m.kind {
            let expected: u8 = v
                .iter()
                .enumerate()
                .map(|(n, (_, spot))| {
                    n as u8
                        + match spot {
                            tzolkin::moves::Placement::Gear(_, pos) => pos.0,
                            tzolkin::moves::Placement::FirstPlayer => 0,
                        }
                })
                .sum();
            assert_eq!(expected, m.corn_cost, "cost mismatch on {m}");
        }
    }
}

/// A move must never cost more corn than the player holds.
#[test]
fn moves_are_affordable() {
    for seed in 0..25u64 {
        let g = Game::new(seed);
        for p in PlayerId::ALL {
            let corn = g.state.players[p.idx()].corn;
            for m in legal_moves(&g.state, p) {
                let budget = if m.beg.is_some() { 3 } else { corn };
                assert!(
                    m.corn_cost <= budget,
                    "seed {seed}: {m} costs {} with {budget} available",
                    m.corn_cost
                );
            }
        }
    }
}

// ---- research ----------------------------------------------------------

/// Go tested `HasLevel(.., 1)` before `HasLevel(.., 3)` with a `>=` predicate,
/// so a maxed agriculture track silently paid 1 corn instead of 3.
#[test]
fn maxed_agriculture_pays_three() {
    let mut g = fresh();
    let p = PlayerId(0);
    // Starting tiles can already have advanced a track, so reset first.
    g.state.research[p.idx()] = [0; 4];
    assert_eq!(g.state.corn_bonus(p, Color::Green), 0);
    g.state.research[p.idx()][Science::Agriculture.idx()] = 1;
    assert_eq!(g.state.corn_bonus(p, Color::Green), 1);
    g.state.research[p.idx()][Science::Agriculture.idx()] = 3;
    assert_eq!(g.state.corn_bonus(p, Color::Green), 3);
}

/// The extraction track never boosts skulls.
#[test]
fn skulls_are_never_boosted() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()][Science::Extraction.idx()] = 3;
    assert_eq!(g.state.resource_bonus(p, Resource::Skull), 0);
    assert_eq!(g.state.resource_bonus(p, Resource::Wood), 1);
}

/// Two paid advances in one action may not spend the same block twice.
#[test]
fn two_advances_cannot_double_spend() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].res = [1, 0, 0, 0];
    g.state.research[p.idx()] = [0, 0, 0, 0];

    for c in tzolkin::options::research_choices(&g.state, p, 2, false) {
        assert!(
            c.affordable(&g.state, p),
            "two advances spent more than the player had: {c}"
        );
    }
}

/// The best a space offers along one axis, over every choice it generates.
fn best_at<F: Fn(&Choice) -> i32>(g: &GameState, p: PlayerId, gear: Gear, pos: u8, f: F) -> i32 {
    choices_at(g, p, gear, Pos(pos))
        .iter()
        .map(|c| f(c))
        .max()
        .unwrap_or(0)
}

/// How much better that best gets when the player holds `levels` and nothing
/// else about the position moves. Zero means the space never asked.
fn research_delta<F: Fn(&Choice) -> i32>(
    g: &mut Game,
    p: PlayerId,
    levels: [u8; 4],
    gear: Gear,
    pos: u8,
    f: F,
) -> i32 {
    g.state.research[p.idx()] = [0; 4];
    let before = best_at(&g.state, p, gear, pos, &f);
    g.state.research[p.idx()] = levels;
    let after = best_at(&g.state, p, gear, pos, &f);
    g.state.research[p.idx()] = [0; 4];
    after - before
}

fn net_points(c: &Choice) -> i32 {
    c.0.iter()
        .filter_map(|e| match e {
            Effect::Points(n) => Some(*n as i32),
            _ => None,
        })
        .sum()
}

/// A bonus computed correctly and never read is invisible, so these four tests
/// pin the *reading* half of every track: the payoff measured as a delta in the
/// best a space offers, at the space the rulebook names and at every route that
/// repeats it -- the free-choice spaces at the top of each gear, and Uxmal 5's
/// mirror -- and pinned at zero everywhere the rulebook excludes.
///
/// Agriculture, rulebook p.12: "+1 corn any time you harvest corn from the
/// jungle (Palenque action 2, 3, 4, or 5; **but not action 1**)" at level 1,
/// "+1 corn every time you fish (Palenque action 1)" at level 2, and "+2 more"
/// -- cumulative, so +3 in total -- at level 3.
#[test]
fn agriculture_reaches_every_palenque_space_and_nothing_else() {
    let mut g = fresh();
    let p = PlayerId(0);
    // A corn face showing over the wood at every tiled space, so both harvests
    // are live and neither irrigation nor a burn is needed to reach the corn.
    for i in 2..=5 {
        g.state.palenque[i] = tzolkin::state::TileStack { corn: 2, wood: 1 };
    }
    // Exactly one corn: enough to buy Uxmal's mirror, not enough for the market
    // at Uxmal 2, which would otherwise out-earn every mirrored harvest and
    // swallow the delta.
    g.state.players[p.idx()].corn = 1;
    g.state.players[p.idx()].res = [0; 4];

    let corn = |c: &Choice| c.net_corn();

    // Fishing is the level-2 bonus. Level 1 must not reach it.
    assert_eq!(research_delta(&mut g, p, [1, 0, 0, 0], Gear::Palenque, 1, corn), 0);
    assert_eq!(research_delta(&mut g, p, [2, 0, 0, 0], Gear::Palenque, 1, corn), 1);

    for pos in 2..=5u8 {
        assert_eq!(
            research_delta(&mut g, p, [1, 0, 0, 0], Gear::Palenque, pos, corn),
            1,
            "Palenque {pos} at agriculture 1"
        );
        assert_eq!(
            research_delta(&mut g, p, [3, 0, 0, 0], Gear::Palenque, pos, corn),
            3,
            "Palenque {pos} at agriculture 3"
        );
    }

    // Every route that repeats a Palenque action: the gear's own free-choice
    // spaces, Uxmal's mirror, and the two Uxmal spaces that repeat the mirror.
    for pos in [6u8, 7] {
        assert_eq!(
            research_delta(&mut g, p, [3, 0, 0, 0], Gear::Palenque, pos, corn),
            3,
            "Palenque {pos}"
        );
    }
    for pos in [5u8, 6, 7] {
        assert_eq!(
            research_delta(&mut g, p, [3, 0, 0, 0], Gear::Uxmal, pos, corn),
            3,
            "Uxmal {pos}"
        );
    }

    // Corn from Yaxchilan is not corn from the jungle.
    for pos in [2u8, 3, 5] {
        assert_eq!(
            research_delta(&mut g, p, [3, 0, 0, 0], Gear::Yaxchilan, pos, corn),
            0,
            "Yaxchilan {pos}"
        );
    }
}

/// Extraction, rulebook p.12: wood from Yaxchilan 1 and Palenque 3-5, stone
/// from Yaxchilan 2 and 5, gold from Yaxchilan 3 and 5 -- and "these
/// technologies only apply to Yaxchilan and Palenque. They do not give you
/// extra resources when you acquire resources in other ways (e.g., from
/// buildings, from the gods, from the Chichen Itza actions, or from the
/// market)."
#[test]
fn extraction_reaches_every_harvest_and_nothing_else() {
    let mut g = fresh();
    let p = PlayerId(0);
    for i in 2..=5 {
        g.state.palenque[i] = tzolkin::state::TileStack { corn: 2, wood: 1 };
    }
    g.state.players[p.idx()].corn = 1;
    g.state.players[p.idx()].res = [0, 0, 0, 1]; // one skull, for Chichen

    let wood = |c: &Choice| c.net_res(Resource::Wood);
    let stone = |c: &Choice| c.net_res(Resource::Stone);
    let gold = |c: &Choice| c.net_res(Resource::Gold);

    assert_eq!(research_delta(&mut g, p, [0, 1, 0, 0], Gear::Yaxchilan, 1, wood), 1);
    assert_eq!(research_delta(&mut g, p, [0, 2, 0, 0], Gear::Yaxchilan, 2, stone), 1);
    assert_eq!(research_delta(&mut g, p, [0, 3, 0, 0], Gear::Yaxchilan, 3, gold), 1);
    assert_eq!(research_delta(&mut g, p, [0, 3, 0, 0], Gear::Yaxchilan, 5, gold), 1);
    assert_eq!(research_delta(&mut g, p, [0, 3, 0, 0], Gear::Yaxchilan, 5, stone), 1);
    for pos in 3..=5u8 {
        assert_eq!(
            research_delta(&mut g, p, [0, 1, 0, 0], Gear::Palenque, pos, wood),
            1,
            "Palenque {pos}"
        );
    }

    // The repeats.
    for pos in [6u8, 7] {
        assert_eq!(research_delta(&mut g, p, [0, 1, 0, 0], Gear::Yaxchilan, pos, wood), 1);
        assert_eq!(research_delta(&mut g, p, [0, 1, 0, 0], Gear::Palenque, pos, wood), 1);
    }
    for pos in [5u8, 6, 7] {
        assert_eq!(
            research_delta(&mut g, p, [0, 1, 0, 0], Gear::Uxmal, pos, wood),
            1,
            "Uxmal {pos}"
        );
    }

    // Chichen 6 hands over a block of the player's choice. It is not a harvest.
    assert_eq!(best_at(&g.state, p, Gear::Chichen, 6, wood), 1, "Chichen 6 gives a block");
    assert_eq!(research_delta(&mut g, p, [0, 3, 0, 0], Gear::Chichen, 6, wood), 0);

    // Neither is the market.
    g.state.players[p.idx()].corn = 10;
    assert_eq!(best_at(&g.state, p, Gear::Uxmal, 2, wood), 5, "ten corn buys five wood");
    assert_eq!(research_delta(&mut g, p, [0, 3, 0, 0], Gear::Uxmal, 2, wood), 0);
}

/// Architecture, rulebook p.12: 1 corn and 2 victory points "any time you
/// construct a building (Tikal action 2 or 4, Uxmal action 4)", and p.9: "you
/// can construct a monument only with Tikal action 4 and the architecture bonus
/// doesn't apply to it."
#[test]
fn architecture_reaches_every_construction_and_no_monument() {
    let mut g = fresh();
    let p = PlayerId(0);
    // Three cards, none of which touches a technology track: a card that hands
    // over an architecture level would move the baseline under the test.
    g.state.buildings_up = [None; tzolkin::state::N_DISPLAY];
    g.state.buildings_up[0] = Some(BuildingId(1)); // 1 wood
    g.state.buildings_up[1] = Some(BuildingId(2)); // 1 wood 2 stone
    g.state.buildings_up[2] = Some(BuildingId(8)); // 2 wood 1 stone
    g.state.players[p.idx()].corn = 30;
    g.state.players[p.idx()].res = [4, 4, 4, 0];

    // Measured over the choices that actually construct something: the
    // free-choice spaces also reach Tikal 3, where an architecture track at
    // level 2 can be pushed to 3 and then cashed for the top-of-track 3 points,
    // which is research paying out but not the build bonus.
    let builds = |c: &Choice| c.0.iter().any(|e| matches!(e, Effect::Build(_)));
    let build_corn = |c: &Choice| if builds(c) { c.net_corn() } else { 0 };
    let build_points = |c: &Choice| if builds(c) { net_points(c) } else { 0 };

    for (gear, pos) in [(Gear::Tikal, 2u8), (Gear::Tikal, 4), (Gear::Uxmal, 4)] {
        assert_eq!(
            research_delta(&mut g, p, [0, 0, 1, 0], gear, pos, build_corn),
            1,
            "{gear:?} {pos} should pay a corn for the building"
        );
    }
    for (gear, pos) in [
        (Gear::Tikal, 2u8),
        (Gear::Tikal, 4),
        (Gear::Tikal, 6),
        (Gear::Tikal, 7),
        (Gear::Uxmal, 4),
        (Gear::Uxmal, 5),
        (Gear::Uxmal, 6),
        (Gear::Uxmal, 7),
    ] {
        assert_eq!(
            research_delta(&mut g, p, [0, 0, 2, 0], gear, pos, build_points),
            2,
            "{gear:?} {pos} should pay two points for the building"
        );
    }

    // A monument is not a building: no corn, no points, and no block off.
    g.state.research[p.idx()] = [0, 0, 3, 0];
    let mut seen = 0;
    for c in choices_at(&g.state, p, Gear::Tikal, Pos(4)) {
        let Some(Effect::TakeMonument(id)) = c.0.iter().copied().find(|e| matches!(e, Effect::TakeMonument(_)))
        else {
            continue;
        };
        seen += 1;
        let listed: i32 = Resource::BLOCKS
            .iter()
            .map(|&r| mdef(id).cost[r.idx()] as i32)
            .sum();
        let paid: i32 = -Resource::BLOCKS.iter().map(|&r| c.net_res(r)).sum::<i32>();
        assert_eq!(paid, listed, "a builder was discounted on a monument: {c}");
        assert_eq!(c.net_corn(), 0, "architecture paid corn for a monument: {c}");
        assert_eq!(net_points(&c), 0, "architecture scored for a monument: {c}");
    }
    assert!(seen > 0, "no affordable monument to check");
}

/// Theology 3, rulebook p.12: "Take 1 more crystal skull any time you get a
/// crystal skull from Yaxchilan (action 4). (Note that this does not apply to
/// crystal skulls you get elsewhere by other means.)"
#[test]
fn theology_three_reaches_only_yaxchilan_four() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 1;
    g.state.players[p.idx()].res = [0; 4];

    let skull = |c: &Choice| c.net_res(Resource::Skull);

    assert_eq!(research_delta(&mut g, p, [0, 0, 0, 3], Gear::Yaxchilan, 4, skull), 1);
    for pos in [6u8, 7] {
        assert_eq!(
            research_delta(&mut g, p, [0, 0, 0, 3], Gear::Yaxchilan, pos, skull),
            1,
            "Yaxchilan {pos}"
        );
    }
    for pos in [5u8, 6, 7] {
        assert_eq!(
            research_delta(&mut g, p, [0, 0, 0, 3], Gear::Uxmal, pos, skull),
            1,
            "Uxmal {pos}"
        );
    }

    // "elsewhere by other means" includes the track's own top-of-track bonus.
    g.state.research[p.idx()] = [0, 0, 0, 3];
    g.state.players[p.idx()].res = [1, 0, 0, 0];
    for c in tzolkin::options::research_choices(&g.state, p, 1, false) {
        assert!(
            c.net_res(Resource::Skull) <= 1,
            "the top-of-track skull was doubled: {c}"
        );
    }
}

// ---- board spaces ------------------------------------------------------

/// Go discarded foresight options with `return Skip()` exactly when the
/// player's own space was full -- the case foresight exists for.
#[test]
fn foresight_survives_a_full_space() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()][Science::Theology.idx()] = 1; // foresight
    g.state.players[p.idx()].res[Resource::Skull.idx()] = 2;
    g.state.chichen_filled |= 1 << 1; // space 1 is used up

    let choices = choices_at(&g.state, p, Gear::Chichen, Pos(1));
    let real: Vec<_> = choices.iter().filter(|c| !c.is_skip()).collect();
    assert!(
        !real.is_empty(),
        "foresight should offer the next space up when this one is full"
    );
    // And it should be spending a skull on space 2, not space 1.
    assert!(real
        .iter()
        .any(|c| c.0.contains(&Effect::FillChichen(Pos(2)))));
}

/// Being devout is an option, not an obligation; Go removed the plain placement.
#[test]
fn devout_keeps_the_plain_placement() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()][Science::Theology.idx()] = 2; // devout
    g.state.players[p.idx()].res = [2, 0, 0, 1];

    let choices = choices_at(&g.state, p, Gear::Chichen, Pos(1));
    let plain = choices.iter().any(|c| {
        c.0.contains(&Effect::Res(Resource::Skull, -1))
            && !c.0.iter().any(|e| matches!(e, Effect::Res(r, n) if *n < 0 && *r != Resource::Skull))
    });
    assert!(plain, "a devout player must still be able to just place a skull");
}

/// Theology level 2, rulebook p.12, in full: "After performing an action on the
/// Chichen Itza gear, you may immediately spend 1 resource block to move up 1
/// step on the temple of your choice. **(If you gained a resource block from
/// your Chichen Itza action, it is available for you to spend in this way.)**"
///
/// The parenthetical is the whole rule: Chichen spaces 6, 8 and 9 hand out a
/// block of your choice, and the block-availability guard read the holding
/// *before* the action, so a devout player with an empty stock could not spend
/// the block the space had just given them.
#[test]
fn devout_may_spend_the_block_the_chichen_action_granted() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()] = [0, 0, 0, 2]; // theology 2: devout
    // One skull and not a single block in hand.
    g.state.players[p.idx()].res = [0, 0, 0, 1];

    // Chichen space 6 pays 8 points, a green temple step, and a block of the
    // player's choice -- the rulebook's own worked example on p.9.
    let choices = choices_at(&g.state, p, Gear::Chichen, Pos(6));
    let devout = choices.iter().any(|c| {
        c.0.contains(&Effect::FillChichen(Pos(6)))
            && c.0
                .iter()
                .any(|e| matches!(e, Effect::Res(r, n) if *r != Resource::Skull && *n < 0))
    });
    assert!(
        devout,
        "a devout player must be able to spend the block Chichen 6 just granted"
    );
}

/// Tikal's top action advances one step on each of two *different* temples.
///
/// Go offered ordered pairs, so each split appeared twice. An earlier pass here
/// over-corrected and allowed both steps on one temple, which the rules forbid.
#[test]
fn tikal_top_takes_two_different_temples() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].res = [3, 0, 0, 0];

    let choices = choices_at(&g.state, p, Gear::Tikal, Pos(5));
    let real: Vec<_> = choices.iter().filter(|c| !c.is_skip()).collect();
    assert!(!real.is_empty());

    for c in &real {
        let mut counts = std::collections::HashMap::new();
        for e in c.0.iter() {
            if let Effect::TempleStep(t, 1) = e {
                *counts.entry(*t).or_insert(0) += 1;
            }
        }
        assert!(
            counts.values().all(|&n| n == 1),
            "both steps landed on one temple: {c}"
        );
        assert_eq!(counts.len(), 2, "should step exactly two temples: {c}");
    }

    // Each unordered pair exactly once: three pairs, one block type spent.
    let mut sorted = real.clone();
    sorted.sort();
    let before = sorted.len();
    sorted.dedup();
    assert_eq!(before, sorted.len(), "duplicate temple-pair choices");
    assert_eq!(real.len(), 3, "three unordered pairs of distinct temples");
}

/// A worker on a gear's entry space can still be picked up; it just does
/// nothing. Go returned no options there, so such a worker was stuck.
#[test]
fn entry_space_can_be_vacated() {
    let g = fresh();
    for gear in Gear::ALL {
        let choices = choices_at(&g.state, PlayerId(0), gear, Pos(0));
        assert!(
            !choices.is_empty(),
            "{} entry space offers no way off the gear",
            gear.name()
        );
    }
}

/// The jungle "dig out the corn" action consumes one tile of each, so it needs
/// both. Go guarded only on wood and drove the corn count negative, which then
/// inverted `corn_showing` because that compares the two counts.
#[test]
fn jungle_never_drives_corn_negative() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.palenque[3] = tzolkin::state::TileStack { corn: 0, wood: 2 };

    for c in choices_at(&g.state, p, Gear::Palenque, Pos(3)) {
        let mut probe = g.state;
        c.apply(&mut probe, p);
        assert!(
            probe.palenque[3].corn == 0,
            "took a corn tile from a space with none: {c}"
        );
    }
}

// ---- flow --------------------------------------------------------------

/// Every unlocked worker eats, including the one on the first player space,
/// which Go's `Wheel_id != -1 || Available` test silently skipped.
#[test]
fn first_player_space_worker_still_eats() {
    let mut g = fresh();
    let p = PlayerId(0);
    let w = GameState::worker_ids(p).next().unwrap();
    g.state.place_on_first_player(w);
    assert_eq!(g.state.n_unlocked(p), 3, "all three unlocked workers count");
    assert!(matches!(g.state.loc(w), WorkerLoc::FirstPlayerSpace));
}

/// Free workers are applied before corn is spent, so a player holding corn
/// still benefits from the buildings that grant them.
#[test]
fn free_workers_apply_before_corn() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 20;
    g.state.players[p.idx()].free_workers = 3;
    let before = g.state.players[p.idx()].corn;

    // Walk to the first food day.
    while g.state.day < 7 {
        g.rotate_for_test();
    }
    let after = g.state.players[p.idx()].corn;
    assert_eq!(
        before, after,
        "three free workers should cover all three unlocked workers"
    );
}

/// Temple tracks clamp at both ends.
#[test]
fn temple_tracks_clamp() {
    let mut g = fresh();
    let p = PlayerId(0);
    for _ in 0..20 {
        g.state.temple_step(p, Temple::Brown, 1);
    }
    assert_eq!(g.state.temple_pos(p, Temple::Brown), 6, "clamped at the top");
    for _ in 0..20 {
        g.state.temple_step(p, Temple::Brown, -1);
    }
    assert_eq!(g.state.temple_pos(p, Temple::Brown), 0, "clamped at the bottom");
}

// ---- data tables -------------------------------------------------------

/// Age 1 and age 2 numbered from 1 independently in Go, so ids collided across
/// decks and the double build excluded the wrong card.
#[test]
fn building_ids_are_globally_unique() {
    let mut seen = std::collections::HashSet::new();
    for d in BUILDINGS.iter() {
        assert!(seen.insert(d.id), "duplicate building id {:?}", d.id);
    }
    assert_eq!(seen.len(), N_BUILDINGS);
    for (i, d) in BUILDINGS.iter().enumerate() {
        assert_eq!(d.id.0 as usize, i + 1, "BUILDINGS must be id-ordered");
    }
}

/// Building 30 was commented out in Go: face up, unbuildable, jamming a slot.
#[test]
fn every_building_is_buildable() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].res = [9, 9, 9, 9];
    g.state.players[p.idx()].corn = 40;

    for d in BUILDINGS.iter() {
        g.state.buildings_up = [Some(d.id), None, None, None, None, None];
        let choices = tzolkin::options::building_choices(&g.state, p, None, true, 1);
        assert!(
            !choices.is_empty(),
            "building {} offers nothing to a player who can afford it",
            d.id.0
        );
        assert!(
            choices.iter().all(|c| c.0.contains(&Effect::Build(d.id))),
            "building {} produced a choice that never builds it",
            d.id.0
        );
    }
}

/// Monuments 6, 10 and 13 all indexed past the end of a lookup table in Go.
/// #10 did it in every four-player game, so end-of-game scoring always panicked.
#[test]
fn every_monument_scores_in_every_reachable_state() {
    let mut g = fresh();
    let p = PlayerId(0);

    // The extremes: nothing, and everything.
    for id in MONUMENTS.iter().map(|m| m.id) {
        let _ = (mdef(id).score)(&g.state, p);
    }

    for q in PlayerId::ALL {
        for w in GameState::worker_ids(q) {
            g.state.workers[w.idx()] = WorkerLoc::Available; // all six in hand
        }
        for s in Science::ALL {
            g.state.research[q.idx()][s.idx()] = 3; // all tracks maxed
        }
        for t in Temple::ALL {
            for _ in 0..12 {
                g.state.temple_step(q, t, 1); // top of every temple
            }
        }
        for d in BUILDINGS.iter() {
            g.state.players[q.idx()].push_building(d.id);
        }
        for m in MONUMENTS.iter() {
            g.state.players[q.idx()].push_monument(m.id);
        }
    }
    g.state.chichen_filled = 0b111_1111_1110;

    for id in MONUMENTS.iter().map(|m| m.id) {
        let score = (mdef(id).score)(&g.state, p);
        assert!(score >= 0, "monument {} scored {score}", id.0);
    }
}

/// Costs must be payable with blocks only -- no building demands a skull.
#[test]
fn no_building_costs_a_skull() {
    for d in BUILDINGS.iter() {
        assert_eq!(
            d.cost[Resource::Skull.idx()],
            0,
            "building {} costs a skull",
            d.id.0
        );
    }
}

// ---- effects -----------------------------------------------------------

/// A choice's label is derived from its effects, so the log cannot describe
/// something other than what ran.
#[test]
fn label_is_derived_from_effects() {
    let c = Choice::of([Effect::Corn(4), Effect::TempleStep(Temple::Brown, 1)]);
    assert_eq!(c.to_string(), "+4 corn, B+1");
    assert_eq!(Choice::skip().to_string(), "skip");
}

/// The whole point of the data representation: state is `Copy`, so snapshotting
/// for search is a memcpy and moves can go in a hash set.
#[test]
fn state_is_copy_and_hashable() {
    fn assert_copy<T: Copy + std::hash::Hash + Eq>() {}
    assert_copy::<GameState>();

    let g = fresh();
    let snapshot = g.state; // a copy, not a borrow
    let mut set = std::collections::HashSet::new();
    set.insert(snapshot);
    assert!(set.contains(&g.state));
    assert!(
        std::mem::size_of::<GameState>() < 1024,
        "GameState is {} bytes; it should stay small enough to memcpy freely",
        std::mem::size_of::<GameState>()
    );
}

// ---- end to end --------------------------------------------------------

/// Games terminate, and the state stays coherent throughout.
#[test]
fn games_complete_with_invariants_held() {
    for seed in 0..50u64 {
        let mut g = Game::new(seed);
        g.run_checked()
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(g.state.over);
        assert_eq!(g.state.day, 27, "seed {seed} ended on day {}", g.state.day);
        validate(&g.state).unwrap();
    }
}

/// The same seed must produce the same game.
#[test]
fn play_is_deterministic() {
    let mut a = Game::new(7);
    let mut b = Game::new(7);
    a.run_random();
    b.run_random();
    assert_eq!(a.state, b.state);
    assert_eq!(a.scores(), b.scores());
}

// ---- retrieval ordering and paying down --------------------------------

/// A worker may take a lower space's action on its own gear, one corn per step
/// down. Go had this commented out as "first attempt broke".
#[test]
fn can_pay_corn_to_take_a_lower_action() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 5;

    let here = tzolkin::moves::choices_for_worker(&g.state, p, Gear::Yaxchilan, Pos(3));

    // Its own space: gold + two corn, free.
    assert!(here
        .iter()
        .any(|c| c.0.contains(&Effect::Res(Resource::Gold, 1)) && c.net_corn() == 2));

    // Two spaces down: wood, for two corn.
    assert!(
        here.iter().any(|c| {
            c.0.contains(&Effect::Res(Resource::Wood, 1)) && c.net_corn() == -2
        }),
        "should be able to pay 2 corn to take Yaxchilan 1's action from space 3"
    );

    // And a player with no corn gets only its own space.
    g.state.players[p.idx()].corn = 0;
    let broke = tzolkin::moves::choices_for_worker(&g.state, p, Gear::Yaxchilan, Pos(3));
    assert!(broke.iter().all(|c| c.net_corn() >= 0), "paid corn it lacks");
}

/// Retrieval order matters, and every legal ordering must be reachable.
///
/// Here the Yaxchilan worker's corn is what makes the Uxmal action affordable,
/// so the move only exists in one direction.
#[test]
fn retrieval_explores_orderings_that_matter() {
    let mut g = fresh();
    let p = PlayerId(0);
    let mut ws = GameState::worker_ids(p);
    let a = ws.next().unwrap();
    let b = ws.next().unwrap();
    for w in ws {
        g.state.workers[w.idx()] = WorkerLoc::Locked;
    }

    g.state.players[p.idx()].corn = 2;
    g.state.place_worker(a, Gear::Yaxchilan, Pos(2)); // +1 stone, +1 corn
    g.state.place_worker(b, Gear::Uxmal, Pos(1)); // pay 3 corn -> temple step

    let both: Vec<_> = legal_moves(&g.state, p)
        .into_iter()
        .filter(|m| {
            matches!(&m.kind, MoveKind::Retrieve(v)
                if v.iter().any(|(w, _)| *w == a) && v.iter().any(|(w, _)| *w == b))
        })
        .collect();
    assert!(!both.is_empty(), "should be able to retrieve both workers");

    // At least one such move must actually pay the 3 corn, which is only
    // possible if the Yaxchilan worker resolved first.
    let paid = both.iter().any(|m| match &m.kind {
        MoveKind::Retrieve(v) => {
            let ai = v.iter().position(|(w, _)| *w == a).unwrap();
            let bi = v.iter().position(|(w, _)| *w == b).unwrap();
            ai < bi && v[bi].1.net_corn() <= -3
        }
        _ => false,
    });
    assert!(
        paid,
        "the ordering that makes the Uxmal action affordable was never generated"
    );
}

/// Distinct moves must reach distinct positions; equivalent orderings collapse.
#[test]
fn retrieval_moves_have_distinct_outcomes() {
    for seed in 0..15u64 {
        let mut g = Game::new(seed);
        for _ in 0..4 {
            g.play_round();
        }
        for p in PlayerId::ALL {
            let mut outcomes = std::collections::HashSet::new();
            for m in legal_moves(&g.state, p) {
                if matches!(m.kind, MoveKind::Retrieve(_)) {
                    let mut probe = g.state;
                    tzolkin::moves::apply_move(&mut probe, p, &m);
                    assert!(
                        outcomes.insert(probe),
                        "seed {seed}: two retrieval moves reach the same state: {m}"
                    );
                }
            }
        }
    }
}

// ---- first player tile -------------------------------------------------

/// The marker stays put when nobody claims the first player space.
#[test]
fn marker_only_moves_when_claimed() {
    let mut g = fresh();
    g.state.first_player = PlayerId(2);
    g.state.first_player_space = None;
    g.resolve_first_player();
    assert_eq!(g.state.first_player, PlayerId(2), "marker moved unclaimed");

    let w = GameState::worker_ids(PlayerId(1)).next().unwrap();
    g.state.place_on_first_player(w);
    g.resolve_first_player();
    assert_eq!(g.state.first_player, PlayerId(1), "claimer takes the marker");
}

/// The first player tile flips back when its owner reaches the top of a temple.
#[test]
fn temple_top_flips_the_tile_back() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].may_skip_day = false;

    // Climbing without reaching the top changes nothing.
    g.state.temple_step(p, Temple::Brown, 1);
    assert!(!g.state.players[p.idx()].may_skip_day);

    // Arriving at the top flips it.
    while g.state.temple_pos(p, Temple::Brown) < 6 {
        g.state.temple_step(p, Temple::Brown, 1);
    }
    assert_eq!(g.state.temple_pos(p, Temple::Brown), 6);
    assert!(
        g.state.players[p.idx()].may_skip_day,
        "reaching the top of a temple must flip the tile back"
    );

    // Falling back down does not.
    g.state.players[p.idx()].may_skip_day = false;
    g.state.temple_step(p, Temple::Brown, -1);
    assert!(!g.state.players[p.idx()].may_skip_day);
}

// ---- sampling ----------------------------------------------------------

/// Every sampled move must survive the same legality check as an enumerated one.
#[test]
fn sampled_moves_are_legal() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);

    for seed in 0..60u64 {
        let mut g = Game::new(seed);
        while !g.state.over {
            g.state.current = g.state.first_player;
            for _ in 0..4 {
                let p = g.state.current;
                if let Some(m) = tzolkin::moves::sample_legal_move(&g.state, p, &mut rng) {
                    check_move(&g.state, p, &m)
                        .unwrap_or_else(|e| panic!("seed {seed}: sampled illegal move {m} -- {e}"));
                    g.play(p, &m);
                }
                validate(&g.state).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
                g.state.current = g.state.current.next(1);
            }
            g.resolve_first_player();
            g.rotate_for_test();
        }
        assert_eq!(g.state.day, 27);
    }
}

/// The sampler must find a move whenever enumeration finds one, and vice versa.
#[test]
fn sampler_agrees_with_enumeration_on_existence() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);

    for seed in 0..30u64 {
        let mut g = Game::new(seed);
        for _ in 0..8 {
            for p in PlayerId::ALL {
                let enumerated = !legal_moves(&g.state, p).is_empty();
                // The sampler is randomised, so give it several attempts before
                // concluding a position has no move.
                let sampled = (0..12).any(|_| {
                    tzolkin::moves::sample_legal_move(&g.state, p, &mut rng).is_some()
                });
                assert_eq!(
                    enumerated, sampled,
                    "seed {seed}: enumeration says {enumerated}, sampler says {sampled}"
                );
            }
            g.play_round();
        }
    }
}

// ---- rules-audit regressions -------------------------------------------

/// Everyone tied for the highest step gets HALF the bonus each -- not a share
/// split between them. My earlier `prize / n_tied` was the regression.
#[test]
fn temple_ties_pay_half_each() {
    let mut g = fresh();
    let t = Temple::Brown;
    let prize = tzolkin::data::temples::TEMPLES[t.idx()].age1_prize;

    // Three players level at step 3, one behind.
    for q in [PlayerId(0), PlayerId(1), PlayerId(2)] {
        g.state.temples[t.idx()][q.idx()] = 3;
    }
    g.state.temples[t.idx()][3] = 0;

    let base = tzolkin::data::temples::TEMPLES[t.idx()].points[3];
    let got = g.state.temple_points(PlayerId(0), 1);
    let others: i16 = Temple::ALL
        .iter()
        .filter(|&&x| x != t)
        .map(|&x| g.state.temple_points_of(PlayerId(0), x, 1))
        .sum();
    assert_eq!(got - others, base + prize / 2, "three-way tie should pay half each");
}

/// Two farm buildings bring feeding to zero corn; Go floored it at 1.
#[test]
fn feeding_can_cost_nothing() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 0;
    g.state.players[p.idx()].worker_discount = 2;
    let before = g.state.players[p.idx()].points;
    g.food_day();
    assert_eq!(
        g.state.players[p.idx()].points, before,
        "a player who feeds for free must not starve"
    );
}

/// A claimer who already holds the marker passes it to their left.
#[test]
fn first_player_self_pass() {
    let mut g = fresh();
    g.state.first_player = PlayerId(1);
    let w = GameState::worker_ids(PlayerId(1)).next().unwrap();
    g.state.place_on_first_player(w);
    g.resolve_first_player();
    assert_eq!(
        g.state.first_player,
        PlayerId(2),
        "holding the marker and claiming the space passes it left"
    );
}

/// A wasted temple step does not make a Chichen space illegal.
#[test]
fn chichen_usable_at_the_top_of_a_temple() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].res[Resource::Skull.idx()] = 1;
    // Park this player at the very top of the yellow track.
    let top = tzolkin::data::temples::TEMPLES[Temple::Yellow.idx()].steps - 1;
    g.state.temples[Temple::Yellow.idx()][p.idx()] = top;

    // Space 9 is the 13-point yellow space.
    let choices = choices_at(&g.state, p, Gear::Chichen, Pos(9));
    assert!(
        choices.iter().any(|c| c.0.contains(&Effect::Points(13))),
        "the step is wasted, but the 13 points are still on offer"
    );
}

/// Advancing past level 3 costs one block, not three.
#[test]
fn top_tier_research_costs_one_block() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()] = [3, 3, 3, 3];
    g.state.players[p.idx()].res = [1, 0, 0, 0];

    let choices = tzolkin::options::research_choices(&g.state, p, 1, false);
    assert!(
        !choices.is_empty(),
        "one block should buy a top-tier payoff"
    );
    assert!(choices
        .iter()
        .all(|c| c.net_res(Resource::Wood) >= -1));
}

/// "Do nothing except pick up the worker" is always available.
#[test]
fn doing_nothing_is_always_an_option() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 20;
    g.state.players[p.idx()].res = [5, 5, 5, 2];

    for gear in Gear::ALL {
        for pos in 0..gear.size() {
            let cs = tzolkin::moves::choices_for_worker(&g.state, p, gear, Pos(pos));
            assert!(
                cs.iter().any(|c| c.is_skip()),
                "{}:{pos} offers no way to decline",
                gear.name()
            );
        }
    }
}

/// The workers-in-play monument counts workers on gears too, and caps at 18.
#[test]
fn workers_in_play_monument() {
    let mut g = fresh();
    let p = PlayerId(0);
    for w in GameState::worker_ids(p) {
        g.state.workers[w.idx()] = WorkerLoc::Available;
    }
    let m6 = MonumentId(6);
    assert_eq!((mdef(m6).score)(&g.state, p), 18, "six in play scores 18");

    // Putting them on gears must not change anything.
    let mut n = 0;
    for w in GameState::worker_ids(p) {
        if n < 3 {
            g.state.place_worker(w, Gear::Yaxchilan, Pos(n));
            n += 1;
        }
    }
    assert_eq!((mdef(m6).score)(&g.state, p), 18, "on a gear is still in play");
}

/// There are exactly 13 crystal skulls and they are conserved.
#[test]
fn skull_supply_is_finite_and_conserved() {
    let mut g = fresh();
    assert_eq!(
        g.state.skulls_remaining as u32
            + PlayerId::ALL
                .iter()
                .map(|&p| g.state.players[p.idx()].get(Resource::Skull) as u32)
                .sum::<u32>(),
        13
    );

    // Drain the bank; further gains do nothing.
    let p = PlayerId(0);
    let taken = g.state.take_skulls(p, 100);
    assert_eq!(taken as u32 + 0, g.state.players[p.idx()].get(Resource::Skull) as u32 - 0);
    assert_eq!(g.state.skulls_remaining, 0);
    let before = g.state.players[p.idx()].get(Resource::Skull);
    Effect::Res(Resource::Skull, 1).apply(&mut g.state, p);
    assert_eq!(g.state.players[p.idx()].get(Resource::Skull), before, "bank is empty");

    // Yaxchilan 4 has no effect once the bank is dry.
    let cs = choices_at(&g.state, PlayerId(1), Gear::Yaxchilan, Pos(4));
    assert!(cs.iter().all(|c| c.is_skip()), "an empty bank grants no skulls");
}

/// The top step of each temple is exclusive.
#[test]
fn temple_top_step_is_exclusive() {
    let mut g = fresh();
    let t = Temple::Brown;
    let top = tzolkin::data::temples::TEMPLES[t.idx()].steps - 1;

    g.state.temples[t.idx()][0] = top;
    g.state.temples[t.idx()][1] = top - 1;

    assert!(
        !g.state.can_temple_step(PlayerId(1), t, 1),
        "the top step is taken"
    );
    g.state.temple_step(PlayerId(1), t, 1);
    assert_eq!(g.state.temple_pos(PlayerId(1), t), top - 1, "blocked below the top");

    // Once the occupant leaves, it opens.
    g.state.temples[t.idx()][0] = 0;
    assert!(g.state.can_temple_step(PlayerId(1), t, 1));
}

/// The market runs both ways.
#[test]
fn market_buys_and_sells() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 4;
    g.state.players[p.idx()].res = [0, 0, 1, 0]; // one gold

    let cs = tzolkin::options::corn_exchange(&g.state, p);
    assert!(
        cs.iter().any(|c| c.net_res(Resource::Gold) == -1 && c.net_corn() >= 4),
        "selling gold for 4 corn must be offered"
    );
    assert!(
        cs.iter().any(|c| c.net_res(Resource::Wood) == 2 && c.net_corn() == -4),
        "buying two wood for 4 corn must be offered"
    );
    assert!(cs.iter().any(|c| c.is_skip()), "exchanging nothing is allowed");
}

/// Ties are broken by workers left on the gears.
#[test]
fn tiebreak_by_workers_on_gears() {
    let mut g = fresh();
    for p in PlayerId::ALL {
        g.state.players[p.idx()].points = 50;
    }
    for w in GameState::worker_ids(PlayerId(2)).take(2) {
        g.state.place_worker(w, Gear::Tikal, Pos(w.0 % 3));
    }
    assert_eq!(g.winners(), vec![PlayerId(2)]);
}

/// With nothing affordable, no workers out and no temple to beg from, the gods
/// take pity: one worker on the cheapest space, all corn to the bank.
///
/// The earlier version of this test claimed to fill the entry spaces and did
/// not, so an ordinary placement always existed and `Pity` was never generated.
#[test]
fn gods_take_pity() {
    let mut g = fresh();
    let p = PlayerId(0);

    // Every cost-0 space taken by somebody else, and the first player space too.
    let mut donor = GameState::worker_ids(PlayerId(1)).chain(GameState::worker_ids(PlayerId(2)));
    for gear in Gear::ALL {
        let w = donor.next().unwrap();
        g.state.place_worker(w, gear, Pos(0));
    }
    g.state.place_on_first_player(donor.next().unwrap());

    // Our player: no corn, no workers out, nothing to beg with.
    g.state.players[p.idx()].corn = 0;
    for t in Temple::ALL {
        g.state.temples[t.idx()][p.idx()] = 0;
    }
    for w in GameState::worker_ids(p) {
        g.state.workers[w.idx()] = WorkerLoc::Available;
    }

    assert!(
        !tzolkin::moves::has_normal_move(&g.state, p),
        "test setup failed: an ordinary move still exists"
    );

    let moves = legal_moves(&g.state, p);
    assert!(!moves.is_empty(), "a player always has something to do");
    assert!(
        moves.iter().all(|m| matches!(m.kind, MoveKind::Pity { .. })),
        "only the pity move should be available here"
    );
    for m in &moves {
        check_move(&g.state, p, m).unwrap_or_else(|e| panic!("{m}: {e}"));
    }

    // And the sampler must find it too, not silently pass.
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let sampled = tzolkin::moves::sample_legal_move(&g.state, p, &mut rng);
    assert!(
        matches!(sampled.map(|m| m.kind), Some(MoveKind::Pity { .. })),
        "the sampler must reach the pity rule"
    );
}

/// Skipping over a food day defers it to the round landed on, rather than
/// feeding everyone at the end of the round left behind.
#[test]
fn extra_day_defers_the_food_day() {
    let mut g = fresh();
    // Park everyone on day 7, one short of the first food day.
    while g.state.day < 7 {
        g.state.advance_days(1);
    }
    for p in PlayerId::ALL {
        g.state.players[p.idx()].corn = 40;
    }
    let corn_before: Vec<u8> = PlayerId::ALL
        .iter()
        .map(|&p| g.state.players[p.idx()].corn)
        .collect();

    // Two days at once, over the food day at 8 and landing on 9.
    g.state.advance_days(2);
    assert_eq!(g.state.day, 9);

    let fed: Vec<bool> = PlayerId::ALL
        .iter()
        .enumerate()
        .map(|(i, &p)| g.state.players[p.idx()].corn < corn_before[i])
        .collect();
    assert!(
        fed.iter().all(|&f| f),
        "the food day passed over must still resolve, on the round landed on"
    );
}

/// The extra day may not push any worker off a gear, and may not run past the
/// end of the calendar.
#[test]
fn extra_day_restrictions() {
    let mut g = fresh();
    let p = PlayerId(0);
    assert!(g.state.may_take_extra_day(p));

    // A worker one space from the end blocks it; one on the very end does not,
    // since the ordinary advance pushes that one off anyway.
    let w = GameState::worker_ids(PlayerId(1)).next().unwrap();
    let last = Gear::Tikal.size() - 1;
    g.state.place_worker(w, Gear::Tikal, Pos(last - 1));
    assert!(!g.state.may_take_extra_day(p), "space {} must block", last - 1);

    g.state.retrieve_worker(w);
    g.state.place_worker(w, Gear::Tikal, Pos(last));
    assert!(g.state.may_take_extra_day(p), "the last space must not block");

    // And not off the end of the calendar.
    g.state.retrieve_worker(w);
    while g.state.day < tzolkin::state::LAST_DAY - 1 {
        g.state.advance_days(1);
    }
    assert!(!g.state.may_take_extra_day(p), "cannot advance past the end");
}

// ---- ui ----------------------------------------------------------------

/// The TUI must render at any plausible terminal size, at any point in a game,
/// without panicking. Ratatui will happily index out of a too-small buffer.
#[test]
fn ui_renders_at_many_sizes() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use tzolkin::eval::Ranking;
    use tzolkin::ui::{self, App, MoveSource};

    for &(w, h) in &[(80u16, 24u16), (100, 30), (132, 44), (200, 60), (60, 20)] {
        for rounds in [0usize, 8, 20, 27] {
            let mut game = Game::new(3);
            for _ in 0..rounds {
                if game.state.over {
                    break;
                }
                game.play_round();
            }

            let p = game.state.current;
            let ranking = tzolkin::eval::rank_all(&game.state, p, 10);
            let n = ranking.moves.len();

            for source in [MoveSource::Sampled, MoveSource::Full, MoveSource::Agent] {
                for exhaustive in [true, false] {
                    let app = App {
                        game: Game::new(3),
                        agent: None,
                        agent_name: "net".into(),
                        thinking: None,
                        last_decisions: Vec::new(),
                        last_played: None,
                        ranking: Ranking {
                            exhaustive,
                            note: "x".repeat(200),
                            ..ranking.clone()
                        },
                        selected: n.saturating_sub(1),
                        source,
                        autoplay: false,
                        status: "x".repeat(200),
                        focus: None,
                    };
                    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                    term.draw(|f| ui::draw(f, &app))
                        .unwrap_or_else(|e| panic!("{w}x{h} after {rounds} rounds: {e}"));
                }
            }
        }
    }
}

/// The two screens the size sweep above cannot reach: the mid-search display,
/// and the sub-decision pane with rows in it.
///
/// Both are new panels with their own width arithmetic, and both are states a
/// dump can only be taken of deliberately — by the time a search returns, the
/// thinking screen is gone. Same sizes as `ui_renders_at_many_sizes`, so a
/// panel that only breaks at 60x20 is caught here too.
#[test]
fn ui_renders_while_thinking_and_after_a_search() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use tzolkin::eval::Ranking;
    use tzolkin::ui::{self, App, Decision, MoveSource, Thinking};

    let mut game = Game::new(3);
    for _ in 0..6 {
        game.play_round();
    }
    let p = game.state.current;
    let ranking = tzolkin::eval::rank_all(&game.state, p, 10);

    // A chain long enough to overflow a short pane, with and without a rival
    // edge, since the runner-up is what the row's width budget is split for.
    let decisions: Vec<Decision> = (0..12)
        .map(|i| Decision {
            phase: "Take".repeat(1 + i % 3),
            chosen: "-1W, -1G, build #7, +1 corn, free worker+1, G+1".into(),
            share: 0.5 + 0.04 * i as f32,
            runner_up: (i % 3 != 0).then(|| ("place first player space".to_string(), 0.11)),
            edges: 1 + i * 7,
            visits: 8192,
            value: -0.9 + 0.15 * i as f32,
        })
        .collect();

    let thinking = Thinking {
        what: "searching R's turn".into(),
        detail: "x".repeat(200),
        elapsed: std::time::Duration::from_secs_f64(3.7),
        stage: Some((2, 2)),
        known: vec!["y".repeat(200), "chose retrieve w0[+3 corn]".into()],
        prior: Some(std::time::Duration::from_secs_f64(2.9)),
    };

    for &(w, h) in &[(80u16, 24u16), (100, 30), (132, 44), (200, 60), (60, 20)] {
        for busy in [None, Some(thinking.clone()), Some(Thinking::default())] {
            for decs in [Vec::new(), decisions.clone()] {
                for source in [MoveSource::Full, MoveSource::Agent] {
                    let app = App {
                        game: Game::new(3),
                        agent: None,
                        agent_name: "mcts8192/heuristic:pt=1:pmin=2:cp=0.02".into(),
                        thinking: busy.clone(),
                        last_decisions: decs.clone(),
                        last_played: Some("R played ".to_string() + &"z".repeat(200)),
                        ranking: Ranking {
                            note: "visit share (%) of 8192 sims — NOT the whole move space".into(),
                            ..ranking.clone()
                        },
                        selected: 0,
                        source,
                        autoplay: false,
                        status: "x".repeat(200),
                        focus: None,
                    };
                    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                    term.draw(|f| ui::draw(f, &app))
                        .unwrap_or_else(|e| panic!("{w}x{h}: {e}"));
                }
            }
        }
    }
}

/// The selection is over the *folded* rows, and every one of them describes
/// something.
///
/// `j`/`k` used to wrap on `Ranking::moves.len()` while `selected_move` indexed
/// the folded list, so past the last row the preview pane went blank. Both ends
/// of the walk are checked because the wrap is where it went wrong.
#[test]
fn selection_stays_on_a_real_row() {
    let mut game = Game::new(3);
    for _ in 0..7 {
        game.play_round();
    }
    let p = game.state.current;
    let mut app = tzolkin::ui::App {
        ranking: tzolkin::eval::rank_all(&game.state, p, 10),
        game,
        agent: None,
        agent_name: String::new(),
        thinking: None,
        last_decisions: Vec::new(),
        last_played: None,
        selected: 0,
        source: tzolkin::ui::MoveSource::Full,
        focus: None,
        autoplay: false,
        status: String::new(),
    };
    let n = app.rows().len();
    assert!(n > 0, "the fixture position should have moves");

    for _ in 0..n * 2 + 3 {
        assert!(app.selected < n, "selection {} left {n} rows", app.selected);
        assert!(app.selected_move().is_some(), "row {} describes nothing", app.selected);
        app.step_selection(1);
    }
    for _ in 0..n * 2 + 3 {
        app.step_selection(-1);
        assert!(app.selected < n, "backwards selection {} left {n} rows", app.selected);
    }
}

/// A `skip` pickup written in three places is one outcome written three ways,
/// and the panel shows it once — with the workers it returns named in the
/// suffix, sorted, so two spellings do not render as two different rows.
#[test]
fn the_shortlist_folds_restated_retrievals() {
    use tzolkin::effect::Choice;
    use tzolkin::ids::WorkerId;
    use tzolkin::moves::{Move, MoveKind, Retrievals};

    let act = Choice::new().with(tzolkin::effect::Effect::Corn(3));
    let skip = Choice::new();
    let mk = |v: Retrievals| Move {
        kind: MoveKind::Retrieve(v),
        beg: None,
        corn_cost: 0,
    };
    let w = |i: u8| WorkerId(i);

    // One effectful pickup and two idle ones, in three different orders.
    let a = mk([(w(0), act.clone()), (w(2), skip.clone()), (w(3), skip.clone())].into_iter().collect());
    let b = mk([(w(2), skip.clone()), (w(0), act.clone()), (w(3), skip.clone())].into_iter().collect());
    let c = mk([(w(3), skip.clone()), (w(2), skip.clone()), (w(0), act.clone())].into_iter().collect());
    // A genuinely different move: one fewer worker comes back.
    let d = mk([(w(0), act.clone()), (w(2), skip.clone())].into_iter().collect());

    assert!(tzolkin::ui::same_outcome(&a, &b));
    assert!(tzolkin::ui::same_outcome(&a, &c));
    assert!(!tzolkin::ui::same_outcome(&a, &d), "a dropped pickup is a real difference");

    // The suffix names the idle workers in a fixed order, or the fold would be
    // showing one outcome under two different labels.
    assert_eq!(tzolkin::ui::row_text(&a), tzolkin::ui::row_text(&c));
    assert!(
        tzolkin::ui::row_text(&a).ends_with("+ w2,w3 to hand"),
        "got {}",
        tzolkin::ui::row_text(&a)
    );

    let g = Game::new(3).state;
    let rows = tzolkin::ui::fold_rows(
        &[(a, 40.0), (b, 20.0), (c, 10.0), (d, 5.0)],
        &g,
        g.current,
    );
    assert_eq!(rows.len(), 2, "three spellings of one outcome, plus one other");
    assert_eq!(rows[0].spellings, 3);
    // The share is summed, not taken from the best spelling: the reader is being
    // told how much of the search went to this *outcome*.
    assert!((rows[0].score - 70.0).abs() < 1e-3, "got {}", rows[0].score);
    assert_eq!(rows[1].spellings, 1);
}

/// Long rows lose their middle, never their ends.
///
/// The fold puts what makes a row different in the suffix, so right-truncation
/// erased exactly the characters a reader is scanning for; `fit_tail` exists
/// for prose whose closing clause is a warning.
#[test]
fn elision_keeps_both_ends() {
    let s = "retrieve w0[+3 corn] w1[-3 corn, G+1] + w2,w3 to hand";
    let cut = tzolkin::ui::fit(s, 30);
    assert_eq!(cut.chars().count(), 30);
    assert!(cut.starts_with("retrieve w0"), "got {cut}");
    assert!(cut.ends_with("to hand"), "got {cut}");
    assert_eq!(tzolkin::ui::fit(s, 200), s, "nothing to cut is nothing to do");

    let note = "mcts8192 · visit share of 8192 sims — NOT the whole move space";
    let tail = tzolkin::ui::fit_tail(note, 40);
    assert_eq!(tail.chars().count(), 40);
    assert!(tail.ends_with("NOT the whole move space"), "got {tail}");
}

/// A worker's lap is arithmetic, and the calendar is part of it.
///
/// A gear advances one space a day, so a worker on `pos` is on `pos + k` on day
/// `day + k` and rides off when that would reach `Gear::size()`. The panel
/// prints this beside the wheel instead of animating it, so the sum has to be
/// the game's and not a plausible-looking one — and it has to stop at day 27,
/// because a worker put on Tikal on day 25 does **not** reach the top of it and
/// a viewer that said `→7` there would be inventing a plan the game has no
/// time for. `docs/FINDINGS-tui.md` T11.
#[test]
fn a_lap_stops_at_the_end_of_the_calendar() {
    use tzolkin::ui::{lap, lap_str};

    // Mid-game: the whole ride is available.
    let (last, off, ends) = lap(Gear::Tikal, Pos(1), 11);
    assert_eq!((last, off, ends), (7, 18, false), "1 + 7 days is the top, then off");
    assert_eq!(lap_str(Gear::Tikal, Pos(1), 11), "1→7 d18");

    // The entry space of the big gear, from the same day.
    let (last, off, ends) = lap(Gear::Chichen, Pos(0), 11);
    assert_eq!((last, off, ends), (10, 22, false));

    // Late: the calendar runs out first, and the panel must say so rather than
    // promise a space the worker never stands on.
    let (last, off, ends) = lap(Gear::Tikal, Pos(2), 25);
    assert_eq!(last, 4, "two days left, so 2 -> 3 -> 4 and no further");
    assert!(ends, "off day {off} is past the last day");
    assert!(lap_str(Gear::Tikal, Pos(2), 25).ends_with("end"), "{}", lap_str(Gear::Tikal, Pos(2), 25));

    // The top space: nowhere left to ride, on any day.
    assert_eq!(lap_str(Gear::Tikal, Pos(7), 3), "7·");
    assert_eq!(lap(Gear::Tikal, Pos(7), 3).0, 7);

    // Never off the end of the gear, at any day, from any space.
    for gear in Gear::ALL {
        for pos in 0..gear.size() {
            for day in 0..=tzolkin::state::LAST_DAY {
                let (last, _, _) = lap(gear, Pos(pos), day);
                assert!(last < gear.size(), "{gear:?} {pos} d{day} rode to {last}");
                assert!(last >= pos, "{gear:?} {pos} d{day} went backwards to {last}");
            }
        }
    }
}

/// The board tells a placed worker from a retrieved one **in a screen dump**.
///
/// Three markings carry it — the glyph, the paint, and the blink rate — and a
/// terminal may honour none of the last two: `Modifier::SLOW_BLINK` is widely
/// ignored, and a dump has no colour at all. So the glyph is the one that has
/// to work, and this test reads the buffer as text, exactly as review does.
/// The same assertion covers the ring and the flat-row fallback, because at
/// 60x20 there is no room for rings and the markers must survive that too.
#[test]
fn the_board_marks_placed_and_retrieved_workers_without_colour() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use tzolkin::ui::{self, App, Marks, MoveSource};

    // Two positions: an opening, where the shortlist is placements, and a
    // late one, where it is retrievals off gears workers are standing on.
    for rounds in [9usize, 14, 20] {
        let mut game = Game::new(7);
        for _ in 0..rounds {
            if game.state.over {
                break;
            }
            game.play_round();
        }
        let p = game.state.current;
        let ranking = tzolkin::eval::rank_all(&game.state, p, 10);
        if ranking.moves.is_empty() {
            continue;
        }

        let n = ranking.moves.len();
        let state = game.state;
        let mut app = App {
            game,
            agent: None,
            agent_name: String::new(),
            thinking: None,
            last_decisions: Vec::new(),
            last_played: None,
            ranking,
            selected: 0,
            source: MoveSource::Full,
            autoplay: false,
            status: String::new(),
            focus: None,
        };
        for sel in 0..n {
            let marks = Marks::of(&state, Some(&app.ranking.moves[sel].0));
            if marks.placed.is_empty() && marks.taken.is_empty() {
                continue;
            }
            app.selected = sel;
            for &(w, h) in &[(80u16, 24u16), (132, 44), (60, 20)] {
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                term.draw(|f| ui::draw(f, &app)).unwrap();
                let buf = term.backend().buffer();
                let text: String = (0..h)
                    .map(|y| {
                        (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>() + "\n"
                    })
                    .collect();

                // The cell marker. `+Y` and `-B` are the two-column forms; the
                // three-column ring writes `+Y+` and `-B-`, which contain them.
                for &(_, _, c) in &marks.placed {
                    assert!(
                        text.contains(&format!("+{c}")),
                        "{w}x{h} r{rounds} sel {sel}: no +{c} anywhere on screen\n{text}"
                    );
                }
                for &(_, _, c) in &marks.taken {
                    assert!(
                        text.contains(&format!("-{c}")),
                        "{w}x{h} r{rounds} sel {sel}: no -{c} anywhere on screen\n{text}"
                    );
                }
                // And the two are never spelled the same way. At the size
                // where the Why pane is drawn whole, the distinction is also
                // written out in words, which is what survives a reader who
                // does not know the glyph convention.
                if (w, h) == (132, 44) {
                    for &(_, _, c) in &marks.placed {
                        assert!(
                            text.contains(&format!("+{c} onto")),
                            "{w}x{h} r{rounds} sel {sel}: +{c} is not said to arrive\n{text}"
                        );
                    }
                    for &(_, _, c) in &marks.taken {
                        assert!(
                            text.contains(&format!("-{c} off")),
                            "{w}x{h} r{rounds} sel {sel}: -{c} is not said to leave\n{text}"
                        );
                    }
                }
            }
        }
    }
}

/// The Board panel says what a space does at **every** size it renders at.
///
/// It was drawing five rings and not one word about any space at 80x24 — the
/// default terminal, and the size where the panel is hardest to fit and most
/// needed. The rings alone are eight numbers in a circle. `FINDINGS-tui.md`
/// T10.1.
#[test]
fn the_board_always_says_what_at_least_one_space_does() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use tzolkin::ui::{self, App, MoveSource};

    for &(w, h) in &[(60u16, 20u16), (80, 24), (100, 30), (132, 44), (200, 60)] {
        for rounds in [0usize, 9, 20] {
            let mut game = Game::new(7);
            for _ in 0..rounds {
                if game.state.over {
                    break;
                }
                game.play_round();
            }
            let p = game.state.current;
            let ranking = tzolkin::eval::rank_all(&game.state, p, 10);
            let mut app = App {
                game,
                agent: None,
                agent_name: String::new(),
                thinking: None,
                last_decisions: Vec::new(),
                last_played: None,
                ranking,
                selected: 0,
                source: MoveSource::Full,
                autoplay: false,
                status: String::new(),
                focus: None,
            };
            for focus in [None, Some(Gear::Chichen), Some(Gear::Palenque)] {
                app.focus = focus;
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                term.draw(|f| ui::draw(f, &app)).unwrap();
                let buf = term.backend().buffer();
                let text: String = (0..h)
                    .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>() + "\n")
                    .collect();
                // `0 enter` is the entry space's label on every gear, and it is
                // the first cell of the space list whichever way the list is
                // laid out — table, flowed sentence, or one truncated row.
                assert!(
                    text.contains("0 enter"),
                    "{w}x{h} r{rounds} focus {focus:?}: the Board panel names no space\n{text}"
                );
                // And the lap line, which is the half of "moving in circles"
                // that a still picture cannot draw.
                assert!(
                    text.contains("rides") || text.contains("move   "),
                    "{w}x{h} r{rounds}: neither the move line nor the rides line survived\n{text}"
                );
            }
        }
    }
}

/// Temples never hides a player.
///
/// `Paragraph` truncates its tail and this panel's tail is step 0, which is
/// where everybody starts — so one row short of the tallest track, the row it
/// dropped was the most-occupied one on the board. At 132x44 that was exactly
/// the case: the panel drew steps 8 down to 1 and three players standing on 0
/// were simply not on screen. `docs/FINDINGS-tui.md` T10.4.
#[test]
fn the_temple_panel_never_truncates_away_an_occupied_step() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use tzolkin::ui::{self, App, MoveSource};

    for &(w, h) in &[(100u16, 30u16), (132, 44), (150, 40), (200, 60)] {
        for rounds in [0usize, 9, 20, 27] {
            let mut game = Game::new(7);
            for _ in 0..rounds {
                if game.state.over {
                    break;
                }
                game.play_round();
            }
            let p = game.state.current;
            let ranking = tzolkin::eval::rank_all(&game.state, p, 10);
            let colors: Vec<char> =
                PlayerId::ALL.iter().map(|q| game.state.players[q.idx()].color.letter()).collect();
            let app = App {
                game,
                agent: None,
                agent_name: String::new(),
                thinking: None,
                last_decisions: Vec::new(),
                last_played: None,
                ranking,
                selected: 0,
                source: MoveSource::Full,
                autoplay: false,
                status: String::new(),
                focus: None,
            };
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| ui::draw(f, &app)).unwrap();
            let buf = term.backend().buffer();
            let rowtext = |y: u16| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>();

            // Find the panel by its title and read only its own columns, so a
            // colour letter elsewhere on the screen cannot stand in for one
            // that is missing here.
            let Some(top) = (0..h).find(|&y| rowtext(y).contains("┌ Temples")) else {
                continue; // too short to draw at all; that is a layout choice
            };
            let line = rowtext(top);
            let x0 = line.find("┌ Temples").expect("just matched") as u16;
            let x0 = line[..x0 as usize].chars().count() as u16;
            let bot = (top + 1..h)
                .find(|&y| rowtext(y)[..].contains('└'))
                .unwrap_or(h - 1);
            let region: String = (top + 1..bot)
                .map(|y| (x0..w).map(|x| buf[(x, y)].symbol()).collect::<String>() + "\n")
                .collect();

            // Every player stands on all three tracks, so every player's letter
            // must appear at least three times inside the panel.
            for c in colors {
                let n = region.matches(c).count();
                assert!(
                    n >= 3,
                    "{w}x{h} r{rounds}: {c} on {} of 3 temple tracks in the panel\n{region}",
                    n
                );
            }
        }
    }
}

/// The preview pane describes exactly what the move does.
#[test]
fn preview_diff_describes_the_move() {
    let g = fresh();
    let p = PlayerId(0);
    let before = g.state;
    let mut after = before;
    after.players[p.idx()].corn += 5;
    after.temple_step(p, Temple::Brown, 1);

    let lines = tzolkin::ui::diff_lines(&before, &after);
    let text: String = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
        .collect();
    assert!(text.contains("corn"), "diff should mention the corn change: {text}");
    assert!(text.contains("temple"), "diff should mention the temple step: {text}");

    let same = tzolkin::ui::diff_lines(&before, &before);
    let text: String = same
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
        .collect();
    assert!(text.contains("no change"), "got: {text}");
}

// ---- starting tiles ----------------------------------------------------

/// The 21 starting wealth tiles, pinned against a physical copy.
///
/// An audit flagged these as missing effects the rulebook's page-16 glossary
/// defines. That glossary is headed "Starting Wealth Tiles *and Building
/// Effects*" and is shared between the two: the choice-bearing symbols belong
/// to the buildings. Checked tile by tile, this table is complete, and no
/// starting tile carries a decision.
#[test]
fn starting_tiles_match_the_components() {
    use tzolkin::data::tiles::{TILES, N_TILES};
    use tzolkin::effect::Effect::*;

    let expected: [&[tzolkin::effect::Effect]; N_TILES] = [
        &[Corn(3), Res(Resource::Wood, 1), FreeWorker(1)],
        &[Corn(6), Res(Resource::Wood, 1), Res(Resource::Stone, 1)],
        &[Corn(2), Res(Resource::Wood, 2), TempleStep(Temple::Green, 1)],
        &[UnlockWorker],
        &[Corn(8), Res(Resource::Gold, 1)],
        &[Corn(4), Res(Resource::Wood, 3)],
        &[Corn(7), Res(Resource::Wood, 2)],
        &[Corn(6), Res(Resource::Stone, 2)],
        &[Corn(3), Res(Resource::Wood, 2), Res(Resource::Stone, 1)],
        &[Res(Resource::Wood, 1), TempleStep(Temple::Green, 1), AdvanceResearch(Science::Extraction)],
        &[Res(Resource::Stone, 1), Res(Resource::Gold, 1), AdvanceResearch(Science::Agriculture)],
        &[Corn(4), Res(Resource::Wood, 1), AdvanceResearch(Science::Extraction)],
        &[Corn(5), Res(Resource::Gold, 1), TempleStep(Temple::Yellow, 1)],
        &[Corn(5), Res(Resource::Stone, 1), TempleStep(Temple::Brown, 1)],
        &[Corn(9), Res(Resource::Stone, 1)],
        &[Corn(2), TempleStep(Temple::Brown, 1), AdvanceResearch(Science::Architecture)],
        &[Corn(3), TempleStep(Temple::Yellow, 1), AdvanceResearch(Science::Agriculture)],
        &[Corn(4), Res(Resource::Wood, 1), Res(Resource::Skull, 1)],
        &[Corn(5), Res(Resource::Stone, 1), AdvanceResearch(Science::Theology)],
        &[Corn(3), Res(Resource::Gold, 1), AdvanceResearch(Science::Architecture)],
        &[Corn(2), Res(Resource::Wood, 2), AdvanceResearch(Science::Theology)],
    ];

    for (i, (got, want)) in TILES.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got, want, "starting tile {} does not match the components", i + 1);
    }
}

/// No starting tile carries a decision, so a flat effect list is the right
/// shape. If a tile ever needs one, this fails and says so.
#[test]
fn no_starting_tile_needs_a_choice() {
    use tzolkin::data::tiles::TILES;
    use tzolkin::effect::Effect;
    for (i, tile) in TILES.iter().enumerate() {
        for e in tile.iter() {
            let choice_free = matches!(
                e,
                Effect::Corn(_)
                    | Effect::Res(..)
                    | Effect::Points(_)
                    | Effect::TempleStep(..)
                    | Effect::AdvanceResearch(_)
                    | Effect::UnlockWorker
                    | Effect::FreeWorker(_)
                    | Effect::WorkerDiscount(_)
            );
            assert!(choice_free, "tile {} carries {e:?}, which is a decision", i + 1);
        }
    }
}

/// Setup produces a coherent state.
#[test]
fn setup_with_tile_choices_is_valid() {
    for seed in 0..40u64 {
        let g = Game::new(seed);
        validate(&g.state).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        for p in PlayerId::ALL {
            assert!(
                g.state.players[p.idx()].corn <= 60,
                "seed {seed}: implausible starting corn"
            );
        }
    }
}

/// Where the placeholder evaluator can and cannot tell moves apart.
///
/// A worker's value only materialises when it is *retrieved*, so a one-ply
/// evaluation of "put a worker on gear X" has almost nothing to score. This
/// prints the spread of `eval::heuristic` across placement candidates against
/// retrieval candidates; run it to see how much of `heuristic:K`'s budget is
/// spent choosing between moves it cannot rank.
///
///     cargo test --release --test rules -- --ignored --nocapture evaluator_discrimination
#[test]
#[ignore]
fn evaluator_discrimination() {
    use rand::SeedableRng;
    use tzolkin::eval::heuristic;
    use tzolkin::moves::{apply_move, sample_legal_move, MoveKind};

    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let (mut place, mut retrieve) = (Vec::new(), Vec::new());

    for seed in 0..40u64 {
        let mut g = Game::new(seed);
        for _ in 0..(seed % 14) {
            if g.state.over {
                break;
            }
            g.play_round();
        }
        let p = g.state.current;

        let (mut ps, mut rs) = (Vec::new(), Vec::new());
        for _ in 0..80 {
            let Some(m) = sample_legal_move(&g.state, p, &mut rng) else {
                break;
            };
            let mut probe = g.state;
            apply_move(&mut probe, p, &m);
            probe.refill_buildings();
            let s = heuristic(&probe, p);
            match m.kind {
                MoveKind::Place(_) => ps.push(s),
                MoveKind::Retrieve(_) => rs.push(s),
                MoveKind::Pity { .. } => {}
            }
        }
        let spread = |v: &[f32]| {
            if v.len() < 2 {
                return None;
            }
            let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            Some(hi - lo)
        };
        if let Some(s) = spread(&ps) {
            place.push(s);
        }
        if let Some(s) = spread(&rs) {
            retrieve.push(s);
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "\nheuristic score spread within one position, over sampled candidates:\n  \
         placement : {:>6.2}  (n={})\n  retrieval : {:>6.2}  (n={})\n  ratio     : {:>6.1}x",
        mean(&place),
        place.len(),
        mean(&retrieve),
        retrieve.len(),
        mean(&retrieve) / mean(&place).max(0.001)
    );
    assert!(!place.is_empty() && !retrieve.is_empty());
}

// ---- evaluator calibration ---------------------------------------------

/// Ordinary least squares with a ridge term, by Gauss-Jordan on the normal
/// equations. `x` rows are already augmented with a 1 for the intercept.
///
/// Ridge rather than plain OLS because the evaluator's terms are correlated by
/// construction — a player with a big engine also holds blocks — and an
/// unregularised solve of a near-singular Gram matrix reports coefficients of
/// ±40 that flip sign between runs. λ is small enough to leave a well-posed
/// column alone and large enough that the answer is stable across seeds.
#[cfg(test)]
fn ridge(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let k = x[0].len();
    let mut a = vec![vec![0.0f64; k + 1]; k];
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..k {
            for j in 0..k {
                a[i][j] += row[i] * row[j];
            }
            a[i][k] += row[i] * yi;
        }
    }
    // No penalty on the intercept (column 0).
    for i in 1..k {
        a[i][i] += lambda * x.len() as f64;
    }
    for i in 0..k {
        let piv = (i..k).max_by(|&r, &s| a[r][i].abs().total_cmp(&a[s][i].abs())).unwrap();
        a.swap(i, piv);
        if a[i][i].abs() < 1e-12 {
            continue;
        }
        let d = a[i][i];
        for j in i..=k {
            a[i][j] /= d;
        }
        for r in 0..k {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for j in i..=k {
                a[r][j] -= f * a[i][j];
            }
        }
    }
    (0..k).map(|i| a[i][k]).collect()
}

/// Invert a small square matrix by Gauss-Jordan with partial pivoting.
#[cfg(test)]
fn invert(a0: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = a0.len();
    let mut a: Vec<Vec<f64>> = (0..k)
        .map(|i| a0[i].iter().copied().chain((0..k).map(|j| f64::from(i == j))).collect())
        .collect();
    for i in 0..k {
        let piv = (i..k).max_by(|&r, &s| a[r][i].abs().total_cmp(&a[s][i].abs())).unwrap();
        a.swap(i, piv);
        if a[i][i].abs() < 1e-12 {
            continue;
        }
        let d = a[i][i];
        for j in 0..2 * k {
            a[i][j] /= d;
        }
        for r in 0..k {
            if r == i || a[r][i] == 0.0 {
                continue;
            }
            let f = a[r][i];
            for j in 0..2 * k {
                a[r][j] -= f * a[i][j];
            }
        }
    }
    a.into_iter().map(|r| r[k..].to_vec()).collect()
}

/// [`ridge`], and a standard error for every coefficient.
///
/// The iid standard error is meaningless on this data and reading these
/// coefficients without one is how a fit of noise gets promoted to a finding.
/// All four seats of a position share one regressor draw, and every position in
/// a game shares one *outcome vector* — 100-odd rows per game whose y is four
/// numbers. So errors are clustered on the game, which is the same argument
/// `bin/arena` makes for treating a rotation block rather than a game as its
/// independent unit. The clustered interval here comes out 5-15x the iid one.
///
/// Sandwich form: `A^-1 (sum_g Xg'eg eg'Xg) A^-1` with `A = X'X + lambda n I`,
/// scaled by `G/(G-1)`. `cl` names each row's game and need not be sorted.
#[cfg(test)]
fn ridge_se(x: &[Vec<f64>], y: &[f64], cl: &[u64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let k = x[0].len();
    let mut a = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..k {
            for j in 0..k {
                a[i][j] += row[i] * row[j];
            }
            xty[i] += row[i] * yi;
        }
    }
    for i in 1..k {
        a[i][i] += lambda * x.len() as f64;
    }
    let ainv = invert(&a);
    let beta: Vec<f64> =
        (0..k).map(|i| (0..k).map(|j| ainv[i][j] * xty[j]).sum()).collect();

    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_unstable_by_key(|&i| cl[i]);
    let mut meat = vec![vec![0.0f64; k]; k];
    let mut score = vec![0.0f64; k];
    let mut groups = 0usize;
    let flush = |score: &mut Vec<f64>, meat: &mut Vec<Vec<f64>>| {
        for i in 0..k {
            for j in 0..k {
                meat[i][j] += score[i] * score[j];
            }
        }
        score.iter_mut().for_each(|v| *v = 0.0);
    };
    let mut cur = None;
    for &i in &order {
        if cur != Some(cl[i]) {
            if cur.is_some() {
                flush(&mut score, &mut meat);
                groups += 1;
            }
            cur = Some(cl[i]);
        }
        let e = y[i] - (0..k).map(|j| beta[j] * x[i][j]).sum::<f64>();
        for j in 0..k {
            score[j] += x[i][j] * e;
        }
    }
    if cur.is_some() {
        flush(&mut score, &mut meat);
        groups += 1;
    }
    let adj = groups as f64 / (groups.max(2) - 1) as f64;
    let se = (0..k)
        .map(|d| {
            let mut v = 0.0;
            for i in 0..k {
                for j in 0..k {
                    v += ainv[d][i] * meat[i][j] * ainv[j][d];
                }
            }
            (v * adj).max(0.0).sqrt()
        })
        .collect();
    (beta, se)
}

/// Is `eval::heuristic` actually predicting the final score, and where is it
/// wrong?
///
///     cargo test --release --test rules -- --ignored --nocapture evaluator_calibration
///     TZ_CAL_GAMES=200 TZ_CAL_AGENT=heuristic:full cargo test --release ...   (slower)
///
/// The evaluator's contract is a number in points that estimates a seat's final
/// score, so the claim is directly testable: play games with a decent agent,
/// record the estimate at every turn root alongside that seat's realised final
/// score, and look at the four things that matter.
///
/// * **Bias and error by day.** A term that decays wrongly shows up as bias
///   that drifts with the calendar, which one pooled number hides.
/// * **Which component carries it.** `eval::components` is a linear
///   decomposition, so the realised score can be regressed on the eight terms.
///   A coefficient of 1.0 means the term is already scaled right; 2.0 means it
///   is worth double what it is being paid; ~0 means it is noise. That is the
///   number that says what to change, and it is why the decomposition exists.
/// * **Ordering.** For a search, ranking the seats right matters more than the
///   magnitude, so the pairwise concordance between estimated and realised
///   standings is reported separately from the error.
/// * **Scale.** `phase.rs::HeuristicEvaluator` squashes with
///   `tanh((raw - mean)/25.0)`, so the spread of the centred estimate has to be
///   compared against the spread of the centred final score.
#[test]
#[ignore]
fn evaluator_calibration() {
    use rand::SeedableRng;
    use rayon::prelude::*;
    use tzolkin::eval::{components, heuristic, margin, Components};
    use tzolkin::record::{flags, parse_agent, play_game, GameConfig};

    let games: u64 = std::env::var("TZ_CAL_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let spec = std::env::var("TZ_CAL_AGENT").unwrap_or_else(|_| "heuristic:32".into());

    /// One recorded turn root: the estimate for every seat, and what every seat
    /// finally scored.
    struct Row {
        /// The game this position came from: every row of one game shares an
        /// outcome vector, so it is the cluster the standard errors use.
        game: u64,
        day: u8,
        state: GameState,
        est: [f32; N_PLAYERS],
        parts: [Components; N_PLAYERS],
        margin: [f32; N_PLAYERS],
        outcome: [f32; N_PLAYERS],
    }

    let started = std::time::Instant::now();
    let rows: Vec<Row> = (0..games)
        .into_par_iter()
        .flat_map(|seed| {
            // One agent instance per game: a searching agent owns per-game state.
            let a = parse_agent(&spec, true).unwrap();
            let agents: [&dyn tzolkin::record::Agent; N_PLAYERS] =
                [a.as_ref(), a.as_ref(), a.as_ref(), a.as_ref()];
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xCA11B);
            let r = play_game(seed, &agents, &GameConfig::evaluation(), &mut rng);
            let outcome: [f32; N_PLAYERS] = std::array::from_fn(|i| r.scores[i] as f32);
            r.nodes
                .iter()
                .filter(|n| n.flags & flags::TURN_ROOT != 0 && !n.state.over)
                .map(|n| Row {
                    game: seed,
                    day: n.state.day,
                    state: n.state,
                    est: std::array::from_fn(|i| heuristic(&n.state, PlayerId(i as u8))),
                    parts: std::array::from_fn(|i| components(&n.state, PlayerId(i as u8))),
                    margin: std::array::from_fn(|i| margin(&n.state, PlayerId(i as u8))),
                    outcome,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let elapsed = started.elapsed().as_secs_f64();
    assert!(!rows.is_empty());

    // ---- helpers -------------------------------------------------------
    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len().max(1) as f64
    }
    fn sd(v: &[f64]) -> f64 {
        if v.len() < 2 {
            return f64::NAN;
        }
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    }
    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let (mx, my) = (mean(x), mean(y));
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        for i in 0..x.len() {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx).powi(2);
            syy += (y[i] - my).powi(2);
        }
        sxy / (sxx * syy).sqrt()
    }

    println!(
        "\n=== evaluator calibration ===\n{games} games of {spec}, \
         {} turn roots, {} (position, seat) pairs, {elapsed:.0}s",
        rows.len(),
        rows.len() * N_PLAYERS,
    );

    // ---- 1. error and bias by day --------------------------------------
    //
    // Buckets of three days. The calendar is 27 long and the interesting
    // structure is at the ends, so finer than a per-age split and coarser than
    // per-day, which at a few hundred games is too thin to read.
    println!(
        "\n-- accuracy by day (estimate vs that seat's final score) --\n\
         {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>6}",
        "days", "n", "est", "actual", "bias", "MAE", "r"
    );
    let bucket_of = |d: u8| (d / 3).min(8) as usize;
    let mut by_bucket: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); 9];
    for r in &rows {
        for i in 0..N_PLAYERS {
            let b = &mut by_bucket[bucket_of(r.day)];
            b.0.push(r.est[i] as f64);
            b.1.push(r.outcome[i] as f64);
        }
    }
    for (b, (est, act)) in by_bucket.iter().enumerate() {
        if est.len() < 10 {
            continue;
        }
        let err: Vec<f64> = est.iter().zip(act).map(|(e, a)| e - a).collect();
        println!(
            "{:>7}  {:>7}  {:>7.1}  {:>7.1}  {:>+7.2}  {:>7.2}  {:>6.3}",
            format!("{}-{}", b * 3, b * 3 + 2),
            est.len(),
            mean(est),
            mean(act),
            mean(&err),
            mean(&err.iter().map(|e| e.abs()).collect::<Vec<_>>()),
            corr(est, act),
        );
    }
    let all_est: Vec<f64> = rows.iter().flat_map(|r| r.est.iter().map(|&x| x as f64)).collect();
    let all_act: Vec<f64> = rows.iter().flat_map(|r| r.outcome.iter().map(|&x| x as f64)).collect();
    let all_err: Vec<f64> = all_est.iter().zip(&all_act).map(|(e, a)| e - a).collect();
    println!(
        "{:>7}  {:>7}  {:>7.1}  {:>7.1}  {:>+7.2}  {:>7.2}  {:>6.3}",
        "all",
        all_est.len(),
        mean(&all_est),
        mean(&all_act),
        mean(&all_err),
        mean(&all_err.iter().map(|e| e.abs()).collect::<Vec<_>>()),
        corr(&all_est, &all_act),
    );

    // ---- 2. which component carries the error --------------------------
    //
    // Regress the realised final score on the eight terms. The fitted
    // coefficient is what the term *should* be multiplied by; the printed
    // `bias` column is that term's own mean signed error contribution,
    // `(coef - 1) * mean(term)`, which is the points per position the current
    // weight is getting wrong.
    let names = Components::NAMES;
    let build = |sel: &dyn Fn(&Row) -> bool| -> (Vec<Vec<f64>>, Vec<f64>, Vec<u64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cl = Vec::new();
        for r in rows.iter().filter(|r| sel(r)) {
            for i in 0..N_PLAYERS {
                let t = r.parts[i].terms();
                let mut row = vec![1.0];
                row.extend(t.iter().map(|&v| v as f64));
                x.push(row);
                y.push(r.outcome[i] as f64);
                cl.push(r.game);
            }
        }
        (x, y, cl)
    };

    println!(
        "\n-- what each term is worth: OLS(final score ~ terms), whole game --\n\
         {:>12}  {:>8}  {:>14}  {:>8}  {:>8}",
        "term", "mean", "coef +/-95%", "should be", "pts/pos"
    );
    let (x, y, cl) = build(&|_| true);
    let (coefs, coef_se) = ridge_se(&x, &y, &cl, 1e-3);
    println!("{:>12}  {:>8}  {:>+8.2}", "(intercept)", "", coefs[0]);
    for (k, name) in names.iter().enumerate() {
        let col: Vec<f64> = x.iter().map(|r| r[k + 1]).collect();
        let m = mean(&col);
        println!(
            "{:>12}  {:>8.2}  {:>14}  {:>8.2}  {:>+8.2}",
            name,
            m,
            format!("{:.2} +/-{:.2}", coefs[k + 1], 1.96 * coef_se[k + 1]),
            coefs[k + 1] * m,
            (1.0 - coefs[k + 1]) * m,
        );
    }

    // The same fit split by phase, because a term that decays wrongly has a
    // coefficient that moves with the calendar.
    for (label, lo, hi) in [("day 0-8", 0u8, 8u8), ("day 9-17", 9, 17), ("day 18-26", 18, 27)] {
        let (x, y, _) = build(&|r| r.day >= lo && r.day <= hi);
        if y.len() < 200 {
            continue;
        }
        let c = ridge(&x, &y, 1e-3);
        print!("\n{label:>12} (n={:>6}) coef:", y.len());
        for (k, name) in names.iter().enumerate() {
            print!("  {name}={:.2}", c[k + 1]);
        }
        println!();
    }

    // ---- 2b. the same fit *within* a position --------------------------
    //
    // This is the one that matters, and the pooled fit above cannot answer it.
    // Two thirds of the variance in a raw term is the calendar (`engine` is
    // mostly `rounds_left`), and the calendar says nothing about who wins. So
    // centre every term and every outcome across the four seats of the *same*
    // position: what is left is exactly the quantity `margin` reduces to, and a
    // coefficient here says what that term is worth as a discriminator.
    //
    // A term whose coefficient is near zero is dead weight in the margin. A
    // negative one is actively pointing the search the wrong way.
    let fe = |sel: &dyn Fn(&Row) -> bool| -> (Vec<Vec<f64>>, Vec<f64>, Vec<u64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cl = Vec::new();
        for r in rows.iter().filter(|r| sel(r)) {
            let t: Vec<[f32; 8]> = (0..N_PLAYERS).map(|i| r.parts[i].terms()).collect();
            let tbar: [f64; 8] = std::array::from_fn(|k| {
                (0..N_PLAYERS).map(|i| t[i][k] as f64).sum::<f64>() / N_PLAYERS as f64
            });
            let ybar = r.outcome.iter().sum::<f32>() as f64 / N_PLAYERS as f64;
            for i in 0..N_PLAYERS {
                let mut row = vec![1.0];
                row.extend((0..8).map(|k| t[i][k] as f64 - tbar[k]));
                x.push(row);
                y.push(r.outcome[i] as f64 - ybar);
                cl.push(r.game);
            }
        }
        (x, y, cl)
    };
    println!(
        "\n-- what each term is worth as a *discriminator*: the same fit with \n\
           every term and outcome centred across the four seats of one position --\n\
         {:>12}  {:>8}  {:>16}  {:>8}",
        "term", "sd", "coef +/-95%", "coef*sd"
    );
    let (x, y, cl) = fe(&|_| true);
    let (fc, fse) = ridge_se(&x, &y, &cl, 1e-3);
    for (k, name) in names.iter().enumerate() {
        let col: Vec<f64> = x.iter().map(|r| r[k + 1]).collect();
        let s = sd(&col);
        println!(
            "{:>12}  {:>8.2}  {:>16}  {:>8.2}",
            name,
            s,
            format!("{:.2} +/-{:.2}", fc[k + 1], 1.96 * fse[k + 1]),
            fc[k + 1] * s,
        );
    }
    println!("  (centred outcome sd {:.2})", sd(&y));

    println!("\n-- discriminator coefficients by day --");
    print!("{:>12}", "days");
    for name in names.iter() {
        print!("  {name:>11}");
    }
    println!();
    println!("  (`.` marks a coefficient whose 95% interval covers 1.0 — \
already scaled right, or too noisy to say)");
    for b in 0..9 {
        let (x, y, cl) = fe(&|r| bucket_of(r.day) == b);
        if y.len() < 200 {
            continue;
        }
        let (c, e) = ridge_se(&x, &y, &cl, 1e-3);
        print!("{:>12}", format!("{}-{}", b * 3, b * 3 + 2));
        for k in 0..8 {
            let flat = (c[k + 1] - 1.0).abs() <= 1.96 * e[k + 1];
            print!("  {:>10.2}{}", c[k + 1], if flat { "." } else { " " });
        }
        println!();
    }

    // ---- 2c. what one unit of each raw holding is worth -----------------
    //
    // The component fit above cannot separate `held` from `liquidation`: both
    // are near-linear in the same block count, so only their sum is
    // identified. This fit sidesteps that by regressing on the *holdings*
    // themselves — one column per resource — again centred across the four
    // seats of a position. The coefficient is then what one more stone, one
    // more skull or one more worker is worth in realised final points, which
    // is exactly the number `src/eval.rs`'s constants are guesses at.
    const RAW: [&str; 18] = [
        "points", "corn", "wood", "stone", "gold", "skull", "corn_tile", "wood_tile",
        "workers", "on_board", "free_wkr", "discount", "temple_pts", "temple_stp",
        "research", "buildings", "monuments", "fp_tile",
    ];
    let raw_of = |g: &GameState, p: PlayerId| -> [f64; 18] {
        let pl = &g.players[p.idx()];
        let temple_pts: i32 = Temple::ALL
            .iter()
            .map(|&t| {
                let step = g.temple_pos(p, t) as usize;
                tzolkin::data::temples::TEMPLES[t.idx()].points[step] as i32
            })
            .sum();
        let temple_stp: u32 = Temple::ALL.iter().map(|&t| g.temple_pos(p, t) as u32).sum();
        [
            pl.points as f64,
            pl.corn as f64,
            pl.get(Resource::Wood) as f64,
            pl.get(Resource::Stone) as f64,
            pl.get(Resource::Gold) as f64,
            pl.get(Resource::Skull) as f64,
            pl.corn_tiles as f64,
            pl.wood_tiles as f64,
            g.n_unlocked(p) as f64,
            g.on_board(p).count() as f64,
            pl.free_workers as f64,
            pl.worker_discount as f64,
            temple_pts as f64,
            temple_stp as f64,
            g.research[p.idx()].iter().map(|&l| l as f64).sum(),
            pl.n_buildings() as f64,
            pl.n_monuments() as f64,
            pl.may_skip_day as u8 as f64,
        ]
    };
    let fe_raw = |sel: &dyn Fn(&Row) -> bool| -> (Vec<Vec<f64>>, Vec<f64>, Vec<u64>) {
        let (mut x, mut y, mut cl) = (Vec::new(), Vec::new(), Vec::new());
        for r in rows.iter().filter(|r| sel(r)) {
            let t: Vec<[f64; 18]> = (0..N_PLAYERS)
                .map(|i| raw_of(&r.state, PlayerId(i as u8)))
                .collect();
            let tbar: [f64; 18] = std::array::from_fn(|k| {
                (0..N_PLAYERS).map(|i| t[i][k]).sum::<f64>() / N_PLAYERS as f64
            });
            let ybar = r.outcome.iter().sum::<f32>() as f64 / N_PLAYERS as f64;
            for i in 0..N_PLAYERS {
                let mut row = vec![1.0];
                row.extend((0..18).map(|k| t[i][k] - tbar[k]));
                x.push(row);
                y.push(r.outcome[i] as f64 - ybar);
                cl.push(r.game);
            }
        }
        (x, y, cl)
    };
    println!(
        "\n-- marginal value of one unit, in realised final points --\n\
         (centred across the four seats of a position. These are *associations*, \n\
          not prices: the columns are collinear -- a seat with more workers is a \n\
          seat that spent corn on them -- so read the sign and the interval, and \n\
          settle the size in the arena.)\n{:>12}  {:>8}  {:>8}  {:>8}  {:>16}",
        "holding", "sd", "day 0-8", "day 18-26", "all +/-95%"
    );
    let (xa, ya, cla) = fe_raw(&|r| r.day <= 8);
    let (xb, yb, clb) = fe_raw(&|r| r.day >= 18);
    let (xall, yall, clall) = fe_raw(&|_| true);
    let ca = ridge_se(&xa, &ya, &cla, 1e-2).0;
    let cb = ridge_se(&xb, &yb, &clb, 1e-2).0;
    let (call, sall) = ridge_se(&xall, &yall, &clall, 1e-2);
    for (k, name) in RAW.iter().enumerate() {
        let col: Vec<f64> = xall.iter().map(|r| r[k + 1]).collect();
        println!(
            "{:>12}  {:>8.2}  {:>8.2}  {:>8.2}  {:>16}",
            name,
            sd(&col),
            ca[k + 1],
            cb[k + 1],
            format!("{:.2} +/-{:.2}", call[k + 1], 1.96 * sall[k + 1]),
        );
    }

    // ---- 3. ordering ---------------------------------------------------
    //
    // For a search, ranking beats calibration: what matters is whether the
    // seat the evaluator likes is the seat that wins. Concordance is over the
    // six seat pairs at each position, skipping pairs that tied on the day.
    let mut agree = 0u64;
    let mut pairs = 0u64;
    let mut top1 = 0u64;
    let mut top1_n = 0u64;
    let mut by_bucket_conc = vec![(0u64, 0u64); 9];
    for r in &rows {
        for i in 0..N_PLAYERS {
            for j in (i + 1)..N_PLAYERS {
                if r.outcome[i] == r.outcome[j] {
                    continue;
                }
                let ok = (r.est[i] > r.est[j]) == (r.outcome[i] > r.outcome[j]);
                pairs += 1;
                agree += ok as u64;
                let b = &mut by_bucket_conc[bucket_of(r.day)];
                b.0 += ok as u64;
                b.1 += 1;
            }
        }
        let lead_est = (0..N_PLAYERS).max_by(|&a, &b| r.est[a].total_cmp(&r.est[b])).unwrap();
        let best = (0..N_PLAYERS).map(|i| r.outcome[i]).fold(f32::MIN, f32::max);
        top1_n += 1;
        top1 += (r.outcome[lead_est] == best) as u64;
    }
    println!(
        "\n-- ordering --\npairwise concordance {:.3} over {pairs} seat pairs \
         (0.5 is a coin flip)\nthe estimator's leader is the eventual winner {:.3} \
         of the time (0.25 is chance)",
        agree as f64 / pairs as f64,
        top1 as f64 / top1_n as f64,
    );
    print!("by day: ");
    for (b, (ok, n)) in by_bucket_conc.iter().enumerate() {
        if *n < 10 {
            continue;
        }
        print!("{}-{}:{:.3}  ", b * 3, b * 3 + 2, *ok as f64 / *n as f64);
    }
    println!();

    // ---- 4. scale ------------------------------------------------------
    //
    // `HeuristicEvaluator` divides the centred estimate by 25 before a tanh.
    // If the centred estimate's spread is far from the centred outcome's, that
    // constant is squashing the wrong range.
    let centred = |v: &[f32; N_PLAYERS]| -> Vec<f64> {
        let m = v.iter().sum::<f32>() as f64 / N_PLAYERS as f64;
        v.iter().map(|&x| x as f64 - m).collect()
    };
    let ce: Vec<f64> = rows.iter().flat_map(|r| centred(&r.est)).collect();
    let ca: Vec<f64> = rows.iter().flat_map(|r| centred(&r.outcome)).collect();
    let mg: Vec<f64> = rows.iter().flat_map(|r| r.margin.iter().map(|&x| x as f64)).collect();
    println!(
        "\n-- scale --\ncentred estimate  sd {:>6.2}   centred final score sd {:>6.2}   \
         r {:.3}\nmargin mean {:>6.2} sd {:>6.2}; tanh(centred/25) uses \
         {:.0}% of its range on 1 sd",
        sd(&ce),
        sd(&ca),
        corr(&ce, &ca),
        mean(&mg),
        sd(&mg),
        100.0 * (sd(&ce) / 25.0).tanh(),
    );

    // How much of the final spread is explained by the estimate at each day.
    println!("\n-- centred estimate vs centred outcome, by day --");
    print!("r: ");
    for b in 0..9 {
        let e: Vec<f64> = rows.iter().filter(|r| bucket_of(r.day) == b).flat_map(|r| centred(&r.est)).collect();
        let a: Vec<f64> = rows.iter().filter(|r| bucket_of(r.day) == b).flat_map(|r| centred(&r.outcome)).collect();
        if e.len() < 10 {
            continue;
        }
        print!("{}-{}:{:.3}({:.1}/{:.1})  ", b * 3, b * 3 + 2, corr(&e, &a), sd(&e), sd(&a));
    }
    println!("\n");
}

// ---- dominated options -------------------------------------------------
//
// These pin the pruning in `options::dominated_dedup`, `affordable_costs` and
// `moves::choices_for_worker`: they assert that a *redundant* option is gone,
// never that a distinct one is, so a rule change that widens the move space
// still passes and only a rule change that narrows it can trip them.

/// A worker on a free-choice space never pays to walk down its own gear.
///
/// Yaxchilan's top spaces repeat the whole gear for nothing, so nothing there
/// can cost corn -- and before the step-down prune, a worker on space 6 offered
/// Yaxchilan 1's single wood for five corn.
#[test]
fn free_choice_space_never_buys_what_it_already_has() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 30;

    for pos in [6u8, 7] {
        for c in tzolkin::moves::choices_for_worker(&g.state, p, Gear::Yaxchilan, Pos(pos)) {
            assert!(
                c.net_corn() >= 0,
                "Yaxchilan:{pos} charged {} corn for an action it gives away: {c}",
                -c.net_corn()
            );
        }
    }
}

/// Uxmal's mirror sells any lower action for one corn, so stepping down to that
/// same action for two or more is the same play at a worse price.
#[test]
fn the_mirror_undercuts_stepping_down() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].corn = 20;

    // Uxmal 3 unlocks a worker; from space 5 the mirror reaches it for one corn
    // and the two-space walk down reaches it for two.
    let unlocks: Vec<Choice> = tzolkin::moves::choices_for_worker(&g.state, p, Gear::Uxmal, Pos(5))
        .into_iter()
        .filter(|c| c.0.contains(&Effect::UnlockWorker))
        .collect();
    assert!(!unlocks.is_empty(), "the mirror should reach Uxmal 3");
    for c in &unlocks {
        assert_eq!(
            c.net_corn(),
            -1,
            "unlocking is available for one corn; this pays more: {c}"
        );
    }
}

/// The architecture discount is not an offer a player may decline: refusing it
/// hands a block back to the bank and changes nothing else.
#[test]
fn a_builder_never_pays_the_listed_price() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()][Science::Architecture.idx()] = 3; // builder
    g.state.players[p.idx()].res = [8, 8, 8, 0];

    let mut checked = 0;
    for c in tzolkin::options::building_choices(&g.state, p, None, true, 0) {
        // `payment_effects` writes the cost immediately before the card.
        let cut = c.0.iter().position(|e| matches!(e, Effect::Build(_))).unwrap();
        let Effect::Build(id) = c.0[cut] else { unreachable!() };
        let listed: i32 = Resource::BLOCKS
            .iter()
            .map(|&r| tzolkin::data::buildings::def(id).cost[r.idx()] as i32)
            .sum();
        if listed == 0 {
            continue;
        }
        let paid: i32 = c.0[..cut]
            .iter()
            .map(|e| match e {
                Effect::Res(_, n) if *n < 0 => -(*n as i32),
                _ => 0,
            })
            .sum();
        assert_eq!(paid, listed - 1, "full price paid with a discount going spare: {c}");
        checked += 1;
    }
    assert!(checked > 0, "no buildable card to check");
}

/// The non-wealth part of a choice, and the net corn, wood, stone, gold and
/// points it moves. Skulls stay in the shape: `take_skulls` caps against a
/// shared bank, so their net does not pin the outcome.
fn shape_and_wealth(c: &Choice) -> (Vec<Effect>, [i32; 5]) {
    let mut shape = Vec::new();
    let mut w = [0i32; 5];
    for e in &c.0 {
        match *e {
            Effect::Corn(n) => w[0] += n as i32,
            Effect::Res(r, n) if r != Resource::Skull => w[1 + r.idx()] += n as i32,
            Effect::Points(n) => w[4] += n as i32,
            other => shape.push(other),
        }
    }
    (shape, w)
}

/// Every option a space offers must reach a position no other option reaches.
///
/// This is ground truth, not a rule about effect lists: two choices that leave
/// the state identical are one move wearing two spellings. `tikal::build_two`
/// used to spell every double build twice, because building A then B and B then
/// A cost the same whenever the architecture discount is not in play -- 6% of
/// every option list on Tikal 4 through 7, measured, and again inside every
/// mirror and every step-down that reaches them.
#[test]
fn no_two_options_at_a_space_reach_the_same_position() {
    for seed in 0..5u64 {
        let mut g = Game::new(seed);
        for _ in 0..14 {
            g.play_round();
            for p in PlayerId::ALL {
                for gear in Gear::ALL {
                    for i in 0..gear.size() {
                        let mut seen: std::collections::HashMap<GameState, Choice> =
                            std::collections::HashMap::new();
                        for c in choices_at(&g.state, p, gear, Pos(i)) {
                            let mut probe = g.state;
                            c.apply(&mut probe, p);
                            if let Some(prev) = seen.insert(probe, c.clone()) {
                                panic!(
                                    "{}:{i} offers [{prev}] and [{c}], which are one position",
                                    gear.name()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// No option may be another option's poorer twin.
///
/// Two choices that agree on everything except how much corn, how many blocks
/// and how many points they move are one decision: the poorer arrives where the
/// richer arrives, holding less, and nothing in the game pays for holding less.
/// The empty choice is exempt -- "pick the worker up and do nothing" is one of
/// the three options the rules name, so it is kept even where a free payout
/// beats it.
#[test]
fn no_option_is_a_strictly_poorer_twin() {
    for seed in 0..5u64 {
        let mut g = Game::new(seed);
        for _ in 0..14 {
            g.play_round();
            for p in PlayerId::ALL {
                for gear in Gear::ALL {
                    for i in 0..gear.size() {
                        let cs = choices_at(&g.state, p, gear, Pos(i));
                        let keyed: Vec<_> = cs.iter().map(shape_and_wealth).collect();
                        for (a, (sa, wa)) in cs.iter().zip(&keyed) {
                            if a.is_skip() {
                                continue;
                            }
                            for (b, (sb, wb)) in cs.iter().zip(&keyed) {
                                if sa != sb {
                                    continue;
                                }
                                let beaten = (0..5).all(|k| wb[k] >= wa[k])
                                    && (0..5).any(|k| wb[k] > wa[k]);
                                assert!(
                                    !beaten,
                                    "{}:{i} offers [{a}] when [{b}] does the same for less",
                                    gear.name()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Yaxchilan 5 hands out the stone of space 2 and the gold of space 3 together,
/// so from the free-choice spaces at the top of the gear those two are dead
/// letters. Wood is not: space 1 gives something space 5 does not.
#[test]
fn a_richer_action_at_the_same_price_retires_the_leaner_one() {
    let g = fresh();
    let p = PlayerId(0);
    let top = choices_at(&g.state, p, Gear::Yaxchilan, Pos(6));

    for dead in [2u8, 3] {
        for c in choices_at(&g.state, p, Gear::Yaxchilan, Pos(dead)) {
            assert!(
                !top.contains(&c),
                "Yaxchilan:6 still offers [{c}], which space 5 swallows whole"
            );
        }
    }
    for live in [1u8, 4, 5] {
        for c in choices_at(&g.state, p, Gear::Yaxchilan, Pos(live)) {
            assert!(
                top.contains(&c),
                "Yaxchilan:6 dropped [{c}], which nothing on the gear covers"
            );
        }
    }
}

/// Building #30's payoff is Uxmal's mirror, and the mirror costs a corn. A
/// player who cannot pay that corn does not thereby lose the right to construct
/// the card -- the privilege is simply unusable, like a temple step on a temple
/// you have already topped. `expand_payoff` returned an empty list for an
/// unusable mirror and `building_choices` reads an empty payoff as "no way to
/// build this", so at zero corn the card fell out of the game.
///
/// Architecture level 1 masked this: the corn the build itself pays arrives
/// before the payoff is expanded, and one corn is exactly the mirror's price.
#[test]
fn a_card_whose_payoff_is_unusable_is_still_buildable() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.buildings_up = [None; tzolkin::state::N_DISPLAY];
    g.state.buildings_up[0] = Some(BuildingId(30)); // 2w 1s 1g, the mirror + 2 VP
    g.state.research[p.idx()] = [0; 4]; // no architecture corn to pay the mirror
    g.state.players[p.idx()].res = [4, 4, 4, 0];
    g.state.players[p.idx()].corn = 0;

    let ways: Vec<_> = tzolkin::options::building_choices(&g.state, p, None, true, 1)
        .into_iter()
        .filter(|c| c.0.contains(&Effect::Build(BuildingId(30))))
        .collect();
    assert!(!ways.is_empty(), "a penniless player may still construct #30");
    assert!(
        ways.iter().all(|c| c.0.contains(&Effect::Points(2))),
        "the card's two points do not depend on the mirror"
    );
}

/// The other half of the same rule: a mirror the player *can* pay for is still
/// a privilege, and `RULES-AUDIT.md` A9 settled for this codebase that a
/// privilege is offered rather than forced -- Uxmal 3, Tikal 2, 4 and 5 were
/// all fixed on exactly that principle. The rulebook's framing is permissive:
/// "Pay 1 corn and immediately perform any action on the Palenque, Yaxchilan,
/// Tikal or Uxmal gear."
///
/// Every choice `mirror_choices` returns leads with the one-corn fee -- it
/// drops the skips its mirrored spaces contribute and prefixes the fee to what
/// is left -- so `Payoff::Mirror` offered no way to construct building #30 and
/// leave the mirror alone. Constructing the card forced the corn *and* forced
/// an action, and the mirror reaches `UnlockWorker`: A9's unwanted Uxmal 3
/// worker, arriving through a card instead of a space.
///
/// The mirror's other caller, Uxmal 5, was never exposed to this and is not
/// changed: `choices_for_worker` prepends the skip there, as it does for every
/// space -- Uxmal 1, 2 and 4 and Tikal 2 and 5 carry no skip in their own lists
/// either. `doing_nothing_is_always_an_option` is what holds that.
///
/// Asserted at the layer that generates the options, because the shipped
/// generator's pruning then hides the fix: Palenque 1 is a plain,
/// inexhaustible corn space, so the mirror always offers "-1 corn, +3 corn" --
/// pure wealth, nothing structural -- which dominates declining on every axis
/// and `dominated_dedup` takes it straight back out. Sound, and the reason this
/// is a contract fix rather than a play-strength one.
#[test]
fn a_mirror_the_player_can_afford_may_still_be_declined() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.buildings_up = [None; tzolkin::state::N_DISPLAY];
    g.state.buildings_up[0] = Some(BuildingId(30)); // 2w 1s 1g, the mirror + 2 VP
    g.state.research[p.idx()] = [0; 4];
    g.state.players[p.idx()].res = [4, 4, 4, 0];
    g.state.players[p.idx()].corn = 5; // enough to pay the mirror several times over

    let ways: Vec<_> = tzolkin::options::building_choices(&g.state, p, None, true, 1)
        .into_iter()
        .filter(|c| c.0.contains(&Effect::Build(BuildingId(30))))
        .collect();
    assert!(ways.len() > 1, "a solvent player has the mirror to spend on");

    // Declining spends no corn at all: the mirror's fee is the only `Corn` a
    // #30 construction can carry here, since the card is paid for in blocks.
    let declined: Vec<_> = ways
        .iter()
        .filter(|c| !c.0.iter().any(|e| matches!(e, Effect::Corn(_))))
        .collect();
    assert_eq!(
        declined.len(),
        1,
        "exactly one way to construct #30 and leave the mirror unused, got {} of {}; \
         e.g. [{}], which is A9's unwanted worker arriving through a card",
        declined.len(),
        ways.len(),
        ways
            .iter()
            .find(|c| c.0.contains(&Effect::UnlockWorker))
            .map_or_else(|| "-".to_string(), |c| c.to_string())
    );
    assert!(
        declined[0].0.contains(&Effect::Points(2)),
        "declining the mirror still takes the card's two points"
    );

    // The premise the fix rests on: every choice the mirror itself returns
    // leads with the fee, so the decline cannot come from inside it and cannot
    // collide with the bare tail either.
    let mirror = tzolkin::spaces::uxmal::mirror_choices(&g.state, p, 1);
    assert!(!mirror.is_empty(), "a solvent player's mirror is not empty");
    assert!(
        mirror.iter().all(|c| c.0.first() == Some(&Effect::Corn(-1))),
        "every mirrored action leads with the one-corn fee"
    );

    // Uxmal 5, the mirror's other caller, is unchanged and covered from
    // outside -- the same skip every other space relies on.
    assert!(
        tzolkin::moves::choices_for_worker(&g.state, p, Gear::Uxmal, Pos(5))
            .iter()
            .any(|c| c.is_skip()),
        "Uxmal 5 offers no way to decline the mirror"
    );
}

/// Tikal action 4, rulebook p.9: "If you construct two buildings, you can apply
/// your architecture technologies to either the first or the second, but one of
/// them must be built without architecture technologies. **If the first
/// building gives you a new architecture technology, you may apply that effect
/// (and any others) to the second building, as long as you applied no
/// architecture effects to the first one.**"
///
/// `build_two` gave the bonus to the first build and nothing to the second, and
/// relied on enumerating both orders to cover "either the first or the second".
/// That covers the choice of *which* card gets a level the player already has,
/// and misses the sentence in bold entirely: building #6 is a gold for a free
/// architecture advance, and the level it hands over is meant to pay for the
/// card built beside it.
#[test]
fn architecture_won_by_the_first_build_pays_for_the_second() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.research[p.idx()] = [0; 4]; // no architecture yet
    // #6 costs 1 gold and gives a free architecture advance; #1 costs 1 wood.
    g.state.buildings_up = [None; tzolkin::state::N_DISPLAY];
    g.state.buildings_up[0] = Some(BuildingId(6));
    g.state.buildings_up[1] = Some(BuildingId(1));
    g.state.players[p.idx()].res = [1, 0, 1, 0];

    let paid = choices_at(&g.state, p, Gear::Tikal, Pos(4)).into_iter().any(|c| {
        c.0.contains(&Effect::Build(BuildingId(6)))
            && c.0.contains(&Effect::Build(BuildingId(1)))
            && c.0.contains(&Effect::AdvanceResearch(Science::Architecture))
            // Architecture level 1 is the only source of corn in this position.
            && c.0.iter().any(|e| matches!(e, Effect::Corn(n) if *n > 0))
    });
    assert!(
        paid,
        "the architecture level #6 hands over must be usable on the second building"
    );
}

/// Building A then B and B then A are the same pair of cards at the same price
/// unless the architecture discount is in play, and `build_two` enumerates both
/// orders. The discount is what makes the orders differ, so this checks the
/// case where it is absent.
#[test]
fn a_double_build_is_offered_once_per_pair() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].res = [9, 9, 9, 0];
    assert!(!g.state.builder(p), "no discount, so the two orders cost the same");

    let mut seen: std::collections::HashMap<GameState, Choice> = std::collections::HashMap::new();
    let mut doubles = 0;
    for c in choices_at(&g.state, p, Gear::Tikal, Pos(4)) {
        if c.0.iter().filter(|e| matches!(e, Effect::Build(_))).count() < 2 {
            continue;
        }
        doubles += 1;
        let mut probe = g.state;
        c.apply(&mut probe, p);
        if let Some(prev) = seen.insert(probe, c.clone()) {
            panic!("the same double build twice: [{prev}] and [{c}]");
        }
    }
    assert!(doubles > 0, "no affordable pair of buildings to check");
}

/// Two advances are one decision however they are spelled: which track goes
/// first, and which block pays for which half, are not choices a player makes.
#[test]
fn two_advances_are_offered_once_per_outcome() {
    let mut g = fresh();
    let p = PlayerId(0);
    g.state.players[p.idx()].res = [2, 2, 2, 0];
    g.state.research[p.idx()] = [0, 1, 3, 0];

    for free in [false, true] {
        let mut seen = std::collections::HashSet::new();
        for c in tzolkin::options::research_choices(&g.state, p, 2, free) {
            let mut probe = g.state;
            c.apply(&mut probe, p);
            assert!(
                seen.insert(probe),
                "two spellings of one pair of advances (free={free}): {c}"
            );
        }
        assert!(!seen.is_empty());
    }
}

/// Ordering the walk permutes it; it may not change what it finds.
///
/// `moves::Priority` exists so a capped caller can be handed the plausible
/// moves first rather than the first ones — the prefix of traversal order is
/// one spine of `retrieve_rec`'s recursion, not a slice of the move space, and
/// `docs/SEARCH.md` §2.2 is the long form of why that matters. Ordering is the
/// sound half of the idea: a filter that drops what the plan dislikes can drop
/// the best move, a permutation cannot drop anything.
///
/// What makes that true here is that `retrieve_rec` is a depth-first walk with
/// a visited set over a DAG — the worker it retrieves never comes back, so no
/// edge can return to its own subtree — and the visited set of such a walk is
/// the reachable set whatever order the edges arrive in. The memo does pick a
/// different *spelling* of a commuting retrieval, which is why this compares
/// reached positions and counts rather than `Move`s.
///
/// The hint is deliberately adversarial: a hash of the option, so it agrees
/// with generation order nowhere.
#[test]
fn a_traversal_hint_reorders_the_walk_and_nothing_else() {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    use std::ops::ControlFlow;
    use tzolkin::moves::{self, Kinds, Placement, Priority};

    fn scramble<H: Hash>(x: H) -> i32 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        x.hash(&mut h);
        (h.finish() >> 33) as i32
    }

    struct Scramble;
    impl Priority for Scramble {
        fn placement(&self, spot: Placement) -> i32 {
            scramble(spot)
        }
        fn worker(&self, gear: Gear, pos: Pos) -> i32 {
            scramble((gear.name(), pos.0, 0x77_6f_72_6bu32))
        }
        fn choice(&self, gear: Gear, pos: Pos, c: &Choice) -> i32 {
            scramble((gear.name(), pos.0, c))
        }
    }

    // Above this a position is skipped rather than compared, so the two walks
    // are never two different prefixes of the same set.
    const MAX_MOVES: usize = 20_000;
    let mut compared = 0usize;
    let mut retrieval_heavy = 0usize;

    for seed in 0..12u64 {
        let mut g = Game::new(seed);
        while !g.state.over {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                for kinds in [Kinds::Placements, Kinds::Retrievals, Kinds::All] {
                    let walk = |order: Option<&Scramble>| {
                        let mut states = HashSet::new();
                        let mut n = 0usize;
                        let mut f = |m: &tzolkin::moves::Move| {
                            let mut probe = g.state;
                            tzolkin::moves::apply_move(&mut probe, p, m);
                            states.insert(probe);
                            n += 1;
                            if n > MAX_MOVES {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        };
                        match order {
                            Some(o) => {
                                let _ = moves::visit_moves_of_by(&g.state, p, kinds, o, &mut f);
                            }
                            None => {
                                let _ = moves::visit_moves_of(&g.state, p, kinds, &mut f);
                            }
                        }
                        (states, n)
                    };
                    let (plain, n_plain) = walk(None);
                    if n_plain > MAX_MOVES {
                        continue;
                    }
                    let (hinted, n_hinted) = walk(Some(&Scramble));
                    assert_eq!(
                        plain, hinted,
                        "seed {seed} day {} {kinds:?}: the hint changed which positions are reachable",
                        g.state.day
                    );
                    assert_eq!(
                        n_plain, n_hinted,
                        "seed {seed} day {} {kinds:?}: the hint changed how many moves are emitted",
                        g.state.day
                    );
                    compared += 1;
                    if kinds == Kinds::Retrievals && n_plain > 200 {
                        retrieval_heavy += 1;
                    }
                }
                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }

    // Guard the guard: the claim is only interesting where the walk is deep
    // enough for the memo to be collapsing orderings in the first place.
    assert!(compared > 3_000, "only {compared} walks compared");
    assert!(
        retrieval_heavy > 100,
        "only {retrieval_heavy} retrieval-heavy positions compared"
    );
    println!(
        "{compared} walks compared, {retrieval_heavy} of them over 200 retrievals"
    );
}

/// `options::pack` must order effects exactly as the derive does.
///
/// The dominance pass groups choices by a digest of their packed structural
/// effects and re-sorts the survivors with `Choice`'s own `Ord`; if the packing
/// disagreed with the derive anywhere, one of the two orders would be a lie and
/// the `TREE_EDGE` index space would move under the search. The vocabulary is
/// small enough to check every pair of a covering sample rather than argue.
#[test]
fn packed_order_matches_derived_ord() {
    use tzolkin::options::pack;
    let mut es: Vec<Effect> = Vec::new();
    for n in [i16::MIN, -300, -7, -1, 0, 1, 7, 300, i16::MAX] {
        es.push(Effect::Corn(n));
    }
    for n in [0u8, 3, 128, 255] {
        es.push(Effect::SetCorn(n));
    }
    for r in Resource::ALL {
        for n in [i8::MIN, -5, -1, 0, 1, 5, i8::MAX] {
            es.push(Effect::Res(r, n));
        }
    }
    for n in [i8::MIN, -1, 0, 1, i8::MAX] {
        es.push(Effect::Points(n));
        es.push(Effect::FreeWorker(n));
        es.push(Effect::WorkerDiscount(n));
        for t in Temple::ALL {
            es.push(Effect::TempleStep(t, n));
        }
    }
    for s in Science::ALL {
        es.push(Effect::AdvanceResearch(s));
    }
    es.push(Effect::UnlockWorker);
    for i in 0..12u8 {
        es.push(Effect::TakePalenqueTile(Pos(i), TileKind::Corn));
        es.push(Effect::TakePalenqueTile(Pos(i), TileKind::Wood));
        es.push(Effect::BurnPalenqueWood(Pos(i)));
        es.push(Effect::FillChichen(Pos(i)));
    }
    for i in 0..N_BUILDINGS as u8 {
        es.push(Effect::Build(BuildingId(i)));
    }
    for i in 0..MONUMENTS.len() as u8 {
        es.push(Effect::TakeMonument(MonumentId(i)));
    }
    for a in &es {
        for b in &es {
            assert_eq!(
                pack(*a).cmp(&pack(*b)),
                a.cmp(b),
                "packed order disagrees for {a:?} vs {b:?} ({} vs {})",
                pack(*a),
                pack(*b)
            );
        }
    }
    // Injective, which is what makes the group digest sound in the first place.
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for e in &es {
        assert!(seen.insert(pack(*e)), "packed collision on {e:?}");
    }
}

/// The fast dominance pass must agree with the rule stated in prose.
///
/// `dominated_dedup` groups on a 64-bit digest for speed. This runs the literal
/// rule -- sort on the filtered effect iterators, group by comparing them
/// exactly, keep the Pareto front of the five wealth axes -- over lists
/// harvested from real positions and shaped the way `choices_for_worker` shapes
/// them, and demands the same sequence out. It is the guard that the digest is
/// an accelerator and never the rule.
#[test]
fn dominated_dedup_matches_the_exact_rule() {
    fn is_wealth(e: &Effect) -> bool {
        matches!(e, Effect::Corn(_) | Effect::Points(_))
            || matches!(e, Effect::Res(r, _) if *r != Resource::Skull)
    }
    fn wealth(c: &Choice) -> [i32; 5] {
        let mut w = [0i32; 5];
        for e in &c.0 {
            match *e {
                Effect::Corn(n) => w[0] += n as i32,
                Effect::Res(r, n) if r != Resource::Skull => w[1 + r.idx()] += n as i32,
                Effect::Points(n) => w[4] += n as i32,
                _ => {}
            }
        }
        w
    }
    fn reference(mut v: Vec<Choice>) -> Vec<Choice> {
        let group = |a: &Choice, b: &Choice| {
            a.is_skip() == b.is_skip()
                && a.0
                    .iter()
                    .filter(|e| !is_wealth(e))
                    .cmp(b.0.iter().filter(|e| !is_wealth(e)))
                    .is_eq()
        };
        v.sort_unstable_by(|a, b| {
            a.is_skip()
                .cmp(&b.is_skip())
                .then_with(|| {
                    a.0.iter()
                        .filter(|e| !is_wealth(e))
                        .cmp(b.0.iter().filter(|e| !is_wealth(e)))
                })
                .then_with(|| {
                    wealth(b).iter().sum::<i32>().cmp(&wealth(a).iter().sum::<i32>())
                })
                .then_with(|| a.cmp(b))
        });
        let mut front: Vec<[i32; 5]> = Vec::new();
        let mut keep = 0usize;
        let mut i = 0usize;
        while i < v.len() {
            let mut j = i + 1;
            while j < v.len() && group(&v[i], &v[j]) {
                j += 1;
            }
            front.clear();
            for r in i..j {
                let w = wealth(&v[r]);
                if front.iter().any(|f| (0..5).all(|k| f[k] >= w[k])) {
                    continue;
                }
                front.push(w);
                v.swap(keep, r);
                keep += 1;
            }
            i = j;
        }
        v.truncate(keep);
        v.sort_unstable();
        v
    }

    let mut lists = 0usize;
    let mut nonempty_prunes = 0usize;
    for seed in 0..6u64 {
        let mut g = Game::new(seed);
        for _ in 0..14 {
            if g.state.over {
                break;
            }
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                for w in g.state.on_board(p).collect::<Vec<_>>() {
                    let Some((gear, pos)) = g.state.loc(w).on_board() else {
                        continue;
                    };
                    // The shape `choices_for_worker` builds: every step-down fee
                    // stacked into one list, which is where the dominance rule
                    // has something to do.
                    let corn = g.state.players[p.idx()].corn;
                    let mut v = vec![Choice::skip()];
                    for j in 0..=pos.0 {
                        let fee = pos.0 - j;
                        if fee > corn {
                            continue;
                        }
                        let mut probe = g.state;
                        probe.players[p.idx()].corn -= fee;
                        for c in choices_at(&probe, p, gear, Pos(j)) {
                            if fee == 0 {
                                v.push(c);
                            } else if !c.is_skip() {
                                v.push(Choice::one(Effect::Corn(-(fee as i16))).chain(&c));
                            }
                        }
                    }
                    let n = v.len();
                    let got = tzolkin::options::dominated_dedup(v.clone());
                    let want = reference(v);
                    assert_eq!(got, want, "dominance disagreed at {gear:?} {pos:?}");
                    lists += 1;
                    if got.len() < n {
                        nonempty_prunes += 1;
                    }
                }
                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }
    // Guard the guard: agreeing on lists that had nothing to prune proves
    // nothing about the rule.
    assert!(lists > 200, "only {lists} lists compared");
    assert!(nonempty_prunes > 100, "only {nonempty_prunes} lists pruned");
    println!("{lists} lists compared, {nonempty_prunes} of them pruned");
}

/// `temple_outlook`'s climb bonus stops where `GameState::temple_ceiling` does.
///
/// The bonus values the *next* step so the search sees climbing as progress
/// between scoring days. It used to test only `step + 1 < steps`, so it paid
/// for a move on to the exclusive top of a temple an opponent was already
/// standing on — a move `temple_step` clamps away. Worth +0.08 [+0.01, +0.14]
/// greedy:64 and +0.19 [+0.02, +0.35] mcts:256 on its own,
/// `docs/FINDINGS-eval.md` F18/F21.
///
/// The arithmetic needs no evaluator constant. Yellow's step points are
/// [-2, 0, 1, 2, 4, 6, 9, 12, 13] and its resource thresholds are 3 and 5, so
/// the pairs (0,1), (1,2) and (6,7) cross no threshold, and with an opponent on
/// the top step `p` never takes a majority prize. What is left in each
/// difference is the step's own points and the climb credit — two equations for
/// the day weight per point and the price of a point of jump, which makes the
/// third pair a prediction rather than a fit.
#[test]
fn evaluator_climb_respects_the_exclusive_top() {
    use tzolkin::data::temples::TEMPLES;
    use tzolkin::eval::components;

    let t = Temple::Yellow;
    let pts = TEMPLES[t.idx()].points;
    let top = (TEMPLES[t.idx()].steps - 1) as usize;

    let temple_at = |step: usize| -> f32 {
        let mut g = fresh();
        g.state.temples[t.idx()][0] = step as u8;
        g.state.temples[t.idx()][1] = top as u8;
        components(&g.state, PlayerId(0)).temple
    };
    let (t0, t1, t2) = (temple_at(0), temple_at(1), temple_at(2));
    let (t6, t7) = (temple_at(6), temple_at(7));

    let d = |a: usize, b: usize| (pts[b] - pts[a]) as f32;
    let jump = |k: usize| (pts[k + 1] - pts[k]) as f32;

    // (1,2): equal jumps either side, so the difference is pure step points.
    let w = (t2 - t1) / d(1, 2);
    // (0,1): the jump falls by one, which prices the climb.
    let c = (d(0, 1) * w - (t1 - t0)) / (jump(0) - jump(1));

    // (6,7) is the prediction. At 7 `p` is on the ceiling of a blocked temple
    // and cannot step again, so the climb credit there must be zero rather than
    // `jump(7)`.
    let want = d(6, 7) * w - jump(6) * c;
    let old = d(6, 7) * w + (jump(7) - jump(6)) * c;
    let got = t7 - t6;
    assert!(
        (want - old).abs() > 1e-2,
        "the two behaviours are indistinguishable here, so the test proves nothing"
    );
    assert!(
        (got - want).abs() < 1e-2,
        "a blocked top must pay no climb: got {got}, want {want}, old behaviour {old}"
    );
}

/// `starvation_risk` counts the corn a player holds, not corn it might earn.
///
/// It used to add `CORN_INCOME_PER_ROUND = 1.9` a round of assumed income
/// before deciding whether a food day was survivable, which forgave any
/// shortfall more than three rounds out — every shortfall at the start of a
/// round block. Setting that to zero is the largest single effect measured on
/// this evaluator since the term re-pricing: **+3.96 greedy:64 and +2.11
/// mcts:256 at 800 blocks**, `docs/FINDINGS-eval.md` F23. The reason it works
/// is that the income is exactly what the search is separately planning for, so
/// assuming it here paid for the same corn twice.
#[test]
fn evaluator_starvation_ignores_income_it_has_not_earned() {
    use tzolkin::eval::components;

    let p = PlayerId(0);
    let broke = {
        let mut g = fresh();
        g.state.players[p.idx()].corn = 0;
        g
    };
    let fed = {
        let mut g = fresh();
        g.state.players[p.idx()].corn = 40;
        g
    };
    // Day 0, so the next food day (`RESOURCE_DAYS[0]`) is eight rounds out --
    // far enough that the old constant projected 15 corn of income and read no
    // risk at all.
    assert_eq!(broke.state.day, 0, "the distance to the food day is the point");
    assert!(
        components(&broke.state, p).starvation > 1.0,
        "no corn and a bill to pay must score as risk however far off the day is"
    );
    assert_eq!(
        components(&fed.state, p).starvation,
        0.0,
        "corn already in hand covers the bill"
    );
}

/// `starvation_risk`'s urgency floor is **unreachable**, and this pins that.
///
/// The term discounts a shortfall that is still several rounds out by
/// `(1.0 / (1.0 + rounds * 0.25)).max(0.3)`. The `.max` never fires: the
/// longest run-up to a food day is the one to `RESOURCE_DAYS[0]` from day 0,
/// and `1 / (1 + 0.25 * 8) = 0.3333`, which is above the floor. Measured as
/// well as derived — `evalab --ab ufl=0.1` reads **+0.0000 in all 656 blocks**,
/// an identity rather than a small effect (`docs/FINDINGS-eval.md` F43a).
///
/// The reason this is worth a test rather than a deletion is the failure mode
/// it guards. Lengthening the calendar, or lowering the 0.25 coefficient, would
/// bring the floor into play — silently, since nothing else changes shape — and
/// the value it would take is a hand number that has never been measured while
/// live. If this test fails, the floor has just become load-bearing and wants
/// sweeping before it is trusted.
#[test]
fn evaluator_starvation_urgency_never_reaches_its_floor() {
    use tzolkin::state::{LAST_DAY, POINT_DAYS, RESOURCE_DAYS};

    const URGENCY: f32 = 0.25; // `starvation_risk`'s coefficient
    const FLOOR: f32 = 0.30; // its `.max(..)`

    let mut worst = f32::INFINITY;
    for day in 0..LAST_DAY {
        let Some(next) = RESOURCE_DAYS
            .iter()
            .chain(POINT_DAYS.iter())
            .copied()
            .filter(|&d| d > day)
            .min()
        else {
            continue;
        };
        let rounds = (next - day) as f32;
        let urgency = 1.0 / (1.0 + rounds * URGENCY);
        assert!(
            urgency > FLOOR,
            "day {day}: the food day is {rounds} rounds out, urgency {urgency} \
             has reached the {FLOOR} floor -- it is no longer dead code and has \
             never been swept while live"
        );
        worst = worst.min(urgency);
    }
    assert!(
        worst < FLOOR + 0.05,
        "the floor is not merely unreached, it is far away ({worst}) -- if the \
         calendar changed, this test is checking nothing"
    );
}

/// The board table is scaled per *gear*, because it prices what a space hands
/// over and never what it charges.
///
/// `evalab --promise`'s `dead` column measures the consequence: the fraction of
/// standing workers for which the evaluator, offered the action under them,
/// declines it. Palenque and Yaxchilan, which only gather, read 0.00 and 0.01;
/// Chichen reads 0.5 to 0.7, because the skull it spends is held at
/// `3.0 + SKULL_PREMIUM` = 5.2 while the bottom of the gear pays 4 printed
/// points. So Chichen was over-paid and Palenque under-paid, and
/// `eval::GEAR_SCALE` is 0.7 and 1.5 there. `docs/FINDINGS-eval.md` F26d, F27.
///
/// Both pins put one worker on the **top** space of a gear, where
/// `board_position` takes no maximum over a ride and every worker is halved by
/// the same top-of-gear discount, so the only thing between two of them is the
/// gear scale.
#[test]
fn evaluator_scales_the_board_table_by_gear() {
    use tzolkin::eval::components;

    let p = PlayerId(0);
    let top_of = |gear: Gear| {
        let mut g = fresh();
        // A skull, so Chichen is not gated down to its no-skull floor.
        g.state.players[p.idx()].res[Resource::Skull.idx()] = 1;
        let w = GameState::worker_ids(p).next().unwrap();
        g.state.place_worker(w, gear, Pos(gear.size() - 1));
        let with = components(&g.state, p).board;
        let mut bare = fresh();
        bare.state.players[p.idx()].res[Resource::Skull.idx()] = 1;
        // The same position without the worker, so what is left is the worker.
        with - components(&bare.state, p).board
    };

    let (palenque, yaxchilan) = (top_of(Gear::Palenque), top_of(Gear::Yaxchilan));
    let (chichen, tikal) = (top_of(Gear::Chichen), top_of(Gear::Tikal));
    for (name, v) in [
        ("palenque", palenque),
        ("yaxchilan", yaxchilan),
        ("chichen", chichen),
        ("tikal", tikal),
    ] {
        assert!(v > 0.0, "{name} credits its worker nothing: the fixture is wrong");
    }

    // The raw table pays Palenque's top 2.6 and Yaxchilan's 3.4, so unscaled
    // this ordering runs the other way. Nothing but `GEAR_SCALE` can flip it.
    assert!(
        palenque > yaxchilan,
        "the corn gear must outprice the resource gear at the top after the \
         re-price: palenque {palenque}, yaxchilan {yaxchilan}"
    );

    // The raw table pays Chichen's top 10.5 against Tikal's 5.2, a ratio of
    // 2.02; at 0.7 it is 1.41. The bracket excludes both the unscaled value and
    // any scale below about 0.6.
    let ratio = chichen / tikal;
    assert!(
        (1.2..1.7).contains(&ratio),
        "chichen/tikal at the top should be about 1.41 after the re-price and \
         2.02 before it, got {ratio}"
    );
}

/// What a **whole research row** is worth to the evaluator, pinned.
///
/// `engine_value` prices research at `research_step_value(s, l) * uses *
/// RESEARCH_SCALE`, and the constant is 0.05 because a full sweep says so on
/// three agents at three depths: `greedy:64` at 800 blocks reads
/// +0.06 / +0.01 / −0.22 * / −0.85 * / −4.28 * at 0.0 / 0.15 / 0.25 / 0.50 /
/// 1.00, and `mcts:1024:cp=0.05` at 250 reads +0.28 / −0.04 / −0.11 / −0.43 /
/// −1.01 *. Not one cell above zero.
///
/// The number this test pins is the thing those sweeps actually varied: **all
/// twelve levels, at the start of the game, are worth about 2.6 points**, which
/// is a fifth of `temple_outlook`'s mean and about 8% of a 33-point estimate.
/// It is here because four different things can move it silently — the twelve
/// entries of `research_step_value`, `RESEARCH_SCALE`, the `uses` horizon, and
/// the `lvl == 2` bonus — and because the *reason* the champion takes 1.3
/// levels of twelve in a game is this number, so anyone who changes it is
/// changing a measured result and should re-measure.
/// `docs/FINDINGS-eval.md` F52, F53, F54, F57.
#[test]
fn evaluator_prices_a_whole_research_row_at_about_two_and_a_half_points() {
    use tzolkin::eval::components;

    let p = PlayerId(0);
    let engine_with = |row: [u8; 4], day: u8| {
        let mut g = fresh();
        g.state.day = day;
        g.state.research[p.idx()] = row;
        components(&g.state, p).engine
    };

    let empty = engine_with([0, 0, 0, 0], 0);
    let full = engine_with([3, 3, 3, 3], 0);
    let whole_row = full - empty;
    assert!(
        (2.0..3.5).contains(&whole_row),
        "twelve research levels on day 0 are worth {whole_row}, not ~2.6 -- \
         `RESEARCH_SCALE`, `research_step_value`, `uses` or the `lvl == 2` \
         bonus has moved, and the sweep behind 0.05 (F54, F57) no longer \
         applies to this evaluator"
    );

    // `uses = (rounds_left / 3).min(7)`: the cap binds while `rounds_left >= 21`
    // — day 0 through day 6 — so an early level is worth the same on either.
    // Every alternative shape measured inert on `greedy:64` at 800 blocks: the
    // early tilt and its exact opposite both read 0.00 +/- 0.09 (F55). The cap
    // is not a bug and it is not free to remove, so it is asserted rather than
    // left to be rediscovered.
    assert_eq!(
        engine_with([3, 3, 3, 3], 0),
        engine_with([3, 3, 3, 3], 6),
        "the `uses` cap should hold research flat over days 0..=6"
    );
    let (d6, d12, d18) = (
        engine_with([3, 3, 3, 3], 6) - engine_with([0, 0, 0, 0], 6),
        engine_with([3, 3, 3, 3], 12) - engine_with([0, 0, 0, 0], 12),
        engine_with([3, 3, 3, 3], 18) - engine_with([0, 0, 0, 0], 18),
    );
    assert!(
        d6 > d12 && d12 > d18 && d18 > 0.0,
        "research must be worth strictly less the later it is taken: \
         day 6 {d6}, day 12 {d12}, day 18 {d18}"
    );
    // Linear after the cap, which is what `uses` says and what every convex
    // alternative failed to beat: the day 6 -> 12 and 12 -> 18 drops are equal.
    let (a, b) = (d6 - d12, d12 - d18);
    assert!(
        (a - b).abs() < 0.05,
        "the decline should be linear in `rounds_left`: {a} then {b}"
    );
}

/// A corn space is not worth the same to everyone.
///
/// `GEAR_SCALE` says the corn gear is worth half again what the hand table
/// pays; `HUNGRY_CORN` says that to a player who cannot pay the next food bill
/// out of the corn in hand it is worth a quarter more again. Once
/// `CORN_INCOME_PER_ROUND` went to zero, corn's worth to such a player is the
/// three points a head `starvation_risk` is charging them, and a flat
/// multiplier cannot say so. `docs/FINDINGS-eval.md` F27b, F29, F29a.
#[test]
fn evaluator_pays_more_for_corn_when_the_food_day_is_unpaid() {
    use tzolkin::eval::components;

    let p = PlayerId(0);
    // What one worker on the top of a gear is worth, differenced against the
    // same position with no worker on the board.
    let worker_worth = |corn: u8, gear: Gear| {
        let mut g = fresh();
        g.state.players[p.idx()].corn = corn;
        let w = GameState::worker_ids(p).next().unwrap();
        g.state.place_worker(w, gear, Pos(gear.size() - 1));
        let with = components(&g.state, p).board;
        let mut bare = fresh();
        bare.state.players[p.idx()].corn = corn;
        with - components(&bare.state, p).board
    };

    // Solve for the feeding bill rather than assuming it: the smallest holding
    // that stops `starvation_risk` firing is what the workers eat.
    let starve = |corn: u8| {
        let mut g = fresh();
        g.state.players[p.idx()].corn = corn;
        components(&g.state, p).starvation
    };
    let bill = (0u8..40).find(|&c| starve(c) == 0.0).expect("nobody ever eats here");
    assert!(bill > 0, "the fixture has no feeding bill, so the test proves nothing");

    let ratio = worker_worth(bill - 1, Gear::Palenque) / worker_worth(bill, Gear::Palenque);
    assert!(
        ratio > 1.05,
        "a worker on the corn gear must be worth more to a player one corn short \
         of the food bill than to one who can pay it: ratio {ratio}"
    );

    // Only the corn gear may move with the corn in hand.
    let (h, f) = (worker_worth(bill - 1, Gear::Yaxchilan), worker_worth(bill, Gear::Yaxchilan));
    assert!((h - f).abs() < 1e-4, "the resource gear moved too: {h} against {f}");
}

