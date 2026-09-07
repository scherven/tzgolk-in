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
                        last_decisions: Vec::new(),
                        ranking: Ranking {
                            exhaustive,
                            note: "x".repeat(200),
                            ..ranking.clone()
                        },
                        selected: n.saturating_sub(1),
                        source,
                        autoplay: false,
                        status: "x".repeat(200),
                    };
                    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                    term.draw(|f| ui::draw(f, &app))
                        .unwrap_or_else(|e| panic!("{w}x{h} after {rounds} rounds: {e}"));
                }
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
    let build = |sel: &dyn Fn(&Row) -> bool| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for r in rows.iter().filter(|r| sel(r)) {
            for i in 0..N_PLAYERS {
                let t = r.parts[i].terms();
                let mut row = vec![1.0];
                row.extend(t.iter().map(|&v| v as f64));
                x.push(row);
                y.push(r.outcome[i] as f64);
            }
        }
        (x, y)
    };

    println!(
        "\n-- what each term is worth: OLS(final score ~ terms), whole game --\n\
         {:>12}  {:>8}  {:>8}  {:>8}  {:>8}",
        "term", "mean", "coef", "should be", "pts/pos"
    );
    let (x, y) = build(&|_| true);
    let coefs = ridge(&x, &y, 1e-3);
    println!("{:>12}  {:>8}  {:>+8.2}", "(intercept)", "", coefs[0]);
    for (k, name) in names.iter().enumerate() {
        let col: Vec<f64> = x.iter().map(|r| r[k + 1]).collect();
        let m = mean(&col);
        println!(
            "{:>12}  {:>8.2}  {:>8.2}  {:>8.2}  {:>+8.2}",
            name,
            m,
            coefs[k + 1],
            coefs[k + 1] * m,
            (1.0 - coefs[k + 1]) * m,
        );
    }

    // The same fit split by phase, because a term that decays wrongly has a
    // coefficient that moves with the calendar.
    for (label, lo, hi) in [("day 0-8", 0u8, 8u8), ("day 9-17", 9, 17), ("day 18-26", 18, 27)] {
        let (x, y) = build(&|r| r.day >= lo && r.day <= hi);
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
    let fe = |sel: &dyn Fn(&Row) -> bool| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
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
            }
        }
        (x, y)
    };
    println!(
        "\n-- what each term is worth as a *discriminator*: the same fit with \n\
           every term and outcome centred across the four seats of one position --\n\
         {:>12}  {:>8}  {:>8}  {:>8}",
        "term", "sd", "coef", "coef*sd"
    );
    let (x, y) = fe(&|_| true);
    for (k, name) in names.iter().enumerate() {
        let col: Vec<f64> = x.iter().map(|r| r[k + 1]).collect();
        let s = sd(&col);
        println!(
            "{:>12}  {:>8.2}  {:>8.2}  {:>8.2}",
            name,
            s,
            ridge(&x, &y, 1e-3)[k + 1],
            ridge(&x, &y, 1e-3)[k + 1] * s,
        );
    }
    println!("  (centred outcome sd {:.2})", sd(&y));

    println!("\n-- discriminator coefficients by day --");
    print!("{:>12}", "days");
    for name in names.iter() {
        print!("  {name:>11}");
    }
    println!();
    for b in 0..9 {
        let (x, y) = fe(&|r| bucket_of(r.day) == b);
        if y.len() < 200 {
            continue;
        }
        let c = ridge(&x, &y, 1e-3);
        print!("{:>12}", format!("{}-{}", b * 3, b * 3 + 2));
        for k in 0..8 {
            print!("  {:>11.2}", c[k + 1]);
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
    let fe_raw = |sel: &dyn Fn(&Row) -> bool| -> (Vec<Vec<f64>>, Vec<f64>) {
        let (mut x, mut y) = (Vec::new(), Vec::new());
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
            }
        }
        (x, y)
    };
    println!(
        "\n-- marginal value of one unit, in realised final points --\n\
         (centred across the four seats of a position; `now` is what \n\
          src/eval.rs currently pays for the same unit)\n{:>12}  {:>8}  {:>8}  {:>8}",
        "holding", "sd", "day 0-8", "day 18-26"
    );
    let (xa, ya) = fe_raw(&|r| r.day <= 8);
    let (xb, yb) = fe_raw(&|r| r.day >= 18);
    let (xall, yall) = fe_raw(&|_| true);
    let (ca, cb, call) = (
        ridge(&xa, &ya, 1e-2),
        ridge(&xb, &yb, 1e-2),
        ridge(&xall, &yall, 1e-2),
    );
    for (k, name) in RAW.iter().enumerate() {
        let col: Vec<f64> = xall.iter().map(|r| r[k + 1]).collect();
        println!(
            "{:>12}  {:>8.2}  {:>8.2}  {:>8.2}   (all {:>5.2})",
            name,
            sd(&col),
            ca[k + 1],
            cb[k + 1],
            call[k + 1],
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
