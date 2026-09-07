//! The evaluator, exhaustive ranking, and the alpha-beta search.

use rand::rngs::StdRng;
use rand::SeedableRng;
use tzolkin::eval::{self, heuristic, margin};
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::{self, check_move, Kinds, Move};
use tzolkin::record::parse_agent;
use tzolkin::search::{
    after_turn, advance_seat, prefers_extra_day, Config, Opponents, Search,
};
use tzolkin::state::GameState;

/// Positions from across a game, so no test is only about the opening.
fn positions(seeds: u64, at: &[usize]) -> Vec<GameState> {
    let mut out = Vec::new();
    for seed in 0..seeds {
        for &rounds in at {
            let mut g = Game::new(seed);
            for _ in 0..rounds {
                if g.state.over {
                    break;
                }
                g.play_round();
            }
            if !g.state.over {
                out.push(g.state);
            }
        }
    }
    out
}

/// A ranking may claim `exhaustive` only if it really did see every move, and
/// must report a short count when it gave up. Both agents and the search carry
/// a deadline on the exhaustive ply, so "it saw everything" is a claim that has
/// to be checked rather than assumed.
fn assert_root_is_honest(r: &tzolkin::eval::Ranking, g: &GameState, p: PlayerId, who: &str) {
    let n = moves::count_legal_moves(g, p);
    if r.exhaustive {
        assert_eq!(r.total, n, "{who} claimed exhaustive on day {} but saw {} of {n}", g.day, r.total);
    } else {
        assert!(
            r.total < n,
            "{who} gave up on day {} yet reports the whole list",
            g.day
        );
    }
    assert!(r.distinct <= r.total);
}

// ---- the evaluator -----------------------------------------------------

/// The whole point of decaying every speculative term: a leaf that is a real
/// result must score as that result, or the search cannot compare a finished
/// game against a guess about one.
#[test]
fn heuristic_is_exact_once_the_game_is_over() {
    for seed in 0..25u64 {
        let mut g = Game::new(seed);
        g.run_sampled();
        assert!(g.state.over);
        let scores = g.state.scores();
        for p in PlayerId::ALL {
            let h = heuristic(&g.state, p);
            assert!(
                (h - scores[p.idx()] as f32).abs() < 1e-3,
                "seed {seed} {p:?}: heuristic {h} but final score {}",
                scores[p.idx()]
            );
        }
    }
}

/// `margin` is a genuine zero-sum reduction: exactly one player can be ahead of
/// the field, and the leader's margin is the negative of the best rival's view
/// of the same position.
#[test]
fn margin_is_antisymmetric_at_the_top() {
    for g in positions(6, &[0, 7, 14, 21]) {
        let hs: Vec<f32> = PlayerId::ALL.iter().map(|&p| heuristic(&g, p)).collect();
        let leader = PlayerId(
            (0..N_PLAYERS)
                .max_by(|&a, &b| hs[a].total_cmp(&hs[b]))
                .unwrap() as u8,
        );
        assert!(margin(&g, leader) >= 0.0, "the leader cannot be behind");
        for p in PlayerId::ALL {
            if p != leader {
                assert!(margin(&g, p) <= 0.0, "{p:?} is not the leader but scores ahead");
            }
        }
    }
}

/// Starving a player has to read as a loss. This is the term the first
/// evaluator had none of, and it is worth 3 points a head in the rules.
#[test]
fn starvation_is_priced() {
    let g = Game::new(4).state;
    let p = PlayerId(0);

    let mut fed = g;
    fed.players[p.idx()].corn = 30;
    let mut starving = g;
    starving.players[p.idx()].corn = 0;

    assert!(
        heuristic(&starving, p) < heuristic(&fed, p),
        "a player with no corn before a food day must score below a fed one"
    );

    // And an extra mouth with nothing to feed it is not an improvement.
    let mut hungry_extra = starving;
    hungry_extra.unlock_worker(p);
    assert!(
        heuristic(&hungry_extra, p) < heuristic(&fed, p),
        "buying a worker you cannot feed must not beat being solvent"
    );
}

// ---- exhaustive ranking ------------------------------------------------

/// `heuristic:full`'s central claim: every legal move was looked at.
#[test]
fn rank_all_sees_every_legal_move() {
    for g in positions(5, &[0, 5, 11, 18, 24]) {
        let p = g.current;
        let r = eval::rank_all(&g, p, 10);
        assert_eq!(
            r.total,
            moves::count_legal_moves(&g, p),
            "rank_all missed moves on day {}",
            g.day
        );
        assert!(r.exhaustive, "rank_all has no deadline and must always finish");
        assert!(r.distinct <= r.total);
        assert!(r.moves.len() <= 10);

        // Best first, and every reported move is playable.
        for w in r.moves.windows(2) {
            assert!(w[0].1 >= w[1].1, "ranking is not sorted");
        }
        for (m, _) in &r.moves {
            check_move(&g, p, m).unwrap_or_else(|e| panic!("day {}: {e}", g.day));
        }
    }
}

/// The shortlist must not be one move written out ten ways.
#[test]
fn the_shortlist_holds_distinct_moves() {
    for g in positions(6, &[0, 3, 9, 16]) {
        let p = g.current;
        let r = eval::rank_all(&g, p, 10);
        for (i, (a, _)) in r.moves.iter().enumerate() {
            for (b, _) in r.moves.iter().skip(i + 1) {
                assert!(
                    !a.same_effect(b),
                    "day {}: `{a}` and `{b}` are the same move",
                    g.day
                );
            }
        }
    }
}

/// Moves `same_effect` calls equal must actually be equal where it matters:
/// they cost the same and they score the same.
#[test]
fn equivalent_moves_score_identically() {
    let mut checked = 0usize;
    for g in positions(4, &[0, 6, 12]) {
        let p = g.current;
        let all = moves::legal_moves(&g, p);
        for (i, a) in all.iter().enumerate() {
            for b in all.iter().skip(i + 1).take(60) {
                if !a.same_effect(b) {
                    continue;
                }
                checked += 1;
                assert_eq!(a.corn_cost, b.corn_cost);
                let (x, y) = (
                    margin(&eval::successor(&g, p, a), p),
                    margin(&eval::successor(&g, p, b), p),
                );
                assert!(
                    (x - y).abs() < 1e-4,
                    "`{a}` scores {x} but its restatement `{b}` scores {y}"
                );
            }
        }
    }
    assert!(checked > 0, "no equivalent pairs found; the test proved nothing");
}

// ---- the split walk ----------------------------------------------------

/// The search caps placements and retrievals separately. That is only sound if
/// the two halves partition the ordinary move space.
#[test]
fn the_two_halves_partition_the_move_space() {
    for g in positions(6, &[0, 4, 10, 17, 23]) {
        let p = g.current;
        let collect = |k| {
            let mut v: Vec<Move> = Vec::new();
            let _ = moves::visit_moves_of(&g, p, k, |m| {
                v.push(m.clone());
                std::ops::ControlFlow::Continue(())
            });
            v.sort();
            v
        };
        let all = collect(Kinds::All);
        let mut split = collect(Kinds::Placements);
        split.extend(collect(Kinds::Retrievals));
        split.sort();
        assert_eq!(all, split, "the halves do not reconstruct the whole on day {}", g.day);

        // And when a normal move exists at all, the whole is the legal set.
        if moves::has_normal_move(&g, p) {
            assert_eq!(all.len(), moves::count_legal_moves(&g, p));
        }
    }
}

// ---- round flow --------------------------------------------------------

/// `advance_seat` is the search's model of what happens between turns. If it
/// disagrees with the driver in `record::play_game`, every value the search
/// backs up is about a different game.
///
/// Both are driven here by the same moves and the same extra-day rule, so any
/// divergence is a real one.
#[test]
fn advance_seat_matches_the_game_driver() {
    for seed in 0..8u64 {
        let mut mine = Game::new(seed).state;
        let mut reference = mine;
        let mut rng = StdRng::seed_from_u64(seed * 31 + 5);
        let mut guard = 0;

        while !mine.over {
            // One round through the reference driver, mirroring `play_game`.
            reference.current = reference.first_player;
            for _ in 0..N_PLAYERS {
                let p = reference.current;
                let Some(m) = moves::sample_legal_move(&reference, p, &mut rng) else {
                    break;
                };
                assert_eq!(mine.current, p, "seat drifted on day {}", mine.day);

                // The same move through both.
                moves::apply_move(&mut reference, p, &m);
                reference.refill_buildings();
                reference.current = reference.current.next(1);

                mine = after_turn(&mine, p, &m);
            }

            let claimer = reference.resolve_first_player();
            let mut days = 1u8;
            if let Some(q) = claimer {
                if reference.may_take_extra_day(q) && prefers_extra_day(&reference, q) {
                    reference.spend_extra_day(q);
                    days = 2;
                }
            }
            reference.advance_days(days);
            reference.current = reference.first_player;

            assert_eq!(mine, reference, "diverged after day {}", reference.day);
            guard += 1;
            assert!(guard < 100, "game did not terminate");
        }
    }
}

/// A round boundary really is crossed: `advance_seat` must move the calendar,
/// not just the seat.
#[test]
fn advance_seat_ends_the_round_on_the_wrap() {
    let g = Game::new(2).state;
    let mut s = g;
    for _ in 0..N_PLAYERS - 1 {
        advance_seat(&mut s);
        assert_eq!(s.day, g.day, "the calendar moved mid-round");
    }
    advance_seat(&mut s);
    assert_eq!(s.day, g.day + 1, "the calendar did not advance at the wrap");
    assert_eq!(s.current, s.first_player);
}

// ---- the search --------------------------------------------------------

/// Whatever it recommends has to be playable, and it has to have looked at
/// everything before recommending it.
#[test]
fn search_returns_a_legal_move_from_the_whole_move_list() {
    let mut s = Search::new(Config::default().with_depth(3).with_budget_ms(400));
    for g in positions(4, &[0, 6, 13, 20, 25]) {
        let p = g.current;
        let r = s.search(&g, p);
        assert!(!r.moves.is_empty(), "no move on day {}", g.day);
        assert_root_is_honest(&r, &g, p, "search");
        for (m, _) in &r.moves {
            check_move(&g, p, m).unwrap_or_else(|e| panic!("day {}: {e}", g.day));
        }
        assert!(s.stats().depth >= 1);
    }
}

/// Searching deeper must not change what the position *is*.
#[test]
fn search_does_not_mutate_the_position() {
    let mut s = Search::new(Config::default().with_depth(4).with_budget_ms(300));
    for g in positions(3, &[0, 9, 18]) {
        let before = g;
        let _ = s.search(&g, g.current);
        assert_eq!(before, g, "search wrote through its own input");
    }
}

/// Asking out of turn answers about the seat asked for.
#[test]
fn search_answers_for_the_seat_it_was_given() {
    let mut s = Search::new(Config::default().with_depth(2).with_budget_ms(200));
    let g = positions(1, &[6]).pop().expect("a mid-game position");
    for p in PlayerId::ALL {
        let r = s.search(&g, p);
        assert_eq!(r.total, moves::count_legal_moves(&g, p), "{p:?}");
        for (m, _) in &r.moves {
            let mut probe = g;
            probe.current = p;
            check_move(&probe, p, m).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        }
    }
}

/// The budget is a budget. A deep search on a wide position must still return.
#[test]
fn the_time_budget_is_respected() {
    let mut s = Search::new(Config {
        max_depth: 40,
        widths: vec![32, 16, 12, 10, 8],
        budget: std::time::Duration::from_millis(250),
        ..Config::default()
    });
    for g in positions(2, &[5, 12]) {
        let t = std::time::Instant::now();
        let r = s.search(&g, g.current);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert!(!r.moves.is_empty());
        // Generous: one interior node can take milliseconds and the budget is
        // only checked between them. What must not happen is running to depth 40.
        assert!(ms < 4_000.0, "a 250 ms budget took {ms:.0} ms on day {}", g.day);
    }
}

/// Every agent that offers a ranking must rank the whole move list, because
/// that is what the panel showing it claims.
#[test]
fn ranking_agents_rank_every_move() {
    for spec in ["heuristic:32", "heuristic:full", "minimax:2:200"] {
        let agent = parse_agent(spec, false).unwrap_or_else(|e| panic!("{spec}: {e}"));
        for g in positions(2, &[0, 8, 16]) {
            let p = g.current;
            let r = agent
                .ranked_moves(&g, p, 10)
                .unwrap_or_else(|| panic!("{spec} offers no ranking"));
            assert_root_is_honest(&r, &g, p, spec);
            for (m, _) in &r.moves {
                check_move(&g, p, m).unwrap_or_else(|e| panic!("{spec}: {e}"));
            }
        }
    }
    // `random` has no ranking to give and must say so rather than inventing one.
    let r = parse_agent("random", false).unwrap();
    assert!(r.ranked_moves(&Game::new(1).state, PlayerId(0), 10).is_none());
}

/// The spec grammar, including the fields that are easy to get wrong: empty
/// fields take the default, and the opponent model has to survive into the name
/// or two very different players are logged as one.
#[test]
fn minimax_specs_parse() {
    use tzolkin::record::AgentSpec;

    for (spec, want) in [
        ("minimax", "minimax:d4:600ms:w12:paranoid"),
        ("minimax:3", "minimax:d3:600ms:w12:paranoid"),
        ("minimax:4:120", "minimax:d4:120ms:w12:paranoid"),
        ("minimax:4:120:8", "minimax:d4:120ms:w8:paranoid"),
        ("minimax:4:120:8:greedy", "minimax:d4:120ms:w8:greedy"),
        ("minimax:6:::greedy", "minimax:d6:600ms:w12:greedy"),
    ] {
        let got = AgentSpec::parse(spec, false)
            .unwrap_or_else(|e| panic!("{spec}: {e}"))
            .name();
        assert_eq!(got, want, "{spec}");
    }

    for bad in [
        "minimax:0",
        "minimax:4:120:0",
        "minimax:4:120:8:cautious",
        "minimax:4:120:8:greedy:9",
        "minimax:x",
    ] {
        assert!(AgentSpec::parse(bad, false).is_err(), "{bad} should not parse");
    }
}

/// Both opponent models have to produce a legal move over the whole move list.
/// They differ in what they assume, not in what they are allowed to play.
#[test]
fn both_opponent_models_search_soundly() {
    for opponents in [Opponents::Paranoid, Opponents::Greedy] {
        let mut s = Search::new(Config {
            opponents,
            ..Config::default().with_depth(4).with_budget_ms(200)
        });
        for g in positions(3, &[0, 7, 15, 22]) {
            let p = g.current;
            let r = s.search(&g, p);
            assert!(!r.moves.is_empty(), "{opponents:?} found nothing on day {}", g.day);
            assert_root_is_honest(&r, &g, p, &format!("{opponents:?}"));
            for (m, _) in &r.moves {
                check_move(&g, p, m)
                    .unwrap_or_else(|e| panic!("{opponents:?} on day {}: {e}", g.day));
            }
        }
    }
}

/// End to end: the search can carry a game to its last day without producing an
/// illegal move or stalling.
#[test]
fn minimax_can_finish_a_game() {
    let agent = parse_agent("minimax:2:60", false).unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let mut state = Game::new(3).state;
    let mut guard = 0;

    while !state.over {
        state.current = state.first_player;
        for _ in 0..N_PLAYERS {
            let p = state.current;
            if let Some(o) = agent.play_turn(&state, p, 0.0, &mut rng) {
                check_move(&state, p, &o.mv).unwrap_or_else(|e| panic!("day {}: {e}", state.day));
                moves::apply_move(&mut state, p, &o.mv);
                state.refill_buildings();
            }
            state.current = state.current.next(1);
        }
        let claimer = state.resolve_first_player();
        let mut days = 1;
        if let Some(q) = claimer {
            if state.may_take_extra_day(q) && agent.extra_day(&state, q, &mut rng).0 {
                state.spend_extra_day(q);
                days = 2;
            }
        }
        state.advance_days(days);
        tzolkin::invariants::validate(&state).unwrap_or_else(|e| panic!("day {}: {e}", state.day));

        guard += 1;
        assert!(guard < 100, "game did not terminate");
    }
    assert!(state.day >= 27);
}
