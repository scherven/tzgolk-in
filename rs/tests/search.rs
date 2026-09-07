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

// ---- the starting-tile draft -------------------------------------------

/// `Game::new` must keep producing exactly the game it always has, or every
/// seeded test in the suite is quietly testing a different board.
#[test]
fn splitting_the_draft_out_did_not_move_any_seeded_game() {
    // The deal is a function of the seed alone, and the random draft keeps two
    // of the four dealt. Both halves have to still hold.
    for seed in 0..30u64 {
        let (_, deal) = Game::new_undrafted(seed);
        let g = Game::new(seed);
        let mut ids: Vec<u8> = deal.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), N_PLAYERS * 4, "seed {seed}: a tile was dealt twice");
        // Something was applied: nobody starts a game with nothing.
        assert!(
            PlayerId::ALL.iter().any(|&p| g.state.players[p.idx()].corn > 0),
            "seed {seed}: the draft applied nothing"
        );
    }
}

/// The whole point of the hook: an agent that picks must beat one that rolls.
#[test]
fn drafting_agents_pick_better_tiles_than_chance() {
    use tzolkin::record::best_pair;

    let mut better = 0usize;
    let mut worse = 0usize;
    for seed in 0..60u64 {
        let (game, deal) = Game::new_undrafted(seed);
        let mut rng = StdRng::seed_from_u64(seed);
        for p in PlayerId::ALL {
            let dealt = deal[p.idx()];

            let chosen = best_pair(&game.state, p, dealt, |s| heuristic(s, p));
            let mut a = game.state;
            for id in chosen {
                for e in tzolkin::data::tiles::TILES[id as usize] {
                    e.apply(&mut a, p);
                }
            }

            // What the old random draft would have taken.
            let rolled = parse_agent("random", false)
                .unwrap()
                .draft(&game.state, p, dealt, &mut rng);
            let mut b = game.state;
            for id in rolled {
                for e in tzolkin::data::tiles::TILES[id as usize] {
                    e.apply(&mut b, p);
                }
            }

            let (x, y) = (heuristic(&a, p), heuristic(&b, p));
            assert!(
                x >= y - 1e-4,
                "seed {seed} {p:?}: the chosen pair {chosen:?} scores {x} but \
                 a random pair {rolled:?} scores {y}"
            );
            if x > y + 1e-4 {
                better += 1;
            } else if x < y - 1e-4 {
                worse += 1;
            }
        }
    }
    assert_eq!(worse, 0);
    assert!(
        better > 60,
        "picking beat rolling only {better} times in {} draws; the hook is not doing anything",
        60 * N_PLAYERS
    );
}

/// What the draft is worth, in evaluator points.
///
///     cargo test --release --test search -- --ignored --nocapture what_the_draft_is_worth
#[test]
#[ignore]
fn what_the_draft_is_worth() {
    use tzolkin::data::tiles::TILES;
    use tzolkin::record::best_pair;

    let apply = |st: &mut GameState, p: PlayerId, pair: [u8; 2]| {
        for id in pair {
            for e in TILES[id as usize] {
                e.apply(st, p);
            }
        }
    };

    let (mut gain, mut spread) = (Vec::new(), Vec::new());
    for seed in 0..400u64 {
        let (game, deal) = Game::new_undrafted(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xBEEF);
        for p in PlayerId::ALL {
            let dealt = deal[p.idx()];
            let score = |pair: [u8; 2]| {
                let mut st = game.state;
                apply(&mut st, p, pair);
                heuristic(&st, p)
            };

            let best = score(best_pair(&game.state, p, dealt, |s| heuristic(s, p)));
            let rolled = score(
                parse_agent("random", false)
                    .unwrap()
                    .draft(&game.state, p, dealt, &mut rng),
            );
            gain.push(best - rolled);

            // The spread across the six pairs says how much was on the table.
            let mut all: Vec<f32> = Vec::new();
            for i in 0..4 {
                for j in (i + 1)..4 {
                    all.push(score([dealt[i], dealt[j]]));
                }
            }
            let lo = all.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = all.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            spread.push(hi - lo);
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let mut sorted = gain.clone();
    sorted.sort_by(f32::total_cmp);
    println!(
        "\nstarting-tile draft, {} deals\n  \
         gain over a random keep : {:>6.2} points mean, {:>6.2} median, {:>6.2} p90\n  \
         best minus worst pair   : {:>6.2} points mean (what was on the table)",
        gain.len(),
        mean(&gain),
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 9 / 10],
        mean(&spread),
    );
}

/// Whatever an agent keeps has to be two distinct tiles it was actually dealt.
#[test]
fn every_agent_drafts_from_what_it_was_dealt() {
    for spec in ["random", "heuristic:32", "heuristic:full", "minimax:2:80"] {
        let agent = parse_agent(spec, false).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for seed in 0..12u64 {
            let (game, deal) = Game::new_undrafted(seed);
            for p in PlayerId::ALL {
                let dealt = deal[p.idx()];
                let kept = agent.draft(&game.state, p, dealt, &mut rng);
                assert_ne!(kept[0], kept[1], "{spec} kept the same tile twice");
                for k in kept {
                    assert!(dealt.contains(&k), "{spec} kept {k}, not dealt {dealt:?}");
                }
            }
        }
    }
}

/// A drafted game is a real game: all 21 tiles are accounted for and the board
/// is playable from it.
#[test]
fn a_drafted_game_is_playable() {
    use tzolkin::record::new_drafted_game;

    let agent = parse_agent("heuristic:32", false).unwrap();
    let refs: [&dyn tzolkin::record::Agent; N_PLAYERS] =
        [&*agent, &*agent, &*agent, &*agent];
    let mut rng = StdRng::seed_from_u64(1);

    for seed in 0..6u64 {
        let state = new_drafted_game(seed, &refs, &mut rng);
        tzolkin::invariants::validate(&state).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert_eq!(state.day, 0);
        assert!(!state.over);
        assert!(moves::has_legal_move(&state, state.current));
        // A drafted board is not an empty one.
        assert!(
            PlayerId::ALL
                .iter()
                .all(|&p| state.players[p.idx()].corn > 0 || state.n_unlocked(p) > 3),
            "seed {seed}: a seat came out of the draft with nothing"
        );
    }
}

/// The spec grammar, including the fields that are easy to get wrong: empty
/// fields take the default, and the opponent model has to survive into the name
/// or two very different players are logged as one.
#[test]
fn minimax_specs_parse() {
    use tzolkin::record::AgentSpec;

    for (spec, want) in [
        ("minimax", "minimax:d8:600ms:w12:greedy"),
        ("minimax:3", "minimax:d3:600ms:w12:greedy"),
        ("minimax:4:120", "minimax:d4:120ms:w12:greedy"),
        ("minimax:4:120:8", "minimax:d4:120ms:w8:greedy"),
        ("minimax:4:120:8:paranoid", "minimax:d4:120ms:w8:paranoid"),
        ("minimax:6:::paranoid", "minimax:d6:600ms:w12:paranoid"),
        // The harness flags are part of the name: an arena run racing two
        // variants has to be able to print which is which.
        ("minimax:8:200::greedy:capw=50", "minimax:d8:200ms:w12:greedy:capw50"),
        ("minimax:8:200::greedy:nocapw", "minimax:d8:200ms:w12:greedy:nocapw"),
        ("minimax:8:200::greedy:ownwidth", "minimax:d8:200ms:w12:greedy:ownwidth"),
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
        "minimax:4:120:8:greedy:capw=0",
        "minimax:4:120:8:greedy:capw=x",
        "minimax:x",
    ] {
        assert!(AgentSpec::parse(bad, false).is_err(), "{bad} should not parse");
    }
}

/// The AB-harness flags have to actually reach the `Config`, or a whole arena
/// run measures the default twice and reports the difference as noise.
#[test]
fn minimax_harness_flags_reach_the_config() {
    use tzolkin::record::AgentSpec;

    let cfg = |spec: &str| {
        AgentSpec::parse(spec, false)
            .unwrap_or_else(|e| panic!("{spec}: {e}"))
            .minimax_config()
            .unwrap_or_else(|| panic!("{spec} did not parse as minimax"))
            .clone()
    };

    assert_eq!(cfg("minimax:8:200::greedy:capw=50").cap_per_width, Some(50));
    assert_eq!(cfg("minimax:8:200::greedy:nocapw").cap_per_width, None);
    // The measured default, not None -- see `Config::cap_per_width`.
    assert_eq!(cfg("minimax:8:200::greedy").cap_per_width, Some(25));
    assert_eq!(cfg("minimax:8:200::greedy:cap=64").interior_cap, 64);
    assert_eq!(cfg("minimax:8:200::greedy:keep=40").keep, 40);
    // Comma-separated, so one spec can carry a whole variant.
    let both = cfg("minimax:8:200::greedy:capw=64,ownwidth,keep=36");
    assert_eq!(both.cap_per_width, Some(64));
    assert!(both.own_width);
    assert_eq!(both.keep, 36);
}

/// Under a width-1 opponent this search cannot prune, and that is a theorem
/// rather than an observation: only a min node lowers beta, every min node has
/// exactly one child, and a node evaluates that child with the beta it
/// inherited — so beta is `INF` everywhere and `alpha >= beta` needs an
/// infinite alpha. Killer moves, a history heuristic, PVS and aspiration
/// windows all act on the cutoff this test says never happens; widening the
/// opponent is what gives them something to work on.
#[test]
fn a_width_one_opponent_makes_alpha_beta_inert() {
    let ps = positions(3, &[0, 9, 18]);

    let mut narrow = Search::new(Config {
        opp_width: 1,
        ..Config::default().with_depth(8).with_budget_ms(200)
    });
    for g in &ps {
        let _ = narrow.search(g, g.current);
        assert_eq!(
            narrow.stats().cutoffs,
            0,
            "greedy width 1 cut something on day {} — beta must have left INF",
            g.day
        );
    }

    // Widening it is the whole point: min nodes gain siblings, beta tightens,
    // and the pruning machinery finally has a bound to cut against.
    let mut hedged = Search::new(Config {
        opp_width: 3,
        ..Config::default().with_depth(8).with_budget_ms(200)
    });
    let mut cuts = 0u64;
    for g in &ps {
        let r = hedged.search(g, g.current);
        cuts += hedged.stats().cutoffs;
        for (m, _) in &r.moves {
            check_move(g, g.current, m)
                .unwrap_or_else(|e| panic!("oppw=3 on day {}: {e}", g.day));
        }
    }
    assert!(
        cuts > 0,
        "a hedged opponent produced no cutoff over {} positions",
        ps.len()
    );
}

/// `cap_per_width` narrows what an interior node enumerates, which is exactly
/// the kind of change that can silently start returning a move the position
/// does not allow. The root stays exhaustive either way -- that contract is not
/// the cap's to break.
#[test]
fn a_width_scaled_cap_still_plays_legally() {
    let mut s = Search::new(Config {
        cap_per_width: Some(8),
        ..Config::default().with_depth(6).with_budget_ms(150)
    });
    for g in positions(3, &[0, 9, 18]) {
        let p = g.current;
        let r = s.search(&g, p);
        assert!(!r.moves.is_empty(), "capw found nothing on day {}", g.day);
        assert_root_is_honest(&r, &g, p, "capw=8");
        for (m, _) in &r.moves {
            check_move(&g, p, m).unwrap_or_else(|e| panic!("capw=8 on day {}: {e}", g.day));
        }
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

// ---- MCTS: priors, the plan seam, and what the TUI is shown -------------

/// The first node of `p`'s turn with at least `want` legal steps.
///
/// Searching at `Phase::Beg` is nearly useless as a test fixture: begging needs
/// corn < 3, so the node is width 1 in most positions and `search_at` returns
/// the forced-move shortcut without running a simulation. The nodes with a real
/// distribution are `Placing` and `Take`, several steps down the chain.
fn wide_node(
    g: &GameState,
    p: PlayerId,
    want: usize,
) -> Option<(GameState, tzolkin::phase::Phase, PlayerId, u8)> {
    use tzolkin::phase::Phase;
    use tzolkin::tree;
    let mut st = *g;
    let mut at = (Phase::Beg, p, 0u8);
    for _ in 0..32 {
        let (phase, turn, done) = at;
        let steps = tree::legal_steps(&st, phase, turn, done);
        if steps.len() >= want {
            return Some((st, phase, turn, done));
        }
        let t = tree::apply_step(&mut st, phase, turn, done, steps.first()?);
        if t.committed() {
            return None;
        }
        at = t.next()?;
    }
    None
}

/// The one-ply prior has to *move* the search, or it is a cost with no effect.
///
/// It is checked as a shift in the visit distribution rather than as a better
/// move: at 96 simulations the two often agree on the argmax, and a test that
/// asserted they disagreed would be asserting that the prior is bad.
#[test]
fn a_one_ply_prior_changes_where_the_search_looks() {
    use tzolkin::mcts::{Mcts, MctsConfig, Priors};
    use tzolkin::phase::{HeuristicEvaluator, Phase};

    let _ = Phase::Beg;
    let mut moved = 0usize;
    let mut checked = 0usize;
    for g in positions(3, &[0, 6, 12, 18]) {
        let Some((st, phase, turn, done)) = wide_node(&g, g.current, 4) else {
            continue;
        };
        let run = |priors: Priors| {
            let cfg = MctsConfig {
                priors,
                dirichlet_eps: 0.0,
                temperature: 0.0,
                ..MctsConfig::default()
            };
            let mut m = Mcts::new(HeuristicEvaluator, cfg);
            m.search_at(&st, phase, turn, done, 96).visits
        };
        let flat = run(Priors::Evaluator);
        let informed = run(Priors::OnePly);
        checked += 1;
        assert_eq!(flat.len(), informed.len(), "the two saw different edge sets");
        if flat != informed {
            moved += 1;
        }
    }
    assert!(checked > 0, "no position had a searchable root");
    assert!(
        moved > 0,
        "the one-ply prior left every visit count identical on {checked} roots, \
         so it is not reaching the tree"
    );
}

/// The seam `src/plan.rs` is meant to steer: a bias raises a prior and never
/// removes an edge, so a heavily favoured edge gains visits while its siblings
/// keep some.
#[test]
fn a_prior_bias_steers_without_excluding() {
    use std::sync::Arc;
    use tzolkin::ids::PlayerId;
    use tzolkin::mcts::{Mcts, MctsConfig, PriorBias};
    use tzolkin::phase::{HeuristicEvaluator, Phase, Step};
    use tzolkin::state::GameState;

    /// Everything the plan does not like is left at 1.0; the last edge of every
    /// node is what it wants.
    struct Last;
    impl PriorBias for Last {
        fn bias(
            &self,
            _s: &GameState,
            _ph: Phase,
            _mover: PlayerId,
            steps: &[Step],
            out: &mut [f32],
        ) {
            if let Some(x) = out.last_mut() {
                *x = 50.0;
            }
            let _ = steps;
        }
        fn name(&self) -> String {
            "last-edge".into()
        }
    }

    let cfg = MctsConfig {
        dirichlet_eps: 0.0,
        temperature: 0.0,
        ..MctsConfig::default()
    };
    let mut steered = 0usize;
    let mut checked = 0usize;
    for g in positions(2, &[0, 9, 18]) {
        let Some((st, phase, turn, done)) = wide_node(&g, g.current, 4) else {
            continue;
        };
        let sims = 128;
        let mut plain = Mcts::new(HeuristicEvaluator, cfg);
        let a = plain.search_at(&st, phase, turn, done, sims);
        let mut biased = Mcts::new(HeuristicEvaluator, cfg);
        biased.set_bias(Some(Arc::new(Last)));
        let b = biased.search_at(&st, phase, turn, done, sims);
        checked += 1;
        assert_eq!(a.visits.len(), b.visits.len());
        // Nothing is excluded: every edge the unbiased search opened is still
        // an edge here, whatever the plan thinks of it.
        for ((sa, _), (sb, _)) in a.visits.iter().zip(b.visits.iter()) {
            assert_eq!(sa, sb, "a bias reordered or dropped an edge");
        }
        let last = a.visits.len() - 1;
        if b.visits[last].1 > a.visits[last].1 {
            steered += 1;
        }
    }
    assert!(checked > 0);
    assert!(
        steered > 0,
        "a 50x bias on the last edge never raised its visit count over {checked} roots"
    );
    assert_eq!(
        {
            let mut m = Mcts::new(HeuristicEvaluator, cfg);
            m.set_bias(Some(Arc::new(Last)));
            m.bias_name()
        },
        Some("last-edge".to_string())
    );
}

/// Every knob a sweep varies has to reach the config *and* the name.
///
/// The alpha-beta work lost a whole experiment to the other half of this:
/// `minimax_label` ignored the harness flags, so an arena run racing two
/// variants printed the same agent name on both sides and its result file
/// cannot be attributed to either.
#[test]
fn mcts_flags_reach_the_config_and_the_label() {
    use tzolkin::mcts::Priors;
    use tzolkin::record::AgentSpec;

    let spec = |s: &str| AgentSpec::parse(s, false).unwrap_or_else(|e| panic!("{s}: {e}"));
    let cfg = |s: &str| *spec(s).mcts_config().expect("not an mcts spec");

    assert_eq!(cfg("mcts:64").priors, Priors::Evaluator);
    assert_eq!(cfg("mcts:64:heuristic:pri=1ply").priors, Priors::OnePly);
    assert_eq!(cfg("mcts:64:pri=1ply").priors, Priors::OnePly);
    assert_eq!(cfg("mcts:64:pri=1ply,ptemp=2.5").prior_temp, 2.5);
    assert_eq!(cfg("mcts:64:pri=1ply,pmin=8").prior_min_edges, 8);
    assert_eq!(cfg("mcts:64:cp=1.5,fpu=0.4,k=16").c_puct_init, 1.5);
    assert_eq!(cfg("mcts:64:cp=1.5,fpu=0.4,k=16").fpu_reduction, 0.4);
    assert_eq!(cfg("mcts:64:cp=1.5,fpu=0.4,k=16").max_edges, 16);
    assert!(!cfg("mcts:64:noreuse").tree_reuse);
    assert!(AgentSpec::parse("mcts:64:nonsense=3", false).is_err());

    // No two variants may share a name, or the progress file cannot say which
    // side of a race a block belongs to.
    let names: Vec<String> = [
        "mcts:64",
        "mcts:64:pri=1ply",
        "mcts:64:pri=1ply,ptemp=2",
        "mcts:64:pri=1ply,pmin=8",
        "mcts:64:cp=1.5",
        "mcts:64:k=16",
        "mcts:64:noreuse",
        "mcts:128",
    ]
    .iter()
    .map(|s| spec(s).name())
    .collect();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "two variants share a name: {names:?}");
}

/// Recording decision nodes and exploring are separate questions.
///
/// `bin/tui` asks for the first and used to get the second: 0.25 Dirichlet
/// noise on its root priors and playout-cap randomisation running seven turns
/// in eight at an eighth of the budget. Checked on the config rather than on
/// play, because the noise is seeded and both searches are deterministic.
#[test]
fn recording_no_longer_forces_self_play_exploration() {
    use std::sync::Arc;
    use tzolkin::mcts::MctsConfig;
    use tzolkin::phase::{Evaluator, HeuristicEvaluator};
    use tzolkin::record::{Exploration, SearchAgent};

    let ev = || Arc::new(HeuristicEvaluator) as Arc<dyn Evaluator>;

    let selfplay = SearchAgent::new(ev(), 64, true, 0);
    assert!(selfplay.mcts_config().dirichlet_eps > 0.0);
    assert!(selfplay.full_share() < 1.0);

    let arena = SearchAgent::new(ev(), 64, false, 0);
    assert_eq!(arena.mcts_config().dirichlet_eps, 0.0);
    assert_eq!(arena.full_share(), 1.0);

    let viewer = SearchAgent::with_config(
        ev(),
        64,
        true,
        Exploration::Off,
        MctsConfig::default(),
        0,
    );
    assert_eq!(viewer.mcts_config().dirichlet_eps, 0.0);
    assert_eq!(viewer.full_share(), 1.0);
}

/// The `agent` view of the TUI, for an MCTS agent.
///
/// It used to fall through to `eval::rank_all` — a one-ply heuristic ranking
/// shown under a heading claiming to be the agent's own.
#[test]
fn mcts_ranks_its_own_turns_and_does_not_claim_the_whole_move_space() {
    use tzolkin::record::parse_analysis_agent;

    let agent = parse_analysis_agent("mcts:96").unwrap();
    let mut ranked = 0usize;
    for g in positions(2, &[0, 8, 16]) {
        let p = g.current;
        let r = agent
            .ranked_moves(&g, p, 10)
            .expect("a search agent has a ranking");
        if r.moves.is_empty() {
            continue;
        }
        ranked += 1;
        for (m, _) in &r.moves {
            check_move(&g, p, m).unwrap_or_else(|e| panic!("mcts ranked an illegal move: {e}"));
        }
        // Best first, and a visit share is a percentage.
        for w in r.moves.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted best first");
        }
        let shown: f32 = r.moves.iter().map(|(_, s)| *s).sum();
        assert!(
            shown > 0.0 && shown <= 100.01,
            "visit shares must be percentages of one distribution, got {shown}"
        );
        // The panel's standing claim is that the shortlist came out of the
        // whole move space. A tree that only holds the paths it walked cannot
        // support that, so it must not imply it.
        assert!(!r.exhaustive);
        assert!(r.note.contains("NOT the whole move space"), "note: {}", r.note);
        assert!(r.distinct <= r.total);
        assert!(r.moves.len() <= r.distinct);
    }
    assert!(ranked > 0, "no position produced a ranking");
    // Two spellings of one turn are one row: retrieval orderings that commute
    // are distinct paths in the tree and the same move to a reader.
    for g in positions(1, &[12]) {
        let p = g.current;
        if let Some(r) = agent.ranked_moves(&g, p, 10) {
            for (i, (a, _)) in r.moves.iter().enumerate() {
                for (b, _) in r.moves.iter().skip(i + 1) {
                    assert!(!a.same_effect(b), "the shortlist restates a move");
                }
            }
        }
    }
}
