//! The early-game research tilt: `eval::tilt_seat`, `record::tilted`, and the
//! `tilt=` / `tiltk=` agent spec.
//!
//! Two things are being pinned here and they pull in opposite directions.
//!
//! 1. **With no tilt installed, `eval::heuristic` is the function it always
//!    was.** Two arena races, a 20,000-game self-play and a training job are
//!    live on it. The proof that matters is cross-binary — `bin/trackrace
//!    --verify` run on a HEAD binary and on this one, output diffed, in
//!    `docs/FINDINGS-tilt.md` §4 — and what is here is the half a test can
//!    hold: the table starts empty, a guard puts back what it found, and one
//!    seat's tilt is invisible to another's estimate.
//! 2. **With one installed, it reaches the parts of `mcts.rs` that decide
//!    which move is even considered.** `Gradient` and `Priors::OnePly` call
//!    `eval::heuristic` directly, past any injected `Evaluator`, so
//!    `the_tilt_reaches_the_edge_ordering` is the test that says this agent is
//!    the intended one rather than a leaf-value-only imitation of it.

use tzolkin::effect::{Choice, Effect};
use tzolkin::eval::{self, ResearchTilt, TiltShape};
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::mcts::Gradient;
use tzolkin::phase::Step;
use tzolkin::record::{self, Agent, AgentSpec};

/// A fresh game with `row` written into seat `p`'s research row, on `day`.
fn with_research(seed: u64, p: PlayerId, row: [u8; 4], day: u8) -> tzolkin::state::GameState {
    let mut g = Game::new(seed).state;
    g.research[p.idx()] = row;
    g.day = day;
    g
}

const AGRI: usize = 0;
const ARCH: usize = 2;

// ---- the mechanism -----------------------------------------------------

/// Nothing is installed until something installs it, and the guard puts back
/// exactly what it found.
///
/// The restore is not housekeeping. `bin/tui` scores its own `full` move list
/// with `eval::margin` between turns, on the same thread that just ran a
/// tilted agent's search; without the restore that list would be silently
/// scored through somebody else's opinion of research.
#[test]
fn the_table_is_empty_until_a_seat_installs_one_and_is_empty_again_after() {
    assert_eq!(eval::tilt_table(), [[0.0; 4]; N_PLAYERS]);
    {
        let _g = eval::tilt_seat(PlayerId(1), [4.0, 0.4, 1.2, 0.8]);
        assert_eq!(eval::tilt_table()[1], [4.0, 0.4, 1.2, 0.8]);
        assert_eq!(eval::tilt_table()[0], [0.0; 4], "only the named seat");
        {
            let _h = eval::tilt_seat(PlayerId(1), [1.0, 0.0, 0.0, 0.0]);
            assert_eq!(eval::tilt_table()[1], [1.0, 0.0, 0.0, 0.0]);
        }
        assert_eq!(eval::tilt_table()[1], [4.0, 0.4, 1.2, 0.8], "nested restore");
    }
    assert_eq!(eval::tilt_table(), [[0.0; 4]; N_PLAYERS]);
}

/// With the tilt off, `heuristic` returns the identical bits.
///
/// This is the weak, in-process half of the claim — the same binary cannot
/// prove it did not change. It is here because it is free and because it
/// catches the mistake that would break the cross-binary proof: a tilt row
/// left installed by a previous test, or a guard that did not fire.
#[test]
fn an_uninstalled_tilt_changes_no_bit_of_the_estimate() {
    for seed in [1u64, 7, 99] {
        for day in [0u8, 7, 14, 21] {
            let g = with_research(seed, PlayerId(0), [1, 2, 3, 1], day);
            let before = eval::heuristic(&g, PlayerId(0));
            {
                let _z = eval::tilt_seat(PlayerId(0), [0.0; 4]);
                assert_eq!(
                    eval::heuristic(&g, PlayerId(0)).to_bits(),
                    before.to_bits(),
                    "an all-zero row is not a tilt"
                );
            }
            assert_eq!(eval::heuristic(&g, PlayerId(0)).to_bits(), before.to_bits());
        }
    }
}

/// The tilt is exactly the research block re-priced, and this is the golden
/// arithmetic that says so.
///
/// `engine_value` prices a held level at `research_step_value(s, l) * uses *
/// RESEARCH_SCALE`, with `uses = min(rounds_left / 3, 7)` and `RESEARCH_SCALE
/// = 0.05`. The tilt adds that same product times `strength * w[s] *
/// early(horizon)`. `eval::research_tilt` keeps its own copy of the `uses`
/// ramp — it has to, because every one of the six unapplied arm patches in
/// `docs/FINDINGS-track-shape.md` has a hunk inside `engine_value` — so if
/// that ramp is ever changed in one place and not the other, this fails.
#[test]
fn the_tilt_is_the_research_block_re_priced() {
    // Agriculture 3 on day 0: uses = min(27/3, 7) = 7, early = 1.
    let g = with_research(5, PlayerId(0), [3, 0, 0, 0], 0);
    let base = eval::heuristic(&g, PlayerId(0));
    let strength = 4.0f32;
    let d = {
        let _t = eval::tilt_seat(PlayerId(0), [strength, 0.0, 0.0, 0.0]);
        eval::heuristic(&g, PlayerId(0)) - base
    };
    // research_step_value(Agriculture, 1..=3) = 0.35 + 0.45 + 0.75.
    let want = (0.35 + 0.45 + 0.75) * 7.0 * 0.05 * strength * 1.0;
    assert!(
        (d - want).abs() < 1e-5,
        "agriculture 3, day 0, strength 4: got {d}, want {want}"
    );

    // And it is linear in the strength, because that is all it is.
    let half = {
        let _t = eval::tilt_seat(PlayerId(0), [strength / 2.0, 0.0, 0.0, 0.0]);
        eval::heuristic(&g, PlayerId(0)) - base
    };
    assert!((half * 2.0 - d).abs() < 1e-4, "{half} * 2 != {d}");
}

/// It is an *early-game* opinion, and it is spent by the end.
///
/// The premium falls twice over: `tilt_early` fades it directly, and the
/// `uses` ramp it multiplies is already fading. What is checked is the shape —
/// flat through day 7, most of it gone by day 21, none of it on the last day.
#[test]
fn the_tilt_is_spent_by_the_end_of_the_calendar() {
    let row = [4.0, 0.0, 0.0, 0.0];
    let d = |day: u8| {
        let g = with_research(5, PlayerId(0), [3, 0, 0, 0], day);
        let base = eval::heuristic(&g, PlayerId(0));
        let _t = eval::tilt_seat(PlayerId(0), row);
        eval::heuristic(&g, PlayerId(0)) - base
    };
    let (d0, d7, d14, d21, d27) = (d(0), d(7), d(14), d(21), d(27));
    // The *premium* is flat through day 7 — `tilt_early` is clamped at 1 until
    // horizon 0.714, and day 7 is 0.741. What it multiplies is not quite: the
    // `uses` cap stops binding at day 7 (`rounds_left >= 21` is days 0..=6), so
    // the product slips 5%. That is `engine_value`'s curve, not the tilt's.
    assert_eq!(eval::tilt_early(27.0 / 27.0), eval::tilt_early(20.0 / 27.0));
    assert!(d7 > 0.94 * d0, "all but the `uses` slip through day 7: {d0} vs {d7}");
    assert!(d14 < 0.5 * d0, "past the knee by day 14: {d14} vs {d0}");
    assert!(d21 < 0.15 * d0, "nearly gone by day 21: {d21} vs {d0}");
    assert_eq!(d27, 0.0, "nothing left on the last day");
    // The curve itself, at the four days R14 measured.
    for (day, want) in [(0u8, 1.0f32), (7, 1.0), (14, 0.674), (21, 0.311)] {
        let h = (27 - day) as f32 / 27.0;
        assert!(
            (eval::tilt_early(h) - want).abs() < 0.01,
            "day {day}: early {} want {want}",
            eval::tilt_early(h)
        );
    }
}

/// A seat's tilt is its own. It does not change what anyone else's position is
/// worth, which is what makes "two of these against two of those" one game
/// rather than one evaluator with a mood.
#[test]
fn the_tilt_is_a_property_of_one_seat() {
    let g = with_research(11, PlayerId(0), [3, 0, 0, 0], 0);
    let mut g = g;
    g.research[1] = [3, 0, 0, 0];
    let before: Vec<f32> = PlayerId::ALL.iter().map(|&p| eval::heuristic(&g, p)).collect();
    let _t = eval::tilt_seat(PlayerId(0), ResearchTilt::new(TiltShape::Agri, 4.0).row());
    let after: Vec<f32> = PlayerId::ALL.iter().map(|&p| eval::heuristic(&g, p)).collect();
    assert!(after[0] > before[0] + 0.1, "seat 0 moved: {before:?} {after:?}");
    for s in 1..N_PLAYERS {
        assert_eq!(
            after[s].to_bits(),
            before[s].to_bits(),
            "seat {s} has the same identical research row and must not move"
        );
    }
}

/// **The test this whole vehicle exists for.**
///
/// `mcts.rs` calls `eval::heuristic` directly in `Gradient::new` (the edge
/// ordering) and in `one_ply` (the prior), neither of which goes through the
/// injected `Evaluator`. Those two decide which moves a node even keeps, so an
/// agent whose tilt reached only the leaf value would be a different and much
/// weaker agent than the one asked for. `Gradient::step` is the one of the two
/// that is reachable from a test, and it prices the research step by probing
/// `heuristic` — so if the tilt shows up here, it is in the ordering.
#[test]
fn the_tilt_reaches_the_edge_ordering() {
    let g = Game::new(3).state;
    let p = PlayerId(0);
    let step = |s: Science| Step::Take(Choice::one(Effect::AdvanceResearch(s)));

    let plain = Gradient::new(&g, p);
    let (base_agri, base_ext) = (
        plain.step(&step(Science::Agriculture)),
        plain.step(&step(Science::Extraction)),
    );

    let tilted = {
        let _t = eval::tilt_seat(p, ResearchTilt::new(TiltShape::Agri, 4.0).row());
        Gradient::new(&g, p)
    };
    // The probe happens inside `Gradient::new`, so the price list is already
    // built by the time the guard drops — which is also how `Mcts` uses it.
    let (agri, ext) = (
        tilted.step(&step(Science::Agriculture)),
        tilted.step(&step(Science::Extraction)),
    );

    // On day 0 the arithmetic is exact: `uses` is 7, `early` is 1, and the
    // first level of Agriculture is priced 0.35, so the tilt adds
    // 0.35 * 7 * 0.05 * (4.0 * 1.00) = 0.49 to the ordering's price for
    // advancing it, and 0.55 * 7 * 0.05 * (4.0 * 0.10) = 0.077 to
    // Extraction's.
    assert!(
        (agri - base_agri - 0.49).abs() < 0.02,
        "the edge ordering must carry the tilt: agriculture {base_agri} -> \
         {agri}, expected +0.49"
    );
    assert!(
        (ext - base_ext - 0.077).abs() < 0.02,
        "extraction has weight 0.10, so it rises a tenth as much: \
         {base_ext} -> {ext}"
    );
    assert!(
        agri > ext,
        "and the ordering now prefers agriculture: {agri} vs {ext}"
    );
}

/// The two shapes are the two opinions, and they disagree about which track.
#[test]
fn the_two_shapes_favour_different_tracks() {
    let agri = TiltShape::Agri.weights();
    let causal = TiltShape::Causal.weights();
    assert_eq!(agri[AGRI], 1.0, "agri normalises on agriculture");
    assert_eq!(causal[ARCH], 1.0, "causal normalises on architecture (R13)");
    assert!(agri[AGRI] > 3.0 * agri[ARCH], "agriculture by far the most");
    // Descending, and much smaller: Architecture, Theology, Extraction — R13's
    // measured order for the three the user did not name.
    assert!(agri[2] > agri[3] && agri[3] > agri[1], "{agri:?}");
    // The causal shape is R13's four numbers over the largest.
    for (i, want) in [
        (0usize, 7.49f32 / 23.52),
        (1, 4.51 / 23.52),
        (3, 15.96 / 23.52),
    ] {
        assert!((causal[i] - want).abs() < 0.005, "{i}: {causal:?}");
    }
}

// ---- the agent spec ----------------------------------------------------

#[test]
fn a_tilted_spec_parses_defaults_and_refuses_nonsense() {
    let t = |s: &str| AgentSpec::parse(s, false).map(|a| a.research_tilt());
    assert_eq!(
        t("mcts:8192:heuristic:deeper,tilt=agri").unwrap(),
        Some(ResearchTilt::new(TiltShape::Agri, ResearchTilt::DEFAULT_STRENGTH))
    );
    assert_eq!(
        t("mcts:2048:heuristic:tilt=causal,tiltk=6").unwrap(),
        Some(ResearchTilt::new(TiltShape::Causal, 6.0))
    );
    // Order-independent: one agent, spelled two ways.
    assert_eq!(
        t("mcts:2048:heuristic:tiltk=6,tilt=causal").unwrap(),
        t("mcts:2048:heuristic:tilt=causal,tiltk=6").unwrap()
    );
    // The strength alone means the default shape, which is the one the brief
    // asked for.
    assert_eq!(
        t("mcts:2048:heuristic:tiltk=2").unwrap(),
        Some(ResearchTilt::new(TiltShape::Agri, 2.0))
    );
    // And the champion is still untilted.
    assert_eq!(t("mcts:8192:heuristic:deeper").unwrap(), None);
    for bad in [
        "mcts:2048:heuristic:tilt=corn",
        "mcts:2048:heuristic:tilt=",
        "mcts:2048:heuristic:tiltk=x",
    ] {
        assert!(AgentSpec::parse(bad, false).is_err(), "{bad} should not parse");
    }
}

/// Two sides of a race must not print the same name, and the name a spec
/// advertises must be the name its instance answers to.
///
/// `examples/nametest.rs` records what this cost the first time: `mcts_label`
/// omitted a flag, both sides printed `mcts2048/heuristic`, and a result table
/// claimed an agent had beaten itself by nine points. The tilt is spelled in
/// two code paths — `mcts_label` for the spec, `TiltedAgent::name` for the
/// instance — so both are checked against each other here.
#[test]
fn the_spec_and_its_instance_agree_on_the_name() {
    let specs = [
        "mcts:8192:heuristic:deeper",
        "mcts:8192:heuristic:deeper,tilt=agri",
        "mcts:8192:heuristic:deeper,tilt=agri,tiltk=2",
        "mcts:8192:heuristic:deeper,tilt=causal",
        "mcts:2048:heuristic:tilt=agri",
    ];
    let mut seen: Vec<(String, &str)> = Vec::new();
    for s in specs {
        let spec = AgentSpec::parse(s, false).expect(s);
        let name = spec.name();
        assert_eq!(name, spec.instance().name(), "spec {s}");
        if let Some((_, other)) = seen.iter().find(|(n, _)| *n == name) {
            panic!("{s} and {other} both print {name}");
        }
        seen.push((name, s));
    }
    // The tilt prints last, so an untilted agent's name is a prefix of its
    // tilted variant's and a results table sorts them together.
    let champ = AgentSpec::parse("mcts:8192:heuristic:deeper", false).unwrap().name();
    let tilted = AgentSpec::parse("mcts:8192:heuristic:deeper,tilt=agri", false)
        .unwrap()
        .name();
    assert_eq!(
        tilted,
        format!("{champ}:tilt=agri:tiltk={}", ResearchTilt::DEFAULT_STRENGTH)
    );
}

// ---- behaviour ---------------------------------------------------------

/// Positions from a real game, one per turn, up to `until` — the window the
/// tilt is actually paid in.
///
/// Played by `heuristic:32` so the corpus costs nothing and does not depend on
/// the agent under test: both sides are then asked the *same* questions.
fn corpus(seed: u64, until: u8) -> Vec<(tzolkin::state::GameState, PlayerId)> {
    use rand::SeedableRng;
    let a = record::parse_agent("heuristic:32", false).expect("spec");
    let refs: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|_| a.as_ref());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x515E);
    let mut game = record::new_drafted_game(seed, &refs, &mut rng);
    let mut out = Vec::new();
    let mut guard = 0;
    while !game.over && game.day <= until && guard < 200 {
        game.current = game.first_player;
        for _ in 0..N_PLAYERS {
            let p = game.current;
            out.push((game, p));
            if let Some(o) = refs[p.idx()].play_turn(&game, p, 0.0, &mut rng) {
                tzolkin::moves::apply_move(&mut game, p, &o.mv);
                game.refill_buildings();
            }
            game.current = game.current.next(1);
        }
        let claimer = game.resolve_first_player();
        let mut days = 1;
        if let Some(p) = claimer {
            if game.may_take_extra_day(p) && refs[p.idx()].extra_day(&game, p, &mut rng).0 {
                game.spend_extra_day(p);
                days = 2;
            }
        }
        game.advance_days(days);
        guard += 1;
    }
    out
}

/// Total research levels seat `p` holds.
fn levels(g: &tzolkin::state::GameState, p: PlayerId) -> u32 {
    g.research[p.idx()].iter().map(|&l| l as u32).sum()
}

/// How many of `corpus`'s positions this agent answers with a move that
/// advances its own research, and how many it is asked.
///
/// Paired by construction: the tilted agent and the plain one are handed the
/// **same positions** and the same rng seed, so the difference between the two
/// counts is the tilt and nothing else. That is what makes this readable at
/// two dozen decisions where an end-of-game level count needs hundreds of
/// games — seat and draft effects were swamping it (the levels a seat finishes
/// with vary by 3x between seats of the *same* agent).
fn research_rate(spec: &str, tilt: Option<ResearchTilt>, corpus: &[(tzolkin::state::GameState, PlayerId)]) -> (usize, usize) {
    use rand::SeedableRng;
    let agent = match tilt {
        Some(t) => record::tilted(record::parse_agent(spec, false).expect("spec"), t),
        None => record::parse_agent(spec, false).expect("spec"),
    };
    let mut took = 0;
    for (i, (g, p)) in corpus.iter().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA55E_7 + i as u64);
        let Some(out) = agent.play_turn(g, *p, 0.0, &mut rng) else {
            continue;
        };
        let mut after = *g;
        tzolkin::moves::apply_move(&mut after, *p, &out.mv);
        if levels(&after, *p) > levels(g, *p) {
            took += 1;
        }
    }
    (took, corpus.len())
}

/// One game with seat 0 played by `spec` (tilted or not) and every other seat
/// by the plain `spec`, returning seat 0's research levels at the end of the
/// tilt's window (day 14) and at the end of the game.
///
/// Paired: the seed, the opponents and the rng stream are identical between
/// the tilted run and the plain one, so seat 0's own choices are the only
/// thing that differs. That is what makes a dozen games readable where the
/// unpaired version needed hundreds — the levels a seat finishes with vary by
/// 3x between *seats of the same agent*.
fn paired_levels(seed: u64, spec: &str, tilt: Option<ResearchTilt>) -> (u32, u32) {
    use rand::SeedableRng;
    let agents: [Box<dyn Agent>; N_PLAYERS] = std::array::from_fn(|s| {
        let a = record::parse_agent(spec, false).expect("spec");
        match (s, tilt) {
            (0, Some(t)) => record::tilted(a, t),
            _ => a,
        }
    });
    let refs: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|s| agents[s].as_ref());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x7117);
    let mut game = record::new_drafted_game(seed, &refs, &mut rng);
    let seat = PlayerId(0);
    let mut at14 = 0;
    let mut guard = 0;
    while !game.over && guard < 200 {
        if game.day <= 14 {
            at14 = levels(&game, seat);
        }
        game.current = game.first_player;
        for _ in 0..N_PLAYERS {
            let p = game.current;
            if let Some(o) = refs[p.idx()].play_turn(&game, p, 0.0, &mut rng) {
                tzolkin::moves::apply_move(&mut game, p, &o.mv);
                game.refill_buildings();
            }
            game.current = game.current.next(1);
        }
        let claimer = game.resolve_first_player();
        let mut days = 1;
        if let Some(p) = claimer {
            if game.may_take_extra_day(p) && refs[p.idx()].extra_day(&game, p, &mut rng).0 {
                game.spend_extra_day(p);
                days = 2;
            }
        }
        game.advance_days(days);
        guard += 1;
    }
    assert!(guard < 200, "seed {seed} hit the round guard");
    (at14, levels(&game, seat))
}

/// Seat 0's research over `n` paired games, tilted and plain.
fn paired_sweep(spec: &str, tilt: Option<ResearchTilt>, n: u64) -> (f32, f32) {
    let mut a = 0u32;
    let mut b = 0u32;
    for seed in 0..n {
        let (d14, end) = paired_levels(6_100 + seed, spec, tilt);
        a += d14;
        b += end;
    }
    (a as f32 / n as f32, b as f32 / n as f32)
}

/// The sweep that chose `ResearchTilt::DEFAULT_STRENGTH`, kept because the
/// next person to doubt the number should re-run it rather than re-derive it.
///
///     cargo test --release --test tilt -- --ignored --nocapture calibrate
///     TILT_SPEC=mcts:2048:heuristic:deeper TILT_N=8 cargo test ... calibrate
///
/// Ignored by default: a few hundred searches, which is a tool and not a test.
/// It is also **not a strength measurement** — it counts research, which is
/// what the tilt is *for*, and says nothing whatever about whether the agent
/// wins more.
#[test]
#[ignore = "calibration tool, not an assertion; run with --ignored"]
fn calibrate_the_strength() {
    let spec = std::env::var("TILT_SPEC").unwrap_or_else(|_| "mcts:256:heuristic:deeper".into());
    let n: u64 = std::env::var("TILT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let seeds: u64 = std::env::var("TILT_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut positions = Vec::new();
    for s in 0..seeds {
        positions.extend(corpus(4_400 + s, 12));
    }
    println!(
        "{spec}\n  {n} paired games (seat 0 tilted, 1-3 plain) + {} decisions from {seeds} games",
        positions.len()
    );
    println!(
        "{:>7} {:>8} | {:>10} {:>10} | {:>14}",
        "k", "shape", "lvls d14", "lvls end", "research moves"
    );
    let (b14, bend) = paired_sweep(&spec, None, n);
    let (bmv, nmv) = research_rate(&spec, None, &positions);
    println!(
        "{:>7} {:>8} | {b14:>10.2} {bend:>10.2} | {bmv:>7} / {nmv}   the champion",
        "-", "-"
    );
    for shape in TiltShape::ALL {
        for k in [4.0f32, 8.0, 12.0, 16.0, 32.0] {
            let t = Some(ResearchTilt::new(shape, k));
            let (l14, lend) = paired_sweep(&spec, t, n);
            let (mv, _) = research_rate(&spec, t, &positions);
            println!(
                "{k:>7} {:>8} | {l14:>10.2} {lend:>10.2} | {mv:>7} / {nmv}",
                shape.name()
            );
        }
    }
}

/// **The calibration, as an assertion.** The default strength has to be big
/// enough that the agent visibly researches more; that is the whole point of
/// it.
///
/// The number it is correcting is the gap R13 measured: a maxed track moves
/// `heuristic` by about 1.2 points against a causal +7.49, so a tilt of a few
/// per cent cannot move a single decision. Paired against the champion over
/// the same seeds, seat 0 finishes the tilt's own window — day 14 — with
/// **more than twice** the research levels.
#[test]
fn the_tilt_changes_what_the_agent_plays() {
    let spec = "mcts:256:heuristic:deeper";
    let n = 12;
    let (plain14, plain_end) = paired_sweep(spec, None, n);
    let (tilt14, tilt_end) = paired_sweep(
        spec,
        Some(ResearchTilt::new(TiltShape::Agri, ResearchTilt::DEFAULT_STRENGTH)),
        n,
    );
    println!(
        "{spec}, {n} paired games: day 14 levels {plain14:.2} -> {tilt14:.2}, \
         final {plain_end:.2} -> {tilt_end:.2}"
    );
    // Deterministic — fixed seeds, fixed rng, fixed agent seeds — so this is
    // a fact about the shipped default and not a sample that might not repeat.
    // The threshold is 1.4x against a measured 1.7x; see
    // `ResearchTilt::DEFAULT_STRENGTH` for the whole sweep.
    assert!(
        tilt14 >= 1.4 * plain14.max(0.05),
        "the tilt must visibly move play toward research by day 14: \
         {tilt14:.2} against the champion's {plain14:.2}"
    );
    assert!(
        tilt_end > plain_end,
        "and must still be ahead at the end: {tilt_end:.2} vs {plain_end:.2}"
    );
}
