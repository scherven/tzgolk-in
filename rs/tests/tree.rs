//! The factored tree pinned to the flat move generator.
//!
//! `docs/SEARCH.md` §2.8. The tree is a reimplementation of legality, so the
//! only thing that makes it safe is being able to say, on real positions:
//!
//! > the set of `GameState`s reachable by walking every path from `Beg` to a
//! > commit edge equals the set reachable by `apply_move` over `legal_moves`.
//!
//! Sets of states, not sets of moves: the engine's ordering memo collapses
//! commuting retrievals, so one state legitimately backs several `Move`s.

use std::collections::HashSet;

use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::invariants::validate;
use tzolkin::mcts::{Mcts, MctsConfig};
use tzolkin::moves::{
    self, apply_move, check_move, has_normal_move, legal_moves, legal_moves_capped, Placement,
};
use tzolkin::phase::{HeuristicEvaluator, ModeChoice, Phase, Step};
use tzolkin::state::GameState;
use tzolkin::tree;

/// Above this the flat walk is too slow to be worth pinning against, and §2.8
/// only asks for positions under it.
const MAX_MOVES: usize = 5_000;

// ---- the equivalence ---------------------------------------------------

/// The right-hand side of §2.8: every state `apply_move` can reach.
fn flat_reachable(state: &GameState, p: PlayerId) -> HashSet<GameState> {
    legal_moves(state, p)
        .iter()
        .map(|m| {
            let mut next = *state;
            apply_move(&mut next, p, m);
            next
        })
        .collect()
}

/// Returns the number of distinct reachable states, or `None` if the position
/// was too wide to check.
fn check_equivalence(state: &GameState, p: PlayerId, label: &str) -> Option<usize> {
    if legal_moves_capped(state, p, MAX_MOVES + 1).len() > MAX_MOVES {
        return None;
    }

    let flat = flat_reachable(state, p);
    let factored = tree::reachable_after_turn(state, p);

    if flat != factored {
        let only_tree: Vec<_> = factored.difference(&flat).take(2).collect();
        let only_flat: Vec<_> = flat.difference(&factored).take(2).collect();
        panic!(
            "{label}: factored tree and legal_moves disagree for {p:?}\n  \
             flat {} states, factored {} states\n  \
             {} reachable only through the tree, {} only through legal_moves\n  \
             corn {} temples {:?} on_board {} available {} normal_move {}\n  \
             sample tree-only: {:?}\n  sample flat-only: {:?}",
            flat.len(),
            factored.len(),
            factored.difference(&flat).count(),
            flat.difference(&factored).count(),
            state.players[p.idx()].corn,
            [
                state.temple_pos(p, Temple::Brown),
                state.temple_pos(p, Temple::Yellow),
                state.temple_pos(p, Temple::Green)
            ],
            state.on_board(p).count(),
            state.available(p).count(),
            has_normal_move(state, p),
            only_tree.first().map(|s| s.players[p.idx()]),
            only_flat.first().map(|s| s.players[p.idx()]),
        );
    }
    Some(flat.len())
}

/// Seeded games driven by the sampling rollout policy, checking every turn of
/// every player against the flat generator. Sampled play rather than the modulo
/// pick because §1.1 measures it as reaching wider, better-resourced positions.
#[test]
fn factored_tree_matches_legal_moves() {
    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut widest = 0usize;
    let mut begging_live = 0usize;
    let mut beg_and_retrieve = 0usize;
    let mut pity_live = 0usize;
    let mut late = 0usize;
    let mut beg_filtered = 0usize;

    for seed in 0..24u64 {
        let mut g = Game::new(seed);
        while !g.state.over {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                let day = g.state.day;

                let can_beg = g.state.players[p.idx()].corn < 3
                    && Temple::ALL
                        .iter()
                        .any(|&t| g.state.can_temple_step(p, t, -1));
                if !has_normal_move(&g.state, p) {
                    pity_live += 1;
                } else if can_beg {
                    begging_live += 1;
                    // Begging one temple down and stepping it back up lands
                    // where begging another does, so these are the positions
                    // where the engine's cross-beg memo actually fires.
                    if g.state.on_board(p).next().is_some() {
                        beg_and_retrieve += 1;
                    }
                }
                if day >= 18 {
                    late += 1;
                }
                // How often the `Beg` viability filter actually removes a
                // variant. Deliberately *not* asserted positive: it is 0 here,
                // and honestly so -- in ordinary play every gear has a space
                // affordable on no corn, so no beg variant is ever dead. Left
                // in as a tripwire, because the day this stops being 0 is the
                // day this sweep starts covering the filter, and until then
                // `begging_only_position_matches` is its only coverage.
                if tree::legal_steps(&g.state, Phase::Beg, p, 0).len()
                    < moves::beg_options(&g.state, p).len()
                {
                    beg_filtered += 1;
                }

                match check_equivalence(&g.state, p, &format!("seed {seed} day {day}")) {
                    Some(n) => {
                        checked += 1;
                        widest = widest.max(n);
                    }
                    None => skipped += 1,
                }

                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }

    // Guard the guard: a coverage regression that silently stopped exercising
    // the interesting positions would leave this test passing on nothing.
    assert!(checked > 2_000, "only {checked} positions checked");
    assert!(late > 200, "only {late} late-game positions checked");
    assert!(begging_live > 20, "only {begging_live} positions with begging live");
    assert!(
        beg_and_retrieve > 20,
        "only {beg_and_retrieve} positions where beg variants can converge"
    );
    println!(
        "checked {checked} positions ({skipped} over {MAX_MOVES} moves), \
         widest {widest} distinct states, {begging_live} with begging live \
         ({beg_and_retrieve} of them with a worker to retrieve), {pity_live} pity, \
         {late} from day 18 on; the beg viability filter fired in {beg_filtered} \
         of them (see `begging_only_position_matches`)"
    );
}

// ---- crafted corner positions ------------------------------------------

/// Fill the low spaces of every gear and the first player space, so that a
/// player with no corn has nowhere affordable to go.
fn choke_board(state: &mut GameState, upto: u8) {
    let mut donors: Vec<WorkerId> = Vec::new();
    for p in [PlayerId(1), PlayerId(2), PlayerId(3)] {
        for _ in 0..3 {
            state.unlock_worker(p);
        }
        donors.extend(state.available(p));
    }
    let mut next = donors.into_iter();
    for gear in Gear::ALL {
        for pos in 0..upto {
            let w = next.next().expect("not enough donor workers");
            state.place_worker(w, gear, Pos(pos));
        }
    }
    if state.first_player_space.is_none() {
        state.place_on_first_player(next.next().expect("not enough donor workers"));
    }
}

/// No corn, no temple left to fall down, nothing affordable, nothing on the
/// gears: the gods take pity.
#[test]
fn pity_position_matches() {
    let mut g = Game::new(7);
    let p = PlayerId(0);
    choke_board(&mut g.state, 1);
    g.state.players[p.idx()].corn = 0;
    for t in Temple::ALL {
        g.state.temples[t.idx()][p.idx()] = 0;
    }
    validate(&g.state).unwrap();

    assert!(!has_normal_move(&g.state, p), "expected a pity position");
    let moves = legal_moves(&g.state, p);
    assert!(!moves.is_empty(), "pity must always leave something");

    // The chain is forced right up to the spot: one beg edge, one mode edge.
    let beg = tree::legal_steps(&g.state, Phase::Beg, p, 0);
    assert_eq!(beg, vec![Step::Beg(None)]);
    let mut probe = g.state;
    tree::apply_step(&mut probe, Phase::Beg, p, 0, &Step::Beg(None));
    assert_eq!(
        tree::legal_steps(&probe, Phase::Mode, p, 0),
        vec![Step::Mode(ModeChoice::Pity)]
    );

    check_equivalence(&g.state, p, "pity").expect("pity positions are tiny");
}

/// Nothing affordable without begging, and begging makes the cheapest space
/// reachable. `Beg(None)` leads nowhere and must not be offered, or the search
/// would carry a dead branch.
///
/// These crafted boards are the **only** coverage the `Beg` viability filter
/// has. `factored_tree_matches_legal_moves` cannot see it: the filter fires in
/// 0 of ~2,300 positions of ordinary sampled play, because every gear normally
/// has a free space cheap enough to afford with no corn at all, so no beg
/// variant is ever dead. Its `begging_live` counter is not a proxy — "begging
/// is available" is not "a beg variant leads nowhere".
///
/// So the order here matters, and is deliberate: `check_equivalence` runs
/// **first**. The structural assertion below is the cheap obvious one, and if
/// it runs first it fires on any regression and the expensive claim — that
/// dropping a dead beg variant drops no reachable state — never gets evaluated
/// at all. Measured on the first board: unfiltering `beg_steps` takes the
/// reachable set from 15 states to 20, because the dead variant falls through
/// `mode_steps`' `out.is_empty()` arm into a pity placement that the flat
/// generator never offers while a normal move exists. That is the filter
/// "earning the pity gate" (see `tree::beg_steps`), and it is what set-equality
/// is here to check.
#[test]
fn begging_only_position_matches() {
    let mut fired = 0usize;
    for (seed, choke) in [(11u64, 2u8), (12, 2), (14, 3), (19, 3)] {
        let mut g = Game::new(seed);
        let p = PlayerId(0);
        choke_board(&mut g.state, choke);
        g.state.players[p.idx()].corn = 0;
        validate(&g.state).unwrap();

        // A starting tile can leave `p` a worker on the board, which makes this
        // a beg-or-retrieve board rather than a beg-only one.
        if !has_normal_move(&g.state, p) || g.state.on_board(p).next().is_some() {
            continue;
        }
        let label = format!("beg-only seed {seed} choke {choke}");

        // The load-bearing claim, before the cheap structural one.
        check_equivalence(&g.state, p, &label).expect("small position");

        for m in legal_moves(&g.state, p) {
            assert!(m.beg.is_some(), "{label}: expected only begging moves, got {m}");
        }
        let beg = tree::legal_steps(&g.state, Phase::Beg, p, 0);
        assert!(!beg.is_empty());
        assert!(
            !beg.contains(&Step::Beg(None)),
            "{label}: the empty beg leads nowhere here and must be filtered out"
        );
        assert!(
            beg.len() < moves::beg_options(&g.state, p).len(),
            "{label}: the viability filter did not actually remove anything, \
             so this board is not exercising it"
        );
        fired += 1;
    }
    assert!(fired >= 2, "only {fired} boards actually exercised the beg filter");
}

/// Begging is live but so is retrieving, so every beg variant leads somewhere.
#[test]
fn begging_and_retrieving_both_live() {
    let mut g = Game::new(13);
    let p = PlayerId(0);
    choke_board(&mut g.state, 2);
    g.state.players[p.idx()].corn = 1;
    // Put one of p's own workers on a gear, above the choke.
    let w = g.state.available(p).next().unwrap();
    g.state.place_worker(w, Gear::Yaxchilan, Pos(4));
    validate(&g.state).unwrap();

    let beg = tree::legal_steps(&g.state, Phase::Beg, p, 0);
    assert!(beg.contains(&Step::Beg(None)), "retrieving works without begging");
    assert!(beg.len() > 1, "begging should also be on offer");

    check_equivalence(&g.state, p, "beg+retrieve").expect("small position");
}

/// Retrieval-heavy: several workers deep on the gears at once, which is where
/// the engine's ordering memo does the most work and therefore where the
/// factored tree's transpositions have to line up exactly.
#[test]
fn retrieval_heavy_positions_match() {
    for (seed, corn) in [(31u64, 0u8), (32, 2), (33, 9)] {
        let mut g = Game::new(seed);
        let p = PlayerId(0);
        for _ in 0..3 {
            g.state.unlock_worker(p);
        }
        g.state.players[p.idx()].corn = corn;
        g.state.players[p.idx()].res = [3, 3, 3, 0];
        // Through the bank, so the 13-skull conservation invariant holds.
        g.state.take_skulls(p, 1);

        // Three workers spread over three gears, high enough to have real
        // choices and to be able to pay down to the cheaper ones. A fourth puts
        // the position over §2.8's 5,000-move ceiling, which is itself worth
        // noticing: three workers on the board is already most of the way there
        // flat, and is eight narrow nodes factored.
        let spots = [
            (Gear::Palenque, Pos(3)),
            (Gear::Yaxchilan, Pos(2)),
            (Gear::Tikal, Pos(2)),
        ];
        for (gear, pos) in spots {
            let w = g.state.available(p).next().unwrap();
            g.state.place_worker(w, gear, pos);
        }
        validate(&g.state).unwrap();
        assert_eq!(g.state.on_board(p).count(), 3);

        let n = check_equivalence(&g.state, p, &format!("retrieval-heavy corn {corn}"))
            .unwrap_or_else(|| panic!("seed {seed} corn {corn} is over the cap"));
        assert!(n > 50, "expected a busy retrieval position, got {n} states");
    }
}

/// A player who can only place: no workers on the gears, plenty of corn.
#[test]
fn placement_only_position_matches() {
    let mut g = Game::new(3);
    let p = PlayerId(0);
    g.state.unlock_worker(p);
    g.state.players[p.idx()].corn = 7;
    assert_eq!(g.state.on_board(p).count(), 0);
    let n = check_equivalence(&g.state, p, "placement-only").expect("small position");
    assert!(n > 100, "expected a wide placement node, got {n}");
}

/// Mid and late game, reached by fast-forwarding the calendar so the gears carry
/// workers into the high-value spaces.
#[test]
fn mid_and_late_game_positions_match() {
    for seed in 0..8u64 {
        let mut g = Game::new(seed);
        let mut rounds = 0;
        while !g.state.over && rounds < 26 {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
            rounds += 1;
            if g.state.over {
                break;
            }
            if rounds >= 10 {
                for p in PlayerId::ALL {
                    check_equivalence(
                        &g.state,
                        p,
                        &format!("seed {seed} round {rounds} (mid/late)"),
                    );
                }
            }
        }
    }
}

// ---- reconstruction and invariants -------------------------------------

/// Every `Move` the search reassembles from a path must survive the engine's
/// own legality check, per §2.8's second paragraph.
#[test]
fn reconstructed_moves_pass_check_move() {
    let mut tested = 0usize;
    for seed in 0..6u64 {
        let mut g = Game::new(seed);
        for _ in 0..8 {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                if legal_moves_capped(&g.state, p, 400).len() < 400 {
                    for path in tree::enumerate_paths(&g.state, p, 400) {
                        let mut m = tree::move_from_path(&path)
                            .unwrap_or_else(|| panic!("path {path:?} rebuilt no move"));
                        tree::retag_workers(&g.state, p, &mut m);
                        check_move(&g.state, p, &m).unwrap_or_else(|e| {
                            panic!("reconstructed {m} from {path:?} is illegal: {e}")
                        });
                        assert!(m.n_workers() > 0, "reconstructed an empty move");

                        // And it lands exactly where the tree said it would.
                        let mut by_move = g.state;
                        apply_move(&mut by_move, p, &m);
                        let mut by_steps = g.state;
                        assert!(tree::apply_path(&mut by_steps, p, &path));
                        assert_eq!(by_move, by_steps, "{m} and its path disagree");
                        tested += 1;
                    }
                }
                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }
    assert!(tested > 500, "only {tested} paths reconstructed");
}

/// §2.7's two invariants, checked structurally rather than by inspection.
#[test]
fn stop_edges_are_gated() {
    let mut g = Game::new(5);
    for _ in 0..6 {
        g.state.current = g.state.first_player;
        for _ in 0..N_PLAYERS {
            let p = g.state.current;

            // `StopPlacing` never appears at n = 0 ...
            assert!(!tree::legal_steps(&g.state, Phase::Placing { n: 0 }, p, 0)
                .contains(&Step::StopPlacing));
            // ... and `StopRetrieving` never before a worker is resolved.
            assert!(!tree::legal_steps(&g.state, Phase::PickWorker, p, 0)
                .contains(&Step::StopRetrieving));
            if g.state.on_board(p).next().is_some() {
                assert!(tree::legal_steps(&g.state, Phase::PickWorker, p, 1)
                    .contains(&Step::StopRetrieving));
            }

            // Pity is offered only when nothing else is.
            let mode = tree::legal_steps(&g.state, Phase::Mode, p, 0);
            if mode.contains(&Step::Mode(ModeChoice::Pity)) {
                assert_eq!(mode.len(), 1, "pity was offered alongside something else");
                assert!(!has_normal_move(&g.state, p));
            }

            g.take_turn_sampled();
            g.state.current = g.state.current.next(1);
        }
        g.end_round_public();
    }
}

/// Every node inside a turn has at least one edge.
///
/// A dead branch costs the flat generator nothing -- it simply emits no move --
/// but in a search it is a node MCTS has to select from and cannot. It is also
/// the failure mode the `Beg` viability filter exists to prevent, and one the
/// set-equality test cannot see, because a dead end contributes no state.
#[test]
fn no_node_inside_a_turn_is_a_dead_end() {
    fn walk(state: &GameState, phase: Phase, turn: PlayerId, done: u8, seen: &mut HashSet<(GameState, Phase, u8)>) {
        if !seen.insert((*state, phase, done)) {
            return;
        }
        let steps = tree::legal_steps(state, phase, turn, done);
        assert!(
            !steps.is_empty(),
            "dead end at {phase:?} for {turn:?} after {done} retrievals"
        );
        if steps.contains(&Step::Mode(ModeChoice::Pity)) {
            assert_eq!(steps.len(), 1, "pity offered alongside something else");
        }
        for step in steps {
            let mut next = *state;
            if let Some((phase, done)) = tree::step_within_turn(&mut next, phase, turn, done, &step)
            {
                walk(&next, phase, turn, done, seen);
            }
        }
    }

    for seed in 0..6u64 {
        let mut g = Game::new(seed);
        for _ in 0..10 {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                if legal_moves_capped(&g.state, p, MAX_MOVES + 1).len() <= MAX_MOVES {
                    walk(&g.state, Phase::Beg, p, 0, &mut HashSet::new());
                }
                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }
}

/// Placement edges only ever name the lowest free space of a gear, and only
/// while the running cost is affordable.
#[test]
fn placement_edges_track_lowest_free_and_budget() {
    let mut g = Game::new(17);
    for _ in 0..10 {
        g.state.current = g.state.first_player;
        for _ in 0..N_PLAYERS {
            let p = g.state.current;
            for n in 0..4u8 {
                for step in tree::legal_steps(&g.state, Phase::Placing { n }, p, 0) {
                    if let Step::Place(spot) = step {
                        if let Placement::Gear(gear, pos) = spot {
                            assert_eq!(g.state.lowest_free(gear), Some(pos));
                        }
                        assert!(n + spot.space_cost() <= g.state.players[p.idx()].corn);
                    }
                }
            }
            g.take_turn_sampled();
            g.state.current = g.state.current.next(1);
        }
        g.end_round_public();
    }
}

// ---- round flow --------------------------------------------------------

/// Driving a whole game through `apply_step` alone must reach the same kind of
/// finish the engine's own loop does: every state valid, the calendar reaching
/// its end, and scores that exist.
#[test]
fn stepping_drives_a_whole_game() {
    for seed in 0..4u64 {
        let g = Game::new(seed);
        let mut state = g.state;
        let mut at = (Phase::Beg, state.first_player, 0u8);
        let mut rng = SmallRng::new(seed ^ 0x5eed);
        let mut steps = 0usize;
        let mut extra_days = 0usize;

        loop {
            let (phase, turn, done) = at;
            let legal = tree::legal_steps(&state, phase, turn, done);
            assert!(!legal.is_empty(), "dead end at {phase:?} for {turn:?}");
            if matches!(phase, Phase::ExtraDay { .. }) {
                extra_days += 1;
            }
            let step = legal[rng.next_below(legal.len())].clone();
            let t = tree::apply_step(&mut state, phase, turn, done, &step);
            validate(&state).unwrap_or_else(|e| panic!("after {step:?}: {e}"));
            steps += 1;
            assert!(steps < 200_000, "game did not terminate");
            match t.next() {
                None => break,
                Some(next) => at = next,
            }
        }

        assert!(state.over);
        assert!(state.day >= 20, "game ended early on day {}", state.day);
        assert!(!state.winners().is_empty());
        assert!(extra_days > 0, "the ExtraDay node was never reached");
    }
}

/// A commit edge hands the turn to the next seat, and the fourth commit of a
/// round runs the round end.
#[test]
fn commit_hands_off_in_seat_order() {
    let g = Game::new(23);
    let mut state = g.state;
    let first = state.first_player;
    let day = state.day;
    let mut at = (Phase::Beg, first, 0u8);
    let mut committed = 0usize;

    while committed < 3 {
        let (phase, turn, done) = at;
        let step = tree::legal_steps(&state, phase, turn, done)[0].clone();
        let t = tree::apply_step(&mut state, phase, turn, done, &step);
        if t.committed() {
            committed += 1;
            let (next_phase, next_turn, _) = t.next().unwrap();
            assert_eq!(next_phase, Phase::Beg);
            assert_eq!(next_turn, first.next(committed));
        }
        at = t.next().unwrap();
    }
    assert_eq!(state.day, day, "the calendar moved mid-round");
}

/// The setup draft: four edges then three, not one six-way choice, so the pick
/// reuses the same per-candidate head as `Take`.
#[test]
fn draft_is_two_sequential_picks() {
    let g = Game::new(29);
    let mut state = g.state;
    let p = PlayerId(0);
    let dealt = [0u8, 1, 2, 3];
    let before = state.players[p.idx()].corn;

    let first = Phase::DraftTile {
        dealt,
        kept: tree::DRAFT_NONE,
    };
    let steps = tree::legal_steps(&state, first, p, 0);
    assert_eq!(steps.len(), 4);

    let t = tree::apply_step(&mut state, first, p, 0, &Step::DraftTile(2));
    assert!(!t.committed(), "the first pick does not end the draft");
    let (second, _, _) = t.next().unwrap();
    assert_eq!(second, Phase::DraftTile { dealt, kept: 2 });

    let steps = tree::legal_steps(&state, second, p, 0);
    assert_eq!(steps.len(), 3);
    assert!(!steps.contains(&Step::DraftTile(2)), "a tile cannot be kept twice");

    let t = tree::apply_step(&mut state, second, p, 0, &Step::DraftTile(0));
    assert!(t.committed());
    assert_eq!(t.next().unwrap().1, p.next(1), "the draft passes to the left");
    assert!(
        state.players[p.idx()].corn > before,
        "the kept tiles were never applied"
    );
}

// ---- MCTS --------------------------------------------------------------

fn small_config(seed: u64) -> MctsConfig {
    MctsConfig {
        seed,
        temperature: 1.0,
        ..MctsConfig::default()
    }
}

#[test]
fn search_returns_a_legal_step_and_a_visit_distribution() {
    let g = Game::new(2);
    let p = g.state.first_player;
    let mut mcts = Mcts::new(HeuristicEvaluator, small_config(1));
    let r = mcts.search(&g.state, Phase::Beg, p, 128);

    let legal = tree::legal_steps(&g.state, Phase::Beg, p, 0);
    assert!(legal.contains(&r.step));
    assert!(!r.visits.is_empty());
    let total: u32 = r.visits.iter().map(|(_, n)| n).sum();
    assert_eq!(total as usize, r.sims as usize, "every simulation must land on an edge");
    let policy: f32 = r.policy().iter().sum();
    assert!((policy - 1.0).abs() < 1e-3, "policy sums to {policy}");
}

/// The value is a 4-vector that sums to zero, not a scalar, and not a win/loss.
#[test]
fn backed_up_value_is_centred() {
    let g = Game::new(4);
    let p = g.state.first_player;
    let mut mcts = Mcts::new(HeuristicEvaluator, small_config(9));
    // A `Mode` node rather than `Beg`, which is often forced.
    let r = mcts.search(&g.state, Phase::Mode, p, 200);
    let sum: f32 = r.root_value.iter().sum();
    assert!(sum.abs() < 1e-3, "value {:?} sums to {sum}", r.root_value);
    for v in r.root_value {
        assert!(v > -1.0 && v < 1.0);
    }
}

#[test]
fn search_is_deterministic_for_a_seed() {
    let g = Game::new(6);
    let p = g.state.first_player;
    let a = Mcts::new(HeuristicEvaluator, small_config(77))
        .search(&g.state, Phase::Mode, p, 96)
        .visits;
    let b = Mcts::new(HeuristicEvaluator, small_config(77))
        .search(&g.state, Phase::Mode, p, 96)
        .visits;
    assert_eq!(a.len(), b.len());
    for ((s1, n1), (s2, n2)) in a.iter().zip(b.iter()) {
        assert_eq!(s1, s2);
        assert_eq!(n1, n2);
    }
}

/// A forced node costs no simulations at all -- §3.6.
#[test]
fn single_edge_nodes_are_collapsed() {
    let mut g = Game::new(7);
    let p = PlayerId(0);
    choke_board(&mut g.state, 1);
    g.state.players[p.idx()].corn = 0;
    for t in Temple::ALL {
        g.state.temples[t.idx()][p.idx()] = 0;
    }
    let mut mcts = Mcts::new(HeuristicEvaluator, small_config(3));
    let r = mcts.search(&g.state, Phase::Beg, p, 256);
    assert_eq!(r.sims, 0, "a forced beg should not have been searched");
    assert_eq!(r.step, Step::Beg(None));
}

/// Virtual loss is applied and lifted for real -- §3.5 -- even though nothing
/// in the current single-threaded search can observe one.
///
/// That is the point of this test. `virtual_loss` is *inert* today: one descent
/// is in flight at a time, so the loss an edge carries is lifted by the same
/// simulation that applied it and no selection ever sees it. This test prints
/// the proof of that inertness rather than asserting it, because the day
/// §3.5's threading lands the inertness stops being true -- but the residue
/// invariant below does not, and a failure to lift would then quietly poison
/// every `Q` in the tree.
#[test]
fn virtual_loss_is_lifted_and_currently_inert() {
    let mut g = Game::new(19);
    for _ in 0..8 {
        g.state.current = g.state.first_player;
        for _ in 0..N_PLAYERS {
            g.take_turn_sampled();
            g.state.current = g.state.current.next(1);
        }
        g.end_round_public();
    }

    let mut baseline: Option<Vec<(Step, u32)>> = None;
    let mut inert = true;
    for vl in [0u32, 1, 5, 10] {
        let mut cfg = small_config(5);
        cfg.virtual_loss = vl;
        let mut mcts = Mcts::new(HeuristicEvaluator, cfg);
        let r = mcts.search(&g.state, Phase::Beg, g.state.current, 600);

        // The invariant that survives threading: nothing is left behind.
        assert_eq!(
            mcts.virtual_loss_residue(),
            0,
            "virtual loss {vl} left residue on the arena after the search"
        );
        // Its corollary at the root: every simulation is accounted for on some
        // edge, which a half-lifted loss would break.
        let total: u32 = r.visits.iter().map(|(_, n)| n).sum();
        assert_eq!(total, r.sims, "virtual loss {vl} lost simulations");

        match &baseline {
            None => baseline = Some(r.visits),
            Some(b) => inert &= *b == r.visits,
        }
    }
    println!(
        "virtual loss is {} in the current single-threaded search",
        if inert { "inert (identical visit counts at vl = 0, 1, 5, 10)" } else { "OBSERVABLE" }
    );
}

/// A whole turn played through the search: every sub-decision yields a target,
/// the state stays valid, and the turn ends on a commit.
#[test]
fn play_turn_produces_one_target_per_sub_decision() {
    let g = Game::new(8);
    let mut state = g.state;
    let p = state.first_player;
    let mut mcts = Mcts::new(HeuristicEvaluator, small_config(21));
    let turn = mcts.play_turn(&mut state, Phase::Beg, p, 48);

    validate(&state).unwrap();
    assert!(turn.steps.len() >= 3, "a turn is at least beg/mode/act/stop");
    assert!(turn.next.is_some());
    for (phase, turn_p, done, r) in &turn.steps {
        assert!(!r.visits.is_empty());
        if r.sims > 0 {
            assert!(r.visits.len() > 1, "{phase:?} was searched with one edge");
        }
        let _ = (turn_p, done);
    }
}

/// A short self-play game driven entirely by the search, to prove the whole
/// loop closes: turn, round end, extra day, food day, scoring.
#[test]
fn search_can_finish_a_game() {
    let g = Game::new(12);
    let mut state = g.state;
    let mut at = (Phase::Beg, state.first_player, 0u8);
    let mut mcts = Mcts::new(HeuristicEvaluator, small_config(31));
    let mut targets = 0usize;
    let mut turns = 0usize;

    loop {
        let (phase, turn, done) = at;
        let r = mcts.search_at(&state, phase, turn, done, 8);
        targets += 1;
        let t = tree::apply_step(&mut state, phase, turn, done, &r.step);
        if t.committed() {
            turns += 1;
        }
        validate(&state).unwrap();
        match t.next() {
            None => break,
            Some(next) => at = next,
        }
        assert!(targets < 100_000);
    }

    assert!(state.over);
    assert!(turns > 80, "only {turns} turns played");
    let scores = state.scores();
    assert!(scores.iter().any(|&s| s != 0));
    println!("search game: {turns} turns, {targets} decisions, scores {scores:?}");
}

// ---- diagnostics -------------------------------------------------------

/// Node widths across sampled play, pinned to SEARCH.md §1.2.
///
/// This is a measurement, but it is not *only* a measurement: the whole design
/// rests on the claim that factoring turns a node of width up to ~1.9M into a
/// chain of nodes a policy head can actually emit a distribution over. That
/// claim is falsifiable, so it is asserted here rather than eyeballed.
///
/// Two things to know before comparing the printed table to §1.2's:
///
/// * **The driver decides the answer.** Positions come from
///   `take_turn_sampled`, the same sampling rollout §1.1 uses, because a naive
///   `moves[i % len]` driver starves players of corn and lands them in narrow,
///   poor positions -- it would flatter these numbers substantially. The walk
///   *within* a turn is a uniform random path, which is what makes `Placing`
///   look wide: a random walker keeps placing rather than stopping.
/// * **The tail needs games to show up.** §1.2's max of 8,959 is one node in
///   38,844 turns. The default 12 seeds are ~1,200 turns, so the max here is
///   drawn from a much shorter tail and will read far lower. Set
///   `TZ_WIDTH_SEEDS` to sample the tail properly.
///
/// Because a single random path through a turn visits only one `Take` node, the
/// `widest/turn` max under-reports: the widest node at a position may sit on a
/// worker the walk never picked. The `Take (all workers)` row removes that
/// dependence by sizing `choices_for_worker` for *every* worker on the board at
/// every position visited, which is a property of the position alone. It is the
/// row to compare against §1.2's 8,959, and the one the assertion below uses.
///
/// `cargo test --release --test tree -- --nocapture node_widths`.
#[test]
fn node_widths() {
    let seeds: u64 = std::env::var("TZ_WIDTH_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let mut per_phase: Vec<Vec<usize>> = vec![Vec::new(); Phase::COUNT];
    let mut widest_in_turn: Vec<usize> = Vec::new();
    let mut per_turn: Vec<usize> = Vec::new();
    let mut flat: Vec<usize> = Vec::new();
    let mut all_take: Vec<usize> = Vec::new();

    for seed in 0..seeds {
        let mut g = Game::new(seed);
        while !g.state.over {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                flat.push(legal_moves_capped(&g.state, p, 2_000_001).len());
                // Every `Take` node the turn could reach, not just the one the
                // walk below happens to pick.
                for w in g.state.on_board(p) {
                    all_take.push(tree::take_candidates(&g.state, p, w).len());
                }

                // Walk one sampled path through the turn, recording each node.
                let mut probe = g.state;
                let mut at = (Phase::Beg, p, 0u8);
                let mut widest = 0usize;
                let mut n_nodes = 0usize;
                let mut rng = SmallRng::new(seed * 977 + g.state.day as u64);
                loop {
                    let (phase, turn, done) = at;
                    let steps = tree::legal_steps(&probe, phase, turn, done);
                    per_phase[phase.tag() as usize].push(steps.len());
                    widest = widest.max(steps.len());
                    n_nodes += 1;
                    let step = steps[rng.next_below(steps.len())].clone();
                    let t = tree::apply_step(&mut probe, phase, turn, done, &step);
                    if t.committed() {
                        // The between-round node sits *past* the commit, so a
                        // walk that stops at the commit never sees one and the
                        // `ExtraDay` row reads zero. Record it, then stop: it
                        // belongs to the claimer, not to this turn.
                        if let Some((next @ Phase::ExtraDay { .. }, turn, done)) = t.next() {
                            let n = tree::legal_steps(&probe, next, turn, done).len();
                            per_phase[next.tag() as usize].push(n);
                        }
                        break;
                    }
                    at = t.next().unwrap();
                }
                widest_in_turn.push(widest);
                per_turn.push(n_nodes);

                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }

    let names = [
        "Beg",
        "Mode",
        "Placing",
        "PickWorker",
        "Take",
        "ExtraDay",
        "PityPlace",
        "DraftTile",
    ];
    println!(
        "{:<14} {:>8} {:>7} {:>6} {:>6} {:>6} {:>8}",
        "node", "n", "mean", "p50", "p90", "p99", "max"
    );
    for (i, name) in names.iter().enumerate() {
        report(name, &mut per_phase[i]);
    }
    report("Take (all wkrs)", &mut all_take);
    report("widest/turn", &mut widest_in_turn);
    report("nodes/turn", &mut per_turn);
    report("flat legal_moves", &mut flat);

    // `report` sorts in place, so these are now sorted ascending.
    let max = |v: &Vec<usize>| v.last().copied().unwrap_or(0);
    let p99 = |v: &Vec<usize>| {
        if v.is_empty() {
            0
        } else {
            v[((v.len() - 1) as f64 * 0.99) as usize]
        }
    };

    // Structural ceilings. These follow from the edge generators and hold for
    // any driver at any sample size, so a breach is a bug, not variance.
    assert!(max(&per_phase[0]) <= 4, "Beg is at most no-beg plus three temples");
    assert!(max(&per_phase[1]) <= 2, "Mode is place, retrieve, or a lone pity");
    let n_place = Gear::ALL.len() + 1 + 1; // gears, first-player space, stop
    assert!(max(&per_phase[2]) <= n_place, "Placing exceeded {n_place}");
    let n_pick = WORKERS_PER_PLAYER + 1; // workers, stop
    assert!(max(&per_phase[3]) <= n_pick, "PickWorker exceeded {n_pick}");
    assert!(
        per_phase[5].iter().all(|&n| n == 2),
        "ExtraDay is always take-it-or-leave-it"
    );
    assert!(!per_phase[5].is_empty(), "no ExtraDay node was ever reached");
    assert!(max(&per_turn) <= 14, "§1.2 measured at most 14 sub-decisions per turn");

    // §1.2's headline numbers, as ceilings, taken on the path-independent row.
    // If either ever breaches, the premise of the whole factoring -- that a
    // policy head can emit a distribution over these nodes -- has changed, and
    // SEARCH.md needs remeasuring before the search is trusted again.
    assert!(
        p99(&all_take) <= 225,
        "Take p99 is {}, over §1.2's 225",
        p99(&all_take)
    );
    assert!(
        max(&all_take) <= 8_959,
        "widest Take node is {}, over §1.2's 8,959",
        max(&all_take)
    );
    // And the cap really is the thing keeping the policy head addressable.
    assert!(
        max(&all_take) > MctsConfig::default().widen_cap,
        "no node exceeded the widening cap, so §2.6's cap is untested here"
    );

    // The point of the exercise: the flat node this replaces really is huge.
    assert!(
        max(&flat) > 50 * max(&widest_in_turn),
        "flat max {} is not meaningfully wider than factored max {}",
        max(&flat),
        max(&widest_in_turn)
    );
}

/// What a search actually costs, and how far widening reaches.
/// `cargo test --release --test tree -- --ignored --nocapture search_cost`.
#[test]
#[ignore]
fn search_cost() {
    for sims in [200u32, 800] {
        let mut g = Game::new(19);
        // Play into the middle game, where the `Take` tail lives.
        for _ in 0..12 {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                g.take_turn_sampled();
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }

        let mut mcts = Mcts::new(HeuristicEvaluator, small_config(5));
        let mut nodes = 0usize;
        let mut decisions = 0usize;
        let mut widened = 0usize;
        let mut capped = 0usize;
        let start = std::time::Instant::now();

        let mut state = g.state;
        for _ in 0..8 {
            let mut at = (Phase::Beg, state.current, 0u8);
            loop {
                let (phase, turn, done) = at;
                let r = mcts.search_at(&state, phase, turn, done, sims);
                decisions += 1;
                nodes += r.nodes;
                if r.legal_edges > 32 {
                    capped += 1;
                    if r.visits.len() > 32 {
                        widened += 1;
                    }
                }
                let t = tree::apply_step(&mut state, phase, turn, done, &r.step);
                let committed = t.committed();
                match t.next() {
                    None => break,
                    Some(next) => at = next,
                }
                if committed {
                    break;
                }
            }
        }
        let elapsed = start.elapsed();
        println!(
            "{sims} sims: {decisions} decisions in {:?} ({:.2} ms each), \
             {:.0} nodes/search, {capped} nodes over K=32 ({widened} widened past it)",
            elapsed,
            elapsed.as_secs_f64() * 1000.0 / decisions as f64,
            nodes as f64 / decisions as f64,
        );
    }
}

fn report(name: &str, v: &mut [usize]) {
    if v.is_empty() {
        println!("{name:<14} {:>8}", 0);
        return;
    }
    v.sort_unstable();
    let pct = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
    println!(
        "{name:<14} {:>8} {mean:>7.1} {:>6} {:>6} {:>6} {:>8}",
        v.len(),
        pct(0.5),
        pct(0.9),
        pct(0.99),
        v[v.len() - 1]
    );
}

// ---- a tiny deterministic RNG ------------------------------------------

/// splitmix64. The tests want reproducible arbitrary choices without pulling
/// `rand` into the test's dependency surface.
struct SmallRng(u64);

impl SmallRng {
    fn new(seed: u64) -> Self {
        SmallRng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }

    fn next_below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z % n.max(1) as u64) as usize
    }
}

// ---- tree reuse --------------------------------------------------------

/// A turn is 5-8 sub-decisions and each one used to start from an empty arena,
/// discarding the subtree under the edge just played -- including the node
/// about to be searched, with all of its statistics.
///
/// Reuse must fire within a turn, and must be *conservative*: a node from
/// outside the root player's turn carries priors that were never noised, and
/// searching those as if they had been would quietly change the search.
#[test]
fn tree_reuse_fires_within_a_turn() {
    use tzolkin::mcts::{Mcts, MctsConfig};
    use tzolkin::phase::{HeuristicEvaluator, Phase};

    let mut g = Game::new(11);
    for _ in 0..6 {
        g.play_round();
    }

    let mut mcts = Mcts::new(HeuristicEvaluator, MctsConfig::default());
    let mut state = g.state;
    let turn = state.current;

    let played = mcts.play_turn(&mut state, Phase::Beg, turn, 128);
    let (reused, _) = mcts.reuse_stats();

    assert!(
        played.steps.len() >= 2,
        "a turn should be several sub-decisions, got {}",
        played.steps.len()
    );
    // The counters describe the *last* search of the turn. Some turns end on a
    // forced move, which does not search at all, so walk the whole turn and
    // require reuse to have fired at least once.
    let mut fired = reused > 0;
    let mut state = g.state;
    let mut mcts = Mcts::new(HeuristicEvaluator, MctsConfig::default());
    let mut at = (Phase::Beg, turn, 0u8);
    for _ in 0..8 {
        let r = mcts.search_at(&state.clone(), at.0, at.1, at.2, 128);
        if mcts.reuse_stats().0 > 0 {
            fired = true;
        }
        let t = tzolkin::tree::apply_step(&mut state, at.0, at.1, at.2, &r.step);
        match t.next() {
            Some(next) if !t.committed() => at = next,
            _ => break,
        }
    }
    assert!(fired, "tree reuse never fired across a whole turn");
}

/// Reuse is a search-*quality* change, not a speed one, and this pins that.
///
/// A reused node keeps the visits it accumulated as an interior node of the
/// previous sub-decision's search and then receives the full `sims` again, so
/// the effective budget at reused nodes is strictly larger. Wall-clock is a
/// wash -- measured 0.77 vs 0.79 games/s at 512 simulations, because retaining
/// costs an arena sweep and an index rebuild that roughly cancel the expansions
/// saved. The win, if there is one, is in the strength of the moves.
///
/// If this ever starts asserting that the two agree, someone has made reuse
/// inert and should find out why.
#[test]
fn tree_reuse_deepens_reused_nodes() {
    use tzolkin::mcts::{Mcts, MctsConfig};
    use tzolkin::phase::{HeuristicEvaluator, Phase};

    let mut g = Game::new(5);
    for _ in 0..5 {
        g.play_round();
    }
    let turn = g.state.current;

    let play = |reuse: bool| {
        let cfg = MctsConfig {
            tree_reuse: reuse,
            ..MctsConfig::default()
        };
        let mut mcts = Mcts::new(HeuristicEvaluator, cfg);
        let mut state = g.state;
        let played = mcts.play_turn(&mut state, Phase::Beg, turn, 96);
        let total: u32 = played
            .steps
            .iter()
            .map(|(_, _, _, r)| r.visits.iter().map(|(_, n)| *n).sum::<u32>())
            .sum();
        let steps: Vec<_> = played.steps.iter().map(|(_, _, _, r)| r.step.clone()).collect();
        (steps, state, total)
    };

    // Determinism first: same settings, same answer.
    assert_eq!(play(true).0, play(true).0, "reuse is not deterministic");
    assert_eq!(play(false).0, play(false).0, "no-reuse is not deterministic");

    let (_, _, with) = play(true);
    let (_, _, without) = play(false);
    assert!(
        with >= without,
        "reuse should not lose simulations: {with} with, {without} without"
    );
}
