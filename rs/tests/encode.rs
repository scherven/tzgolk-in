//! Encoder and network tests.
//!
//! The one that matters most is `deck_order_is_invisible`. `GameState` carries
//! the undrawn deck in shuffled order; an encoder that read it would train an
//! agent that sees the future and does not transfer to a real table. That
//! failure is silent — the net would simply get better — so it needs a test
//! that fails loudly the moment someone wires `Deck::ids` in.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tzolkin::data::buildings::N_BUILDINGS;
use tzolkin::data::monuments::N_MONUMENTS;
use tzolkin::effect::{Choice, Effect};
use tzolkin::encode::*;
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::{beg_options, choices_for_worker};
use tzolkin::net::{Arch, BatchConfig, BatchedEvaluator, Gemm, Net, Portable, Query, Unbatched};
use tzolkin::phase::{Evaluation, Evaluator, Phase};
use tzolkin::state::GameState;

fn played(seed: u64, rounds: usize) -> Game {
    let mut g = Game::new(seed);
    for _ in 0..rounds {
        if g.state.over {
            break;
        }
        g.play_round();
    }
    g
}

/// A spread of phases, so every head gets exercised.
fn phases(g: &GameState, p: PlayerId) -> Vec<Phase> {
    let mut v = vec![
        Phase::Beg,
        Phase::Mode,
        Phase::Placing { n: 0 },
        Phase::Placing { n: 2 },
        Phase::PickWorker,
        Phase::PityPlace,
        Phase::ExtraDay { claimer: p.next(1) },
        Phase::DraftTile {
            dealt: [0, 3, 7, 11],
            kept: 0,
        },
    ];
    if let Some(w) = g.on_board(p).next() {
        v.push(Phase::Take { worker: w });
    }
    v
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

#[test]
fn widths_and_offsets() {
    assert_eq!(RESERVED_OFF + RESERVED_W, D_IN);
    assert_eq!(BOARD_W, N_CELLS * CELL_CH);
    assert_eq!(N_WHO, N_CELLS + 1);
    // The debug build asserts every block's width on the way out of `encode`.
    for seed in 0..8 {
        let g = played(seed, 4);
        for p in PlayerId::ALL {
            for ph in phases(&g.state, p) {
                let v = encoded(&g.state, p, ph);
                assert_eq!(v.len(), D_IN);
                assert!(v.iter().all(|x| x.is_finite()));
                assert!(v[RESERVED_OFF..].iter().all(|&x| x == 0.0));
                assert!(v.iter().all(|&x| x.abs() < 50.0), "feature blew up");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hidden information
// ---------------------------------------------------------------------------

/// Permuting the *undrawn* tail of every deck must not move a single float.
///
/// This is the test that would fail if someone later fed `Deck::ids` to the
/// network. It is deliberately not vacuous: it asserts the two states differ as
/// `GameState`s before asserting their encodings do not.
#[test]
fn deck_order_is_invisible() {
    let mut rng = StdRng::seed_from_u64(99);
    let mut checked = 0usize;

    for seed in 0..12u64 {
        for rounds in [0usize, 3, 9] {
            let g = played(seed, rounds);
            if g.state.over {
                continue;
            }
            let base = g.state;

            for _ in 0..4 {
                let mut other = base;
                // Shuffle only what has not been drawn. Everything before the
                // cursor is public: it is face up or in somebody's tableau.
                shuffle_tail(&mut other.age1.ids, other.age1.next as usize, &mut rng);
                shuffle_tail(&mut other.age2.ids, other.age2.next as usize, &mut rng);
                shuffle_tail(
                    &mut other.monument_deck.ids,
                    other.monument_deck.next as usize,
                    &mut rng,
                );
                if other == base {
                    continue;
                }
                checked += 1;

                for p in PlayerId::ALL {
                    for ph in phases(&base, p) {
                        assert_eq!(
                            encoded(&base, p, ph),
                            encoded(&other, p, ph),
                            "seed {seed}: the encoder can see the undrawn deck"
                        );
                    }
                }
            }
        }
    }
    assert!(checked > 10, "the permutation never actually changed anything");
}

/// The same guarantee for `encode_choice`, which is a much easier place to leak
/// it from.
///
/// `encode_choice` works by *applying* the candidate to a probe copy of the
/// state and diffing summaries. Several `Effect`s refill the display, so the
/// probe advances a deck cursor and a card that was face-down becomes face-up
/// inside the probe. If any summarised quantity depended on that card, the
/// candidate's 96 features would carry the identity of the next card in the
/// deck — a strictly worse leak than the state encoder's would be, because it
/// is per-candidate and the pointer head reads it directly.
#[test]
fn deck_order_is_invisible_to_choice_features() {
    let mut rng = StdRng::seed_from_u64(4242);
    let mut checked = 0usize;
    let mut a = vec![0.0f32; D_CHOICE];
    let mut b = vec![0.0f32; D_CHOICE];

    for seed in 0..10u64 {
        for rounds in [4usize, 9, 14] {
            let g = played(seed, rounds);
            if g.state.over {
                continue;
            }
            let base = g.state;
            let mut other = base;
            shuffle_tail(&mut other.age1.ids, other.age1.next as usize, &mut rng);
            shuffle_tail(&mut other.age2.ids, other.age2.next as usize, &mut rng);
            shuffle_tail(
                &mut other.monument_deck.ids,
                other.monument_deck.next as usize,
                &mut rng,
            );
            if other == base {
                continue;
            }

            for p in PlayerId::ALL {
                for w in base.on_board(p) {
                    let Some((gr, pos)) = base.loc(w).on_board() else {
                        continue;
                    };
                    let cs = choices_for_worker(&base, p, gr, pos);
                    // The candidate *list* must not depend on it either. That is
                    // `moves.rs`, not this file, but a divergence here would make
                    // the feature comparison below meaningless.
                    assert_eq!(
                        cs,
                        choices_for_worker(&other, p, gr, pos),
                        "the candidate list depends on the undrawn deck"
                    );
                    for c in &cs {
                        encode_choice(&base, p, c, &mut a);
                        encode_choice(&other, p, c, &mut b);
                        assert_eq!(a, b, "choice features can see the undrawn deck");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 50, "only {checked} candidates compared");
}

fn shuffle_tail<const N: usize>(ids: &mut [u8; N], from: usize, rng: &mut StdRng) {
    if from >= N {
        return;
    }
    for i in (from + 1..N).rev() {
        let j = rng.gen_range(from..=i);
        ids.swap(i, j);
    }
}

/// The cursor *is* public, and it is encoded — so drawing a card must move the
/// encoding even when the deck contents cannot.
#[test]
fn deck_cursor_is_visible() {
    let g = played(3, 2);
    let mut after = g.state;
    let _ = after.age1.draw();
    assert_ne!(
        encoded(&g.state, PlayerId(0), Phase::Mode),
        encoded(&after, PlayerId(0), Phase::Mode)
    );
}

/// The unseen mask is `ALL \ (face_up ∪ owned)`, computed the way any player at
/// the table could compute it.
#[test]
fn unseen_mask_matches_a_hand_count() {
    for seed in 0..6u64 {
        let g = played(seed, 7);
        let s = &g.state;
        let mut seen_b = 0u32;
        for id in s.face_up_buildings() {
            seen_b |= 1 << (id.0 - 1);
        }
        for pl in &s.players {
            seen_b |= pl.buildings;
        }
        let want_b = (N_BUILDINGS as u32) - seen_b.count_ones();

        let mut seen_m = 0u32;
        for id in s.face_up_monuments() {
            seen_m |= 1 << (id.0 - 1);
        }
        for pl in &s.players {
            seen_m |= pl.monuments as u32;
        }
        let want_m = (N_MONUMENTS as u32) - seen_m.count_ones();

        let v = encoded(s, PlayerId(0), Phase::Mode);
        // Counts live immediately after each mask; find them by summing the
        // mask bits instead of hard-coding an offset.
        let b_count: f32 = v[GLOBAL_OFF..GLOBAL_OFF + GLOBAL_W]
            .iter()
            .copied()
            .filter(|&x| x == want_b as f32 / N_BUILDINGS as f32)
            .count() as f32;
        assert!(b_count >= 1.0, "unseen-building count missing");
        let m_count = v[GLOBAL_OFF..GLOBAL_OFF + GLOBAL_W]
            .iter()
            .filter(|&&x| x == want_m as f32 / N_MONUMENTS as f32)
            .count();
        assert!(m_count >= 1, "unseen-monument count missing");
    }
}

// ---------------------------------------------------------------------------
// Perspective
// ---------------------------------------------------------------------------

/// Slot `k` of the player block seen by `p` is slot 0 seen by `p.next(k)`.
#[test]
fn player_block_rotates() {
    for seed in 0..6u64 {
        let g = played(seed, 5);
        for p in PlayerId::ALL {
            let a = encoded(&g.state, p, Phase::Mode);
            for k in 0..N_PLAYERS {
                let b = encoded(&g.state, p.next(k), Phase::Mode);
                let lo = PLAYERS_OFF + k * PLAYER_W;
                assert_eq!(
                    &a[lo..lo + PLAYER_W],
                    &b[PLAYERS_OFF..PLAYERS_OFF + PLAYER_W],
                    "seat {k} did not rotate"
                );
            }
        }
    }
}

/// The board's ownership channels rotate too: the same physical worker reads as
/// "mine" to its owner and as some offset to everyone else.
#[test]
fn board_ownership_rotates() {
    let g = played(5, 6);
    let s = &g.state;
    let occupied: Vec<(Gear, Pos, PlayerId)> = Gear::ALL
        .iter()
        .flat_map(|&gr| {
            (0..gr.size()).filter_map(move |q| {
                s.gears[gr.idx()]
                    .at(Pos(q))
                    .map(|w| (gr, Pos(q), w.owner()))
            })
        })
        .collect();
    assert!(!occupied.is_empty(), "no workers on the board to check");

    for (gr, pos, owner) in occupied {
        let base = BOARD_OFF + cell(gr, pos) * CELL_CH;
        for p in PlayerId::ALL {
            let v = encoded(s, p, Phase::Mode);
            let off = seat_off(p, owner);
            for k in 0..N_PLAYERS {
                assert_eq!(
                    v[base + 2 + k],
                    if k == off { 1.0 } else { 0.0 },
                    "ownership channel {k} wrong from seat {p:?}"
                );
            }
        }
    }
}

/// `Color` must never reach the encoding: it is a display attribute, and
/// encoding it would break the rotation the value head depends on.
#[test]
fn colour_is_not_encoded() {
    let g = played(11, 4);
    let mut recoloured = g.state;
    for (i, pl) in recoloured.players.iter_mut().enumerate() {
        pl.color = Color::ALL[(i + 1) % N_PLAYERS];
    }
    for p in PlayerId::ALL {
        assert_eq!(
            encoded(&g.state, p, Phase::Mode),
            encoded(&recoloured, p, Phase::Mode)
        );
    }
}

// ---------------------------------------------------------------------------
// Choice features
// ---------------------------------------------------------------------------

#[test]
fn choice_features_are_state_deltas() {
    let g = played(4, 6);
    let p = PlayerId(0);
    let mut f = vec![0.0; D_CHOICE];

    // A skip changes nothing, so every delta *scalar* is zero. The
    // thermometers are not: `0 >= -1` is true, which is exactly what makes a
    // threshold linearly separable, and §8.3 leans on feature 0 (a constant
    // bias) and feature 1 (an explicit flag) to keep "do nothing" from being an
    // all-zero key that scores zero regardless of context.
    encode_choice(&g.state, p, &Choice::skip(), &mut f);
    assert_eq!(f[0], 1.0, "bias");
    assert_eq!(f[1], 1.0, "is_skip");
    assert_eq!(f[2], 0.0, "n_effects");
    const DELTA_SCALARS: &[usize] = &[
        3, 10, 14, 18, 22, 26, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 62, 77, 83, 84, 85, 86, 87,
    ];
    for &i in DELTA_SCALARS {
        assert_eq!(f[i], 0.0, "a skip moved delta feature {i}");
    }
    assert!(f[90..].iter().all(|&x| x == 0.0), "reserved tail is not zero");
    // A zero delta still lights every threshold at or below zero.
    assert_eq!((f[4], f[5], f[6], f[7]), (1.0, 1.0, 1.0, 0.0));

    // +6 corn moves the corn scalar and its thermometer, nothing else.
    let mut f2 = vec![0.0; D_CHOICE];
    encode_choice(&g.state, p, &Choice::one(Effect::Corn(6)), &mut f2);
    assert_eq!(f2[1], 0.0);
    assert!((f2[3] - 0.6).abs() < 1e-6, "Δcorn/10");
    assert_eq!(f2[8], 1.0, "Δcorn >= +4");

    // The features measure what *happened*, not what was asked for: a skull
    // grant against an empty bank is worth nothing, and only a delta knows it.
    let mut empty = g.state;
    empty.skulls_remaining = 0;
    let mut f3 = vec![0.0; D_CHOICE];
    let mut f4 = vec![0.0; D_CHOICE];
    let skull = Choice::one(Effect::Res(Resource::Skull, 1));
    encode_choice(&g.state, p, &skull, &mut f3);
    encode_choice(&empty, p, &skull, &mut f4);
    let sk = 10 + 3 * 4; // the skull resource's scalar slot
    assert!(f3[sk] > 0.0, "a skull from a full bank is a gain");
    assert_eq!(f4[sk], 0.0, "a skull from an empty bank is not");
}

#[test]
fn choice_features_never_mutate_the_state() {
    let g = played(6, 6);
    let p = PlayerId(1);
    let before = g.state;
    let mut f = vec![0.0; D_CHOICE];
    for w in g.state.on_board(p) {
        let Some((gr, pos)) = g.state.loc(w).on_board() else {
            continue;
        };
        for c in choices_for_worker(&g.state, p, gr, pos) {
            encode_choice(&g.state, p, &c, &mut f);
            assert!(f.iter().all(|x| x.is_finite()));
        }
    }
    assert_eq!(before, g.state);
}

// ---------------------------------------------------------------------------
// Edge reconstruction
// ---------------------------------------------------------------------------

/// `phase::Evaluator` hands over only `n_edges`, so the encoder re-derives the
/// edge list from the engine. When it can, the reconstruction must have the
/// right arity; when it cannot, it must say `Uniform` rather than guess.
#[test]
fn edges_reconstruct_or_abstain() {
    for seed in 0..8u64 {
        let g = played(seed, 5);
        if g.state.over {
            continue;
        }
        // One append-only buffer for every pointer node, exactly as
        // `evaluate_batch` uses it: `EdgeSpec::Pointer` hands back an offset
        // into this rather than its own `Vec`.
        let mut feats: Vec<f32> = Vec::new();
        for p in PlayerId::ALL {
            let n_beg = beg_options(&g.state, p).len();
            match edges(&g.state, p, Phase::Beg, n_beg, &mut feats) {
                EdgeSpec::Fixed { head, idx } => {
                    assert_eq!(head, Head::Beg);
                    assert_eq!(idx.len(), n_beg);
                    assert_eq!(idx[0], 0, "the no-beg edge comes first");
                    assert!(idx.iter().all(|&i| (i as usize) < Head::Beg.arity()));
                }
                _ => assert_eq!(n_beg, 1),
            }

            let board = g.state.on_board(p).count();
            for (n, want_stop) in [(board, false), (board + 1, true)] {
                if n == 0 {
                    continue;
                }
                match edges(&g.state, p, Phase::PickWorker, n, &mut feats) {
                    EdgeSpec::Fixed { head, idx } => {
                        assert_eq!(head, Head::Who);
                        assert_eq!(idx.len(), n);
                        assert!(idx.iter().all(|&i| (i as usize) < Head::Who.arity()));
                        assert_eq!(
                            idx.last() == Some(&(N_CELLS as u16)),
                            want_stop || board == 0
                        );
                    }
                    _ => panic!("PickWorker should always reconstruct"),
                }
            }

            if let Some(w) = g.state.on_board(p).next() {
                let (gr, pos) = g.state.loc(w).on_board().unwrap();
                let cs = choices_for_worker(&g.state, p, gr, pos);
                let before = feats.len();
                match edges(&g.state, p, Phase::Take { worker: w }, cs.len(), &mut feats) {
                    EdgeSpec::Pointer { off, n } => {
                        assert_eq!(n, cs.len());
                        // Appended, at the offset reported, and nothing else
                        // touched: a batch of these has to stack.
                        assert_eq!(off, before);
                        assert_eq!(feats.len(), off + n * D_CHOICE);
                        // The bias feature makes an all-zero row distinguishable
                        // from a real skip, which §8.3 needs for "do nothing".
                        assert!((0..n).all(|i| feats[off + i * D_CHOICE] == 1.0));
                    }
                    _ => panic!("Take should always produce candidates"),
                }
            }

            // A width that cannot be produced by any budget must abstain rather
            // than emit a scrambled mapping.
            let before = feats.len();
            assert!(matches!(
                edges(&g.state, p, Phase::Placing { n: 0 }, 99, &mut feats),
                EdgeSpec::Uniform
            ));
            assert_eq!(feats.len(), before, "a non-pointer edge set wrote rows");
        }
    }
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

fn check_eval(e: &Evaluation, n: usize) {
    assert_eq!(e.priors.len(), n);
    assert!(e.priors.iter().all(|p| p.is_finite() && *p >= 0.0));
    if n > 0 {
        let s: f32 = e.priors.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "priors sum to {s}");
    }
    assert!(e.value.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    let mean: f32 = e.value.iter().sum::<f32>() / N_PLAYERS as f32;
    assert!(mean.abs() < 1e-5, "value is not centred: {mean}");
}

#[test]
fn random_net_is_usable_immediately() {
    let net = Net::random(Arch::MAIN, 1);
    let g = played(2, 8);
    let mut out = Vec::new();
    for p in PlayerId::ALL {
        for ph in phases(&g.state, p) {
            for n in [1usize, 2, 5, 7] {
                let e = net.evaluate_one(&g.state, ph, p, n);
                check_eval(&e, n);
            }
        }
    }
    // And through the batched path, which is the one search should call.
    let qs: Vec<Query> = phases(&g.state, PlayerId(0))
        .into_iter()
        .map(|ph| Query::new(&g.state, ph, PlayerId(0), 3))
        .collect();
    net.evaluate_batch(&qs, &mut out);
    assert_eq!(out.len(), qs.len());
    for e in &out {
        check_eval(e, 3);
    }
}

/// Batched and single-position evaluation must agree exactly: the search will
/// mix them, and a discrepancy would be a transposition-table poisoning bug.
///
/// **The batch is deliberately ragged.** Every pointer node in it contributes a
/// different number of candidates to one shared key matrix, and the top-`K`
/// filter means the number of rows a node contributes is not even its edge
/// count. Getting the offsets wrong there silently mixes one node's candidates
/// into another's softmax — a bug that would look like a slightly worse policy
/// rather than a crash, so this is the test that has to catch it. A batch of
/// uniform-width `Mode` nodes catches nothing.
#[test]
fn batch_matches_single() {
    let net = Net::random(Arch::MAIN, 7);
    let gs: Vec<Game> = (0..6).map(|s| played(s, 4)).collect();
    let mut qs: Vec<Query> = Vec::new();
    let mut takes = 0usize;
    let mut cands: Vec<Vec<Choice>> = Vec::new();

    for (i, g) in gs.iter().enumerate() {
        let p = PlayerId((i % 4) as u8);
        qs.push(Query::new(&g.state, Phase::Mode, p, 2));
        qs.push(Query::new(&g.state, Phase::Beg, p, beg_options(&g.state, p).len()));
        // Ragged pointer rows, from real positions.
        for w in g.state.on_board(p) {
            let Some((gr, pos)) = g.state.loc(w).on_board() else {
                continue;
            };
            let cs = choices_for_worker(&g.state, p, gr, pos);
            if cs.is_empty() {
                continue;
            }
            qs.push(Query::new(&g.state, Phase::Take { worker: w }, p, cs.len()));
            takes += 1;
            cands.push(cs);
        }
        // A pointer node of a different phase, so `step_emb` is exercised too.
        qs.push(Query::new(
            &g.state,
            Phase::DraftTile {
                dealt: [1, 2, 3, 4],
                kept: 0,
            },
            p,
            4,
        ));
    }
    assert!(takes >= 6, "only {takes} Take nodes; the batch is not ragged");
    // Interleave so no two pointer nodes are adjacent in the batch.
    qs.reverse();

    let mut batched = Vec::new();
    net.evaluate_batch(&qs, &mut batched);
    assert_eq!(batched.len(), qs.len());
    for (i, q) in qs.iter().enumerate() {
        let one = net.evaluate_one(q.state, q.phase, q.turn, q.n_edges);
        assert_eq!(one.priors.len(), batched[i].priors.len());
        for k in 0..N_PLAYERS {
            assert!((one.value[k] - batched[i].value[k]).abs() < 1e-5);
        }
        for k in 0..one.priors.len() {
            assert!(
                (one.priors[k] - batched[i].priors[k]).abs() < 1e-5,
                "row {i} ({:?}) edge {k}: {} batched vs {} alone",
                q.phase,
                batched[i].priors[k],
                one.priors[k]
            );
        }
    }
}

/// The same, but with every pointer node's candidate list supplied by the
/// caller, which is the path a search actually takes.
#[test]
fn supplied_candidates_batch_ragged() {
    let net = Net::random(Arch::MAIN, 11);
    let gs: Vec<Game> = (0..5).map(|s| played(s + 20, 6)).collect();
    let mut cands: Vec<(usize, PlayerId, WorkerId, Vec<Choice>)> = Vec::new();
    for (i, g) in gs.iter().enumerate() {
        for p in PlayerId::ALL {
            for w in g.state.on_board(p) {
                let Some((gr, pos)) = g.state.loc(w).on_board() else {
                    continue;
                };
                let cs = choices_for_worker(&g.state, p, gr, pos);
                if !cs.is_empty() {
                    cands.push((i, p, w, cs));
                }
            }
        }
    }
    assert!(cands.len() >= 8);
    let qs: Vec<Query> = cands
        .iter()
        .map(|(i, p, w, cs)| Query {
            state: &gs[*i].state,
            phase: Phase::Take { worker: *w },
            turn: *p,
            n_edges: cs.len(),
            candidates: Some(cs),
            steps: None,
        })
        .collect();

    let mut batched = Vec::new();
    net.evaluate_batch(&qs, &mut batched);
    for (i, q) in qs.iter().enumerate() {
        let mut one = Vec::new();
        net.evaluate_batch(std::slice::from_ref(q), &mut one);
        for k in 0..q.n_edges {
            assert!(
                (one[0].priors[k] - batched[i].priors[k]).abs() < 1e-6,
                "row {i} edge {k}"
            );
        }
    }
}

/// A pre-built candidate list must give the same answer as re-deriving one, so
/// a search can hand over what it already has and skip ~35 µs.
#[test]
fn supplied_candidates_match_rederived() {
    let net = Net::random(Arch::MAIN, 3);
    let g = played(9, 7);
    let p = PlayerId(0);
    let Some(w) = g.state.on_board(p).next() else {
        return;
    };
    let (gr, pos) = g.state.loc(w).on_board().unwrap();
    let cs = choices_for_worker(&g.state, p, gr, pos);
    let ph = Phase::Take { worker: w };

    let mut a = Vec::new();
    net.evaluate_batch(&[Query::new(&g.state, ph, p, cs.len())], &mut a);
    let mut b = Vec::new();
    net.evaluate_batch(
        &[Query {
            state: &g.state,
            phase: ph,
            turn: p,
            n_edges: cs.len(),
            candidates: Some(&cs),
            steps: None,
        }],
        &mut b,
    );
    for k in 0..cs.len() {
        assert!((a[0].priors[k] - b[0].priors[k]).abs() < 1e-6);
    }
}

/// The pointer head must stay a distribution when stage 1 discards a tail.
#[test]
fn pointer_head_covers_long_candidate_lists() {
    let net = Net::random(Arch::MAIN, 4);
    let mut widest = 0usize;
    for seed in 0..12u64 {
        for rounds in [6usize, 10, 16] {
            let g = played(seed, rounds);
            if g.state.over {
                continue;
            }
            for p in PlayerId::ALL {
                let board: Vec<WorkerId> = g.state.on_board(p).collect();
                for w in board {
                    let Some((gr, pos)) = g.state.loc(w).on_board() else {
                        continue;
                    };
                    let cs = choices_for_worker(&g.state, p, gr, pos);
                    widest = widest.max(cs.len());
                    let e = net.evaluate_one(&g.state, Phase::Take { worker: w }, p, cs.len());
                    check_eval(&e, cs.len());
                    assert!(
                        e.priors.iter().all(|&x| x > 0.0),
                        "a discarded candidate got probability zero"
                    );
                }
            }
        }
    }
    // Stage 1 only starts discarding past 64, so the interesting path needs a
    // genuinely wide node to have been reached.
    assert!(widest > 64, "widest candidate list was only {widest}");
}

#[test]
fn gemm_backends_agree() {
    // Raw kernel first, on shapes with awkward tails and non-trivial strides.
    let mut rng = StdRng::seed_from_u64(5);
    for &(m, n, k) in &[(1, 7, 13), (3, 5, 64), (8, 512, 512), (17, 51, 96)] {
        let (lda, ldb, ldc) = (k + 3, k + 1, n + 2);
        let a: Vec<f32> = (0..m * lda).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..n * ldb).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut c1 = vec![0.5f32; m * ldc];
        let mut c2 = c1.clone();
        Portable.sgemm_nt(m, n, k, &a, lda, &b, ldb, 0.0, &mut c1, ldc);
        naive_ref(m, n, k, &a, lda, &b, ldb, &mut c2, ldc);
        for i in 0..m {
            for j in 0..n {
                let (x, y) = (c1[i * ldc + j], c2[i * ldc + j]);
                assert!((x - y).abs() < 1e-3, "{m}x{n}x{k} at {i},{j}: {x} vs {y}");
            }
        }
    }

    // Then the whole net, which is what actually has to match across machines.
    let g = played(1, 6);
    let a = Net::random(Arch::MAIN, 42);
    let b = Net::random(Arch::MAIN, 42).with_gemm(Box::new(Portable));
    for p in PlayerId::ALL {
        for ph in phases(&g.state, p) {
            let ea = a.evaluate_one(&g.state, ph, p, 4);
            let eb = b.evaluate_one(&g.state, ph, p, 4);
            for k in 0..N_PLAYERS {
                assert!(
                    (ea.value[k] - eb.value[k]).abs() < 2e-4,
                    "{} vs {}: {:?} / {:?}",
                    a.backend(),
                    b.backend(),
                    ea.value,
                    eb.value
                );
            }
            for k in 0..ea.priors.len() {
                assert!((ea.priors[k] - eb.priors[k]).abs() < 2e-4);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn naive_ref(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a[i * lda + t] * b[j * ldb + t];
            }
            c[i * ldc + j] = s;
        }
    }
}

#[test]
fn safetensors_round_trips() {
    let dir = std::env::temp_dir().join("tzolkin-net-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("gen_test.safetensors");

    let a = Net::random(Arch::SMALL, 11);
    a.save(&path).unwrap();
    let b = Net::load(&path).unwrap();
    assert_eq!(a.arch, b.arch);
    assert_eq!(a.n_params(), b.n_params());

    let g = played(8, 5);
    for p in PlayerId::ALL {
        let ea = a.evaluate_one(&g.state, Phase::Mode, p, 2);
        let eb = b.evaluate_one(&g.state, Phase::Mode, p, 2);
        assert_eq!(ea.value, eb.value);
        assert_eq!(ea.priors, eb.priors);
    }

    // Every tensor the manifest names is present, at the documented shape, and
    // nothing else is.
    let bundle = a.bundle();
    let manifest = Net::tensor_manifest(Arch::SMALL);
    assert_eq!(bundle.tensors.len(), manifest.len());
    for (name, shape) in manifest {
        let t = bundle.tensors.get(&name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t.shape, shape, "{name}");
        assert_eq!(t.data.len(), shape.iter().product::<usize>());
    }
    assert_eq!(bundle.meta.get("format").map(String::as_str), Some("tzolkin-net"));
    let _ = std::fs::remove_file(&path);
}

/// A checkpoint from a different board geometry must be refused, not loaded
/// with its first layer reading the wrong columns.
#[test]
fn stale_geometry_is_refused() {
    let mut b = Net::random(Arch::SMALL, 2).bundle();
    b.meta.insert("d_in".into(), "2048".into());
    let err = match Net::from_bundle(b, "stale") {
        Err(e) => e,
        Ok(_) => panic!("a checkpoint with the wrong D_IN loaded"),
    };
    assert!(err.to_string().contains("d_in"), "{err}");
}

/// The container's own version, as distinct from the architecture's.
///
/// A file from a later revision may put things where this build does not expect
/// them, and a checkpoint that loads wrong is worse than one that refuses.
#[test]
fn newer_format_version_is_refused() {
    let good = Net::random(Arch::SMALL, 2).bundle();
    assert!(Net::from_bundle(good.clone(), "v").is_ok());


    let mut b = good.clone();
    b.meta.insert(
        "format_version".into(),
        (tzolkin::net::FORMAT_VERSION + 1).to_string(),
    );
    let err = match Net::from_bundle(b, "v") {
        Err(e) => e,
        Ok(_) => panic!("a checkpoint from a newer container revision loaded"),
    };
    assert!(err.to_string().contains("format_version"), "{err}");

    let mut b = good;
    b.meta.insert("format".into(), "something-else".into());
    let err = match Net::from_bundle(b, "v") {
        Err(e) => e,
        Ok(_) => panic!("a checkpoint from another format loaded"),
    };
    assert!(err.to_string().contains("format"), "{err}");
}

/// The on-disk bytes must be what every other safetensors implementation
/// expects, because the file this writes is the one the training side reads and
/// the one it writes back.
#[test]
fn safetensors_bytes_are_conformant() {
    let bytes = Net::random(Arch::SMALL, 13).bundle().to_bytes();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;

    // `8 + N` is a multiple of 8: the reference writer pads the JSON with
    // spaces to guarantee it, and an mmap-based reader needs the data section
    // aligned. Nothing here reads the file that way; the Python side does.
    assert_eq!((8 + n) % 8, 0, "data section is not 8-byte aligned");

    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let obj = hdr.as_object().unwrap();
    assert!(obj.contains_key("__metadata__"));

    // Offsets are half-open, relative to `8 + N`, contiguous, gap-free and in
    // increasing order, and every tensor is F32.
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        assert_eq!(v["dtype"], "F32", "{k}");
        let o = v["data_offsets"].as_array().unwrap();
        let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        let elems: usize = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .product();
        assert_eq!(b - a, elems * 4, "{k}");
        spans.push((a, b, k.clone()));
    }
    spans.sort_unstable();
    let mut end = 0usize;
    for (a, b, k) in &spans {
        assert_eq!(*a, end, "gap or overlap before {k}");
        end = *b;
    }
    assert_eq!(8 + n + end, bytes.len(), "trailing bytes after the last tensor");

    // And it round-trips through our own reader.
    let back = tzolkin::net::Bundle::parse(&bytes).unwrap();
    assert_eq!(back.tensors.len(), spans.len());
}

#[test]
fn parameter_count_is_in_budget() {
    let main = Net::random(Arch::MAIN, 0);
    let small = Net::random(Arch::SMALL, 0);
    println!(
        "params: main {} ({}), small {}",
        main.n_params(),
        main.backend(),
        small.n_params()
    );
    // LEARNING.md §4.2 budgets ~4.22 M for `main` and ~1.9 M for `small`.
    assert!((3_900_000..4_700_000).contains(&main.n_params()));
    assert!((1_500_000..2_400_000).contains(&small.n_params()));
}

/// `edges_for` and `edges` must agree wherever `edges` is willing to answer.
///
/// They are two derivations of the same mapping: `edges_for` reads the `Step`s
/// the tree enumerated, `edges` rebuilds them from the engine and checks the
/// length. A disagreement means one of them is wrong, and if it is `edges` the
/// symptom is a policy silently applied to the wrong edges — no panic, no
/// wrong-length assertion, just a worse net. So this walks real `legal_steps`
/// output and diffs the two.
///
/// It also counts how often `edges` gives up. Every `Uniform` there is a node
/// whose policy head contributed nothing, which is invisible without this.
#[test]
fn steps_and_rederivation_agree() {
    use tzolkin::tree::legal_steps;

    let mut compared = 0usize;
    let mut abstained = 0usize;
    let mut pointer_nodes = 0usize;

    for seed in 0..10u64 {
        for rounds in [2usize, 6, 11] {
            let g = played(seed, rounds);
            if g.state.over {
                continue;
            }
            let p = g.state.current;
            for ph in phases(&g.state, p) {
                let steps = legal_steps(&g.state, ph, p, 0);
                if steps.is_empty() {
                    continue;
                }
                let mut fa = Vec::new();
                let mut fb = Vec::new();
                let a = edges_for(&g.state, p, ph, &steps, &mut fa);
                let b = edges(&g.state, p, ph, steps.len(), &mut fb);
                match (&a, &b) {
                    (EdgeSpec::Fixed { head: ha, idx: ia }, EdgeSpec::Fixed { head: hb, idx: ib }) => {
                        assert_eq!(ha, hb, "{ph:?}");
                        assert_eq!(ia, ib, "{ph:?}: head slots differ; one of the two is wrong");
                        compared += 1;
                    }
                    (EdgeSpec::Pointer { n: na, .. }, EdgeSpec::Pointer { n: nb, .. }) => {
                        assert_eq!(na, nb, "{ph:?}");
                        assert_eq!(fa, fb, "{ph:?}: candidate features differ");
                        compared += 1;
                        pointer_nodes += 1;
                    }
                    (_, EdgeSpec::Uniform) => abstained += 1,
                    (EdgeSpec::Uniform, _) => {
                        panic!("{ph:?}: edges_for abstained where edges did not")
                    }
                    _ => panic!("{ph:?}: the two derivations gave different kinds"),
                }
            }
        }
    }
    assert!(compared > 40, "only {compared} nodes compared");
    assert!(pointer_nodes > 5, "no pointer nodes in the comparison");
    // Not an assertion, a record: this is what carrying the edges buys.
    println!("edges() abstained on {abstained} of {} nodes", compared + abstained);
}

// ---------------------------------------------------------------------------
// The batching evaluator
// ---------------------------------------------------------------------------

/// Many threads blocking in the batch-1 `Evaluator::evaluate` must get exactly
/// what one thread calling `Unbatched` would get, and the batcher must actually
/// have batched.
///
/// The second half is the one that rots silently. `COMPUTE.md` §2.6: *"if the
/// mean batch size is not close to the configured maximum, none of §5's numbers
/// are happening, and it is the only symptom you will get."* A queue that works
/// and a queue that serves every request alone are indistinguishable from the
/// outside — same answers, twentieth of the throughput — so the histogram is
/// part of the contract, not diagnostics.
#[test]
fn batched_evaluator_matches_unbatched() {
    use std::sync::Arc;

    let net = Arc::new(Net::random(Arch::MAIN, 5));
    let gs: Vec<Game> = (0..8).map(|s| played(s + 40, 5)).collect();

    // A ragged work list: fixed-arity heads and pointer nodes together.
    let mut work: Vec<(usize, PlayerId, Phase, usize)> = Vec::new();
    for (i, g) in gs.iter().enumerate() {
        for p in PlayerId::ALL {
            work.push((i, p, Phase::Mode, 2));
            for w in g.state.on_board(p) {
                let Some((gr, pos)) = g.state.loc(w).on_board() else {
                    continue;
                };
                let n = choices_for_worker(&g.state, p, gr, pos).len();
                if n > 0 {
                    work.push((i, p, Phase::Take { worker: w }, n));
                }
            }
        }
    }
    assert!(work.len() >= 32);

    let solo = Unbatched(Arc::clone(&net));
    let want: Vec<Evaluation> = work
        .iter()
        .map(|&(i, p, ph, n)| solo.evaluate(&gs[i].state, ph, p, n))
        .collect();

    let threads = 8usize;
    let pool = BatchedEvaluator::new(
        Arc::clone(&net),
        BatchConfig {
            max_batch: 32,
            max_wait: std::time::Duration::from_millis(2),
            threads: 1,
        },
    );

    // Each thread walks the whole list, so every request has peers in flight.
    // A `wait_timeout`-based batcher that only ever saw one request would still
    // pass the equality half of this test.
    let got: Vec<Vec<Evaluation>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                let ev = pool.handle();
                let work = &work;
                let gs = &gs;
                sc.spawn(move || {
                    work.iter()
                        .map(|&(i, p, ph, n)| ev.evaluate(&gs[i].state, ph, p, n))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for (t, run) in got.iter().enumerate() {
        assert_eq!(run.len(), want.len());
        for (j, e) in run.iter().enumerate() {
            assert_eq!(e.priors.len(), want[j].priors.len(), "thread {t} row {j}");
            for k in 0..N_PLAYERS {
                assert!((e.value[k] - want[j].value[k]).abs() < 1e-6);
            }
            for k in 0..e.priors.len() {
                assert!(
                    (e.priors[k] - want[j].priors[k]).abs() < 1e-6,
                    "thread {t} row {j} edge {k}"
                );
            }
        }
    }

    let st = pool.stats();
    assert_eq!(st.evals, (threads * work.len()) as u64);
    assert!(
        st.mean_batch() > 1.5,
        "the batcher never batched: {}",
        st.report()
    );
    // Nothing may have taken the inline path while the pool was alive.
    assert_eq!(st.bypassed, 0);
    // And the `Net` itself must not have seen a single one-at-a-time call from
    // the pooled path -- only the `Unbatched` reference run above.
    assert_eq!(net.unbatched_calls(), want.len() as u64);
}

/// `Evaluator::evaluate_many` is now the method the trait wants implemented, so
/// it has to be right for every way into the network.
///
/// Three things are checked. That the trait's batched entry point agrees with
/// the single-node one wherever the single-node one is able to answer at all
/// (it is not, at the ~15% of nodes where the re-derivation abstains — those
/// get a uniform prior and are skipped here). That a `BatchHandle` gives the
/// same answers as the bare `Net`. And that `evaluate_many` through the handle
/// actually produced *one* batch rather than N.
#[test]
fn evaluate_many_carries_the_edges() {
    use std::sync::Arc;
    use tzolkin::phase::Query as PQuery;
    use tzolkin::tree::legal_steps;

    let net = Arc::new(Net::random(Arch::MAIN, 17));
    let gs: Vec<Game> = (0..6).map(|s| played(s + 60, 7)).collect();

    let mut edge_sets: Vec<(usize, PlayerId, Phase, Vec<tzolkin::phase::Step>)> = Vec::new();
    for (i, g) in gs.iter().enumerate() {
        let p = g.state.current;
        for ph in phases(&g.state, p) {
            let st = legal_steps(&g.state, ph, p, 0);
            if st.len() > 1 {
                edge_sets.push((i, p, ph, st));
            }
        }
    }
    assert!(edge_sets.len() >= 12, "{} edge sets", edge_sets.len());

    let qs: Vec<PQuery> = edge_sets
        .iter()
        .map(|(i, p, ph, st)| PQuery {
            state: &gs[*i].state,
            phase: *ph,
            turn: *p,
            edges: st,
        })
        .collect();

    let many = net.evaluate_many(&qs);
    assert_eq!(many.len(), qs.len());
    for (e, (_, _, _, st)) in many.iter().zip(&edge_sets) {
        check_eval(e, st.len());
    }

    // Where the count-only path can reconstruct the same edge list, the two
    // must agree; where it abstains it hands back a uniform prior, which is
    // exactly the thing carrying the edges fixes.
    let mut agreed = 0usize;
    for (e, (i, p, ph, st)) in many.iter().zip(&edge_sets) {
        let mut probe = Vec::new();
        if matches!(
            edges(&gs[*i].state, *p, *ph, st.len(), &mut probe),
            EdgeSpec::Uniform
        ) {
            continue;
        }
        let one = net.evaluate_one(&gs[*i].state, *ph, *p, st.len());
        for k in 0..st.len() {
            assert!(
                (one.priors[k] - e.priors[k]).abs() < 1e-5,
                "{ph:?} edge {k}: {} many vs {} single",
                e.priors[k],
                one.priors[k]
            );
        }
        agreed += 1;
    }
    assert!(agreed > 8, "only {agreed} nodes were comparable");

    // And through the queue, which clones the edge list into the request.
    let pool = BatchedEvaluator::new(
        Arc::clone(&net),
        BatchConfig {
            max_batch: 256,
            max_wait: std::time::Duration::from_millis(1),
            threads: 1,
        },
    );
    let via = pool.handle().evaluate_many(&qs);
    assert_eq!(via.len(), many.len());
    for (a, b) in via.iter().zip(&many) {
        assert_eq!(a.priors.len(), b.priors.len());
        for k in 0..a.priors.len() {
            assert!((a.priors[k] - b.priors[k]).abs() < 1e-6);
        }
        for k in 0..N_PLAYERS {
            assert!((a.value[k] - b.value[k]).abs() < 1e-6);
        }
    }
    let st = pool.stats();
    assert_eq!(st.evals, qs.len() as u64);
    assert_eq!(
        st.batches, 1,
        "evaluate_many serialised into {} batches: {}",
        st.batches,
        st.report()
    );
}

/// A handle that outlives its pool must not park forever.
#[test]
fn handle_outliving_the_pool_still_answers() {
    use std::sync::Arc;
    let net = Arc::new(Net::random(Arch::SMALL, 6));
    let g = played(4, 3);
    let ev = {
        let pool = BatchedEvaluator::new(Arc::clone(&net), BatchConfig::default());
        pool.handle()
    };
    let e = ev.evaluate(&g.state, Phase::Mode, PlayerId(0), 2);
    check_eval(&e, 2);
    assert_eq!(ev.stats().bypassed, 1);
}

/// Print the exact tensor list a checkpoint must contain.
///
/// `cargo test --release --test encode -- --ignored --nocapture manifest`.
///
/// This is the seam with the training side, and the seam is a list of names and
/// shapes rather than prose. Run it, paste the output into whatever writes the
/// `state_dict`, and the file loads. Every weight is `[out, in]` — `nn.Linear`
/// exactly — so nothing is transposed on either side.
#[test]
#[ignore]
fn manifest() {
    for (label, arch) in [("MAIN", Arch::MAIN), ("SMALL", Arch::SMALL)] {
        let m = Net::tensor_manifest(arch);
        let total: usize = m.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
        println!("# {label}: {} tensors, {total} parameters", m.len());
        for (n, sh) in &m {
            println!("{n:32} {sh:?}");
        }
        println!();
    }
    println!(
        "input geometry: D_IN {D_IN} = board {BOARD_W} + players {PLAYERS_W} \
         + global {GLOBAL_W} + bslots {BSLOTS_W} + mslots {MSLOTS_W} + reserved {RESERVED_W}"
    );
    println!("D_CHOICE {D_CHOICE}, N_WHO {N_WHO}, N_PLACE {N_PLACE}, N_CELLS {N_CELLS}");
}

// ---------------------------------------------------------------------------
// Throughput
// ---------------------------------------------------------------------------

/// Not a correctness test. `cargo test --release --test encode -- --ignored
/// --nocapture throughput` to measure. Set `VECLIB_MAXIMUM_THREADS=1` first or
/// Accelerate spawns its own pool and the numbers mean nothing.
#[test]
#[ignore]
fn throughput() {
    use std::sync::Arc;

    let net = Net::random(Arch::MAIN, 0);
    let g = played(1, 10);
    println!("backend: {}, params: {}", net.backend(), net.n_params());

    // Best of three: this machine is not quiet, and the interesting number is
    // the throughput the search can actually get, not the median under load.
    fn sweep<'a>(net: &Net, label: &str, mk: &dyn Fn(usize) -> Vec<Query<'a>>) {
        println!("-- {label}");
        for &b in &[1usize, 8, 32, 64, 128, 256, 512] {
            let qs = mk(b);
            let mut out = Vec::new();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let iters = (4_000 / b).max(6);
                for _ in 0..3 {
                    net.evaluate_batch(&qs, &mut out);
                }
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    net.evaluate_batch(&qs, &mut out);
                }
                best = best.min(t.elapsed().as_secs_f64() / iters as f64);
            }
            println!(
                "batch {b:4}: {:9.0} evals/s   {:8.1} us/call   {:6.2} us/eval",
                b as f64 / best,
                best * 1e6,
                best / b as f64 * 1e6
            );
        }
    }

    // The trunk alone: `Mode` is two edges off a fixed-arity head.
    sweep(&net, "trunk only (Phase::Mode, 2 edges)", &|b| {
        (0..b)
            .map(|i| Query::new(&g.state, Phase::Mode, PlayerId((i % 4) as u8), 2))
            .collect()
    });

    // Trunk plus the two-stage pointer head, which is the realistic mix: a turn
    // resolves one `Take` per worker placed or retrieved. The per-node form of
    // the key MLP issues two GEMM calls per row here; the batched form issues
    // two per batch, and the gap between this sweep and the one above is what
    // that costs.
    let mut takes: Vec<(PlayerId, WorkerId, usize)> = PlayerId::ALL
        .iter()
        .flat_map(|&p| {
            g.state
                .on_board(p)
                .filter_map(move |w| {
                    let (gr, pos) = g.state.loc(w).on_board()?;
                    let n = choices_for_worker(&g.state, p, gr, pos).len();
                    (n > 1).then_some((p, w, n))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // Widest first: a batch of width-2 nodes measures the trunk, not the head.
    takes.sort_unstable_by_key(|t| std::cmp::Reverse(t.2));
    takes.truncate(8);
    if takes.is_empty() {
        println!("-- no Take nodes in this position; pointer sweep skipped");
    } else {
        let widths: Vec<usize> = takes.iter().map(|t| t.2).collect();
        println!(
            "-- trunk + pointer head ({} distinct Take nodes, widths {:?})",
            takes.len(),
            widths
        );
        sweep(&net, "trunk + pointer (Phase::Take)", &|b| {
            (0..b)
                .map(|i| {
                    let (p, w, n) = takes[i % takes.len()];
                    Query::new(&g.state, Phase::Take { worker: w }, p, n)
                })
                .collect()
        });
    }

    // End to end through the trait: N game threads parked in the batch-1
    // `Evaluator::evaluate`, one batcher thread. This is the number that
    // matters, because it is the one self-play gets.
    let shared = Arc::new(Net::random(Arch::MAIN, 0));
    for &(games, batch, batchers) in &[
        (1usize, 1usize, 1usize),
        (16, 16, 1),
        (64, 64, 1),
        (128, 128, 1),
        (128, 128, 2),
        (256, 128, 2),
    ] {
        let pool = BatchedEvaluator::new(
            Arc::clone(&shared),
            BatchConfig {
                max_batch: batch,
                max_wait: std::time::Duration::from_micros(200),
                threads: batchers,
            },
        );
        let per = 3_000 / games.max(1) + 20;
        let t = std::time::Instant::now();
        std::thread::scope(|sc| {
            for gi in 0..games {
                let ev = pool.handle();
                let st = &g.state;
                sc.spawn(move || {
                    for _ in 0..per {
                        let _ = ev.evaluate(st, Phase::Mode, PlayerId((gi % 4) as u8), 2);
                    }
                });
            }
        });
        let secs = t.elapsed().as_secs_f64();
        let s = pool.stats();
        println!(
            "games {games:4} batch<={batch:4} batchers {batchers}: {:9.0} evals/s  mean batch {:6.1}  wait {:5.0} us",
            s.evals as f64 / secs,
            s.mean_batch(),
            s.mean_wait_us()
        );
    }

    // The encoder is on the same critical path, so measure it separately.
    let t = std::time::Instant::now();
    let mut buf = vec![0.0; D_IN];
    for i in 0..20_000 {
        encode(&g.state, PlayerId((i % 4) as u8), Phase::Mode, &mut buf);
    }
    println!(
        "encode: {:.2} us each",
        t.elapsed().as_secs_f64() / 20_000.0 * 1e6
    );

    let p = PlayerId(0);
    let first = g.state.on_board(p).next();
    if let Some(w) = first {
        let (gr, pos) = g.state.loc(w).on_board().unwrap();
        let cs = choices_for_worker(&g.state, p, gr, pos);
        let mut f = vec![0.0; D_CHOICE];
        let t = std::time::Instant::now();
        for _ in 0..20_000 {
            for c in &cs {
                encode_choice(&g.state, p, c, &mut f);
            }
        }
        println!(
            "encode_choice: {:.0} ns each ({} candidates)",
            t.elapsed().as_secs_f64() / (20_000 * cs.len()) as f64 * 1e9,
            cs.len()
        );
    }
}
