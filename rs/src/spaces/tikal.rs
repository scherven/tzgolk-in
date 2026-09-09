//! Tikal: research, construction and temples.

use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::options::{building_choices, monument_choices, research_choices};
use crate::state::GameState;

pub fn at(g: &GameState, p: PlayerId, pos: Pos) -> Vec<Choice> {
    at_d(g, p, pos, super::MAX_DEPTH)
}

/// `depth` bounds how far build-another and mirror chains may nest.
pub fn at_d(g: &GameState, p: PlayerId, pos: Pos, depth: u8) -> Vec<Choice> {
    match pos.0 {
        0 => super::entry_space(),
        1 => or_skip(research_choices(g, p, 1, false)),
        2 => or_skip(building_choices(g, p, None, true, depth)),
        // "Advance 1 or 2 technology levels" -- Go required exactly two, so a
        // player who could afford only one advance got nothing.
        3 => {
            let mut v = research_choices(g, p, 1, false);
            v.extend(research_choices(g, p, 2, false));
            or_skip(crate::options::dedup_unordered(v))
        }
        4 => or_skip(build_two(g, p, depth)),
        5 => or_skip(two_temple_steps(g, p)),
        _ => {
            let mut out = Vec::new();
            for i in 0..6 {
                out.extend(at_d(g, p, Pos(i), depth));
            }
            out
        }
    }
}

/// Build one building, or two, or take a monument.
///
/// The second build is generated against a probe state with the first already
/// applied. In Go this meant a full `Game.Clone()` per building option -- 609k
/// of them in a two-minute run, which is what the profile was showing. Here the
/// probe is a memcpy of a `Copy` struct.
fn build_two(g: &GameState, p: PlayerId, depth: u8) -> Vec<Choice> {
    let mut out = Vec::new();
    let mut pairs = Vec::new();

    for first in building_choices(g, p, None, true, depth) {
        out.push(first.clone());

        let built = first.0.iter().find_map(|e| match e {
            Effect::Build(id) => Some(*id),
            _ => None,
        });

        let mut probe = *g;
        first.apply(&mut probe, p);

        // The second building gets no architecture bonus.
        for second in building_choices(&probe, p, built, false, 0) {
            pairs.push(first.clone().chain(&second));
        }
    }

    // "If the first building gives you a new architecture technology, you may
    // apply that effect (and any others) to the second building, as long as you
    // applied no architecture effects to the first one."
    //
    // Enumerating both orders covers the *other* half of that rule -- picking
    // which card spends a level the player already holds -- because the two
    // orders differ only in who gets the discount. It cannot cover this half:
    // building #6 is one gold for a free architecture advance, and the level it
    // hands over is not in `g`, only in the probe. So the plain-first pairing is
    // generated a second time here, with the bonus on the second card and read
    // off the state the first card left behind.
    //
    // Guarded on the level actually moving, so this adds nothing to the width of
    // an ordinary double build: without the guard every pair would appear a
    // third time, spelled the way the loop above already spells its mirror.
    let arch = g.research[p.idx()][Science::Architecture.idx()];
    for first in building_choices(g, p, None, false, depth) {
        let mut probe = *g;
        first.apply(&mut probe, p);
        if probe.research[p.idx()][Science::Architecture.idx()] <= arch {
            continue;
        }
        let built = first.0.iter().find_map(|e| match e {
            Effect::Build(id) => Some(*id),
            _ => None,
        });
        for second in building_choices(&probe, p, built, true, 0) {
            pairs.push(first.clone().chain(&second));
        }
    }

    // Both orders of every pair are enumerated above, and the two orders reach
    // the same position whenever the architecture discount is not in play --
    // which is most of the game. See `options::dedup_by_position` for why the
    // states are compared rather than the effect lists. Only the pairs are
    // walked: a single build already comes deduplicated out of
    // `building_choices`, and a pair can never collide with a single or with a
    // monument, because they disagree on the cards-owned bitsets.
    // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
    if crate::options::wide_pruning_on() {
        pairs = crate::options::dedup_by_position(g, p, pairs);
    }
    out.append(&mut pairs);
    out.extend(monument_choices(g, p));
    out
}

/// Pay one block, then advance one step on each of two *different* temples.
///
/// Go enumerated ordered pairs, so every split appeared twice; this takes each
/// unordered pair once. Both steps may not go on the same temple -- an earlier
/// pass here allowed that, which is a rule violation, not a missing option.
fn two_temple_steps(g: &GameState, p: PlayerId) -> Vec<Choice> {
    let mut out = Vec::new();
    for block in Resource::BLOCKS {
        if g.players[p.idx()].get(block) == 0 {
            continue;
        }
        for (i, &a) in Temple::ALL.iter().enumerate() {
            for &b in &Temple::ALL[i + 1..] {
                if !g.can_temple_step(p, a, 1) {
                    continue;
                }
                let mut probe = *g;
                probe.temple_step(p, a, 1);
                if !probe.can_temple_step(p, b, 1) {
                    continue;
                }
                out.push(Choice::of([
                    Effect::Res(block, -1),
                    Effect::TempleStep(a, 1),
                    Effect::TempleStep(b, 1),
                ]));
            }
        }
    }
    out
}

fn or_skip(v: Vec<Choice>) -> Vec<Choice> {
    if v.is_empty() {
        vec![Choice::skip()]
    } else {
        v
    }
}
