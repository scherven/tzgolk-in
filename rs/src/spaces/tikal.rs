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
            or_skip(crate::options::dedup(v))
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
            out.push(first.clone().chain(&second));
        }
    }

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
