//! Board spaces as generators.
//!
//! A space is `fn(&GameState, PlayerId) -> Vec<Choice>` -- it answers "what can
//! this player do here, right now?" against the live state. That is the shape
//! the game needs, because almost every space's legal choices depend on research
//! tier, temple position and tile supply.
//!
//! What it *returns* is data. Values like `3 + corn_bonus(...)` are resolved
//! here, at generation time, and stored in the effect.

pub mod chichen;
pub mod palenque;
pub mod tikal;
pub mod uxmal;
pub mod yaxchilan;

use crate::effect::Choice;
use crate::ids::*;
use crate::state::GameState;

/// How deep a build-another / mirror chain may recurse.
pub const MAX_DEPTH: u8 = 1;

pub fn choices_at(g: &GameState, p: PlayerId, gear: Gear, pos: Pos) -> Vec<Choice> {
    let v = raw_at(g, p, gear, pos);
    // `dominated_dedup` rather than `dedup` because a free-choice space stacks
    // several routes to one action on top of each other: Uxmal 6 offers the
    // unlock for nothing (space 3) and again for a corn (space 5's mirror), and
    // the same doubling runs through every action the mirror can reach.
    crate::options::dominated_dedup(v)
}

/// The space's list before the dominance pass.
///
/// `choices_for_worker` stacks one of these per step-down fee and runs
/// `dominated_dedup` over the concatenation, and dominance is transitive: a
/// choice beaten inside one space's list is beaten inside the union, and the
/// lexicographic tie-break between two spellings of one position picks the same
/// survivor either way. So the inner pass cannot change the answer there, only
/// the size of the list the outer pass sees -- which makes it a throughput
/// question, measured in `docs/FINDINGS-generation.md`, not a rules one.
pub fn raw_at(g: &GameState, p: PlayerId, gear: Gear, pos: Pos) -> Vec<Choice> {
    match gear {
        Gear::Palenque => palenque::at(g, p, pos),
        Gear::Yaxchilan => yaxchilan::at(g, p, pos),
        Gear::Tikal => tikal::at(g, p, pos),
        Gear::Uxmal => uxmal::at(g, p, pos),
        Gear::Chichen => chichen::at(g, p, pos),
    }
}

/// The spaces that repeat every action below them for free: 6 and 7 on the
/// small gears, 10 on Chichen Itza.
///
/// This mirrors the catch-all arm of each space module and has to move with it.
/// `choices_for_worker` reads it to skip the pay-to-step-down walk, which from
/// one of these spaces can only re-buy what the space already gives away.
pub fn is_free_choice(gear: Gear, pos: Pos) -> bool {
    match gear {
        Gear::Chichen => pos.0 >= 10,
        _ => pos.0 >= 6,
    }
}

/// The two spaces at the top of each small gear repeat every action below them.
pub fn mirror(g: &GameState, p: PlayerId, gear: Gear, upto: u8) -> Vec<Choice> {
    let mut out = Vec::new();
    for i in 0..upto {
        out.extend(match gear {
            Gear::Palenque => palenque::at(g, p, Pos(i)),
            Gear::Yaxchilan => yaxchilan::at(g, p, Pos(i)),
            Gear::Tikal => tikal::at(g, p, Pos(i)),
            Gear::Uxmal => uxmal::at(g, p, Pos(i)),
            Gear::Chichen => chichen::at(g, p, Pos(i)),
        });
    }
    out
}

/// The entry space of every gear has no action of its own.
///
/// Go returned an empty option list here, and since retrieval enumerates over
/// options, a worker on the entry space could never be picked up at all -- it
/// had to ride to the top of the gear. An explicit skip keeps the retrieval
/// legal and does nothing, which is what the rules describe.
pub fn entry_space() -> Vec<Choice> {
    vec![Choice::skip()]
}
