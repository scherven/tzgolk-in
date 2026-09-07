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
    let v = match gear {
        Gear::Palenque => palenque::at(g, p, pos),
        Gear::Yaxchilan => yaxchilan::at(g, p, pos),
        Gear::Tikal => tikal::at(g, p, pos),
        Gear::Uxmal => uxmal::at(g, p, pos),
        Gear::Chichen => chichen::at(g, p, pos),
    };
    crate::options::dedup(v)
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
