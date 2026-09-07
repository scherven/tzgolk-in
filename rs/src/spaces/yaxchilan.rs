//! Yaxchilan: raw resources.

use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::state::GameState;

pub fn at(g: &GameState, p: PlayerId, pos: Pos) -> Vec<Choice> {
    match pos.0 {
        0 => super::entry_space(),
        1 => vec![Choice::one(Effect::Res(
            Resource::Wood,
            1 + g.resource_bonus(p, Resource::Wood),
        ))],
        2 => vec![Choice::of([
            Effect::Res(Resource::Stone, 1 + g.resource_bonus(p, Resource::Stone)),
            Effect::Corn(1),
        ])],
        3 => vec![Choice::of([
            Effect::Res(Resource::Gold, 1 + g.resource_bonus(p, Resource::Gold)),
            Effect::Corn(2),
        ])],
        // The extraction track never boosts skulls, but theology level 3 grants
        // an extra one whenever you take a skull from *this* space. With an
        // empty bank the action has no effect at all.
        4 => {
            if g.skulls_remaining == 0 {
                vec![Choice::skip()]
            } else {
                let n = 1 + i8::from(g.has_level(p, Science::Theology, 3));
                vec![Choice::one(Effect::Res(Resource::Skull, n))]
            }
        }
        5 => vec![Choice::of([
            Effect::Res(Resource::Gold, 1 + g.resource_bonus(p, Resource::Gold)),
            Effect::Res(Resource::Stone, 1 + g.resource_bonus(p, Resource::Stone)),
            Effect::Corn(2),
        ])],
        _ => super::mirror(g, p, Gear::Yaxchilan, 6),
    }
}
