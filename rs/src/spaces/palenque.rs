//! Palenque: corn and wood from the jungle.

use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::state::GameState;

/// (corn yield, wood yield) for the three jungle spaces.
const JUNGLE: [(i8, i8); 3] = [(5, 2), (7, 3), (9, 4)];

pub fn at(g: &GameState, p: PlayerId, pos: Pos) -> Vec<Choice> {
    match pos.0 {
        0 => super::entry_space(),
        // A plain corn space: no tile involved, so nothing can exhaust it.
        1 => vec![Choice::one(Effect::Corn(
            (3 + g.corn_bonus(p, Color::Blue)) as i16,
        ))],
        2 => {
            let corn = (4 + g.corn_bonus(p, Color::Green)) as i16;
            let stack = g.palenque[2];
            if stack.corn > 0 {
                vec![Choice::of([
                    Effect::Corn(corn),
                    Effect::TakePalenqueTile(Pos(2), TileKind::Corn),
                ])]
            } else if g.irrigation(p) {
                vec![Choice::one(Effect::Corn(corn))]
            } else {
                vec![Choice::skip()]
            }
        }
        3..=5 => jungle(g, p, pos),
        _ => super::mirror(g, p, Gear::Palenque, 6),
    }
}

fn jungle(g: &GameState, p: PlayerId, pos: Pos) -> Vec<Choice> {
    let (corn_yield, wood_yield) = JUNGLE[pos.idx() - 3];
    let corn = (corn_yield + g.corn_bonus(p, Color::Green)) as i16;
    let wood = wood_yield + g.resource_bonus(p, Resource::Wood);
    let stack = g.palenque[pos.idx()];
    let mut out = Vec::new();

    if stack.wood > 0 {
        out.push(Choice::of([
            Effect::Res(Resource::Wood, wood),
            Effect::TakePalenqueTile(pos, TileKind::Wood),
        ]));

        // Dig the corn out from under a wood tile and anger the gods for it.
        // This consumes one of each, so it needs both to be present. Go guarded
        // only on wood, which drove the corn count negative -- and once negative
        // it inverted `corn_showing`, since that compares the two counts.
        //
        // The burned wood goes back to the box, so it is *not* a
        // `TakePalenqueTile`: burned wood does not count for the wood-tile
        // monument.
        if stack.corn > 0 {
            for t in Temple::ALL {
                if g.can_temple_step(p, t, -1) {
                    out.push(Choice::of([
                        Effect::Corn(corn),
                        Effect::TakePalenqueTile(pos, TileKind::Corn),
                        Effect::BurnPalenqueWood(pos),
                        Effect::TempleStep(t, -1),
                    ]));
                }
            }
        }
    }

    if stack.corn_showing() {
        out.push(Choice::of([
            Effect::Corn(corn),
            Effect::TakePalenqueTile(pos, TileKind::Corn),
        ]));
    }

    // Irrigation harvests corn with no tile at all -- but only when there is no
    // corn tile to be had. With one showing you take the tile, which is worth
    // four points on a monument.
    if g.irrigation(p) && !stack.corn_showing() {
        out.push(Choice::one(Effect::Corn(corn)));
    }

    if out.is_empty() {
        out.push(Choice::skip());
    }
    out
}
