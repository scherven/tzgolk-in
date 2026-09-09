//! Chichen Itza: spend skulls for temples and points.

use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::state::GameState;

struct Spot {
    temple: Temple,
    points: i8,
    /// Whether the space also grants a block of the player's choice.
    block: bool,
}

const SPOTS: [Spot; 9] = [
    Spot { temple: Temple::Brown,  points: 4,  block: false },
    Spot { temple: Temple::Brown,  points: 5,  block: false },
    Spot { temple: Temple::Brown,  points: 6,  block: false },
    Spot { temple: Temple::Green,  points: 7,  block: false },
    Spot { temple: Temple::Green,  points: 8,  block: false },
    Spot { temple: Temple::Green,  points: 8,  block: true  },
    Spot { temple: Temple::Yellow, points: 10, block: false },
    Spot { temple: Temple::Yellow, points: 11, block: true  },
    Spot { temple: Temple::Yellow, points: 13, block: true  },
];

pub fn at(g: &GameState, p: PlayerId, pos: Pos) -> Vec<Choice> {
    match pos.0 {
        // The entry space has no action of its own, but foresight still lets a
        // player look one space up from it.
        0 => {
            let mut out = super::entry_space();
            if g.foresight(p) {
                out.extend(spot(g, p, Pos(1), false));
            }
            out
        }
        1..=9 => {
            let v = spot(g, p, pos, true);
            if v.is_empty() {
                vec![Choice::skip()]
            } else {
                v
            }
        }
        // Space 10 is Chichen's free-choice space: any action on this gear.
        _ => {
            let mut out = Vec::new();
            for i in 0..=9 {
                out.extend(at(g, p, Pos(i)));
            }
            out
        }
    }
}

fn spot(g: &GameState, p: PlayerId, pos: Pos, can_foresee: bool) -> Vec<Choice> {
    let mut out = Vec::new();

    // Foresight lets a blocked player use the next space up. Go accumulated
    // these and then hit `return Skip()` when the space was full or the player
    // was out of skulls -- discarding them in exactly the case they exist for.
    if can_foresee && g.foresight(p) {
        if pos.0 < 9 {
            out.extend(spot(g, p, Pos(pos.0 + 1), false));
        } else {
            for i in 1..=9 {
                out.extend(spot(g, p, Pos(i), false));
            }
        }
    }

    if g.chichen_is_full(pos) || g.players[p.idx()].get(Resource::Skull) == 0 {
        return out;
    }

    // A temple step that cannot be taken is *wasted*, not refused: the skull
    // still goes down and the points and block are still collected. Gating on
    // it made the 13-point space unusable to anyone atop the yellow temple.
    let s = &SPOTS[pos.idx() - 1];

    let mut cores: Vec<Choice> = Vec::new();
    let base = [
        Effect::Res(Resource::Skull, -1),
        Effect::TempleStep(s.temple, 1),
        Effect::Points(s.points),
        Effect::FillChichen(pos),
    ];
    if s.block {
        for b in Resource::BLOCKS {
            cores.push(Choice::of(base).with(Effect::Res(b, 1)));
        }
    } else {
        cores.push(Choice::of(base));
    }

    // A devout player *may* pay a block for one more step. Go made this
    // mandatory once the track was reached, which removed the plain placement.
    //
    // The block is checked against the position the *core* reaches, not the one
    // the turn started in, because the rule says so in as many words: "If you
    // gained a resource block from your Chichen Itza action, it is available
    // for you to spend in this way." Spaces 6, 8 and 9 hand out a block of the
    // player's choice, so reading the pre-action stock denied the step to
    // exactly the player the sentence was written for -- one who arrives with
    // none. The temple test already probed the post-core state, since the
    // core's own step can be the one that reaches the top.
    out.extend(cores.iter().cloned());
    if g.devout(p) {
        for core in &cores {
            let mut probe = *g;
            core.apply(&mut probe, p);
            for b in Resource::BLOCKS {
                if probe.players[p.idx()].get(b) == 0 {
                    continue;
                }
                for t in Temple::ALL {
                    if !probe.can_temple_step(p, t, 1) {
                        continue;
                    }
                    out.push(core.clone().with(Effect::Res(b, -1)).with(Effect::TempleStep(t, 1)));
                }
            }
        }
    }

    out
}
