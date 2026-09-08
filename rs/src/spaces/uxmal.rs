//! Uxmal: trade, workers, and the mirror.

use crate::data::buildings::def as bdef;
use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::options::{corn_exchange, dedup};
use crate::state::GameState;

pub fn at(g: &GameState, p: PlayerId, pos: Pos) -> Vec<Choice> {
    at_d(g, p, pos, super::MAX_DEPTH)
}

pub fn at_d(g: &GameState, p: PlayerId, pos: Pos, depth: u8) -> Vec<Choice> {
    match pos.0 {
        0 => super::entry_space(),
        1 => or_skip(pay_for_temple(g, p)),
        2 => or_skip(corn_exchange(g, p)),
        3 => vec![Choice::one(Effect::UnlockWorker)],
        4 => or_skip(build_with_corn(g, p)),
        5 => or_skip(mirror_choices(g, p, depth)),
        _ => {
            let mut out = Vec::new();
            for i in 0..6 {
                out.extend(at_d(g, p, Pos(i), depth));
            }
            out
        }
    }
}

/// Three corn for one temple step.
fn pay_for_temple(g: &GameState, p: PlayerId) -> Vec<Choice> {
    if g.players[p.idx()].corn < 3 {
        return Vec::new();
    }
    Temple::ALL
        .iter()
        .filter(|&&t| g.can_temple_step(p, t, 1))
        .map(|&t| Choice::of([Effect::Corn(-3), Effect::TempleStep(t, 1)]))
        .collect()
}

/// Buy a building outright with corn: two corn per block of its cost, less two
/// for a maxed architecture track.
fn build_with_corn(g: &GameState, p: PlayerId) -> Vec<Choice> {
    let corn = g.players[p.idx()].corn as i32;
    let mut out = Vec::new();

    for id in g.face_up_buildings() {
        let d = bdef(id);
        let blocks: i32 = Resource::BLOCKS.iter().map(|&r| d.cost[r.idx()] as i32).sum();
        let mut price = blocks * 2;
        if g.builder(p) {
            price -= 2;
        }
        let price = price.max(0);
        if price > corn {
            continue;
        }

        let mut base = Choice::of([Effect::Corn(-(price as i16)), Effect::Build(id)]);
        base.0.extend_from_slice(&g.build_bonus(p));

        let mut probe = *g;
        base.apply(&mut probe, p);
        for tail in crate::options::expand_payoff(&probe, p, id, &d.payoff, 0) {
            out.push(base.clone().chain(&tail));
        }
    }
    dedup(out)
}

/// The mirror: pay one corn to perform one action on the Palenque, Yaxchilan,
/// Tikal or Uxmal gear.
///
/// Go omitted Palenque, which is where the corn is. This copies the *space
/// definitions* rather than the board, so it offers the strongest version of
/// each action regardless of what is showing.
pub fn mirror_choices(g: &GameState, p: PlayerId, depth: u8) -> Vec<Choice> {
    if depth == 0 || g.players[p.idx()].corn == 0 {
        return Vec::new();
    }
    // Everything mirrored is generated against a state with the one-corn fee
    // already paid, so a mirrored action cannot spend corn the fee consumed.
    let mut probe = *g;
    probe.players[p.idx()].corn -= 1;

    let mut all = Vec::new();
    for i in 0..6 {
        all.extend(super::palenque::at(&probe, p, Pos(i)));
        all.extend(super::yaxchilan::at(&probe, p, Pos(i)));
        all.extend(super::tikal::at_d(&probe, p, Pos(i), depth - 1));
    }
    for i in 1..5 {
        all.extend(at_d(&probe, p, Pos(i), depth - 1));
    }

    // Prefixed in place rather than rebuilt: the fee has to lead so that
    // `Choice::affordable` sees it before the action spends, and this list is
    // the widest one generation builds -- the whole of Palenque, Yaxchilan and
    // Tikal -- so a second copy of every choice is a malloc and a free each.
    all.retain(|c| !c.is_skip());
    for c in all.iter_mut() {
        c.0.insert(0, Effect::Corn(-1));
    }
    all
}

fn or_skip(v: Vec<Choice>) -> Vec<Choice> {
    if v.is_empty() {
        vec![Choice::skip()]
    } else {
        v
    }
}
