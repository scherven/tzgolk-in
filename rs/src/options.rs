//! Shared option builders: research, buildings, monuments.
//!
//! Everything here runs at *generation* time against an immutable `&GameState`,
//! and everything it returns is fully-resolved data.

use crate::data::buildings::{def as bdef, Payoff};
use crate::data::monuments::{def as mdef, MONUMENTS};
use crate::effect::{Choice, Effect, Effects};
use crate::ids::*;
use crate::state::GameState;

/// Every distinct way to pay `n` blocks out of `res`.
///
/// Combinations, not permutations: the Go version recursed over all three block
/// types at every level, so paying wood-then-stone and stone-then-wood were two
/// separate options all the way up the tree.
pub fn pay_blocks(res: [u8; 4], n: u8) -> Vec<Bundle> {
    fn go(res: &[u8; 4], idx: usize, left: u8, cur: Bundle, out: &mut Vec<Bundle>) {
        if left == 0 {
            out.push(cur);
            return;
        }
        if idx >= 3 {
            return;
        }
        for take in 0..=left.min(res[idx]) {
            let mut c = cur;
            c[idx] = take as i8;
            go(res, idx + 1, left - take, c, out);
        }
    }
    let mut out = Vec::new();
    go(&res, 0, n, EMPTY, &mut out);
    out
}

fn payment_effects(pay: Bundle, into: &mut Effects) {
    for r in Resource::BLOCKS {
        if pay[r.idx()] > 0 {
            into.push(Effect::Res(r, -pay[r.idx()]));
        }
    }
}

/// The one-off payoff for advancing a track that is already at level 3.
fn top_payoffs(g: &GameState, p: PlayerId, s: Science) -> Vec<Effects> {
    let mut out: Vec<Effects> = Vec::new();
    match s {
        Science::Agriculture => {
            for t in Temple::ALL {
                if g.can_temple_step(p, t, 1) {
                    out.push([Effect::TempleStep(t, 1)].into_iter().collect());
                }
            }
            // A privilege you cannot use is wasted, not withheld -- the payoff
            // must still be reachable so the advance itself can be taken.
            if out.is_empty() {
                out.push([Effect::TempleStep(Temple::Brown, 1)].into_iter().collect());
            }
        }
        Science::Extraction => {
            // Two blocks, same or different. Go enumerated ordered pairs, so
            // every mixed pair appeared twice.
            for (i, &a) in Resource::BLOCKS.iter().enumerate() {
                for &b in &Resource::BLOCKS[i..] {
                    out.push([Effect::Res(a, 1), Effect::Res(b, 1)].into_iter().collect());
                }
            }
        }
        Science::Architecture => out.push([Effect::Points(3)].into_iter().collect()),
        Science::Theology => {
            out.push([Effect::Res(Resource::Skull, 1)].into_iter().collect())
        }
    }
    out
}

/// `n` research advances, each on any track. `free` waives the block cost.
pub fn research_choices(g: &GameState, p: PlayerId, n: u8, free: bool) -> Vec<Choice> {
    let mut out = Vec::new();
    let res = g.players[p.idx()].res;
    let lvls = g.research[p.idx()];
    recurse(g, p, res, lvls, n, free, &Choice::new(), &mut out);
    dedup(out)
}

#[allow(clippy::too_many_arguments)]
fn recurse(
    g: &GameState,
    p: PlayerId,
    res: [u8; 4],
    lvls: [u8; 4],
    n: u8,
    free: bool,
    acc: &Choice,
    out: &mut Vec<Choice>,
) {
    if n == 0 {
        out.push(acc.clone());
        return;
    }
    for s in Science::ALL {
        let lvl = lvls[s.idx()];
        // Levels 1/2/3 cost 1/2/3 blocks; the one-off payoff past level 3
        // costs 1. Go charged 3 for it, which put it nearly out of reach.
        let blocks = if free {
            0
        } else if lvl >= 3 {
            1
        } else {
            lvl + 1
        };
        for pay in pay_blocks(res, blocks) {
            // Spend now so a second advance in the same choice cannot spend the
            // same block twice. The Go recursion passed the *unspent* bundle
            // down the level-3 branch, so it could.
            let mut res2 = res;
            for r in Resource::BLOCKS {
                res2[r.idx()] -= pay[r.idx()] as u8;
            }

            if lvl < 3 {
                let mut next = acc.clone();
                payment_effects(pay, &mut next.0);
                next.0.push(Effect::AdvanceResearch(s));
                let mut l2 = lvls;
                l2[s.idx()] += 1;
                recurse(g, p, res2, l2, n - 1, free, &next, out);
            } else {
                for gain in top_payoffs(g, p, s) {
                    let mut next = acc.clone();
                    payment_effects(pay, &mut next.0);
                    let mut res3 = res2;
                    for e in &gain {
                        if let Effect::Res(r, d) = e {
                            res3[r.idx()] = (res3[r.idx()] as i32 + *d as i32).max(0) as u8;
                        }
                    }
                    next.0.extend_from_slice(&gain);
                    recurse(g, p, res3, lvls, n - 1, free, &next, out);
                }
            }
        }
    }
}

/// One free advance on a named track: the level-3 payoff if the track is maxed,
/// otherwise a plain advance.
///
/// The Go `FreeResearch` returned a list here too, but every caller took `[0]`
/// blindly, silently picking an arbitrary top payoff.
pub fn free_track(g: &GameState, p: PlayerId, s: Science) -> Vec<Choice> {
    if g.has_level(p, s, 3) {
        top_payoffs(g, p, s).into_iter().map(Choice).collect()
    } else {
        vec![Choice::one(Effect::AdvanceResearch(s))]
    }
}

/// Every way to construct one of the face-up buildings.
///
/// `exclude` skips a card mid-way through a double build. `bonus` applies the
/// architecture-track reward, which the second half of a double build does not
/// get.
pub fn building_choices(
    g: &GameState,
    p: PlayerId,
    exclude: Option<BuildingId>,
    bonus: bool,
    depth: u8,
) -> Vec<Choice> {
    let mut out = Vec::new();

    for id in g.face_up_buildings() {
        if Some(id) == exclude {
            continue;
        }
        let d = bdef(id);
        for cost in affordable_costs(g, p, d.cost, bonus) {
            let mut base = Choice::new();
            payment_effects(cost, &mut base.0);
            base.0.push(Effect::Build(id));
            if bonus {
                base.0.extend_from_slice(&g.build_bonus(p));
            }

            // Expand the payoff against a state that already has the cost paid
            // and the card taken, so anything the payoff spends is checked
            // against what is actually left.
            let mut probe = *g;
            base.apply(&mut probe, p);

            for tail in expand_payoff(&probe, p, id, &d.payoff, depth) {
                out.push(base.clone().chain(&tail));
            }
        }
    }
    dedup(out)
}

/// The listed cost, plus every one-block discount the architecture track allows.
///
/// `discount` is off for the second half of a double build, which costs full
/// price and grants no corn or victory points.
fn affordable_costs(g: &GameState, p: PlayerId, cost: Bundle, discount: bool) -> Vec<Bundle> {
    let player = &g.players[p.idx()];
    let mut out = Vec::new();
    if player.can_pay(cost) {
        out.push(cost);
    }
    if discount && g.builder(p) {
        for r in Resource::BLOCKS {
            if cost[r.idx()] > 0 {
                let mut c = cost;
                c[r.idx()] -= 1;
                if player.can_pay(c) && !out.contains(&c) {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// Turn a building's `Payoff` into the concrete choices it offers.
pub fn expand_payoff(
    g: &GameState,
    p: PlayerId,
    self_id: BuildingId,
    payoff: &Payoff,
    depth: u8,
) -> Vec<Choice> {
    match payoff {
        Payoff::Fixed(effects) => vec![Choice::of(effects.iter().copied())],
        Payoff::FreeTrack(s, tail) => free_track(g, p, *s)
            .into_iter()
            .map(|c| c.chain(&Choice::of(tail.iter().copied())))
            .collect(),
        Payoff::FreeAny(n, tail) => research_choices(g, p, *n, true)
            .into_iter()
            .map(|c| c.chain(&Choice::of(tail.iter().copied())))
            .collect(),
        Payoff::BuildAnother => {
            if depth == 0 {
                return vec![Choice::skip()];
            }
            let mut out = Vec::new();
            let steppable: Vec<Temple> = Temple::ALL
                .iter()
                .copied()
                .filter(|&t| g.can_temple_step(p, t, 1))
                .collect();
            for second in building_choices(g, p, Some(self_id), true, depth - 1) {
                if steppable.is_empty() {
                    // The step is wasted, but the build still happens.
                    out.push(second.clone());
                } else {
                    for &t in &steppable {
                        out.push(second.clone().with(Effect::TempleStep(t, 1)));
                    }
                }
            }
            if out.is_empty() {
                out.push(Choice::skip());
            }
            out
        }
        Payoff::CornExchange(tail) => corn_exchange(g, p)
            .into_iter()
            .map(|c| c.chain(&Choice::of(tail.iter().copied())))
            .collect(),
        Payoff::Mirror(tail) => crate::spaces::uxmal::mirror_choices(g, p, depth)
            .into_iter()
            .map(|c| c.chain(&Choice::of(tail.iter().copied())))
            .collect(),
    }
}

/// Every face-up monument the player can afford.
pub fn monument_choices(g: &GameState, p: PlayerId) -> Vec<Choice> {
    let player = &g.players[p.idx()];
    g.face_up_monuments()
        .filter(|&id| player.can_pay(mdef(id).cost))
        .map(|id| {
            let mut c = Choice::new();
            payment_effects(mdef(id).cost, &mut c.0);
            c.0.push(Effect::TakeMonument(id));
            c
        })
        .collect()
}

/// Exchange corn and blocks, in either direction.
///
/// The table is symmetric: wood is worth 2 corn, stone 3, gold 4. A player may
/// sell blocks for corn, buy blocks with corn, or do both in one exchange --
/// selling gold to fund wood is a normal play. Doing nothing is included.
///
/// The Go version enumerated only corn -> blocks, and even that was broken: it
/// mixed the recursion's remaining budget with the player's total corn and
/// dropped the base case out of every recursive branch, so it never offered
/// "exchange nothing". It was left commented out as "first attempt broke".
pub fn corn_exchange(g: &GameState, p: PlayerId) -> Vec<Choice> {
    const PRICES: [(Resource, u8); 3] = [
        (Resource::Wood, 2),
        (Resource::Stone, 3),
        (Resource::Gold, 4),
    ];
    let player = &g.players[p.idx()];

    // Every way to sell some of what is held.
    let mut sells: Vec<([u8; 3], u32)> = Vec::new();
    for w in 0..=player.get(Resource::Wood) {
        for st in 0..=player.get(Resource::Stone) {
            for gd in 0..=player.get(Resource::Gold) {
                let gained = w as u32 * 2 + st as u32 * 3 + gd as u32 * 4;
                sells.push(([w, st, gd], gained));
            }
        }
    }

    let mut out = Vec::new();
    for (sold, gained) in sells {
        let budget = (player.corn as u32 + gained).min(u8::MAX as u32) as u8;

        // Every way to spend that budget on blocks. Buying back something just
        // sold is legal but pointless, so those pairings are skipped.
        fn buys(budget: u8, idx: usize, cur: [u8; 3], out: &mut Vec<[u8; 3]>) {
            if idx == 3 {
                out.push(cur);
                return;
            }
            let price = PRICES[idx].1;
            for take in 0..=budget / price {
                let mut c = cur;
                c[idx] = take;
                buys(budget - take * price, idx + 1, c, out);
            }
        }
        let mut bought = Vec::new();
        buys(budget, 0, [0; 3], &mut bought);

        for buy in bought {
            if (0..3).any(|i| sold[i] > 0 && buy[i] > 0) {
                continue;
            }
            let spent: u32 = (0..3).map(|i| buy[i] as u32 * PRICES[i].1 as u32).sum();
            if spent > player.corn as u32 + gained {
                continue;
            }
            let net = gained as i32 - spent as i32;

            let mut c = Choice::new();
            for i in 0..3 {
                let delta = buy[i] as i8 - sold[i] as i8;
                if delta != 0 {
                    c.0.push(Effect::Res(PRICES[i].0, delta));
                }
            }
            if net != 0 {
                c.0.push(Effect::Corn(net as i16));
            }
            out.push(c);
        }
    }
    dedup(out)
}

/// Sort and deduplicate. Generation is naturally redundant -- several routes
/// reach the same bundle of effects -- and identical choices are worth
/// collapsing before they multiply through move generation.
pub fn dedup(mut v: Vec<Choice>) -> Vec<Choice> {
    v.sort();
    v.dedup();
    v
}

/// Monument definitions are static; this is here so callers don't reach past
/// the module for the count.
pub fn n_monuments() -> usize {
    MONUMENTS.len()
}
