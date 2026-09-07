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
    recurse(g, p, res, lvls, n, free, 0, EMPTY, &Choice::new(), &mut out);
    dedup(out)
}

/// Two spellings of one decision, collapsed.
///
/// `floor` is the lowest track index this step may still take. Advancing
/// agriculture then extraction and extraction then agriculture are one
/// decision wearing two spellings: the two advances are independent, and both
/// orders reach every total block payment, because `pay_blocks` offers every
/// split of it. Only a *maxed* track breaks that independence -- its one-off
/// payoff can hand back blocks, and blocks in hand may be what pays for a
/// track this walk has already passed -- so a payoff that grants a resource
/// reopens the whole range. Nothing else an advance does is visible to a later
/// one: `top_payoffs` reads the unmutated `g`, and a level change is only ever
/// read by the same track.
///
/// `owed` is the same argument applied to the *bill*. Paying wood for the
/// first advance and stone for the second reaches the position paying stone
/// then wood reaches, so the payments are carried as one running bundle and
/// settled in a single block, and the two splits become the same `Choice`.
/// The bundle is settled early -- before a payoff that grants a resource --
/// because deferring past that point would let a choice read as affordable on
/// blocks the payoff had not handed over yet.
#[allow(clippy::too_many_arguments)]
fn recurse(
    g: &GameState,
    p: PlayerId,
    res: [u8; 4],
    lvls: [u8; 4],
    n: u8,
    free: bool,
    floor: usize,
    owed: Bundle,
    acc: &Choice,
    out: &mut Vec<Choice>,
) {
    if n == 0 {
        let mut done = acc.clone();
        payment_effects(owed, &mut done.0);
        out.push(done);
        return;
    }
    for s in Science::ALL {
        // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
        if pruning_on() && s.idx() < floor {
            continue;
        }
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
            let mut owed2 = owed;
            for r in Resource::BLOCKS {
                owed2[r.idx()] += pay[r.idx()];
            }
            // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
            let defer = pruning_on();

            if lvl < 3 {
                let mut next = acc.clone();
                if !defer {
                    payment_effects(pay, &mut next.0);
                }
                next.0.push(Effect::AdvanceResearch(s));
                let mut l2 = lvls;
                l2[s.idx()] += 1;
                let carry = if defer { owed2 } else { EMPTY };
                recurse(g, p, res2, l2, n - 1, free, s.idx(), carry, &next, out);
            } else {
                for gain in top_payoffs(g, p, s) {
                    let mut next = acc.clone();
                    // Extraction's payoff is two blocks, which may be exactly
                    // what an earlier track needs; a skull cannot pay for
                    // research, but the bill is settled for any resource
                    // rather than resting on that.
                    let reopen = gain.iter().any(|e| matches!(e, Effect::Res(..)));
                    let carry = if defer && !reopen { owed2 } else { EMPTY };
                    if !defer || reopen {
                        payment_effects(if defer { owed2 } else { pay }, &mut next.0);
                    }
                    let mut res3 = res2;
                    for e in &gain {
                        if let Effect::Res(r, d) = e {
                            res3[r.idx()] = (res3[r.idx()] as i32 + *d as i32).max(0) as u8;
                        }
                    }
                    next.0.extend_from_slice(&gain);
                    let next_floor = if reopen { 0 } else { s.idx() };
                    recurse(g, p, res3, lvls, n - 1, free, next_floor, carry, &next, out);
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

/// Every one-block discount the architecture track allows, or the listed cost
/// when it allows none.
///
/// `discount` is off for the second half of a double build, which costs full
/// price and grants no corn or victory points.
///
/// *Which* block to knock off is a real decision, so all of them are offered.
/// Declining the discount altogether is not: the card, its payoff and the
/// architecture bonus are identical either way and the extra block is simply
/// handed back to the bank, where it scores nothing and buys nothing. So the
/// full price is only generated when no discount is available -- a track short
/// of level 3, the second half of a double build, or a cost the player cannot
/// quite reduce. That the payoff is unaffected is not an assumption: it is
/// expanded against a probe with the cost already paid, and every generator is
/// monotone in the player's holdings, so the cheaper branch's payoffs are a
/// superset of the dearer one's.
fn affordable_costs(g: &GameState, p: PlayerId, cost: Bundle, discount: bool) -> Vec<Bundle> {
    let player = &g.players[p.idx()];
    let mut out = Vec::new();
    // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
    if !pruning_on() && player.can_pay(cost) {
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
    if out.is_empty() && player.can_pay(cost) {
        out.push(cost);
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

// SCRATCH: measurement switch, delete with src/bin/movestats.rs.
//
// Three levels rather than two, so one run can price each generation of rules
// against the *same* positions: 0 generates everything, 1 is the corn-axis
// dominance that was already here, 2 adds this pass's rules. Phase 1 of
// `movestats` walks its games at level 0, so the position set does not move
// when the rules do.
pub static PRUNE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
// SCRATCH: measurement switch, delete with src/bin/movestats.rs.
pub fn prune_level() -> u8 {
    PRUNE.load(std::sync::atomic::Ordering::Relaxed)
}
// SCRATCH: measurement switch, delete with src/bin/movestats.rs.
pub fn pruning_on() -> bool {
    prune_level() >= 1
}
// SCRATCH: measurement switch, delete with src/bin/movestats.rs.
pub fn wide_pruning_on() -> bool {
    prune_level() >= 2
}

/// Drop every choice that lands in a position an earlier one already reaches.
///
/// Ground truth rather than a rule about effect lists: two choices that leave
/// `g` in the same state are the same move for every purpose downstream, and
/// all that is lost is one of two spellings in the move log.
///
/// The expensive kind of redundancy this exists for is `tikal::build_two`.
/// Building A then B and B then A are the same pair of cards, and unless the
/// architecture discount is in play they cost the same too, so an unbuilder's
/// double build is generated exactly twice -- measured at 6% of every option
/// list on Tikal 4, 5, 6 and 7, and again inside every mirror that reaches
/// them. `dedup` cannot see it because the two effect lists are permutations,
/// and a multiset key would not be sound: `TempleStep` clamps, so a choice
/// carrying both a step up and a step down on one temple depends on their
/// order, and building #30's mirror can reach Palenque's corn dig.
///
/// A `GameState` is `Copy`, so the probe is a memcpy; `FxHashSet` for the same
/// reason `mcts` uses it -- 320 bytes of key is where SipHash starts to cost
/// more than the work it protects.
pub fn dedup_by_position(g: &GameState, p: PlayerId, v: Vec<Choice>) -> Vec<Choice> {
    let mut seen: rustc_hash::FxHashSet<GameState> =
        rustc_hash::FxHashSet::with_capacity_and_hasher(v.len(), Default::default());
    v.into_iter()
        .filter(|c| {
            let mut probe = *g;
            c.apply(&mut probe, p);
            seen.insert(probe)
        })
        .collect()
}

/// Sort and deduplicate. Generation is naturally redundant -- several routes
/// reach the same bundle of effects -- and identical choices are worth
/// collapsing before they multiply through move generation.
pub fn dedup(mut v: Vec<Choice>) -> Vec<Choice> {
    v.sort();
    v.dedup();
    v
}

/// Sort, deduplicate, and drop every choice another choice strictly beats.
///
/// Two choices that agree on every effect *except* how much corn, how many
/// blocks and how many points they move are not two decisions: whoever takes
/// the one that moves less of all five arrives at exactly the position the
/// other reaches, poorer. Nothing in the game pays a player for holding *less*
/// of any of them -- blocks and skulls have no upkeep and no hand limit, points
/// are pure victory points, and the one rule that reads a corn *ceiling* is
/// begging, which requires fewer than three, sets the count to exactly three
/// and charges a step *down* a temple, so holding three or more is better on
/// both axes than being allowed to beg. The poorer choice is therefore
/// dominated, not merely unattractive.
///
/// This is the same argument `choices_for_worker` already made for "paying corn
/// to reach a space that does nothing", widened twice: first from the empty
/// choice to every choice, then from corn alone to everything liquid. Both
/// widenings pay off because the board offers the *same action at two prices*
/// all over the place once a worker is high on a gear -- Uxmal's mirror sells
/// any lower action for one corn while stepping down to that action costs one
/// corn per space -- and because the free-choice spaces stack whole gears on
/// top of each other, where a strictly fatter version of an action is usually
/// sitting a few spaces up. Yaxchilan 5 hands out the gold of space 3 and the
/// stone of space 2 together, so from Yaxchilan 6 those two spaces are dead
/// letters; and a worker on Uxmal 6 can mirror Palenque 1 for a net two corn,
/// which beats selling a block for two corn at Uxmal 2.
///
/// The wealth deltas commute with everything else in an affordable choice -- no
/// generated effect reads corn, blocks or points at execution time, `SetCorn`
/// is never generated (only begging emits it, and begging lives on the `Move`),
/// and generation never emits a sequence whose running balance dips below zero
/// -- `generated_moves_are_legal` runs `check_move`, and so `Choice::affordable`,
/// over every move of 40 seeded games -- so `apply`'s clamp never fires on a
/// generated choice and the net *is* the position reached.
/// Skulls are deliberately **not** a wealth axis: `take_skulls` caps against a
/// shared bank, so the net does not determine the outcome and a bigger gain
/// also changes `skulls_remaining`, which is not this player's to trade away.
/// They stay in the key and must match exactly.
///
/// The key is compared *in order* rather than as a multiset, which leaves two
/// spellings of the same bundle standing: `TempleStep` clamps, so re-ordering
/// is not free in general and is not worth the proof here.
///
/// The empty choice is exempt. "Pick the worker up and do nothing" is one of
/// the three options the rules name, not a degenerate corn gain, and every
/// space is required to offer it (`doing_nothing_is_always_an_option`) --
/// without the exemption a space whose action is pure corn, like Palenque 1,
/// would lose it to its own payout.
pub fn dominated_dedup(mut v: Vec<Choice>) -> Vec<Choice> {
    // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
    if !pruning_on() {
        return dedup(v);
    }
    // Order by "what it does, ignoring the price", then by total wealth moved.
    // A dominator moves at least as much on every axis, so it also moves at
    // least as much in total: sorting by the sum descending puts every
    // dominator ahead of everything it beats, which turns the Pareto front into
    // one forward pass. The key is compared as an iterator rather than
    // materialised, because this runs at every node of the retrieval walk and a
    // per-choice allocation there is not free.
    // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
    let wide = wide_pruning_on();
    let group = |a: &Choice, b: &Choice| {
        a.is_skip() == b.is_skip()
            && a.0
                .iter()
                .filter(|e| !is_wealth(e, wide))
                .cmp(b.0.iter().filter(|e| !is_wealth(e, wide)))
                .is_eq()
    };
    v.sort_unstable_by(|a, b| {
        a.is_skip()
            .cmp(&b.is_skip())
            .then_with(|| {
                a.0.iter()
                    .filter(|e| !is_wealth(e, wide))
                    .cmp(b.0.iter().filter(|e| !is_wealth(e, wide)))
            })
            .then_with(|| {
                wealth(b, wide)
                    .iter()
                    .sum::<i32>()
                    .cmp(&wealth(a, wide).iter().sum::<i32>())
            })
            .then_with(|| a.cmp(b))
    });

    // Compact in place: `keep` is the write cursor and never runs ahead of the
    // read cursor, so the survivors of the group being scanned are always the
    // slice this compares against.
    let mut front: Vec<[i32; N_WEALTH]> = Vec::new();
    let mut keep = 0usize;
    let mut i = 0usize;
    while i < v.len() {
        let mut j = i + 1;
        while j < v.len() && group(&v[i], &v[j]) {
            j += 1;
        }
        front.clear();
        for r in i..j {
            let w = wealth(&v[r], wide);
            // Equal vectors count as dominated: identical key and identical net
            // means an identical position, so the first spelling stands for all
            // of them. That collapses pairs plain `dedup` misses, such as the
            // mirror's `-1 corn, +3 corn` against a bare `+2 corn`.
            if front.iter().any(|f| (0..N_WEALTH).all(|k| f[k] >= w[k])) {
                continue;
            }
            front.push(w);
            v.swap(keep, r);
            keep += 1;
        }
        i = j;
    }
    v.truncate(keep);
    // Back into `Choice` order, which is what every other generator returns and
    // what makes the traversal order of `legal_moves` stable.
    v.sort_unstable();
    v
}

/// Corn, the three block types, and points.
const N_WEALTH: usize = 5;

/// Whether an effect only moves liquid wealth, and so is priced rather than
/// structural. Skulls are excluded on purpose -- see `dominated_dedup`.
///
/// SCRATCH: `wide` is the measurement switch. False keeps corn as the only
/// priced axis, which is what the rule was before this pass; delete the
/// parameter with src/bin/movestats.rs.
#[inline]
fn is_wealth(e: &Effect, wide: bool) -> bool {
    match e {
        Effect::Corn(_) => true,
        Effect::Points(_) => wide,
        Effect::Res(r, _) => wide && *r != Resource::Skull,
        _ => false,
    }
}

/// Net corn, wood, stone, gold and points a choice moves.
#[inline]
fn wealth(c: &Choice, wide: bool) -> [i32; N_WEALTH] {
    let mut w = [0i32; N_WEALTH];
    for e in &c.0 {
        match *e {
            Effect::Corn(n) => w[0] += n as i32,
            Effect::Res(r, n) if wide && r != Resource::Skull => w[1 + r.idx()] += n as i32,
            Effect::Points(n) if wide => w[4] += n as i32,
            _ => {}
        }
    }
    w
}

/// A linear price on the resolved effect vocabulary: the prior a `Choice` can
/// be given **without applying it**.
///
/// # Why this is enough
///
/// `effect.rs` resolves every value at generation time -- `Effect::Corn(7)` is
/// seven corn, not "however much the agriculture level implies" -- so a
/// `Choice` is already a summary of what it does. Nothing further has to travel
/// alongside it. Measured over 6,238 `Take` nodes of >= 8 edges (353,429 edges,
/// 24 seeded games), pricing a choice this way costs **16.0 ns an edge against
/// 123.0 ns** for the `apply_step` plus `eval::heuristic` that
/// `mcts::Priors::OnePly` pays -- 7.7x, and 6.4x at the nodes wider than
/// `widen_cap`, where over half the edge mass sits and the expansion cost
/// actually lives.
///
/// The five-axis net that `dominated_dedup` already computes for every choice
/// is *not* enough on its own, which is the part that had to be measured rather
/// than assumed: pricing only corn, blocks and points leaves 0.247 heuristic
/// points of regret in the first `max_edges` against 0.057 for the whole
/// vocabulary, and drops the best edge at a truncated node three times as
/// often. The structural effects -- a temple step, a research advance, a card --
/// carry real ordering information, and they are exactly what the wealth vector
/// throws away in order to be a sound dominance key.
///
/// # Why the prices are the caller's
///
/// A fixed table would be a second opinion about the value function. The
/// numbers that worked are `eval::heuristic`'s own local gradient: probe `+1`
/// of each axis against the position once, and price a choice as the dot
/// product. Fifteen `heuristic` calls, 1.8 us, paid once per *turn* and reused
/// at every `Take` node of it -- 0.71 us amortised per node. Rebuilding the
/// gradient at each node is 4x dearer and buys 0.057 -> 0.035 points of regret,
/// so the turn-level probe is the one to build. Ordering is sound and filtering
/// is not, so a stale gradient costs simulations and never legality.
///
/// Deliberately not priced: `Effect::Res(Skull, _)` is a normal axis here even
/// though `dominated_dedup` excludes it, because ordering is not dominance --
/// a skull genuinely is worth something, it just is not this player's to trade
/// away.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EffectPrice {
    pub corn: f32,
    /// Wood, stone, gold, skull, in `Resource::ALL` order.
    pub res: [f32; 4],
    pub points: f32,
    /// Brown, yellow, green, in `Temple::ALL` order.
    pub temple: [f32; 3],
    pub science: [f32; 4],
    pub unlock_worker: f32,
    pub free_worker: f32,
    pub worker_discount: f32,
    /// The card- and space-naming effects. Probing each distinct one against
    /// the state is worth almost nothing -- a per-card probe moves the regret
    /// in the first `max_edges` from 0.057 to 0.052 points and costs 26% more
    /// per edge -- so one constant apiece is the shape that earns its keep.
    pub palenque_tile: f32,
    pub burn_wood: f32,
    pub fill_chichen: f32,
    pub build: f32,
    pub monument: f32,
}

impl EffectPrice {
    /// Exhaustive with no wildcard arm, for the same reason `Effect::tag` is:
    /// a new effect must break the build here rather than silently price at
    /// zero and drop out of every prior in the search.
    #[inline]
    pub fn of(&self, e: Effect) -> f32 {
        match e {
            Effect::Corn(n) => n as f32 * self.corn,
            // Never generated -- only begging emits it, and begging lives on
            // the `Move` rather than in a `Choice` -- so the delta reading is
            // unreachable rather than wrong.
            Effect::SetCorn(n) => n as f32 * self.corn,
            Effect::Res(r, n) => n as f32 * self.res[r.idx()],
            Effect::Points(n) => n as f32 * self.points,
            Effect::TempleStep(t, n) => n as f32 * self.temple[t.idx()],
            Effect::AdvanceResearch(s) => self.science[s.idx()],
            Effect::UnlockWorker => self.unlock_worker,
            Effect::FreeWorker(n) => n as f32 * self.free_worker,
            Effect::WorkerDiscount(n) => n as f32 * self.worker_discount,
            Effect::TakePalenqueTile(..) => self.palenque_tile,
            Effect::BurnPalenqueWood(_) => self.burn_wood,
            Effect::FillChichen(_) => self.fill_chichen,
            Effect::Build(_) => self.build,
            Effect::TakeMonument(_) => self.monument,
        }
    }

    /// One choice, priced. Linear, so it misses every interaction between the
    /// effects of one choice; that is what makes it a hint and not an
    /// evaluation.
    #[inline]
    pub fn choice(&self, c: &Choice) -> f32 {
        c.0.iter().map(|&e| self.of(e)).sum()
    }
}

/// Monument definitions are static; this is here so callers don't reach past
/// the module for the count.
pub fn n_monuments() -> usize {
    MONUMENTS.len()
}
