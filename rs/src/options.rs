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
pub fn pay_blocks(res: [u8; 4], n: u8) -> Splits {
    fn go(res: &[u8; 4], idx: usize, left: u8, cur: Bundle, out: &mut Splits) {
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
    let mut out = Splits::new();
    go(&res, 0, n, EMPTY, &mut out);
    out
}

/// Every way to split one bill across the three block types.
///
/// The dearest research advance costs three blocks, which splits ten ways, so
/// this never touches the heap in the base game -- and `recurse` asks for one
/// per science track per level, which was a malloc and a free apiece.
pub type Splits = smallvec::SmallVec<[Bundle; 12]>;

fn payment_effects(pay: Bundle, into: &mut Effects) {
    for r in Resource::BLOCKS {
        if pay[r.idx()] > 0 {
            into.push(Effect::Res(r, -pay[r.idx()]));
        }
    }
}

/// The level-3 payoffs of one track: at most one per temple or per track.
pub type Payoffs = smallvec::SmallVec<[Effects; 4]>;

/// The one-off payoff for advancing a track that is already at level 3.
fn top_payoffs(g: &GameState, p: PlayerId, s: Science) -> Payoffs {
    let mut out: Payoffs = Payoffs::new();
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
    dedup_unordered(out)
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
fn affordable_costs(g: &GameState, p: PlayerId, cost: Bundle, discount: bool) -> Splits {
    let player = &g.players[p.idx()];
    let mut out = Splits::new();
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
            // Inline: three temples, so this never reaches the heap. Threading
            // an output buffer through the whole payoff expansion was tried
            // and measured neutral (11.26 s against 11.30, four interleaved
            // rounds); the allocations that matter were the ones inside the
            // recursion, not the one per payoff.
            let steppable: smallvec::SmallVec<[Temple; 3]> = Temple::ALL
                .iter()
                .copied()
                .filter(|&t| g.can_temple_step(p, t, 1))
                .collect();
            for second in building_choices(g, p, Some(self_id), true, depth - 1) {
                if steppable.is_empty() {
                    // The step is wasted, but the build still happens.
                    out.push(second);
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
        Payoff::Mirror(tail) => {
            let mut out: Vec<Choice> = crate::spaces::uxmal::mirror_choices(g, p, depth)
                .into_iter()
                .map(|c| c.chain(&Choice::of(tail.iter().copied())))
                .collect();
            // The mirror is a privilege, not a cost, so the bare tail is
            // always on offer and the mirrored actions are extras beside it.
            //
            // Two ways that used to go wrong, and they are one rule. A player
            // who *cannot* pay the corn -- or whose chain is at its depth bound
            // -- still constructs the card and still takes the tail;
            // `mirror_choices` returns an empty list in both cases and
            // `building_choices` reads an empty payoff as "no way to build
            // this", so card #30 dropped out of the game entirely at zero corn.
            // Architecture level 1 masked that: the corn the build itself pays
            // lands before the payoff is expanded, and one corn is exactly the
            // mirror's price. A player who *can* pay and would rather not was
            // the other half -- every choice `mirror_choices` returns leads
            // with the fee, so constructing #30 forced the corn and forced an
            // action, and the mirror reaches `UnlockWorker`. That is
            // `RULES-AUDIT.md` A9's unwanted Uxmal 3 worker arriving through a
            // card. Uxmal 5, the mirror's other caller, was never exposed to
            // it: `choices_for_worker` prepends the skip there, the same way it
            // does for Uxmal 1, 2 and 4 and Tikal 2 and 5, none of which carry
            // one in their own list either.
            //
            // Pushed unconditionally rather than as an `is_empty` fallback:
            // both halves are the same sentence of the rulebook, and no choice
            // `mirror_choices` returns can collide with the bare tail, since it
            // drops the mirrored spaces' skips and prefixes the fee to what is
            // left.
            out.push(Choice::of(tail.iter().copied()));
            out
        }
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

    let mut out = Vec::new();
    // One buffer for the whole exchange rather than one per sale: the sell
    // loop runs (wood+1)(stone+1)(gold+1) times and each pass used to allocate
    // and free its own list of purchases.
    let mut bought: Vec<[u8; 3]> = Vec::new();
    // Every way to sell some of what is held, crossed with every way to spend
    // the proceeds.
    for w in 0..=player.get(Resource::Wood) {
        for st in 0..=player.get(Resource::Stone) {
            for gd in 0..=player.get(Resource::Gold) {
        let sold = [w, st, gd];
        let gained = w as u32 * 2 + st as u32 * 3 + gd as u32 * 4;
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
        bought.clear();
        buys(budget, 0, [0; 3], &mut bought);

        for &buy in &bought {
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
        }
    }
    dedup_unordered(out)
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
    // SCRATCH: `TZ_PRUNE` pins the level for a whole process, which is how the
    // "does the dominance rule earn its cost under MCTS" run reaches `arena` --
    // a binary this workstream does not own and cannot add a flag to. Read once
    // through a `Once`; `movestats` still stores directly and wins, because it
    // stores after this has already fired.
    static ENV: std::sync::Once = std::sync::Once::new();
    ENV.call_once(|| {
        if let Some(v) = std::env::var("TZ_PRUNE").ok().and_then(|v| v.parse().ok()) {
            PRUNE.store(v, std::sync::atomic::Ordering::Relaxed);
        }
    });
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
    // Unstable: the only elements the comparator calls equal are *identical*
    // choices, so which copy survives is not a question, and the stable sort's
    // scratch buffer was showing up as `driftsort` plus a malloc at every
    // `research_choices` and `corn_exchange` call.
    v.sort_unstable();
    v.dedup();
    v
}

/// Collapse exact duplicates, imposing no order.
///
/// Every list built in this module is on its way into `dominated_dedup`, and
/// that pass's output is a function of its input *multiset*: the sweep orders
/// by group and by total wealth moved, two choices can only prune each other
/// when their nets are equal, and the equal case picks the lexicographically
/// first spelling explicitly. So the canonical order `dedup` used to leave
/// behind was never read by anything downstream -- and finding duplicates by
/// sorting was 35% of all generation time, spread over `research_choices`,
/// `corn_exchange` and `tikal::at_d`.
///
/// A digest collision leaves a duplicate standing rather than dropping a
/// distinct choice, and a surviving duplicate is collapsed by `dominated_dedup`
/// anyway, so the hash is an accelerator here too.
pub fn dedup_unordered(mut v: Vec<Choice>) -> Vec<Choice> {
    if v.len() < 2 {
        return v;
    }
    // Short lists are the common case and a hash map for eight elements costs
    // more than looking at all of them; `keep` never runs ahead of `r`, so the
    // survivors are always the prefix being compared against.
    if v.len() <= 16 {
        let mut keep = 0usize;
        'outer: for r in 0..v.len() {
            for k in 0..keep {
                if v[k] == v[r] {
                    continue 'outer;
                }
            }
            v.swap(keep, r);
            keep += 1;
        }
        v.truncate(keep);
        return v;
    }
    let mut seen = SCRATCH.take_seen();
    seen.clear();
    let mut keep = 0usize;
    for r in 0..v.len() {
        let d = choice_digest(&v[r]);
        if let Some(&i) = seen.get(&d) {
            if v[i as usize] == v[r] {
                continue;
            }
        }
        seen.insert(d, keep as u32);
        v.swap(keep, r);
        keep += 1;
    }
    v.truncate(keep);
    SCRATCH.put_seen(seen);
    v
}

/// An order-sensitive digest of a whole choice, wealth included.
#[inline]
fn choice_digest(c: &Choice) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ c.0.len() as u64;
    for &e in &c.0 {
        h ^= pack(e) as u64;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
    }
    h
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
    // One element cannot dominate anything and is already in order; the p50
    // space returns a list this short, so the early exit is most of the calls.
    if v.len() < 2 {
        return v;
    }
    // SCRATCH: measurement switch, delete with src/bin/movestats.rs.
    let wide = wide_pruning_on();

    // Summarise every choice once instead of re-deriving the key inside a
    // comparator. The old shape spent 21% of all generation time in
    // `Iterator::cmp_by`, comparing two filtered effect iterators O(n log n)
    // times per list, and another 15% in the comparator that drove it; the
    // summary is one linear pass and the sort then compares two `u64`s.
    //
    // Deliberately one straight-line body rather than a `sweep` helper and a
    // stack-array path for short lists: both were tried and both measured
    // slower (12.57 s against 12.67 and 12.87, four interleaved rounds), so
    // the shared buffers are already cheaper than anything that avoids them.
    let mut scratch = SCRATCH.take();
    let keys = &mut scratch.keys;
    keys.clear();
    keys.extend(v.iter().map(|c| Key::of(c, wide)));

    let idx = &mut scratch.idx;
    idx.clear();
    idx.extend(0..v.len() as u32);
    // Group, then total wealth moved descending: a dominator moves at least as
    // much on every axis and so also in total, which puts every dominator ahead
    // of everything it beats and turns the Pareto front into one forward pass.
    idx.sort_unstable_by(|&a, &b| {
        let (ka, kb) = (&keys[a as usize], &keys[b as usize]);
        ka.group.cmp(&kb.group).then_with(|| kb.sum.cmp(&ka.sum))
    });

    let keep = &mut scratch.keep;
    keep.clear();
    keep.resize(v.len(), false);
    let front = &mut scratch.front;
    let mut i = 0usize;
    while i < idx.len() {
        let g = keys[idx[i] as usize].group;
        let mut j = i + 1;
        while j < idx.len() && keys[idx[j] as usize].group == g {
            j += 1;
        }
        front.clear();
        for s in &idx[i..j] {
            let r = *s as usize;
            let w = keys[r].w;
            // `group` is a 64-bit digest, so a collision would put two genuinely
            // different actions in one group and could drop a legal move.
            // Confirming the real key before a prune fires -- and only then,
            // which over 27M edges is a few compares per list rather than
            // n log n -- makes the digest an accelerator and never the rule.
            let mut dominated = false;
            for (f, fw) in front.iter_mut() {
                if !(0..N_WEALTH).all(|k| fw[k] >= w[k]) || !same_group(&v[*f], &v[r], wide) {
                    continue;
                }
                // Equal vectors count as dominated: identical key and identical
                // net means an identical position, so one spelling stands for
                // all of them, which collapses pairs plain `dedup` misses --
                // the mirror's `-1 corn, +3 corn` against a bare `+2 corn`.
                //
                // Which spelling is settled here rather than in the sort. Two
                // choices can only dominate each other when their nets are
                // equal (a dominator's total is at least the dominated one's,
                // and the sweep runs in total order), so the lexicographic
                // tie-break the comparator used to carry was doing work for
                // exactly this case -- and paying a `Choice` compare on every
                // one of the n log n sort steps to do it.
                if *fw == w && v[r] < v[*f] {
                    keep[*f] = false;
                    keep[r] = true;
                    *f = r;
                }
                dominated = true;
                break;
            }
            if dominated {
                continue;
            }
            front.push((r, w));
            keep[r] = true;
        }
        i = j;
    }

    // Compact in place, then back into `Choice` order -- what every other
    // generator returns, and what makes the traversal order of `legal_moves`
    // stable.
    let mut w = 0usize;
    for r in 0..v.len() {
        if keep[r] {
            v.swap(w, r);
            w += 1;
        }
    }
    v.truncate(w);
    SCRATCH.put(scratch);
    v.sort_unstable();
    v
}

/// What a choice looks like to the dominance test, computed in one pass.
///
/// `group` digests the *structural* effects in order -- everything that is not
/// liquid wealth -- with the skip flag folded in, because "pick the worker up
/// and do nothing" is one of the three options the rules name and must never be
/// priced away by a space whose action is pure corn (`Palenque 1`). `w` is the
/// net the dominance test compares and `sum` orders the sweep.
#[derive(Clone, Copy)]
struct Key {
    group: u64,
    w: [i32; N_WEALTH],
    sum: i32,
}

impl Key {
    #[inline]
    fn of(c: &Choice, wide: bool) -> Key {
        // A distinct seed rather than a flag field: the skip is the only empty
        // effect list, so seeding it apart is what keeps it out of the group of
        // every all-wealth choice at zero extra cost in the sweep.
        let mut group: u64 = if c.0.is_empty() { 0x9e37_79b9_7f4a_7c15 } else { 0 };
        let mut w = [0i32; N_WEALTH];
        for &e in &c.0 {
            match e {
                Effect::Corn(n) => {
                    w[0] += n as i32;
                    continue;
                }
                Effect::Res(r, n) if wide && r != Resource::Skull => {
                    w[1 + r.idx()] += n as i32;
                    continue;
                }
                Effect::Points(n) if wide => {
                    w[4] += n as i32;
                    continue;
                }
                _ => {}
            }
            group ^= pack(e) as u64;
            group = group.wrapping_mul(0xff51_afd7_ed55_8ccd);
            group ^= group >> 33;
        }
        let sum = w[0] + w[1] + w[2] + w[3] + w[4];
        Key { group, w, sum }
    }
}

/// The exact test the digest stands in for: same skip flag, same structural
/// effects in the same order. Only reached when a prune is about to fire.
#[inline]
fn same_group(a: &Choice, b: &Choice, wide: bool) -> bool {
    a.0.is_empty() == b.0.is_empty()
        && a.0
            .iter()
            .filter(|e| !is_wealth(e, wide))
            .cmp(b.0.iter().filter(|e| !is_wealth(e, wide)))
            .is_eq()
}

/// One effect as a `u32`, order-isomorphic to `Effect`'s derived `Ord`.
///
/// The vocabulary is 14 variants whose widest payload is a 16-bit corn delta,
/// so tag and payload fit a word with room to spare, and every field is either
/// a fieldless enum (declaration order, which is what the derive compares) or a
/// `u8` newtype. Biasing the signed fields keeps the packed order the same as
/// the derived one; `packed_order_matches_derived_ord` in `tests/rules.rs`
/// holds that.
#[inline]
pub fn pack(e: Effect) -> u32 {
    let (tag, payload): (u32, u32) = match e {
        Effect::Corn(n) => (0, (n as i32 + 32_768) as u32),
        Effect::SetCorn(n) => (1, n as u32),
        Effect::Res(r, n) => (2, ((r as u32) << 9) | (n as i32 + 128) as u32),
        Effect::Points(n) => (3, (n as i32 + 128) as u32),
        Effect::TempleStep(t, n) => (4, ((t as u32) << 9) | (n as i32 + 128) as u32),
        Effect::AdvanceResearch(s) => (5, s as u32),
        Effect::UnlockWorker => (6, 0),
        Effect::FreeWorker(n) => (7, (n as i32 + 128) as u32),
        Effect::WorkerDiscount(n) => (8, (n as i32 + 128) as u32),
        Effect::TakePalenqueTile(pos, k) => (9, ((pos.0 as u32) << 9) | k as u32),
        Effect::BurnPalenqueWood(pos) => (10, pos.0 as u32),
        Effect::FillChichen(pos) => (11, pos.0 as u32),
        Effect::Build(b) => (12, b.0 as u32),
        Effect::TakeMonument(m) => (13, m.0 as u32),
    };
    (tag << 20) | payload
}

/// Buffers `dominated_dedup` would otherwise allocate four times per call.
///
/// The allocator is 14.6% of a running search and generation is where most of
/// that comes from, so the working set of a pass that runs at every `Take` node
/// is worth keeping. Taken out of the cell rather than borrowed: a future
/// nested call then allocates its own instead of panicking on the borrow.
#[derive(Default)]
struct Scratch {
    keys: Vec<Key>,
    idx: Vec<u32>,
    keep: Vec<bool>,
    front: Vec<(usize, [i32; N_WEALTH])>,
    seen: rustc_hash::FxHashMap<u64, u32>,
}

thread_local! {
    static SCRATCH_CELL: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

struct ScratchSlot;
const SCRATCH: ScratchSlot = ScratchSlot;

impl ScratchSlot {
    #[inline]
    fn take(&self) -> Scratch {
        // Taken out of the cell rather than borrowed across the body: nothing
        // nests today -- `raw_at` finishes before `dominated_dedup` starts --
        // and a future nesting then allocates its own buffers instead of
        // aliasing these.
        SCRATCH_CELL.with(|c| std::mem::take(&mut *c.borrow_mut()))
    }
    #[inline]
    fn put(&self, s: Scratch) {
        SCRATCH_CELL.with(|c| {
            let mut slot = c.borrow_mut();
            if slot.keys.capacity() < s.keys.capacity() {
                *slot = s;
            }
        });
    }
    #[inline]
    fn take_seen(&self) -> rustc_hash::FxHashMap<u64, u32> {
        SCRATCH_CELL.with(|c| std::mem::take(&mut c.borrow_mut().seen))
    }
    #[inline]
    fn put_seen(&self, m: rustc_hash::FxHashMap<u64, u32>) {
        SCRATCH_CELL.with(|c| {
            if let Ok(mut slot) = c.try_borrow_mut() {
                if slot.seen.capacity() < m.capacity() {
                    slot.seen = m;
                }
            }
        });
    }
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
