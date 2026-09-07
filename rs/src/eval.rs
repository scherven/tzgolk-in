//! A hand-written position evaluator, and exhaustive move ranking on top of it.
//!
//! [`heuristic`] estimates a player's **final score**. The contract is the one
//! the value head will inherit — `(&GameState, PlayerId) -> f32`, in points —
//! so swapping a trained net in stays local.
//!
//! The shape it is built to is worth stating, because it is what the first
//! version got wrong. A Tzolk'in position is not a pile of resources; it is an
//! *engine* plus a set of *scheduled payouts*. So the estimate is
//!
//! ```text
//!   points already banked
//! + what liquidation pays today   (exact: the same arithmetic as `end_game`)
//! + the temple payouts still to come, projected from where everyone stands
//! + the engine, valued at what it can still convert before the calendar ends
//! - the food the engine will fail to pay for
//! ```
//!
//! The third and fourth terms are the ones that decay: `engine_value` is scaled
//! by the rounds remaining and reaches zero on the last day, which makes
//! `heuristic` **exactly** `state.scores()[p]` once `over` is set. A leaf
//! evaluation and a real result are therefore the same number, which is what
//! lets the search compare a forced win against a heuristic guess.
//!
//! [`margin`] is the zero-sum reduction the search runs on: how far ahead of the
//! best opponent a player stands. It is what makes denying an opponent — a
//! contested temple top, the last monument, the skull bank — score as a gain
//! rather than as nothing at all.

use crate::data::monuments::def as mdef;
use crate::data::temples::TEMPLES;
use crate::ids::*;
use crate::moves::Move;
use crate::state::{GameState, LAST_DAY, POINT_DAYS, RESOURCE_DAYS};
use std::ops::ControlFlow;

// ---- tuning ------------------------------------------------------------

/// The base game's liquidation rate: four corn to a point.
const CORN_PER_POINT: f32 = 4.0;

/// What a *spendable* corn is worth beyond its liquidation value. Corn buys
/// placement depth, feeds workers and mirrors actions, none of which the 4:1
/// conversion sees. Applies to the first [`LIQUID_CORN`] only — a hoard past
/// what a player can plausibly spend really is worth just 1/4 a point each.
const CORN_PREMIUM: f32 = 0.10;
const LIQUID_CORN: f32 = 16.0;

/// Per-block premium over liquidation value, for the same reason: blocks build,
/// and a building is worth more than the three corn its stone converts to.
const BLOCK_PREMIUM: f32 = 0.55;

/// A skull liquidates at 3 (`GameState::end_game`) but *spends* at Chichen for
/// 4 to 13 points plus a temple step. This is the premium for holding one while
/// a Chichen space and the time to reach it both remain.
const SKULL_PREMIUM: f32 = 2.2;

/// Points lost per worker that goes unfed on a food day.
const STARVE_POINTS: f32 = 3.0;

/// Corn a player can expect to bring in per round, used only to decide whether
/// a food day is survivable. Deliberately pessimistic: the penalty should fire
/// on positions that are genuinely short, not on every position that is not
/// already holding the whole bill.
const CORN_INCOME_PER_ROUND: f32 = 1.9;

/// Value of one worker-action, in points. Calibrated against the space table
/// below, whose mid-range entries sit around 2-3.
const ACTION_VALUE: f32 = 2.4;

/// Rounds a worker typically spends riding a gear between actions.
const ROUNDS_PER_ACTION: f32 = 2.6;

// ---- the evaluator -----------------------------------------------------

/// Estimated final score for `p`, in points.
///
/// Exact once `g.over`: every speculative term is scaled by the rounds left.
pub fn heuristic(g: &GameState, p: PlayerId) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = pl.points as f32;

    // Liquidation, as `end_game` would compute it right now. Exact.
    v += liquidation(g, p);

    if g.over {
        return v;
    }

    let rounds_left = (LAST_DAY.saturating_sub(g.day)) as f32;
    // 1.0 at the start of the game, 0.0 on the last day. Everything speculative
    // is scaled by some function of this.
    let horizon = rounds_left / LAST_DAY as f32;

    v += held_premium(g, p, rounds_left);
    v += temple_outlook(g, p);
    v += engine_value(g, p, rounds_left, horizon);
    v += board_position(g, p, rounds_left);
    v += monument_outlook(g, p, horizon);
    v -= starvation_risk(g, p);

    v
}

/// How far ahead of the best opponent `p` stands.
///
/// This is the quantity the search maximises. Using it rather than the raw
/// estimate is what makes a move that costs an opponent more than it costs you
/// — taking the monument they were saving for, standing on a temple top,
/// emptying the skull bank — read as progress.
pub fn margin(g: &GameState, p: PlayerId) -> f32 {
    let mine = heuristic(g, p);
    let best_other = PlayerId::ALL
        .iter()
        .filter(|&&q| q != p)
        .map(|&q| heuristic(g, q))
        .fold(f32::NEG_INFINITY, f32::max);
    mine - best_other
}

// ---- components --------------------------------------------------------

/// Exactly what `GameState::end_game` would add to this player's points.
fn liquidation(g: &GameState, p: PlayerId) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = (pl.total_corn() / 4) as f32;
    v += pl.get(Resource::Skull) as f32 * 3.0;
    for id in pl.monument_ids() {
        v += (mdef(id).score)(g, p) as f32;
    }
    v
}

/// What holdings are worth *over* their liquidation value, because they can
/// still be spent. Goes to zero as the calendar runs out.
fn held_premium(g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
    let pl = &g.players[p.idx()];
    // Spending takes a turn, so the premium is gone well before the last day.
    let spendable = (rounds_left / 6.0).min(1.0);
    if spendable <= 0.0 {
        return 0.0;
    }

    let mut v = (pl.corn as f32).min(LIQUID_CORN) * CORN_PREMIUM * spendable;

    // Blocks build, and buildings are the densest points in the game. A spread
    // of block types is worth more than the same count of one type, because
    // building costs are mixed — three of each beats nine wood.
    let counts = [
        pl.get(Resource::Wood) as f32,
        pl.get(Resource::Stone) as f32,
        pl.get(Resource::Gold) as f32,
    ];
    let total: f32 = counts.iter().sum();
    v += total.min(9.0) * BLOCK_PREMIUM * spendable;
    // Reward breadth: the minimum across types is what a mixed cost is limited
    // by. Worth about one extra block each.
    let breadth = counts.iter().copied().fold(f32::INFINITY, f32::min);
    v += breadth.min(3.0) * 0.5 * spendable;

    // Skulls are Chichen fuel. Only while there is a space left to spend them on
    // and a turn in which to do it.
    let chichen_left = 9u32.saturating_sub(g.chichen_filled.count_ones()) as f32;
    if chichen_left > 0.0 {
        let usable = (pl.get(Resource::Skull) as f32).min(chichen_left);
        v += usable * SKULL_PREMIUM * spendable;
    }

    // Palenque tiles feed monuments #7 and #8 at 4 points each. Even without
    // one in hand they are a live option while monuments remain.
    if g.face_up_monuments().next().is_some() {
        v += (pl.corn_tiles + pl.wood_tiles) as f32 * 0.25 * spendable;
    }

    v
}

/// The temple payouts still on the calendar, projected from where everyone
/// stands right now.
///
/// Both kinds of scoring day are modelled: the point days pay
/// `temple_points` (which already handles the majority prize and its tie
/// split), the resource days hand out blocks and skulls. Each is discounted for
/// how far away it is, because standings move.
fn temple_outlook(g: &GameState, p: PlayerId) -> f32 {
    let mut v = 0.0;

    for (n, &d) in POINT_DAYS.iter().filter(|&&d| d > g.day).enumerate() {
        // `advance_days` pays out at the age current when the day resolves, and
        // bumps the age afterwards.
        let age = 1 + POINT_DAYS.iter().filter(|&&x| x < d).count() as u8;
        let pts = g.temple_points(p, age) as f32;
        // The nearest payout is close to locked in; a later one is a forecast.
        v += pts * if n == 0 { 0.85 } else { 0.55 };
    }

    for (n, &_d) in RESOURCE_DAYS.iter().filter(|&&d| d > g.day).enumerate() {
        let mut haul = 0.0;
        for t in Temple::ALL {
            let step = g.temple_pos(p, t);
            for &(at, r) in TEMPLES[t.idx()].resources {
                if step >= at {
                    haul += match r {
                        // A skull is worth its liquidation plus its Chichen use.
                        Resource::Skull => 3.0 + SKULL_PREMIUM,
                        other => other.corn_value() as f32 / CORN_PER_POINT + BLOCK_PREMIUM,
                    };
                }
            }
        }
        v += haul * if n == 0 { 0.85 } else { 0.55 };
    }

    // A step is worth more than its current payout when it is one short of a
    // jump, and the tracks are steep at the top. Value the *next* step so the
    // search sees climbing as progress even between scoring days.
    let mut climb = 0.0;
    for t in Temple::ALL {
        let d = &TEMPLES[t.idx()];
        let step = g.temple_pos(p, t) as usize;
        if (step + 1) < d.steps as usize {
            climb += (d.points[step + 1] - d.points[step]) as f32;
        }
    }
    v += climb * 0.10;

    v
}

/// Workers, research and the permanent food discounts: everything that converts
/// remaining rounds into points.
fn engine_value(g: &GameState, p: PlayerId, rounds_left: f32, horizon: f32) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = 0.0;

    // Each worker takes roughly one action every `ROUNDS_PER_ACTION` rounds,
    // and each action is worth about `ACTION_VALUE`. That is the whole reason
    // to buy a fourth, fifth and sixth worker — and the reason they stop being
    // worth buying near the end.
    let actions_each = (rounds_left / ROUNDS_PER_ACTION).min(6.0);
    let workers = g.n_unlocked(p) as f32;
    v += workers * actions_each * ACTION_VALUE * 0.5;

    // Feeding is a cost per food day, not per round; the permanent discounts
    // are worth the corn they save over every food day still to come.
    let food_days_left = RESOURCE_DAYS
        .iter()
        .chain(POINT_DAYS.iter())
        .filter(|&&d| d > g.day)
        .count() as f32;
    let saved = pl.free_workers as f32 * 2.0
        + (pl.worker_discount as f32).min(2.0) * workers;
    v += saved * food_days_left / CORN_PER_POINT;

    // Research. Each level is valued at what it actually pays, scaled by how
    // many actions remain to use it on.
    let uses = (rounds_left / 3.0).min(7.0);
    for s in Science::ALL {
        let lvl = g.level(p, s);
        for l in 1..=lvl {
            v += research_step_value(s, l) * uses * 0.5;
        }
        // Monument #11 pays 9/20/33 for maxed tracks and #12 pays 3 a level, so
        // a track one short of the top is worth finishing.
        if lvl == 2 && horizon > 0.15 {
            v += 0.4;
        }
    }

    // Buildings are mostly one-shot and their payoff is already in the state.
    // What survives is that three monuments count them, so the pile itself has
    // a residual value.
    v += pl.n_buildings() as f32 * 0.45;

    v
}

/// Per-use value of one research level, in points.
///
/// These are the real effects from `research.rs`, priced: an extra corn is
/// 1/4 a point plus its spending premium, an extra block rather more.
fn research_step_value(s: Science, level: u8) -> f32 {
    match (s, level) {
        // +1 corn on green Palenque spaces.
        (Science::Agriculture, 1) => 0.35,
        // +1 on blue, and irrigation: corn from an exhausted jungle space.
        (Science::Agriculture, 2) => 0.45,
        // +3 corn on green — the biggest single corn step in the game.
        (Science::Agriculture, 3) => 0.75,
        // One extra block per gather, by type. Gold is worth the most and is
        // gated behind the two below it.
        (Science::Extraction, 1) => 0.55,
        (Science::Extraction, 2) => 0.65,
        (Science::Extraction, 3) => 0.75,
        // +1 corn, then +2 points, on every build. Then a block off the price.
        (Science::Architecture, 1) => 0.30,
        (Science::Architecture, 2) => 0.85,
        (Science::Architecture, 3) => 0.80,
        // Foresight: use the next Chichen space when yours is taken.
        (Science::Theology, 1) => 0.45,
        // Devout: a block buys one more temple step alongside a skull.
        (Science::Theology, 2) => 0.65,
        // A second skull from Yaxchilan 4.
        (Science::Theology, 3) => 0.90,
        _ => 0.0,
    }
}

/// What this player's workers are standing on.
///
/// A worker's value is the action it will eventually take, not its distance
/// along a gear. A worker one space from a double build is worth far more than
/// one three spaces along Palenque, and the old `pos * 0.2` could not say so.
fn board_position(g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
    let mut v = 0.0;

    for w in g.on_board(p) {
        let Some((gear, pos)) = g.loc(w).on_board() else {
            continue;
        };
        let last = gear.size() - 1;
        // How far it can ride before the calendar ends or it falls off the top.
        let reach = last.min(pos.0.saturating_add(rounds_left as u8));

        // It can act now, or wait for something better up the gear. Waiting
        // costs tempo and risks the gear turning past the calendar's end.
        let now = space_value(g, p, gear, pos.0);
        let mut best_later: f32 = 0.0;
        for j in (pos.0 + 1)..=reach {
            best_later = best_later.max(space_value(g, p, gear, j) * 0.85);
        }
        let mut worth = now.max(best_later);

        // A worker on the top space is picked up by the next rotation whether
        // its owner wants it or not: take it this turn or lose the action.
        if pos.0 == last {
            worth *= 0.5;
        }
        // And nothing is worth anything once the calendar is done.
        if rounds_left <= 0.0 {
            worth = 0.0;
        }
        v += worth;
    }

    // The first player space: the corn pot, the marker, and the extra day.
    if let Some(w) = g.first_player_space {
        if w.owner() == p {
            v += g.accumulated_corn as f32 / CORN_PER_POINT + 1.2;
        }
    }
    // The unused first-player tile is a free calendar day, which is a whole
    // extra round of gear movement for everyone but is claimed by its holder.
    if g.players[p.idx()].may_skip_day && rounds_left > 2.0 {
        v += 0.6;
    }

    v
}

/// Point-equivalent of the action at one board space.
///
/// Hand-priced against the space generators in `src/spaces`. Gated where the
/// action needs something the player may not have: Chichen without a skull is
/// worth nothing, and neither is a build with no blocks.
fn space_value(g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
    let pl = &g.players[p.idx()];
    match gear {
        // Corn and wood. The jungle spaces climb 5/7/9 corn or 2/3/4 wood.
        Gear::Palenque => match pos {
            0 => 0.0,
            1 => 0.9,
            2 => 1.6,
            3 => 1.6,
            4 => 2.1,
            5 => 2.6,
            _ => 2.6,
        },
        // Raw resources, and the game's only repeatable skull.
        Gear::Yaxchilan => match pos {
            0 => 0.0,
            1 => 0.8,
            2 => 1.3,
            3 => 1.8,
            // A skull is 3 points liquidated and much more at Chichen — but
            // only if the bank still has one.
            4 => {
                if g.skulls_remaining > 0 {
                    3.0 + SKULL_PREMIUM * 0.5
                } else {
                    0.2
                }
            }
            5 => 2.7,
            _ => 3.4,
        },
        // Research, construction and temples: the densest gear in the game.
        Gear::Tikal => {
            let can_build = pl.n_blocks() >= 2;
            match pos {
                0 => 0.0,
                1 => 2.0,
                2 => if can_build { 3.2 } else { 0.4 },
                3 => 3.4,
                4 => if can_build { 5.2 } else { 0.6 },
                // Two temple steps, for one block.
                5 => if pl.n_blocks() >= 1 { 4.0 } else { 0.3 },
                _ => 5.2,
            }
        }
        Gear::Uxmal => match pos {
            0 => 0.0,
            1 => if pl.corn >= 3 { 2.2 } else { 0.3 },
            2 => 1.1,
            // Buying a worker is the strongest early action on the board and
            // worthless once there is no time to use it.
            3 => {
                let locked = WORKERS_PER_PLAYER - g.n_unlocked(p);
                if locked > 0 {
                    3.4
                } else {
                    0.2
                }
            }
            4 => if pl.corn >= 4 { 3.0 } else { 0.3 },
            5 => if pl.corn >= 1 { 4.0 } else { 0.2 },
            _ => 4.0,
        },
        // Chichen pays in points directly, but only for a player holding a
        // skull and only on a space nobody has used.
        Gear::Chichen => {
            if pl.get(Resource::Skull) == 0 {
                return 0.1;
            }
            if pos >= 1 && pos <= 9 && g.chichen_is_full(Pos(pos)) {
                // Foresight steps up to the next space instead.
                return if g.foresight(p) { 3.0 } else { 0.1 };
            }
            // The printed rewards, 4..13 points, less the skull spent.
            match pos {
                0 => 0.0,
                1 => 3.0,
                2 => 3.8,
                3 => 4.6,
                4 => 5.4,
                5 => 6.2,
                6 => 6.8,
                7 => 8.0,
                8 => 9.0,
                9 => 10.5,
                _ => 10.5,
            }
        }
    }
}

/// A face-up monument the player is close to affording is worth a fraction of
/// what it would score.
///
/// Monuments are the largest single scores in the game and the row is shared,
/// so "I am two gold from a 20-point card" is a real feature of a position.
fn monument_outlook(g: &GameState, p: PlayerId, horizon: f32) -> f32 {
    if horizon <= 0.10 {
        return 0.0;
    }
    let pl = &g.players[p.idx()];
    let mut best: f32 = 0.0;

    for id in g.face_up_monuments() {
        let d = mdef(id);
        // Blocks still needed. Skulls in a monument cost are counted the same
        // way; #13 wants one.
        let short: i32 = Resource::ALL
            .iter()
            .map(|&r| (d.cost[r.idx()] as i32 - pl.get(r) as i32).max(0))
            .sum();
        if short > 4 {
            continue;
        }
        // What it would pay if taken now. Self-counting monuments (#2/#4/#5,
        // which count themselves) come out one card short, which is the right
        // direction to be wrong in.
        let pays = (d.score)(g, p) as f32;
        if pays <= 0.0 {
            continue;
        }
        // Full value when it is already affordable, falling away with distance.
        best = best.max(pays * 0.45 / (1.0 + short as f32));
    }

    best * horizon.min(1.0)
}

/// Points expected to be lost to unfed workers on the next food day.
///
/// This is the term whose absence let the old evaluator buy a sixth worker with
/// two corn in hand and call it progress. The rule costs 3 points a head, which
/// is more than most single actions are worth.
fn starvation_risk(g: &GameState, p: PlayerId) -> f32 {
    let Some(next) = RESOURCE_DAYS
        .iter()
        .chain(POINT_DAYS.iter())
        .copied()
        .filter(|&d| d > g.day)
        .min()
    else {
        return 0.0;
    };

    let pl = &g.players[p.idx()];
    let mouths = g.n_unlocked(p);
    let free = (pl.free_workers as usize).min(mouths);
    let each = 2u8.saturating_sub(pl.worker_discount);
    if each == 0 {
        return 0.0;
    }
    let owed = (mouths - free) as f32 * each as f32;

    let rounds = (next - g.day) as f32;
    let expected = pl.corn as f32 + rounds * CORN_INCOME_PER_ROUND;
    let short = owed - expected;
    if short <= 0.0 {
        return 0.0;
    }
    // Every `each` corn short is one worker unfed.
    let unfed = short / each as f32;
    // Discount a shortfall that is still several rounds out — there is time to
    // fix it, but the search should still see it coming.
    let urgency = (1.0 / (1.0 + rounds * 0.25)).max(0.3);
    unfed * STARVE_POINTS * urgency
}

// ---- ranking -----------------------------------------------------------

/// A scored, ordered slice of a position's move list.
#[derive(Clone, Debug, Default)]
pub struct Ranking {
    /// Best first. Scores are [`margin`]-scale points unless a search says
    /// otherwise.
    pub moves: Vec<(Move, f32)>,
    /// Legal moves at this position, counting every spelling.
    pub total: usize,
    /// Legal moves that do something different from each other, as far as the
    /// scan could tell — `total` less the restatements it dropped. Equal to
    /// `total` when nothing was deduplicated.
    pub distinct: usize,
    /// Whether `total` is the whole move list rather than a capped walk.
    pub exhaustive: bool,
    /// What produced the ranking, for the status line.
    pub note: String,
}

impl Ranking {
    pub fn best(&self) -> Option<&Move> {
        self.moves.first().map(|(m, _)| m)
    }
}

/// Keeps the best `k` distinct moves of a stream without holding the rest.
///
/// The point of this type: the widest node measured is ~1.9M moves at 264 bytes
/// each, so the exhaustive pass has to *score* every move without ever *owning*
/// more than `k` of them. Amortised O(n): the buffer grows to `2k` and is
/// pruned by a single sort.
///
/// Distinctness is [`Move::same_effect`], not `Eq`. Without it a shortlist of
/// ten is routinely five moves written out twice — placement enumerates every
/// assignment of interchangeable workers to the same set of spaces — and the
/// search wastes most of its root beam re-deepening one position.
struct TopK {
    k: usize,
    buf: Vec<(Move, f32)>,
    /// The lowest score currently kept, once the buffer has been pruned once.
    floor: f32,
    /// Moves dropped as restatements of one already held.
    dupes: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            k: k.max(1),
            buf: Vec::with_capacity(k.max(1) * 2),
            floor: f32::NEG_INFINITY,
            dupes: 0,
        }
    }

    fn offer(&mut self, m: &Move, score: f32) {
        // The score test comes first: it rejects the overwhelming majority for
        // the price of one comparison, so the linear distinctness scan below
        // only runs on moves that were going to be kept anyway.
        if score <= self.floor {
            return;
        }
        if self.buf.iter().any(|(x, _)| x.same_effect(m)) {
            self.dupes += 1;
            return;
        }
        self.buf.push((m.clone(), score));
        if self.buf.len() >= self.k * 2 {
            self.prune();
        }
    }

    fn prune(&mut self) {
        self.buf.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.buf.truncate(self.k);
        if self.buf.len() == self.k {
            self.floor = self.buf[self.k - 1].1;
        }
    }

    fn finish(mut self) -> Vec<(Move, f32)> {
        self.buf.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.buf.truncate(self.k);
        self.buf
    }
}

/// The state a move leads to, as the next player will actually see it.
///
/// `refill_buildings` is part of `Game::play`, so leaving it out evaluates a
/// building row no player is ever shown.
#[inline]
pub fn successor(g: &GameState, p: PlayerId, m: &Move) -> GameState {
    let mut probe = *g;
    crate::moves::apply_move(&mut probe, p, m);
    probe.refill_buildings();
    probe
}

/// Rank an already-materialised move list by one-ply [`margin`].
pub fn rank(g: &GameState, p: PlayerId, moves: &[Move]) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = moves
        .iter()
        .enumerate()
        .map(|(i, m)| (i, margin(&successor(g, p, m), p)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Score **every** legal move with `score` and keep the best `keep`.
///
/// `score` is handed the successor state, already refilled. This is the
/// primitive `heuristic:full` and the search root are both built on: no cap, no
/// sampling, no traversal-order bias. It is affordable because scoring is
/// streamed — `visit_legal_moves` hands over one move at a time, it is applied
/// to a copy of the state, scored, and dropped unless it is good enough to keep.
///
/// Cost is one `score` call per legal move, so an evaluator that is not cheap
/// has no business here: the widest measured node is ~1.9M moves.
pub fn rank_all_by<F>(g: &GameState, p: PlayerId, keep: usize, mut score: F) -> Ranking
where
    F: FnMut(&GameState) -> f32,
{
    let started = std::time::Instant::now();
    let mut top = TopK::new(keep);
    let mut total = 0usize;

    let _ = crate::moves::visit_legal_moves(g, p, |m| {
        total += 1;
        top.offer(m, score(&successor(g, p, m)));
        ControlFlow::Continue(())
    });

    let ms = started.elapsed().as_secs_f64() * 1e3;
    let dupes = top.dupes;
    Ranking {
        moves: top.finish(),
        total,
        distinct: total - dupes,
        exhaustive: true,
        note: format!("all {total} moves scored in {ms:.0} ms"),
    }
}

/// [`rank_all_by`] against the built-in [`margin`].
pub fn rank_all(g: &GameState, p: PlayerId, keep: usize) -> Ranking {
    rank_all_by(g, p, keep, |s| margin(s, p))
}

/// The same, stopping after `cap` moves.
///
/// Not a sample: it takes the *first* `cap` in traversal order, which is
/// systematically "place one worker low on Palenque". Use it where a bounded
/// prefix is genuinely wanted, not where a representative slice is.
pub fn rank_capped(g: &GameState, p: PlayerId, keep: usize, cap: usize) -> Ranking {
    let mut top = TopK::new(keep);
    let mut total = 0usize;

    let flow = crate::moves::visit_legal_moves(g, p, |m| {
        total += 1;
        top.offer(m, margin(&successor(g, p, m), p));
        if total >= cap {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });

    let capped = flow.is_break();
    let dupes = top.dupes;
    Ranking {
        moves: top.finish(),
        total,
        distinct: total - dupes,
        exhaustive: !capped,
        note: if capped {
            format!("first {total} moves in traversal order")
        } else {
            format!("all {total} moves scored")
        },
    }
}
