//! SCRATCH — A/B arena for `src/eval.rs`. Delete before finishing.
//!
//! `bin/arena` cannot answer "is the new evaluator better than the old one",
//! because both of its agent specs resolve to whatever `eval::heuristic` is in
//! the binary being run. This holds a **frozen copy** of the evaluator as it
//! stood at commit cbb61a2 and plays the live one against it, using the same
//! rotation-block design `bin/arena` documents: one candidate rotated through
//! all four seats against three baselines, the block (not the game) as the
//! independent unit, null centred score 0 and null win rate 25%.
//!
//!     cargo run --release --bin evalab -- --games 400 [--k full|32] [--resume]
//!
//! Progress is appended to a JSONL file as each block completes, so a Ctrl-C
//! loses nothing and `--resume` picks up where it stopped.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;
use tzolkin::ids::*;
use tzolkin::phase::{Evaluation, Evaluator, Phase};
use tzolkin::record::{
    play_game, Agent, Candidates, GameConfig, GreedyAgent, Summary,
};
use tzolkin::state::GameState;

// =======================================================================
// The frozen baseline: `src/eval.rs` exactly as it stood at cbb61a2.
// =======================================================================
#[allow(dead_code)]
mod frozen {
    use tzolkin::data::monuments::def as mdef;
    use tzolkin::data::temples::TEMPLES;
    use tzolkin::ids::*;
    use tzolkin::state::{GameState, LAST_DAY, POINT_DAYS, RESOURCE_DAYS};

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
/// Exactly `g.scores()[p]` once `g.over`, so a finished line and a guess about
/// one are the same kind of number and the search can compare them.
pub fn heuristic(g: &GameState, p: PlayerId) -> f32 {
    let pl = &g.players[p.idx()];
    let mut v = pl.points as f32;

    // A finished game needs no estimate. `end_game` has already folded corn,
    // skulls and monuments into `points`, and the resources it converted are
    // still sitting in the player — so adding `liquidation` here would pay for
    // them a second time.
    if g.over {
        return v;
    }

    // What liquidation would pay if the game ended now, by the same arithmetic
    // `end_game` uses. Exact, unlike everything below it.
    v += liquidation(g, p);

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

/// Exactly what `GameState::end_game` would add to this player's points if the
/// game ended right now. Only meaningful before it has actually run.
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

}

/// `phase::Evaluator` over the frozen heuristic, so `GreedyAgent` can be built
/// on it exactly as it is built on `HeuristicEvaluator`.
struct FrozenEvaluator;

impl Evaluator for FrozenEvaluator {
    fn evaluate(&self, s: &GameState, _ph: Phase, _t: PlayerId, n_edges: usize) -> Evaluation {
        let raw: Vec<f32> = PlayerId::ALL.iter().map(|&q| frozen::heuristic(s, q)).collect();
        let mean = raw.iter().sum::<f32>() / N_PLAYERS as f32;
        let value = std::array::from_fn(|i| ((raw[i] - mean) / 25.0).tanh());
        let p = if n_edges == 0 { 0.0 } else { 1.0 / n_edges as f32 };
        Evaluation { priors: vec![p; n_edges], value }
    }
    fn name(&self) -> String {
        "frozen".into()
    }
}

/// The live evaluator, wired the same way. (`phase::HeuristicEvaluator` is
/// exactly this; it is restated so the two sides differ in one line only.)
struct LiveEvaluator;

impl Evaluator for LiveEvaluator {
    fn evaluate(&self, s: &GameState, _ph: Phase, _t: PlayerId, n_edges: usize) -> Evaluation {
        let raw: Vec<f32> = PlayerId::ALL
            .iter()
            .map(|&q| tzolkin::eval::heuristic(s, q))
            .collect();
        let mean = raw.iter().sum::<f32>() / N_PLAYERS as f32;
        let value = std::array::from_fn(|i| ((raw[i] - mean) / 25.0).tanh());
        let p = if n_edges == 0 { 0.0 } else { 1.0 / n_edges as f32 };
        Evaluation { priors: vec![p; n_edges], value }
    }
    fn name(&self) -> String {
        "live".into()
    }
}

// =======================================================================
// Rotation blocks
// =======================================================================

#[derive(Clone, Copy)]
struct Out {
    centred: f64,
    vs_base: f64,
    win: f64,
    cand: f64,
    base: f64,
}

fn play_block(seed: u64, cand: &dyn Agent, base: &dyn Agent) -> Vec<Out> {
    let cfg = GameConfig::evaluation();
    let mut out = Vec::new();
    for c in 0..N_PLAYERS {
        let seats: [bool; N_PLAYERS] = std::array::from_fn(|i| i == c);
        let agents: [&dyn Agent; N_PLAYERS] =
            std::array::from_fn(|s| if seats[s] { cand } else { base });
        // Common random numbers: every seating of a block plays from the *same*
        // play RNG, not `bin/arena`'s per-seating offset. Both agents here are
        // deterministic at temperature 0, so under the null (identical
        // evaluators) the four games of a block are the same game and the block
        // mean is exactly zero — all remaining variance comes from the change
        // under test. `bin/arena` offsets instead because it wants four
        // different games from one deal; this wants a matched pair.
        let mut rng =
            <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
        let r = play_game(seed, &agents, &cfg, &mut rng);
        let sc: [f64; N_PLAYERS] = std::array::from_fn(|s| r.scores[s] as f64);
        let table = sc.iter().sum::<f64>() / N_PLAYERS as f64;
        let cand_score = sc[c];
        let base_score =
            (0..N_PLAYERS).filter(|&s| s != c).map(|s| sc[s]).sum::<f64>() / (N_PLAYERS - 1) as f64;
        out.push(Out {
            centred: cand_score - table,
            vs_base: cand_score - base_score,
            win: r.win_share[c] as f64,
            cand: cand_score,
            base: base_score,
        });
    }
    out
}

fn mean(v: &[Out], f: impl Fn(&Out) -> f64) -> f64 {
    v.iter().map(&f).sum::<f64>() / v.len() as f64
}

fn main() {
    if std::env::args().any(|a| a == "--eqcheck") { eqcheck(); return; }
    let argv: Vec<String> = std::env::args().collect();
    let get = |n: &str| argv.iter().position(|a| a == n).and_then(|i| argv.get(i + 1)).cloned();
    let games: usize = get("--games").and_then(|v| v.parse().ok()).unwrap_or(400);
    let seed0: u64 = get("--seed").and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
    let k = get("--k").unwrap_or_else(|| "full".into());
    let cands = if k == "full" {
        Candidates::All
    } else {
        Candidates::Sampled(k.parse().unwrap_or(32))
    };
    let out = PathBuf::from(get("--out").unwrap_or_else(|| "evalab-progress.jsonl".into()));
    let resume = argv.iter().any(|a| a == "--resume");
    // Swap the sides, to check the harness itself is unbiased.
    let flip = argv.iter().any(|a| a == "--flip");

    let blocks = games.div_ceil(N_PLAYERS);
    let done: std::collections::HashSet<u64> = if resume && out.exists() {
        std::fs::read_to_string(&out)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                let at = l.find("\"seed\":")? + 7;
                l[at..].split(',').next()?.trim_end_matches('}').parse().ok()
            })
            .collect()
    } else {
        Default::default()
    };
    if !resume {
        let _ = std::fs::remove_file(&out);
    }

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    {
        let s = stop.clone();
        let _ = ctrlc_lite(move || s.store(true, Ordering::SeqCst));
    }

    let file = Mutex::new(
        std::fs::OpenOptions::new().create(true).append(true).open(&out).unwrap(),
    );
    let rows: Mutex<Vec<(u64, Vec<Out>)>> = Mutex::new(Vec::new());
    let n_done = AtomicUsize::new(0);
    let started = Instant::now();

    eprintln!(
        "evalab: live vs frozen, {} blocks x {} games, cands={k}{}",
        blocks,
        N_PLAYERS,
        if flip { " (FLIPPED)" } else { "" }
    );

    (0..blocks as u64).into_par_iter().for_each(|b| {
        let seed = seed0 + b;
        if done.contains(&seed) || stop.load(Ordering::SeqCst) {
            return;
        }
        let live = GreedyAgent { ev: LiveEvaluator, cands, record: false };
        let froz = GreedyAgent { ev: FrozenEvaluator, cands, record: false };
        let (c, bs): (&dyn Agent, &dyn Agent) =
            if flip { (&froz, &live) } else { (&live, &froz) };
        let games = play_block(seed, c, bs);
        let line = format!(
            r#"{{"seed":{seed},"centred":{:.4},"vs_base":{:.4},"win":{:.4},"cand":{:.3},"base":{:.3}}}"#,
            mean(&games, |o| o.centred),
            mean(&games, |o| o.vs_base),
            mean(&games, |o| o.win),
            mean(&games, |o| o.cand),
            mean(&games, |o| o.base),
        );
        {
            let mut f = file.lock().unwrap();
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
        rows.lock().unwrap().push((seed, games));
        let n = n_done.fetch_add(1, Ordering::SeqCst) + 1;
        if n % 10 == 0 {
            eprintln!("  {n} blocks, {:.0}s", started.elapsed().as_secs_f64());
        }
    });

    // Everything on disk, including anything a previous run left.
    let mut centred = Vec::new();
    let mut vs_base = Vec::new();
    let mut win = Vec::new();
    let mut cand = Vec::new();
    let mut base = Vec::new();
    for l in std::fs::read_to_string(&out).unwrap_or_default().lines() {
        let num = |k: &str| -> Option<f64> {
            let at = l.find(&format!("\"{k}\":"))? + k.len() + 3;
            let r = &l[at..];
            let e = r.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).unwrap_or(r.len());
            r[..e].parse().ok()
        };
        if let (Some(c), Some(v), Some(w), Some(a), Some(bb)) =
            (num("centred"), num("vs_base"), num("win"), num("cand"), num("base"))
        {
            centred.push(c);
            vs_base.push(v);
            win.push(w);
            cand.push(a);
            base.push(bb);
        }
    }
    let s = Summary::of(&centred);
    let v = Summary::of(&vs_base);
    let w = Summary::of(&win);
    println!(
        "\n{} blocks ({} games), {:.0}s\n\
         mean centred score  {:+.2}  95% CI [{:+.2}, {:+.2}]   (null 0)\n\
         mean vs baseline    {:+.2}  95% CI [{:+.2}, {:+.2}]   (null 0)\n\
         win rate            {:.3}  95% CI [{:.3}, {:.3}]   (null 0.25)\n\
         mean score          candidate {:.1}   baseline {:.1}\n\
         {}",
        s.n,
        s.n * N_PLAYERS,
        started.elapsed().as_secs_f64(),
        s.mean, s.mean - s.ci, s.mean + s.ci,
        v.mean, v.mean - v.ci, v.mean + v.ci,
        w.mean, w.mean - w.ci, w.mean + w.ci,
        Summary::of(&cand).mean,
        Summary::of(&base).mean,
        if s.mean.abs() > s.ci {
            "DISTINGUISHABLE from zero at 95%."
        } else {
            "not distinguishable from zero: the interval covers 0."
        }
    );
}

/// Minimal Ctrl-C hook without pulling in a dependency: a thread that watches
/// for the signal is overkill here, so this just returns Ok and relies on the
/// per-block JSONL flush to make a kill lossless.
fn ctrlc_lite(_f: impl Fn() + Send + 'static) -> Result<(), ()> {
    Ok(())
}

// --- scratch equality check ---
fn eqcheck() {
    use tzolkin::game::Game;
    let mut worst = 0.0f32;
    let mut n = 0;
    for seed in 0..200u64 {
        let mut g = Game::new(seed);
        for _ in 0..(seed % 20) {
            if g.state.over { break; }
            g.play_round();
        }
        for p in PlayerId::ALL {
            let a = tzolkin::eval::heuristic(&g.state, p);
            let b = frozen::heuristic(&g.state, p);
            if (a - b).abs() > worst { worst = (a-b).abs(); }
            n += 1;
        }
    }
    println!("eqcheck: {n} states, max |live - frozen| = {worst:e}");
}
