//! `GameState` -> `[f32; D_IN]`, and `Choice` -> `[f32; D_CHOICE]`.
//!
//! Written against `docs/LEARNING.md` §1 and §2.3. Three properties are load
//! bearing and everything else follows from them:
//!
//! **Flat, not spatial.** The gears are not a board. A convolution shares
//! weights across positions, which is right exactly when positions are
//! exchangeable; Tikal 3 and Tikal 4 are unrelated actions and Palenque and
//! Chichen Itza are unrelated gears. So block A is 55 independent cells and the
//! one genuine sequence structure — rotation, and falling off the top — is
//! handed over as an explicit channel rather than left to be rediscovered.
//!
//! **Perspective-relative.** Everything per-player is rotated by
//! `(owner - p) mod 4`, so one set of weights reads every seat and the same
//! position can be encoded four times for four value targets. `Color` is never
//! encoded: it is a display attribute and encoding it would break the rotation.
//!
//! **Blind to the deck.** `Deck::ids` holds the *undrawn* cards in shuffled
//! order. A network fed that reads the future, and an agent trained on it does
//! not transfer to a table with humans at it. Nothing here touches `.ids` — it
//! gets the cursors, which say how many cards have left, and the unseen mask,
//! `ALL \ (face_up ∪ owned)`, which any player at the table can compute.
//!
//! `tests/encode.rs` pins this twice, by permuting the undrawn tail and
//! asserting bit-identical output: `deck_order_is_invisible` for the state
//! encoder, and `deck_order_is_invisible_to_choice_features` for
//! [`encode_choice`], which is the easier place to leak from — it *applies* the
//! candidate to a probe copy, so effects that refill the display turn a
//! face-down card face-up inside the probe. Nothing summarised there depends on
//! it, and the test is what keeps that true.
//!
//! Sizes track the engine: `MAX_GEAR_SPACES` has already moved twice (8/11 ->
//! 7/10 -> 8/11) and block A is sized from it at compile time. `D_IN` is held
//! fixed at 3072 by a reserved zero tail that absorbs the difference, so that
//! kind of change costs a recompile rather than a discarded checkpoint. The
//! reserved tail is wired into the global stem (see `net.rs`) precisely so that
//! a feature added there trains without changing any weight shape.

use crate::data::buildings::{def as bdef, Payoff, N_BUILDINGS};
use crate::data::monuments::{def as mdef, N_MONUMENTS};
use crate::data::temples::TEMPLES;
use crate::data::tiles::TILES;
use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::moves::{beg_options, choices_for_worker, pity_moves, Placement};
use crate::phase::{ModeChoice, Phase, Step};
use crate::state::{GameState, MAX_GEAR_SPACES, N_DISPLAY, N_SKULLS, POINT_DAYS, RESOURCE_DAYS};
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

pub const N_GEARS: usize = 5;
/// One cell per (gear, position) pair, including the ones that do not exist on
/// the smaller gears. Channel 0 says which are real; keeping the rectangle means
/// the block is a compile-time constant rather than a ragged offset table.
pub const N_CELLS: usize = N_GEARS * MAX_GEAR_SPACES;
pub const CELL_CH: usize = 12;

pub const BOARD_W: usize = N_CELLS * CELL_CH;
pub const PLAYER_W: usize = 282;
pub const PLAYERS_W: usize = N_PLAYERS * PLAYER_W;
pub const GLOBAL_W: usize = 254;
pub const BSLOT_W: usize = 73;
pub const BSLOTS_W: usize = N_DISPLAY * BSLOT_W;
pub const MSLOT_W: usize = 45;
pub const MSLOTS_W: usize = N_DISPLAY * MSLOT_W;

pub const BOARD_OFF: usize = 0;
pub const PLAYERS_OFF: usize = BOARD_OFF + BOARD_W;
pub const GLOBAL_OFF: usize = PLAYERS_OFF + PLAYERS_W;
pub const BSLOTS_OFF: usize = GLOBAL_OFF + GLOBAL_W;
pub const MSLOTS_OFF: usize = BSLOTS_OFF + BSLOTS_W;
pub const RESERVED_OFF: usize = MSLOTS_OFF + MSLOTS_W;

/// Held fixed across rules changes on purpose. See the module docs.
pub const D_IN: usize = 3072;
pub const RESERVED_W: usize = D_IN - RESERVED_OFF;

/// Per-candidate width for the `Take` / `DraftTile` pointer head.
pub const D_CHOICE: usize = 96;

/// Arity of the cell head: one slot per cell plus `STOP`.
pub const N_WHO: usize = N_CELLS + 1;
/// `[Palenque, Yaxchilan, Tikal, Uxmal, Chichen, FirstPlayer, STOP]`.
pub const N_PLACE: usize = 7;

const _: () = {
    assert!(RESERVED_OFF < D_IN, "the live blocks overflowed D_IN");
    // 322 spare dimensions at the current geometry. If a rules change eats
    // these, D_IN moves and every checkpoint is discarded; that is the tradeoff
    // the reserve exists to postpone.
    assert!(RESERVED_W >= 64, "reserved tail nearly exhausted; raise D_IN");
};

// ---------------------------------------------------------------------------
// Feature writer
// ---------------------------------------------------------------------------

/// A cursor over the output slice.
///
/// Every block asserts its own width on the way out, so a mis-sized feature
/// shifts nothing downstream: it fails at the block boundary in the debug build
/// and in `tests/encode.rs`. Hand-maintained offset constants for a 3,072-wide
/// vector would not survive a week of edits.
struct W<'a> {
    buf: &'a mut [f32],
    i: usize,
}

impl<'a> W<'a> {
    #[inline]
    fn new(buf: &'a mut [f32]) -> Self {
        W { buf, i: 0 }
    }

    #[inline]
    fn f(&mut self, v: f32) {
        self.buf[self.i] = v;
        self.i += 1;
    }

    #[inline]
    fn flag(&mut self, b: bool) {
        self.f(if b { 1.0 } else { 0.0 });
    }

    /// One-hot over `n` slots; `k >= n` or `None` writes all zeros.
    #[inline]
    fn onehot(&mut self, k: Option<usize>, n: usize) {
        let base = self.i;
        if let Some(k) = k {
            if k < n {
                self.buf[base + k] = 1.0;
            }
        }
        self.i += n;
    }

    /// `x >= t` for a ladder of thresholds.
    ///
    /// Every threshold in the rules — `corn < 3` gates begging, `corn >= 3` the
    /// Uxmal temple buy, `corn >= price` each build — becomes linearly separable
    /// at the first layer. A single normalised magnitude makes a small net find
    /// those slowly if at all, and the extra width is free.
    #[inline]
    fn therm(&mut self, x: f32, ts: &[f32]) {
        for &t in ts {
            self.flag(x >= t);
        }
    }

    /// `n` may exceed the mask's width: the padding bits are the room a new
    /// card leaves in an existing checkpoint (32 buildings written into 40, 13
    /// monuments into 16), and they read as zero.
    #[inline]
    fn bits(&mut self, mask: u32, n: usize) {
        for b in 0..n {
            self.flag(b < 32 && mask & (1u32 << b) != 0);
        }
    }

    #[inline]
    fn skip(&mut self, n: usize) {
        self.i += n;
    }

    #[inline]
    fn done(self, expect: usize) {
        debug_assert_eq!(self.i, expect, "block width mismatch");
    }
}

const CORN_T: [f32; 16] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 25.0, 30.0, 40.0,
];
const RES_T: [f32; 7] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0];
const POINT_T: [f32; 13] = [
    -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 140.0,
];
const SMALL_T: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

// ---------------------------------------------------------------------------
// The state encoder
// ---------------------------------------------------------------------------

/// Perspective-relative encoding of `g` as seen by `p`, at `phase`.
///
/// `p` need not be `g.current`: the seat offset of `g.current` from `p` is one
/// of the features, which is what lets one trunk pass answer for all four
/// players and what makes the 4x perspective augmentation on the value head
/// legitimate rather than a lie.
///
/// `phase` is not optional. A mid-turn state — two workers placed, deciding the
/// third — is a perfectly good `GameState`, but nothing in it says whether we
/// are choosing a gear, choosing a worker to pick up, or choosing that worker's
/// action.
pub fn encode(g: &GameState, p: PlayerId, phase: Phase, out: &mut [f32]) {
    assert_eq!(out.len(), D_IN, "encode buffer must be exactly D_IN wide");
    out.fill(0.0);

    encode_board(g, p, &mut out[BOARD_OFF..BOARD_OFF + BOARD_W]);
    for s in 0..N_PLAYERS {
        let base = PLAYERS_OFF + s * PLAYER_W;
        encode_player(g, p.next(s), &mut out[base..base + PLAYER_W]);
    }
    encode_global(g, p, phase, &mut out[GLOBAL_OFF..GLOBAL_OFF + GLOBAL_W]);
    for i in 0..N_DISPLAY {
        let base = BSLOTS_OFF + i * BSLOT_W;
        encode_building_slot(g, p, g.buildings_up[i], &mut out[base..base + BSLOT_W]);
    }
    for i in 0..N_DISPLAY {
        let base = MSLOTS_OFF + i * MSLOT_W;
        encode_monument_slot(g, p, g.monuments_up[i], &mut out[base..base + MSLOT_W]);
    }
    // The reserved tail stays zero. It is wired into the global stem so that a
    // feature added here trains against an existing checkpoint.
}

/// Allocating form, for tests and one-off calls.
pub fn encoded(g: &GameState, p: PlayerId, phase: Phase) -> Vec<f32> {
    let mut v = vec![0.0; D_IN];
    encode(g, p, phase, &mut v);
    v
}

// ---- block A: the gears ------------------------------------------------

fn encode_board(g: &GameState, p: PlayerId, out: &mut [f32]) {
    let mut w = W::new(out);
    for gear in Gear::ALL {
        let size = gear.size() as usize;
        let lowest = g.lowest_free(gear);
        for pos in 0..MAX_GEAR_SPACES {
            let exists = pos < size;
            let occ = if exists {
                g.gears[gear.idx()].at(Pos(pos as u8))
            } else {
                None
            };

            w.flag(exists);
            w.flag(exists && occ.is_none());
            // Ownership rotated into perspective order: slot 0 is always mine.
            // Worker *identity* within a player is never encoded — `place_rec`
            // consumes available workers in id order precisely because they are
            // interchangeable, so identity carries no information.
            let owner_off = occ.map(|wk| seat_off(p, wk.owner()));
            for k in 0..N_PLAYERS {
                w.flag(owner_off == Some(k));
            }
            // The unique legal placement target on this gear: a worker always
            // enters on `lowest_free`, so choosing a gear chooses the space.
            w.flag(exists && lowest == Some(Pos(pos as u8)));
            // Rotation is a nonlinear function of (pos, gear.size()), so the
            // rounds-before-falling-off count is handed over rather than left
            // for the net to rediscover from two one-hots.
            w.f(if exists {
                (size - 1 - pos) as f32 / 10.0
            } else {
                0.0
            });
            w.f(pos as f32 / 10.0);
            w.flag(gear == Gear::Chichen && exists && g.chichen_is_full(Pos(pos as u8)));
            let (pc, pw) = if gear == Gear::Palenque && pos < g.palenque.len() {
                (g.palenque[pos].corn as f32, g.palenque[pos].wood as f32)
            } else {
                (0.0, 0.0)
            };
            w.f(pc / 4.0);
            w.f(pw / 4.0);
        }
    }
    w.done(BOARD_W);
}

// ---- block B: one player -----------------------------------------------

fn encode_player(g: &GameState, q: PlayerId, out: &mut [f32]) {
    let pl = &g.players[q.idx()];
    let mut w = W::new(out);

    // corn (17)
    w.f((pl.corn as f32).min(60.0) / 60.0);
    w.therm(pl.corn as f32, &CORN_T);

    // resources (32)
    for r in Resource::ALL {
        let v = pl.get(r) as f32;
        w.f(v.min(12.0) / 12.0);
        w.therm(v, &RES_T);
    }

    // points (14)
    w.f((pl.points as f32 - 60.0) / 60.0);
    w.therm(pl.points as f32, &POINT_T);

    // corn / wood tiles (14)
    for v in [pl.corn_tiles, pl.wood_tiles] {
        w.f(v as f32 / 6.0);
        w.therm(v as f32, &SMALL_T);
    }

    // free workers (7), worker discount (3), first-player tile (1)
    w.f(pl.free_workers as f32 / 6.0);
    w.therm(pl.free_workers as f32, &SMALL_T);
    w.onehot(Some((pl.worker_discount as usize).min(2)), 3);
    w.flag(pl.may_skip_day);

    // holdings bitsets, padded past the current card counts (40 + 16)
    w.bits(pl.buildings, 40);
    w.bits(pl.monuments as u32, 16);

    // research (20)
    for s in Science::ALL {
        w.onehot(Some(g.level(q, s) as usize), 4);
    }
    for s in Science::ALL {
        w.f(g.level(q, s) as f32 / 3.0);
    }

    // temples (45). Rank, not just position: `gain_temple_points` pays a prize
    // to everyone level with the highest player and splits it, so the value of a
    // step is relative. Handing over "alone at the top / tied / mid / bottom"
    // beats asking a 512-wide trunk to compute three argmaxes.
    for t in Temple::ALL {
        w.onehot(Some(g.temple_pos(q, t) as usize), 9);
    }
    for t in Temple::ALL {
        w.f(g.temple_pos(q, t) as f32 / 8.0);
    }
    for t in Temple::ALL {
        let mine = g.temple_pos(q, t);
        let hi = PlayerId::ALL
            .iter()
            .map(|&o| g.temple_pos(o, t))
            .max()
            .unwrap_or(0);
        let lo = PlayerId::ALL
            .iter()
            .map(|&o| g.temple_pos(o, t))
            .min()
            .unwrap_or(0);
        let tied = PlayerId::ALL
            .iter()
            .filter(|&&o| g.temple_pos(o, t) == hi)
            .count();
        let rank = if mine == hi && tied == 1 {
            0
        } else if mine == hi {
            1
        } else if mine > lo {
            2
        } else {
            3
        };
        w.onehot(Some(rank), 4);
    }
    for t in Temple::ALL {
        let mine = g.temple_pos(q, t) as f32;
        let hi = PlayerId::ALL
            .iter()
            .map(|&o| g.temple_pos(o, t))
            .max()
            .unwrap_or(0) as f32;
        w.f((hi - mine) / 8.0);
    }

    // workers (47)
    let mut avail = 0usize;
    let mut board = 0usize;
    let mut locked = 0usize;
    let mut on_first = false;
    let mut per_gear = [0usize; N_GEARS];
    for wk in GameState::worker_ids(q) {
        match g.loc(wk) {
            crate::state::WorkerLoc::Locked => locked += 1,
            crate::state::WorkerLoc::Available => avail += 1,
            crate::state::WorkerLoc::FirstPlayerSpace => on_first = true,
            crate::state::WorkerLoc::OnGear { gear, .. } => {
                board += 1;
                per_gear[gear.idx()] += 1;
            }
        }
    }
    w.onehot(Some(avail.min(6)), 7);
    w.onehot(Some(board.min(6)), 7);
    w.onehot(Some(locked.min(6)), 7);
    w.flag(on_first);
    for n in per_gear {
        w.onehot(Some(n.min(4)), 5);
    }

    // seat role (1)
    w.flag(q == g.first_player);

    // feeding (17)
    let mouths = g.n_unlocked(q);
    let each = 2u8.saturating_sub(pl.worker_discount) as usize;
    let to_pay = mouths.saturating_sub(pl.free_workers as usize);
    let bill = to_pay * each;
    let shortfall = if each == 0 {
        0
    } else {
        to_pay.saturating_sub(pl.corn as usize / each)
    };
    w.onehot(Some(mouths.min(6)), 7);
    w.f(bill as f32 / 12.0);
    w.f((pl.corn as f32 - bill as f32) / 12.0);
    w.flag(shortfall > 0);
    w.onehot(Some(shortfall.min(6)), 7);

    // projections (5): what the calendar already owes this player
    let pt_age = if g.day < POINT_DAYS[0] { 1 } else { 2 };
    w.f(g.temple_points(q, pt_age) as f32 / 20.0);
    let mut due = [0f32; 4];
    for t in Temple::ALL {
        let step = g.temple_pos(q, t);
        for &(at, r) in TEMPLES[t.idx()].resources {
            if step >= at {
                due[r.idx()] += 1.0;
            }
        }
    }
    for r in Resource::ALL {
        w.f(due[r.idx()] / 2.0);
    }

    // holdings value (3)
    w.f(pl.total_corn() as f32 / 60.0);
    let mscore: i32 = pl.monument_ids().map(|id| (mdef(id).score)(g, q)).sum();
    w.f(mscore as f32 / 40.0);
    w.f(pl.get(Resource::Skull) as f32 * 3.0 / 20.0);

    w.done(PLAYER_W);
}

// ---- block C: global ---------------------------------------------------

fn encode_global(g: &GameState, p: PlayerId, phase: Phase, out: &mut [f32]) {
    let mut w = W::new(out);

    // calendar (34 + 2 + 12 + 1)
    w.onehot(Some(g.day as usize), 32);
    w.f(g.day as f32 / 27.0);
    w.f((crate::state::LAST_DAY.saturating_sub(g.day)) as f32 / 27.0);
    w.onehot(Some((g.age as usize).saturating_sub(1).min(1)), 2);
    let (to_food, food_kind) = next_food_day(g.day);
    w.onehot(Some(to_food.min(8)), 9);
    w.onehot(food_kind, 3);
    w.flag(g.day + 1 >= crate::state::LAST_DAY);

    // accumulated corn (14)
    w.f(g.accumulated_corn as f32 / 12.0);
    for t in 0..13 {
        w.flag(g.accumulated_corn as usize >= t);
    }

    // seats (5 + 4 + 4 + 4)
    match g.first_player_space {
        None => w.onehot(Some(0), 5),
        Some(wk) => w.onehot(Some(1 + seat_off(p, wk.owner())), 5),
    }
    w.onehot(Some(seat_off(p, g.first_player)), 4);
    w.onehot(Some(seat_off(p, g.current)), 4);
    w.onehot(Some(seat_off(g.first_player, g.current)), 4);

    // supply (15 + 12 + 8)
    w.f(g.skulls_remaining as f32 / N_SKULLS as f32);
    for t in 0..14 {
        w.flag(g.skulls_remaining as usize >= t);
    }
    w.bits(g.chichen_filled as u32, MAX_GEAR_SPACES);
    w.f(g.chichen_filled.count_ones() as f32 / 9.0);
    let corn_left: u32 = g.palenque.iter().map(|s| s.corn as u32).sum();
    let wood_left: u32 = g.palenque.iter().map(|s| s.wood as u32).sum();
    w.f(corn_left as f32 / 16.0);
    w.f(wood_left as f32 / 12.0);
    for pos in 2..=5usize {
        w.flag(g.palenque[pos].corn_showing());
    }
    w.flag(corn_left > 0);
    w.flag(wood_left > 0);

    // ---- hidden information ------------------------------------------
    //
    // The cursors and the unseen set, and nothing else. `Deck::ids` past
    // `next` is the shuffled future; reading it trains a cheat that cannot
    // transfer to a table with a real deck on it.
    let mut seen_b: u32 = 0;
    for id in g.face_up_buildings() {
        seen_b |= 1 << (id.0 - 1);
    }
    for pl in &g.players {
        seen_b |= pl.buildings;
    }
    let all_b: u32 = if N_BUILDINGS >= 32 {
        u32::MAX
    } else {
        (1u32 << N_BUILDINGS) - 1
    };
    let unseen_b = all_b & !seen_b;
    w.bits(unseen_b, 40);
    w.f(unseen_b.count_ones() as f32 / N_BUILDINGS as f32);

    let mut seen_m: u32 = 0;
    for id in g.face_up_monuments() {
        seen_m |= 1 << (id.0 - 1);
    }
    for pl in &g.players {
        seen_m |= pl.monuments as u32;
    }
    let unseen_m = ((1u32 << N_MONUMENTS) - 1) & !seen_m;
    w.bits(unseen_m, 16);
    w.f(unseen_m.count_ones() as f32 / N_MONUMENTS as f32);

    w.f(g.age1.next as f32 / 14.0);
    w.f(g.age2.next as f32 / 18.0);
    w.f(g.monument_deck.next as f32 / 13.0);

    // ---- phase (78) ---------------------------------------------------
    w.onehot(Some(phase.tag() as usize), Phase::COUNT);
    w.onehot(
        match phase {
            Phase::Placing { n } => Some((n as usize).min(6)),
            _ => None,
        },
        7,
    );
    w.onehot(
        match phase {
            Phase::Take { worker } => g.loc(worker).on_board().map(|(gr, ps)| cell(gr, ps)),
            _ => None,
        },
        N_WHO,
    );
    w.onehot(
        match phase {
            Phase::ExtraDay { claimer } => Some(seat_off(p, claimer)),
            _ => None,
        },
        4,
    );
    w.onehot(
        match phase {
            Phase::DraftTile { kept, .. } => Some((kept.count_ones() as usize).min(2)),
            _ => None,
        },
        3,
    );

    w.done(GLOBAL_W);
}

// ---- display slots -----------------------------------------------------

fn payoff_kind(pay: &Payoff) -> usize {
    match pay {
        Payoff::Fixed(_) => 0,
        Payoff::FreeTrack(..) => 1,
        Payoff::FreeAny(..) => 2,
        Payoff::BuildAnother => 3,
        Payoff::CornExchange(_) => 4,
        Payoff::Mirror(_) => 5,
    }
}

fn color_idx(c: Color) -> usize {
    match c {
        Color::Red => 0,
        Color::Green => 1,
        Color::Blue => 2,
        Color::Yellow => 3,
    }
}

/// Nothing here says what a card *does* beyond its payoff kind: a dense layer
/// over the slot already has per-id weights, so "building 22 is the corn
/// exchange" is learned rather than transcribed. What is worth transcribing is
/// what needs a *join* — who can afford it — because that is a comparison the
/// trunk would otherwise have to synthesise from two distant blocks.
fn encode_building_slot(g: &GameState, p: PlayerId, id: Option<BuildingId>, out: &mut [f32]) {
    let mut w = W::new(out);
    let Some(id) = id else {
        w.skip(BSLOT_W - 1);
        w.flag(true);
        w.done(BSLOT_W);
        return;
    };
    let d = bdef(id);
    w.onehot(Some((id.0 - 1) as usize), 40);
    for r in Resource::ALL {
        w.f(d.cost[r.idx()] as f32 / 3.0);
    }
    for r in Resource::BLOCKS {
        w.therm(d.cost[r.idx()] as f32, &[1.0, 2.0, 3.0, 4.0]);
    }
    w.onehot(Some(color_idx(d.color)), 4);
    w.onehot(Some(payoff_kind(&d.payoff)), 8);
    for s in 0..N_PLAYERS {
        w.flag(g.players[p.next(s).idx()].can_pay(d.cost));
    }
    w.flag(false);
    w.done(BSLOT_W);
}

/// The monument slot carries `score(&GameState, seat)` evaluated right now for
/// all four seats. Those functions are pure reads and calling six of them four
/// times is a few hundred nanoseconds; handing the net "monument 11 is worth 20
/// to me and 9 to the leader" is far cheaper than teaching it `TABLE[maxed]`.
fn encode_monument_slot(g: &GameState, p: PlayerId, id: Option<MonumentId>, out: &mut [f32]) {
    let mut w = W::new(out);
    let Some(id) = id else {
        w.skip(16 + 4 + 12);
        w.flag(true);
        w.skip(MSLOT_W - (16 + 4 + 12 + 1));
        w.done(MSLOT_W);
        return;
    };
    let d = mdef(id);
    w.onehot(Some((id.0 - 1) as usize), 16);
    for r in Resource::ALL {
        w.f(d.cost[r.idx()] as f32 / 4.0);
    }
    for r in Resource::BLOCKS {
        w.therm(d.cost[r.idx()] as f32, &[1.0, 2.0, 3.0, 4.0]);
    }
    w.flag(false);
    w.onehot(Some(color_idx(d.color)), 4);
    for s in 0..N_PLAYERS {
        w.flag(g.players[p.next(s).idx()].can_pay(d.cost));
    }
    for s in 0..N_PLAYERS {
        w.f((d.score)(g, p.next(s)) as f32 / 20.0);
    }
    w.done(MSLOT_W);
}

// ---- shared helpers ----------------------------------------------------

#[inline]
pub fn seat_off(p: PlayerId, q: PlayerId) -> usize {
    (q.idx() + N_PLAYERS - p.idx()) % N_PLAYERS
}

#[inline]
pub fn cell(gear: Gear, pos: Pos) -> usize {
    gear.idx() * MAX_GEAR_SPACES + pos.idx()
}

/// Rounds until the next food day, and what kind it is: 0 resources, 1 points.
fn next_food_day(day: u8) -> (usize, Option<usize>) {
    let mut best: Option<(u8, usize)> = None;
    for (kind, days) in [(0usize, &RESOURCE_DAYS), (1usize, &POINT_DAYS)] {
        for &d in days {
            if d > day && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, kind));
            }
        }
    }
    match best {
        Some((d, kind)) => ((d - day) as usize, Some(kind)),
        None => (8, Some(2)),
    }
}

// ---------------------------------------------------------------------------
// The candidate featuriser
// ---------------------------------------------------------------------------

/// Everything the choice featuriser measures, read off a state.
#[derive(Clone, Copy)]
struct Summary {
    corn: i32,
    res: [i32; 4],
    points: i32,
    temple: [i32; 3],
    research: [i32; 4],
    free_workers: i32,
    discount: i32,
    available: i32,
    corn_tiles: i32,
    wood_tiles: i32,
    pal_corn: i32,
    pal_wood: i32,
    chichen: u16,
    skulls_left: i32,
    proj_points: i32,
    proj_res: [i32; 4],
    feed_short: i32,
}

fn summarise(g: &GameState, p: PlayerId) -> Summary {
    let pl = &g.players[p.idx()];
    let pt_age = if g.day < POINT_DAYS[0] { 1 } else { 2 };
    let mut proj_res = [0i32; 4];
    for t in Temple::ALL {
        let step = g.temple_pos(p, t);
        for &(at, r) in TEMPLES[t.idx()].resources {
            if step >= at {
                proj_res[r.idx()] += 1;
            }
        }
    }
    let mouths = g.n_unlocked(p);
    let each = 2u8.saturating_sub(pl.worker_discount) as usize;
    let to_pay = mouths.saturating_sub(pl.free_workers as usize);
    let feed_short = if each == 0 {
        0
    } else {
        to_pay.saturating_sub(pl.corn as usize / each) as i32
    };

    Summary {
        corn: pl.corn as i32,
        res: std::array::from_fn(|i| pl.res[i] as i32),
        points: pl.points as i32,
        temple: std::array::from_fn(|i| g.temples[i][p.idx()] as i32),
        research: std::array::from_fn(|i| g.research[p.idx()][i] as i32),
        free_workers: pl.free_workers as i32,
        discount: pl.worker_discount as i32,
        available: g.available(p).count() as i32,
        corn_tiles: pl.corn_tiles as i32,
        wood_tiles: pl.wood_tiles as i32,
        pal_corn: g.palenque.iter().map(|s| s.corn as i32).sum(),
        pal_wood: g.palenque.iter().map(|s| s.wood as i32).sum(),
        chichen: g.chichen_filled,
        skulls_left: g.skulls_remaining as i32,
        proj_points: g.temple_points(p, pt_age) as i32,
        proj_res,
        feed_short,
    }
}

/// Delta features for one candidate at `(g, p)`.
///
/// Computed as `probe = *g; c.apply(&mut probe, p);` and then differencing a
/// fixed set of summaries — **not** by parsing `Effect` variants. That is the
/// single most rules-robust decision in the design: adding an `Effect`,
/// changing what a space grants, or making the market two-way costs this head
/// nothing, because it measures what actually happened rather than what was
/// requested. `take_skulls` clamping is the standing example: `Res(Skull, +2)`
/// against an empty bank is worth zero, and only the delta knows that.
///
/// The one exception is the "wasted" flags at 36-38, which need the *intent* to
/// compare the outcome against, and so do read `Effect::TempleStep`.
pub fn encode_choice(g: &GameState, p: PlayerId, c: &Choice, out: &mut [f32]) {
    assert_eq!(out.len(), D_CHOICE);
    out.fill(0.0);

    let before = summarise(g, p);
    let mut probe = *g;
    c.apply(&mut probe, p);
    let after = summarise(&probe, p);

    let mut requested = [0i32; 3];
    for e in &c.0 {
        if let Effect::TempleStep(t, d) = *e {
            requested[t.idx()] += d as i32;
        }
    }
    let built = c.0.iter().find_map(|e| match e {
        Effect::Build(id) => Some(*id),
        _ => None,
    });
    let took = c.0.iter().find_map(|e| match e {
        Effect::TakeMonument(id) => Some(*id),
        _ => None,
    });

    let d_corn = after.corn - before.corn;
    let d_res: [i32; 4] = std::array::from_fn(|i| after.res[i] - before.res[i]);
    let d_points = after.points - before.points;

    let mut w = W::new(out);
    // 0-2
    w.f(1.0);
    w.flag(c.is_skip());
    w.f(c.0.len() as f32 / 8.0);
    // 3-9
    w.f(d_corn as f32 / 10.0);
    w.therm(d_corn as f32, &[-8.0, -4.0, -1.0, 1.0, 4.0, 8.0]);
    // 10-25
    for i in 0..4 {
        w.f(d_res[i] as f32 / 3.0);
        w.therm(d_res[i] as f32, &[-1.0, 1.0, 2.0]);
    }
    // 26-32
    w.f(d_points as f32 / 10.0);
    w.therm(d_points as f32, &[-4.0, -1.0, 1.0, 4.0, 8.0, 13.0]);
    // 33-38
    for i in 0..3 {
        w.f((after.temple[i] - before.temple[i]) as f32 / 2.0);
    }
    for i in 0..3 {
        let moved = after.temple[i] - before.temple[i];
        w.flag(requested[i] != 0 && moved != requested[i]);
    }
    // 39-42
    for i in 0..4 {
        w.f((after.research[i] - before.research[i]) as f32);
    }
    // 43-47
    w.f((after.free_workers - before.free_workers) as f32);
    w.f((after.discount - before.discount) as f32);
    w.f((after.available - before.available) as f32);
    w.f((after.corn_tiles - before.corn_tiles) as f32);
    w.f((after.wood_tiles - before.wood_tiles) as f32);
    // 48-49: separates a burn from a take
    w.f((after.pal_corn - before.pal_corn) as f32);
    w.f((after.pal_wood - before.pal_wood) as f32);
    // 50-61
    let newly = after.chichen & !before.chichen;
    w.f(newly.count_ones() as f32);
    w.bits(newly as u32, MAX_GEAR_SPACES);
    // 62-76
    w.flag(built.is_some());
    for r in Resource::BLOCKS {
        w.f(built.map_or(0.0, |id| bdef(id).cost[r.idx()] as f32 / 3.0));
    }
    w.onehot(built.map(|id| color_idx(bdef(id).color)), 4);
    w.onehot(built.map(|id| payoff_kind(&bdef(id).payoff)), 6);
    w.f(built.map_or(0.0, |id| {
        Resource::BLOCKS
            .iter()
            .map(|&r| bdef(id).cost[r.idx()] as f32)
            .sum::<f32>()
            / 6.0
    }));
    // 77-82
    w.flag(took.is_some());
    for r in Resource::ALL {
        w.f(took.map_or(0.0, |id| mdef(id).cost[r.idx()] as f32 / 4.0));
    }
    w.f(took.map_or(0.0, |id| (mdef(id).score)(&probe, p) as f32 / 20.0));
    // 83
    w.f((before.skulls_left - after.skulls_left) as f32);
    // 84-85
    let corn_equiv = d_corn
        + 2 * d_res[Resource::Wood.idx()]
        + 3 * d_res[Resource::Stone.idx()]
        + 4 * d_res[Resource::Gold.idx()];
    w.f(corn_equiv as f32 / 10.0);
    w.f((d_points as f32 + corn_equiv as f32 / 4.0 + 3.0 * d_res[Resource::Skull.idx()] as f32)
        / 10.0);
    // 86-87
    w.f((after.proj_points - before.proj_points) as f32 / 10.0);
    let proj_corn: i32 = Resource::ALL
        .iter()
        .map(|&r| (after.proj_res[r.idx()] - before.proj_res[r.idx()]) * r.corn_value().max(3))
        .sum();
    w.f(proj_corn as f32 / 10.0);
    // 88-89
    w.flag(after.feed_short > 0);
    w.f(after.corn as f32 / 20.0);
    // 90-95 reserved
    w.skip(6);
    w.done(D_CHOICE);
}

/// `n * D_CHOICE`, row-major, one row per candidate.
pub fn encode_choices(g: &GameState, p: PlayerId, cs: &[Choice], out: &mut Vec<f32>) {
    out.clear();
    out.resize(cs.len() * D_CHOICE, 0.0);
    for (i, c) in cs.iter().enumerate() {
        encode_choice(g, p, c, &mut out[i * D_CHOICE..(i + 1) * D_CHOICE]);
    }
}

// ---------------------------------------------------------------------------
// Edge reconstruction
// ---------------------------------------------------------------------------

/// Which fixed-arity head answers a phase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Head {
    /// `[none, Brown, Yellow, Green]`
    Beg,
    /// `[place, retrieve, pity]`
    Mode,
    /// `[Pal, Yax, Tik, Uxm, Chi, FirstPlayer, STOP]`
    Place,
    /// `gear * MAX_GEAR_SPACES + pos`, last index `STOP`
    Who,
    /// `[decline, take]`
    ExtraDay,
}

impl Head {
    pub fn arity(self) -> usize {
        match self {
            Head::Beg => 4,
            Head::Mode => 3,
            Head::Place => N_PLACE,
            Head::Who => N_WHO,
            Head::ExtraDay => 2,
        }
    }
}

/// How to turn head logits into a prior per edge.
pub enum EdgeSpec {
    /// `idx[e]` is the head slot that edge `e` corresponds to. Everything not
    /// named is masked to `-inf`, so no gradient reaches a move that cannot be
    /// made and the head never spends capacity ranking one.
    Fixed { head: Head, idx: SmallVec<[u16; 8]> },
    /// `n` candidate feature rows, `D_CHOICE` wide, were **appended** to the
    /// buffer the caller passed in; row `e` of that append is edge `e`, and
    /// `off` is where the append started. The candidate list *is* the legal
    /// set, so there is nothing to mask.
    ///
    /// An offset into a caller-owned buffer rather than an owned `Vec` so that
    /// a batched caller ends up with every candidate in the whole batch in one
    /// matrix: the pointer head's key MLP is then two GEMMs for the batch
    /// instead of two per node, which at batch 128 is the difference between
    /// 4 calls and 512.
    Pointer { off: usize, n: usize },
    /// The edge list could not be reconstructed; fall back to uniform.
    Uniform,
}

/// Rebuild the edge list the search enumerated, so each edge can be mapped onto
/// a head slot or a candidate feature row.
///
/// `phase::Evaluator` passes only `n_edges`, not the edges themselves, so this
/// re-derives them from the same engine calls `docs/SEARCH.md` §2.7 names as
/// each node's source. **Every path is length-checked against `n_edges` and
/// degrades to `Uniform` on any disagreement** — a wrong-length prior would be
/// a silent scrambling of the policy, whereas a uniform one is merely weak.
/// `Query::candidates` lets a caller hand over the candidate list it already
/// built and skip the `Take` recompute entirely.
///
/// `feats` is an append-only scratch buffer for the pointer variant: it is
/// never read or cleared here, only extended, so one buffer serves a whole
/// batch. `Fixed` and `Uniform` leave it untouched.
pub fn edges(
    g: &GameState,
    p: PlayerId,
    phase: Phase,
    n_edges: usize,
    feats: &mut Vec<f32>,
) -> EdgeSpec {
    if n_edges == 0 {
        return EdgeSpec::Uniform;
    }
    match phase {
        Phase::Beg => {
            let opts = beg_options(g, p);
            if opts.len() != n_edges {
                return EdgeSpec::Uniform;
            }
            let idx = opts
                .iter()
                .map(|o| match o {
                    None => 0u16,
                    Some(t) => 1 + t.idx() as u16,
                })
                .collect();
            EdgeSpec::Fixed {
                head: Head::Beg,
                idx,
            }
        }
        // A single edge gets prior 1.0 whatever the head says, and pity is only
        // ever offered alone, so the two-edge case is unambiguously
        // [place, retrieve] without re-deriving affordability.
        Phase::Mode => match n_edges {
            1 => EdgeSpec::Uniform,
            2 => EdgeSpec::Fixed {
                head: Head::Mode,
                idx: SmallVec::from_slice(&[0, 1]),
            },
            _ => EdgeSpec::Uniform,
        },
        Phase::Placing { n } => place_edges(g, n, n_edges),
        Phase::PityPlace => {
            let spots = pity_moves(g, p);
            if spots.len() != n_edges {
                return EdgeSpec::Uniform;
            }
            let mut idx = SmallVec::new();
            for m in &spots {
                let crate::moves::MoveKind::Pity { spot, .. } = &m.kind else {
                    return EdgeSpec::Uniform;
                };
                idx.push(place_slot(*spot));
            }
            EdgeSpec::Fixed {
                head: Head::Place,
                idx,
            }
        }
        Phase::PickWorker => {
            let board: SmallVec<[WorkerId; 6]> = g.on_board(p).collect();
            let stop = match n_edges.checked_sub(board.len()) {
                Some(0) => false,
                Some(1) => true,
                _ => return EdgeSpec::Uniform,
            };
            let mut idx: SmallVec<[u16; 8]> = SmallVec::new();
            for wk in &board {
                let Some((gr, ps)) = g.loc(*wk).on_board() else {
                    return EdgeSpec::Uniform;
                };
                idx.push(cell(gr, ps) as u16);
            }
            if stop {
                idx.push(N_CELLS as u16);
            }
            EdgeSpec::Fixed {
                head: Head::Who,
                idx,
            }
        }
        Phase::Take { worker } => {
            let Some((gr, ps)) = g.loc(worker).on_board() else {
                return EdgeSpec::Uniform;
            };
            let cs = choices_for_worker(g, p, gr, ps);
            pointer_spec(g, p, &cs, n_edges, feats)
        }
        Phase::ExtraDay { .. } => {
            if n_edges == 2 {
                EdgeSpec::Fixed {
                    head: Head::ExtraDay,
                    idx: SmallVec::from_slice(&[0, 1]),
                }
            } else {
                EdgeSpec::Uniform
            }
        }
        Phase::DraftTile { dealt, kept } => {
            let cs = draft_candidates(g, p, dealt, kept, n_edges);
            pointer_spec(g, p, &cs, n_edges, feats)
        }
    }
}

/// The same, from the edges themselves rather than from a count.
///
/// **Prefer this wherever the caller has the edge list**, which since
/// `phase::Query` carries `&[Step]` is everywhere in the search. It is not a
/// faster version of [`edges`]; it is a version with no way to be wrong.
///
/// [`edges`] re-derives the edge list from the engine and checks its length
/// against `n_edges`, because a count was all `Evaluator::evaluate` used to
/// pass. That leaves two failure modes this function does not have. The engine
/// can produce a list of the right length in a *different order* — the priors
/// then line up with the wrong edges, silently, and no length check catches it.
/// And when the lengths disagree the only safe answer is `Uniform`, so a policy
/// head quietly stops contributing at exactly the nodes where the
/// reconstruction is hardest. Here every edge is mapped from the object the
/// tree actually enumerated, in the tree's own order.
///
/// It is also cheaper: no `choices_for_worker`, no `beg_options`, no
/// `place_edges` threshold search. `Take` in particular is the whole candidate
/// enumeration, and it is already in hand.
pub fn edges_for(
    g: &GameState,
    p: PlayerId,
    phase: Phase,
    steps: &[Step],
    feats: &mut Vec<f32>,
) -> EdgeSpec {
    if steps.is_empty() {
        return EdgeSpec::Uniform;
    }
    // The head is a property of the phase; the steps only choose slots within
    // it. A step that does not belong to its phase's head is a tree bug, and
    // `Uniform` is a better answer than a scrambled one.
    let head = match phase {
        Phase::Beg => Head::Beg,
        Phase::Mode => Head::Mode,
        Phase::Placing { .. } | Phase::PityPlace => Head::Place,
        Phase::PickWorker => Head::Who,
        Phase::ExtraDay { .. } => Head::ExtraDay,
        Phase::Take { .. } => {
            let off = feats.len();
            feats.resize(off + steps.len() * D_CHOICE, 0.0);
            for (i, st) in steps.iter().enumerate() {
                let Step::Take(c) = st else {
                    feats.truncate(off);
                    return EdgeSpec::Uniform;
                };
                let r = off + i * D_CHOICE;
                encode_choice(g, p, c, &mut feats[r..r + D_CHOICE]);
            }
            return EdgeSpec::Pointer {
                off,
                n: steps.len(),
            };
        }
        Phase::DraftTile { .. } => {
            let off = feats.len();
            feats.resize(off + steps.len() * D_CHOICE, 0.0);
            for (i, st) in steps.iter().enumerate() {
                let Step::DraftTile(id) = *st else {
                    feats.truncate(off);
                    return EdgeSpec::Uniform;
                };
                let c = tile_choice(g, p, id);
                let r = off + i * D_CHOICE;
                encode_choice(g, p, &c, &mut feats[r..r + D_CHOICE]);
            }
            return EdgeSpec::Pointer {
                off,
                n: steps.len(),
            };
        }
    };

    let mut idx: SmallVec<[u16; 8]> = SmallVec::new();
    for st in steps {
        let slot = match (head, st) {
            (Head::Beg, Step::Beg(None)) => 0,
            (Head::Beg, Step::Beg(Some(t))) => 1 + t.idx() as u16,
            (Head::Mode, Step::Mode(ModeChoice::Place)) => 0,
            (Head::Mode, Step::Mode(ModeChoice::Retrieve)) => 1,
            (Head::Mode, Step::Mode(ModeChoice::Pity)) => 2,
            (Head::Place, Step::Place(sp)) => place_slot(*sp),
            (Head::Place, Step::Pity(sp)) => place_slot(*sp),
            (Head::Place, Step::StopPlacing) => 6,
            (Head::Who, Step::PickWorker(w)) => match g.loc(*w).on_board() {
                Some((gr, ps)) => cell(gr, ps) as u16,
                None => return EdgeSpec::Uniform,
            },
            (Head::Who, Step::StopRetrieving) => N_CELLS as u16,
            (Head::ExtraDay, Step::ExtraDay(false)) => 0,
            (Head::ExtraDay, Step::ExtraDay(true)) => 1,
            _ => return EdgeSpec::Uniform,
        };
        debug_assert!((slot as usize) < head.arity());
        idx.push(slot);
    }
    EdgeSpec::Fixed { head, idx }
}

/// The tiles still on offer. `kept` is read as a bitmask over `dealt`; if that
/// does not give the width the caller reported, fall back to "the last
/// `n_edges` dealt", which is at least the right length.
///
/// A starting tile becomes a `Choice`, which is the payoff for having made
/// `encode_choice` a state-delta function: the draft head is new parameters,
/// not new feature code. No starting tile carries a decision, so the whole
/// tile is one candidate.
pub fn draft_candidates(
    g: &GameState,
    p: PlayerId,
    dealt: [u8; 4],
    kept: u8,
    n_edges: usize,
) -> Vec<Choice> {
    let by_mask: Vec<u8> = (0..4)
        .filter(|i| kept & (1 << i) == 0)
        .map(|i| dealt[i])
        .collect();
    let ids = if by_mask.len() == n_edges {
        by_mask
    } else {
        dealt[4usize.saturating_sub(n_edges.min(4))..].to_vec()
    };
    ids.iter().map(|&id| tile_choice(g, p, id)).collect()
}

/// One starting tile as a `Choice`, which is what lets the draft head be the
/// same pointer machinery plus one embedding row.
///
/// Drafting happens before anyone has research, but this is also called from
/// tests against mid-game states, and `AdvanceResearch` is debug-asserted never
/// to be applied at level 3.
pub fn tile_choice(g: &GameState, p: PlayerId, id: u8) -> Choice {
    let mut c = Choice::new();
    for e in TILES.get(id as usize).copied().unwrap_or(&[]) {
        if let Effect::AdvanceResearch(sc) = *e {
            if g.level(p, sc) >= 3 {
                continue;
            }
        }
        c.0.push(*e);
    }
    c
}

fn pointer_spec(
    g: &GameState,
    p: PlayerId,
    cs: &[Choice],
    n_edges: usize,
    feats: &mut Vec<f32>,
) -> EdgeSpec {
    let off = feats.len();
    feats.resize(off + n_edges * D_CHOICE, 0.0);
    for i in 0..n_edges {
        // A caller that capped the edge list keeps a prefix; a longer list than
        // expected leaves the tail as a neutral all-zero row, which the bias
        // feature at index 0 makes distinguishable from a real skip.
        if let Some(c) = cs.get(i) {
            let r = off + i * D_CHOICE;
            encode_choice(g, p, c, &mut feats[r..r + D_CHOICE]);
        }
    }
    EdgeSpec::Pointer { off, n: n_edges }
}

fn place_slot(spot: Placement) -> u16 {
    match spot {
        Placement::Gear(gear, _) => gear.idx() as u16,
        Placement::FirstPlayer => 5,
    }
}

/// `Placing`'s edges are the gears whose lowest free space is affordable, then
/// the first-player space, then `STOP`.
///
/// `Phase::Placing` carries `n` but not the corn already committed, so the
/// budget is not recoverable. It does not have to be: the filter is
/// `pos <= budget - n - spent`, so the legal gear set is always the gears whose
/// `lowest_free` is at or below some threshold. Solving for the threshold that
/// reproduces `n_edges` recovers the exact list without knowing the budget.
fn place_edges(g: &GameState, n: u8, n_edges: usize) -> EdgeSpec {
    let mut avail: SmallVec<[(usize, u8); 5]> = SmallVec::new();
    for gear in Gear::ALL {
        if let Some(pos) = g.lowest_free(gear) {
            avail.push((gear.idx(), pos.0));
        }
    }
    let fp_free = g.first_player_space.is_none();
    let stop = n > 0;

    for &with_fp in &[true, false] {
        if with_fp && !fp_free {
            continue;
        }
        for t in 0..=MAX_GEAR_SPACES as u8 {
            let keep = avail.iter().filter(|&&(_, pos)| pos <= t).count();
            let total = keep + usize::from(with_fp) + usize::from(stop);
            if total != n_edges {
                continue;
            }
            let mut idx: SmallVec<[u16; 8]> = SmallVec::new();
            for &(gi, pos) in &avail {
                if pos <= t {
                    idx.push(gi as u16);
                }
            }
            if with_fp {
                idx.push(5);
            }
            if stop {
                idx.push(6);
            }
            return EdgeSpec::Fixed {
                head: Head::Place,
                idx,
            };
        }
    }
    EdgeSpec::Uniform
}
