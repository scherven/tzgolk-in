//! The game state: one `Copy` struct, no heap allocation, no interior pointers.
//!
//! Snapshotting for search is `let saved = *state;` — a memcpy of a few hundred
//! bytes. This replaces the Go version's `Freeze`/`Save`/`Load` table, which
//! deep-cloned the whole game (609,124 times in a two-minute run) under keys
//! computed as `10000 * ply` and `20000 * ply`, values that collide.

use crate::data::temples::TEMPLES;
use crate::ids::*;

/// Where a worker is. In the Go version this was three fields (`Available`,
/// `Wheel_id`, `Position`) whose combinations encoded location implicitly, and
/// two bugs came directly from that: the panel counted `Wheel_id > 0` (missing
/// every worker on Palenque, gear 0) and food day counted
/// `Wheel_id != -1 || Available` (missing the worker on the first player space,
/// which is neither). As one sum type, both are `match` arms you cannot omit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum WorkerLoc {
    /// Not yet bought; not fed on food days.
    Locked,
    /// In hand, available to place. Fed on food days.
    Available,
    OnGear { gear: Gear, pos: Pos },
    FirstPlayerSpace,
}

impl WorkerLoc {
    /// Every worker the player owns except the ones still locked. This is the
    /// set that must be fed.
    #[inline]
    pub fn is_unlocked(self) -> bool {
        !matches!(self, WorkerLoc::Locked)
    }

    #[inline]
    pub fn on_board(self) -> Option<(Gear, Pos)> {
        match self {
            WorkerLoc::OnGear { gear, pos } => Some((gear, pos)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Player {
    pub color: Color,
    pub corn: u8,
    pub res: [u8; 4],
    pub points: i16,
    pub corn_tiles: u8,
    pub wood_tiles: u8,
    /// Workers fed for free on a food day.
    pub free_workers: u8,
    /// Reduction in the 2-corn per-worker food cost.
    pub worker_discount: u8,
    /// The first-player tile's unused face: whether this player may still take
    /// the extra calendar advance when they hold the first player space.
    pub may_skip_day: bool,
    /// Bitset over global `BuildingId` (bit `id - 1`).
    pub buildings: u32,
    /// Bitset over `MonumentId` (bit `id - 1`).
    pub monuments: u16,
}

impl Player {
    pub fn new(color: Color) -> Self {
        Player {
            color,
            corn: 0,
            res: [0; 4],
            points: 0,
            corn_tiles: 0,
            wood_tiles: 0,
            free_workers: 0,
            worker_discount: 0,
            may_skip_day: true,
            buildings: 0,
            monuments: 0,
        }
    }

    #[inline]
    pub fn get(&self, r: Resource) -> u8 {
        self.res[r.idx()]
    }

    #[inline]
    pub fn can_pay(&self, cost: Bundle) -> bool {
        Resource::ALL
            .iter()
            .all(|&r| self.res[r.idx()] as i32 >= cost[r.idx()] as i32)
    }

    /// Total blocks held, used by several monuments.
    pub fn n_blocks(&self) -> u32 {
        Resource::BLOCKS.iter().map(|&r| self.get(r) as u32).sum()
    }

    pub fn push_building(&mut self, id: BuildingId) {
        self.buildings |= 1 << (id.0 - 1);
    }

    pub fn push_monument(&mut self, id: MonumentId) {
        self.monuments |= 1 << (id.0 - 1);
    }

    pub fn has_building(&self, id: BuildingId) -> bool {
        self.buildings & (1 << (id.0 - 1)) != 0
    }

    pub fn n_buildings(&self) -> u32 {
        self.buildings.count_ones()
    }

    pub fn n_monuments(&self) -> u32 {
        self.monuments.count_ones()
    }

    pub fn building_ids(&self) -> impl Iterator<Item = BuildingId> + '_ {
        (0..32u8).filter_map(move |b| {
            (self.buildings & (1 << b) != 0).then_some(BuildingId(b + 1))
        })
    }

    pub fn monument_ids(&self) -> impl Iterator<Item = MonumentId> + '_ {
        (0..16u8).filter_map(move |b| {
            (self.monuments & (1 << b) != 0).then_some(MonumentId(b + 1))
        })
    }

    /// End-of-game corn value of everything liquid, per the base game's
    /// "4 corn = 1 point" conversion.
    pub fn total_corn(&self) -> i32 {
        let mut c = self.corn as i32;
        for r in Resource::BLOCKS {
            c += self.get(r) as i32 * r.corn_value();
        }
        c
    }
}

/// Corn and wood tiles remaining on one Palenque space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct TileStack {
    pub corn: u8,
    pub wood: u8,
}

impl TileStack {
    /// Wood tiles sit on top; a corn face is exposed only once the wood on this
    /// space is outnumbered.
    #[inline]
    pub fn corn_showing(self) -> bool {
        self.corn > self.wood
    }
}

pub const EMPTY_SPACE: u8 = 0xFF;

/// The largest gear (Chichen Itza) has eleven worker spaces.
pub const MAX_GEAR_SPACES: usize = 11;

/// Worker occupancy of one gear. `EMPTY_SPACE` marks a free space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GearState {
    pub occ: [u8; MAX_GEAR_SPACES],
}

impl GearState {
    pub fn empty() -> Self {
        GearState {
            occ: [EMPTY_SPACE; MAX_GEAR_SPACES],
        }
    }

    #[inline]
    pub fn at(&self, pos: Pos) -> Option<WorkerId> {
        let v = self.occ[pos.idx()];
        (v != EMPTY_SPACE).then_some(WorkerId(v))
    }

    #[inline]
    pub fn is_free(&self, pos: Pos) -> bool {
        self.occ[pos.idx()] == EMPTY_SPACE
    }
}

/// A shuffled draw pile of ids, drawn from the front.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Deck<const N: usize> {
    pub ids: [u8; N],
    pub next: u8,
}

impl<const N: usize> Deck<N> {
    pub fn new(ids: [u8; N]) -> Self {
        Deck { ids, next: 0 }
    }

    pub fn remaining(&self) -> usize {
        N - self.next as usize
    }

    pub fn draw(&mut self) -> Option<u8> {
        if self.remaining() == 0 {
            return None;
        }
        let v = self.ids[self.next as usize];
        self.next += 1;
        Some(v)
    }
}

pub const N_DISPLAY: usize = 6;

/// Every crystal skull in the game.
pub const N_SKULLS: u8 = 13;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GameState {
    pub players: [Player; N_PLAYERS],
    pub workers: [WorkerLoc; N_WORKERS],
    pub gears: [GearState; 5],

    /// `temples[temple][player]` = step index on that track.
    pub temples: [[u8; N_PLAYERS]; 3],
    /// `research[player][science]` = 0..=3.
    pub research: [[u8; 4]; N_PLAYERS],

    /// Indexed by `Pos`; only spaces 2..=5 carry tiles.
    pub palenque: [TileStack; 8],
    /// Bitset over Chichen positions that have been used.
    pub chichen_filled: u16,

    pub age1: Deck<14>,
    pub age2: Deck<18>,
    pub monument_deck: Deck<13>,
    pub buildings_up: [Option<BuildingId>; N_DISPLAY],
    pub monuments_up: [Option<MonumentId>; N_DISPLAY],

    pub first_player_space: Option<WorkerId>,
    pub accumulated_corn: u8,

    /// Crystal skulls still in the bank. There are exactly 13 in the game and
    /// no more; skulls spent at Chichen Itza stay on the board and never come
    /// back, so this only ever falls.
    pub skulls_remaining: u8,

    pub current: PlayerId,
    pub first_player: PlayerId,
    pub age: u8,
    /// 0..=26. The base game's calendar is 27 days long.
    pub day: u8,
    pub over: bool,
}

/// The calendar is 27 rounds long. `day` counts completed rounds, so it runs
/// 0..=27 and the game ends once round 27 has resolved.
pub const LAST_DAY: u8 = 27;
/// Rounds after which every player is fed and the temples pay out resources.
pub const RESOURCE_DAYS: [u8; 2] = [8, 21];
/// Rounds after which every player is fed and the temples pay out points.
pub const POINT_DAYS: [u8; 2] = [14, 27];

impl GameState {
    // ---- temples -------------------------------------------------------

    #[inline]
    pub fn temple_pos(&self, p: PlayerId, t: Temple) -> u8 {
        self.temples[t.idx()][p.idx()]
    }

    /// The highest step `p` may occupy on `t` right now.
    ///
    /// The top step is exclusive: once someone stands on it, nobody else may
    /// reach it until they come back down.
    fn temple_ceiling(&self, p: PlayerId, t: Temple) -> u8 {
        let top = TEMPLES[t.idx()].steps - 1;
        let taken = PlayerId::ALL
            .iter()
            .any(|&q| q != p && self.temples[t.idx()][q.idx()] == top);
        if taken {
            top - 1
        } else {
            top
        }
    }

    /// Move along a temple track, clamped to what the player may actually reach.
    ///
    /// Reaching the top of a temple turns the player's first-player tile back
    /// over, making the extra calendar day available to them again. Go only did
    /// this when a step would have overshot the top, so arriving exactly on the
    /// top step did not flip the tile.
    ///
    /// Clamping is deliberate rather than a guard: a privilege that would climb
    /// higher than the player can go is *wasted*, not refused, so callers may
    /// apply a step that goes nowhere.
    pub fn temple_step(&mut self, p: PlayerId, t: Temple, d: i8) {
        let ceiling = self.temple_ceiling(p, t) as i32;
        let cur = self.temples[t.idx()][p.idx()] as i32;
        let next = (cur + d as i32).clamp(0, ceiling);
        self.temples[t.idx()][p.idx()] = next as u8;

        // On *reaching* the top, not on a step that goes nowhere because the
        // player is already there -- otherwise a player camped on a temple top
        // could refresh the privilege indefinitely.
        let top = (TEMPLES[t.idx()].steps - 1) as i32;
        if d > 0 && next == top && cur < top {
            self.players[p.idx()].may_skip_day = true;
        }
    }

    /// Whether a step would actually move the player. A step that would not is
    /// still legal to attempt -- it is simply wasted -- so this is for
    /// generating *useful* options, never for gating legality.
    pub fn can_temple_step(&self, p: PlayerId, t: Temple, d: i8) -> bool {
        let cur = self.temple_pos(p, t);
        match d.signum() {
            1 => cur < self.temple_ceiling(p, t),
            -1 => cur > 0,
            _ => false,
        }
    }

    /// Points from one temple alone, used by tests and by `temple_points`.
    pub fn temple_points_of(&self, p: PlayerId, t: Temple, age: u8) -> i16 {
        let step = self.temple_pos(p, t);
        let d = &TEMPLES[t.idx()];
        let mut gained = d.points[step as usize];
        let highest = PlayerId::ALL
            .iter()
            .map(|&q| self.temple_pos(q, t))
            .max()
            .unwrap_or(0);
        if step == highest {
            let tied = PlayerId::ALL
                .iter()
                .filter(|&&q| self.temple_pos(q, t) == highest)
                .count();
            let prize = if age == 1 { d.age1_prize } else { d.age2_prize };
            gained += if tied > 1 { prize / 2 } else { prize };
        }
        gained
    }

    /// Points this player scores from all three temples on a scoring day.
    pub fn temple_points(&self, p: PlayerId, age: u8) -> i16 {
        Temple::ALL
            .iter()
            .map(|&t| self.temple_points_of(p, t, age))
            .sum()
    }

    // ---- round flow ------------------------------------------------------

    /// The player on the first player space takes the accumulated corn and the
    /// marker.
    ///
    /// Returns the player now entitled to decide on the extra calendar day, if
    /// anyone. That is a genuine decision, not a side effect: a search has to be
    /// able to reason about it, so it is surfaced rather than resolved here.
    pub fn resolve_first_player(&mut self) -> Option<PlayerId> {
        let Some(w) = self.first_player_space.take() else {
            // Nobody claimed it, so the marker stays where it is and one corn
            // is placed on the space for next round.
            self.accumulated_corn += 1;
            return None;
        };

        let p = w.owner();
        self.workers[w.idx()] = WorkerLoc::Available;

        let corn = self.accumulated_corn;
        self.players[p.idx()].corn += corn;
        self.accumulated_corn = 0;

        // A claimer who already holds the marker passes it to their left.
        self.first_player = if self.first_player == p {
            p.next(1)
        } else {
            p
        };

        // Offered to the claimer; the decision itself belongs to the caller.
        Some(p)
    }

    /// Whether `p` may advance the calendar an extra day.
    ///
    /// It is allowed immediately before a food day — the new round simply
    /// becomes the food day that was skipped over. What it may not do is push
    /// any worker, of any player, off the end of a gear.
    pub fn may_take_extra_day(&self, p: PlayerId) -> bool {
        self.players[p.idx()].may_skip_day
            && !self.worker_on_penultimate_space()
            // There is no day past the end of the calendar to advance into.
            && self.day + 2 <= LAST_DAY
    }

    /// Spend the first-player tile to advance the calendar a second day.
    /// Spend the first-player tile. Call *before* the day advance; the caller
    /// then advances two days in one go so a food day passed over is resolved
    /// on the round landed on, not on the one left behind.
    pub fn spend_extra_day(&mut self, p: PlayerId) {
        debug_assert!(self.may_take_extra_day(p));
        self.players[p.idx()].may_skip_day = false;
    }

    /// Advance the calendar one day and resolve whatever that day brings.
    ///
    /// Go checked the day *before* incrementing it, so every food day fired one
    /// rotation early.
    pub fn advance_day(&mut self) {
        self.advance_days(1);
    }

    /// Advance the calendar `n` days as a single step.
    ///
    /// At most one food day resolves, on the round landed on. Skipping over a
    /// food day does not feed everyone at the end of the round left behind --
    /// the new round *becomes* the food day that was passed.
    pub fn advance_days(&mut self, n: u8) {
        if self.over || n == 0 {
            return;
        }
        let n = n.min(LAST_DAY - self.day);
        if n == 0 {
            return;
        }
        for _ in 0..n {
            self.rotate_gears();
            self.day += 1;
        }

        // Whichever food day was traversed resolves here, on the round landed
        // on. Advancing two days over one does not resolve it twice, nor early.
        let from = self.day + 1 - n;
        let hit = |days: &[u8; 2]| days.iter().any(|d| (from..=self.day).contains(d));

        if hit(&RESOURCE_DAYS) {
            self.food_day();
            self.gain_temple_resources();
        } else if hit(&POINT_DAYS) {
            self.food_day();
            for p in PlayerId::ALL {
                self.gain_temple_points(p);
            }

            self.age += 1;
            if self.age == 2 {
                // A fresh row from the age 2 deck.
                self.buildings_up = [None; N_DISPLAY];
                self.refill_buildings();
            } else {
                // The gear turns once more before final scoring, which can push
                // workers off and so change the tiebreaker.
                self.rotate_gears();
                self.end_game();
            }
        }

        if self.day >= LAST_DAY {
            self.over = true;
        }
    }

    pub fn food_day(&mut self) {
        for p in PlayerId::ALL {
            let mouths = self.n_unlocked(p);
            let pl = &mut self.players[p.idx()];

            let free = (pl.free_workers as usize).min(mouths);
            let to_pay = mouths - free;

            // Two farm buildings bring the per-worker cost to zero, at which
            // point everyone eats for free. Go floored the cost at 1.
            let each = (2u8).saturating_sub(pl.worker_discount);
            let paid = if each == 0 {
                to_pay
            } else {
                let affordable = (pl.corn / each) as usize;
                let paid = to_pay.min(affordable);
                pl.corn -= (paid as u8) * each;
                paid
            };

            let unpaid = to_pay - paid;
            pl.points -= 3 * unpaid as i16;
        }
    }

    /// Pay out temple resources to everyone.
    ///
    /// Skulls are handled together because of the supply rule: if the bank
    /// cannot cover everyone who qualifies, nobody gets one.
    fn gain_temple_resources(&mut self) {
        let mut skull_claimants: Vec<PlayerId> = Vec::new();

        for p in PlayerId::ALL {
            for t in Temple::ALL {
                let step = self.temple_pos(p, t);
                for &(at, r) in crate::data::temples::TEMPLES[t.idx()].resources {
                    if step < at {
                        continue;
                    }
                    if r == Resource::Skull {
                        skull_claimants.push(p);
                    } else {
                        self.players[p.idx()].res[r.idx()] += 1;
                    }
                }
            }
        }

        // If the bank cannot cover everyone who qualifies, nobody gets one.
        if (skull_claimants.len() as u8) <= self.skulls_remaining {
            for p in skull_claimants {
                self.take_skulls(p, 1);
            }
        }
    }

    fn gain_temple_points(&mut self, p: PlayerId) {
        let gained = self.temple_points(p, self.age);
        self.players[p.idx()].points += gained;
    }

    /// Final scoring. Sets `over`.
    pub fn end_game(&mut self) {
        for p in PlayerId::ALL {
            let pl = &self.players[p.idx()];
            let mut extra = (pl.total_corn() / 4) as i16;
            extra += pl.get(Resource::Skull) as i16 * 3;

            let mons: Vec<MonumentId> = pl.monument_ids().collect();
            for id in mons {
                extra += (crate::data::monuments::def(id).score)(self, p) as i16;
            }
            self.players[p.idx()].points += extra;
        }
        self.over = true;
    }

    /// Who won. Most points; ties broken by workers left on the gears; still
    /// tied and they all win.
    pub fn winners(&self) -> Vec<PlayerId> {
        let best = self.players.iter().map(|p| p.points).max().unwrap();
        let tied: Vec<PlayerId> = PlayerId::ALL
            .iter()
            .copied()
            .filter(|p| self.players[p.idx()].points == best)
            .collect();
        if tied.len() == 1 {
            return tied;
        }
        let most = tied
            .iter()
            .map(|&p| self.on_board(p).count())
            .max()
            .unwrap_or(0);
        tied.into_iter()
            .filter(|&p| self.on_board(p).count() == most)
            .collect()
    }

    /// Final scores. Only meaningful once the game is over.
    pub fn scores(&self) -> [i16; N_PLAYERS] {
        std::array::from_fn(|i| self.players[i].points)
    }

    // ---- crystal skulls -------------------------------------------------

    /// Draw up to `n` skulls from the bank, returning how many were actually
    /// available.
    pub fn take_skulls(&mut self, p: PlayerId, n: u8) -> u8 {
        let got = n.min(self.skulls_remaining);
        self.skulls_remaining -= got;
        self.players[p.idx()].res[Resource::Skull.idx()] += got;
        got
    }

    // ---- research ------------------------------------------------------

    #[inline]
    pub fn level(&self, p: PlayerId, s: Science) -> u8 {
        self.research[p.idx()][s.idx()]
    }

    #[inline]
    pub fn has_level(&self, p: PlayerId, s: Science, n: u8) -> bool {
        self.level(p, s) >= n
    }

    // ---- workers -------------------------------------------------------

    pub fn worker_ids(p: PlayerId) -> impl Iterator<Item = WorkerId> {
        let base = p.idx() * WORKERS_PER_PLAYER;
        (base..base + WORKERS_PER_PLAYER).map(|i| WorkerId(i as u8))
    }

    #[inline]
    pub fn loc(&self, w: WorkerId) -> WorkerLoc {
        self.workers[w.idx()]
    }

    /// Workers that must be fed on a food day: everything except locked.
    pub fn n_unlocked(&self, p: PlayerId) -> usize {
        Self::worker_ids(p)
            .filter(|&w| self.loc(w).is_unlocked())
            .count()
    }

    pub fn available(&self, p: PlayerId) -> impl Iterator<Item = WorkerId> + '_ {
        Self::worker_ids(p).filter(move |&w| self.loc(w) == WorkerLoc::Available)
    }

    pub fn on_board(&self, p: PlayerId) -> impl Iterator<Item = WorkerId> + '_ {
        Self::worker_ids(p).filter(move |&w| self.loc(w).on_board().is_some())
    }

    pub fn unlock_worker(&mut self, p: PlayerId) {
        if let Some(w) = Self::worker_ids(p).find(|&w| self.loc(w) == WorkerLoc::Locked) {
            self.workers[w.idx()] = WorkerLoc::Available;
        }
    }

    pub fn place_worker(&mut self, w: WorkerId, gear: Gear, pos: Pos) {
        debug_assert!(self.gears[gear.idx()].is_free(pos), "space already occupied");
        self.gears[gear.idx()].occ[pos.idx()] = w.0;
        self.workers[w.idx()] = WorkerLoc::OnGear { gear, pos };
    }

    pub fn place_on_first_player(&mut self, w: WorkerId) {
        debug_assert!(self.first_player_space.is_none());
        self.first_player_space = Some(w);
        self.workers[w.idx()] = WorkerLoc::FirstPlayerSpace;
    }

    pub fn retrieve_worker(&mut self, w: WorkerId) {
        if let WorkerLoc::OnGear { gear, pos } = self.loc(w) {
            self.gears[gear.idx()].occ[pos.idx()] = EMPTY_SPACE;
        }
        self.workers[w.idx()] = WorkerLoc::Available;
    }

    /// Whether any worker -- anyone's -- sits where a *second* day's rotation
    /// would carry it off its gear.
    ///
    /// That is the second-to-last space: spaces 6 and 9 in the rulebook's
    /// numbering. A worker on the last space does not block the privilege,
    /// because the ordinary daily advance pushes it off regardless.
    pub fn worker_on_penultimate_space(&self) -> bool {
        Gear::ALL
            .iter()
            .any(|&gear| !self.gears[gear.idx()].is_free(Pos(gear.size() - 2)))
    }

    /// Workers always enter a gear on its lowest free space.
    pub fn lowest_free(&self, gear: Gear) -> Option<Pos> {
        (0..gear.size())
            .map(Pos)
            .find(|&pos| self.gears[gear.idx()].is_free(pos))
    }

    // ---- decks ---------------------------------------------------------

    pub fn take_building(&mut self, id: BuildingId) {
        if let Some(slot) = self
            .buildings_up
            .iter_mut()
            .find(|s| **s == Some(id))
        {
            *slot = None;
        } else {
            debug_assert!(false, "built a card that was not face up: {id:?}");
        }
    }

    pub fn take_monument(&mut self, id: MonumentId) {
        if let Some(slot) = self.monuments_up.iter_mut().find(|s| **s == Some(id)) {
            *slot = None;
        } else {
            debug_assert!(false, "took a monument that was not face up: {id:?}");
        }
    }

    /// Refill the building row from the current age's deck. Called once at the
    /// end of a turn, never mid-choice, so that a double-build cannot reach a
    /// card that was dealt by its own first half.
    pub fn refill_buildings(&mut self) {
        for i in 0..N_DISPLAY {
            if self.buildings_up[i].is_none() {
                let drawn = if self.age == 1 {
                    self.age1.draw()
                } else {
                    self.age2.draw()
                };
                self.buildings_up[i] = drawn.map(BuildingId);
            }
        }
    }

    pub fn face_up_buildings(&self) -> impl Iterator<Item = BuildingId> + '_ {
        self.buildings_up.iter().filter_map(|s| *s)
    }

    pub fn face_up_monuments(&self) -> impl Iterator<Item = MonumentId> + '_ {
        self.monuments_up.iter().filter_map(|s| *s)
    }

    // ---- calendar ------------------------------------------------------

    #[inline]
    pub fn chichen_is_full(&self, pos: Pos) -> bool {
        self.chichen_filled & (1 << pos.0) != 0
    }

    /// Advance every gear one space. A worker on the last space of a gear comes
    /// back to its owner's hand.
    pub fn rotate_gears(&mut self) {
        for gear in Gear::ALL {
            let size = gear.size() as usize;
            let old = self.gears[gear.idx()].occ;
            let mut new = [EMPTY_SPACE; MAX_GEAR_SPACES];
            for pos in 0..size {
                let w = old[pos];
                if w == EMPTY_SPACE {
                    continue;
                }
                if pos + 1 >= size {
                    // Falls off the top and returns to the player.
                    self.workers[w as usize] = WorkerLoc::Available;
                } else {
                    new[pos + 1] = w;
                    self.workers[w as usize] = WorkerLoc::OnGear {
                        gear,
                        pos: Pos((pos + 1) as u8),
                    };
                }
            }
            self.gears[gear.idx()].occ = new;
        }
    }
}
