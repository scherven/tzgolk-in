//! The closed effect vocabulary.
//!
//! Every mutation the base game can make to a player or the board is one of
//! these 13 variants. Options are *generated* by closures (see [`crate::spaces`])
//! but *executed* as data, which is what makes moves `Clone + Eq + Hash` and the
//! whole `GameState` a plain `Copy` struct.
//!
//! Crucially, an effect carries values that were already resolved at generation
//! time. `Effect::Corn(7)` means seven corn, not "however much corn the player's
//! agriculture level implies when this finally runs". In the Go version those
//! were re-derived inside the closure against a state that generation had
//! mutated and rolled back, so the logged description and the executed effect
//! could legitimately disagree.

use crate::ids::*;
use crate::state::GameState;
use smallvec::{smallvec, SmallVec};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Effect {
    /// Gain (or spend, if negative) corn.
    ///
    /// Wider than the other deltas because the market can move a whole hoard in
    /// one exchange, and corn itself is a `u8`.
    Corn(i16),
    /// Set corn to an absolute value. Only used by begging.
    SetCorn(u8),
    Res(Resource, i8),
    Points(i8),
    /// Move on a temple track. Clamped at both ends.
    TempleStep(Temple, i8),
    /// Advance one science track by one. Never emitted at level 3; the
    /// level-3 payoffs are enumerated as concrete effects instead.
    AdvanceResearch(Science),
    /// Make one locked worker available.
    UnlockWorker,
    /// Permanently feed one more worker for free on food days.
    FreeWorker(i8),
    /// Permanently reduce the per-worker food cost.
    WorkerDiscount(i8),
    /// Consume a tile from a Palenque space and credit the player's tile count.
    TakePalenqueTile(Pos, TileKind),
    /// Discard a wood tile to reach the corn beneath it. The tile goes back to
    /// the box, so it does not count toward the wood-tile monument.
    BurnPalenqueWood(Pos),
    /// Mark a Chichen Itza skull space as used for the rest of the game.
    FillChichen(Pos),
    /// Move a face-up building to the player. Payment and any architecture
    /// bonus are separate effects in the same choice.
    Build(BuildingId),
    /// Move a face-up monument to the player.
    TakeMonument(MonumentId),
}

impl Effect {
    /// How many variants the vocabulary has. The network embeds effects by
    /// index, so this is a hard interface number.
    pub const COUNT: usize = 14;

    /// Dense index for featurisation.
    ///
    /// Deliberately an exhaustive match with no wildcard arm: adding a variant
    /// must break the build here rather than silently mis-index an embedding.
    pub fn tag(self) -> u8 {
        match self {
            Effect::Corn(_) => 0,
            Effect::SetCorn(_) => 1,
            Effect::Res(..) => 2,
            Effect::Points(_) => 3,
            Effect::TempleStep(..) => 4,
            Effect::AdvanceResearch(_) => 5,
            Effect::UnlockWorker => 6,
            Effect::FreeWorker(_) => 7,
            Effect::WorkerDiscount(_) => 8,
            Effect::TakePalenqueTile(..) => 9,
            Effect::BurnPalenqueWood(_) => 10,
            Effect::FillChichen(_) => 11,
            Effect::Build(_) => 12,
            Effect::TakeMonument(_) => 13,
        }
    }

    /// Apply to `g` on behalf of `p`.
    pub fn apply(self, g: &mut GameState, p: PlayerId) {
        let i = p.idx();
        match self {
            Effect::Corn(n) => {
                g.players[i].corn = (g.players[i].corn as i32 + n as i32).max(0) as u8;
            }
            Effect::SetCorn(n) => g.players[i].corn = n,
            Effect::Res(Resource::Skull, n) if n > 0 => {
                // There are exactly 13 crystal skulls. Once the bank is empty
                // an action that would grant one simply has no effect.
                g.take_skulls(p, n as u8);
            }
            Effect::Res(r, n) => {
                let slot = &mut g.players[i].res[r.idx()];
                *slot = (*slot as i32 + n as i32).max(0) as u8;
            }
            Effect::Points(n) => g.players[i].points += n as i16,
            Effect::TempleStep(t, d) => g.temple_step(p, t, d),
            Effect::AdvanceResearch(s) => {
                let lvl = &mut g.research[i][s.idx()];
                debug_assert!(*lvl < 3, "AdvanceResearch emitted at max level");
                *lvl = (*lvl + 1).min(3);
            }
            Effect::UnlockWorker => g.unlock_worker(p),
            Effect::FreeWorker(n) => {
                g.players[i].free_workers = (g.players[i].free_workers as i8 + n).max(0) as u8
            }
            Effect::WorkerDiscount(n) => {
                g.players[i].worker_discount =
                    (g.players[i].worker_discount as i8 + n).clamp(0, 2) as u8
            }
            Effect::TakePalenqueTile(pos, kind) => {
                let stack = &mut g.palenque[pos.idx()];
                match kind {
                    TileKind::Corn => {
                        debug_assert!(stack.corn > 0, "took a corn tile from an empty space");
                        stack.corn = stack.corn.saturating_sub(1);
                        g.players[i].corn_tiles += 1;
                    }
                    TileKind::Wood => {
                        debug_assert!(stack.wood > 0, "took a wood tile from an empty space");
                        stack.wood = stack.wood.saturating_sub(1);
                        g.players[i].wood_tiles += 1;
                    }
                }
            }
            Effect::BurnPalenqueWood(pos) => {
                let stack = &mut g.palenque[pos.idx()];
                debug_assert!(stack.wood > 0, "burned wood from a space with none");
                stack.wood = stack.wood.saturating_sub(1);
            }
            Effect::FillChichen(pos) => g.chichen_filled |= 1 << pos.0,
            Effect::Build(id) => {
                g.take_building(id);
                g.players[i].push_building(id);
            }
            Effect::TakeMonument(id) => {
                g.take_monument(id);
                g.players[i].push_monument(id);
            }
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Effect::Corn(n) => write!(f, "{n:+} corn"),
            Effect::SetCorn(n) => write!(f, "corn={n}"),
            Effect::Res(r, n) => write!(f, "{n:+}{}", r.letter()),
            Effect::Points(n) => write!(f, "{n:+}pt"),
            Effect::TempleStep(t, d) => write!(f, "{}{d:+}", t.letter()),
            Effect::AdvanceResearch(s) => write!(f, "{}+1", s.letter()),
            Effect::UnlockWorker => f.write_str("unlock"),
            Effect::FreeWorker(n) => write!(f, "free worker{n:+}"),
            Effect::WorkerDiscount(n) => write!(f, "worker cost{:+}", -n),
            Effect::TakePalenqueTile(pos, k) => {
                let k = if k == TileKind::Corn { 'c' } else { 'w' };
                write!(f, "tile {k}@{}", pos.0)
            }
            Effect::BurnPalenqueWood(pos) => write!(f, "burn w@{}", pos.0),
            Effect::FillChichen(pos) => write!(f, "fill@{}", pos.0),
            Effect::Build(id) => write!(f, "build #{}", id.0),
            Effect::TakeMonument(id) => write!(f, "monument #{}", id.0),
        }
    }
}

/// Inline capacity covers every choice the base game generates; the largest are
/// the double-build on Tikal and the theology-discounted Chichen plays.
pub type Effects = SmallVec<[Effect; 8]>;

/// One fully-resolved thing a player may do at one board space.
///
/// A `Choice` is entirely data. Its label is *derived* from its effects, so the
/// move log cannot describe something other than what ran.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Choice(pub Effects);

impl Choice {
    pub fn new() -> Self {
        Choice(Effects::new())
    }

    /// The explicit "take the space but do nothing", which several spaces need
    /// so that a worker can be retrieved from them at all.
    pub fn skip() -> Self {
        Choice(Effects::new())
    }

    pub fn of<I: IntoIterator<Item = Effect>>(effects: I) -> Self {
        Choice(effects.into_iter().collect())
    }

    pub fn one(effect: Effect) -> Self {
        Choice(smallvec![effect])
    }

    pub fn is_skip(&self) -> bool {
        self.0.is_empty()
    }

    pub fn with(mut self, effect: Effect) -> Self {
        self.0.push(effect);
        self
    }

    /// Concatenate, used where one space's action is another's plus a surcharge
    /// (the Uxmal mirror, the theology discount on Chichen).
    pub fn chain(mut self, other: &Choice) -> Self {
        self.0.extend_from_slice(&other.0);
        self
    }


    /// Whether this choice can be paid for out of `p`'s current holdings.
    ///
    /// `apply` clamps at zero so it can never corrupt the state, but clamping
    /// would silently paper over a generation bug. The fuzz harness checks this
    /// before every application so an unaffordable choice is a loud failure.
    pub fn affordable(&self, g: &GameState, p: PlayerId) -> bool {
        let pl = &g.players[p.idx()];
        let mut corn = pl.corn as i32;
        let mut res = [
            pl.res[0] as i32,
            pl.res[1] as i32,
            pl.res[2] as i32,
            pl.res[3] as i32,
        ];
        for e in &self.0 {
            match *e {
                Effect::Corn(n) => corn += n as i32,
                Effect::SetCorn(n) => corn = n as i32,
                Effect::Res(r, n) => res[r.idx()] += n as i32,
                _ => {}
            }
            if corn < 0 || res.iter().any(|&v| v < 0) {
                return false;
            }
        }
        true
    }

    pub fn apply(&self, g: &mut GameState, p: PlayerId) {
        for e in &self.0 {
            e.apply(g, p);
        }
    }

    /// Net change to a single resource across this choice, used by generation to
    /// check affordability without applying anything.
    pub fn net_res(&self, r: Resource) -> i32 {
        self.0
            .iter()
            .filter_map(|e| match e {
                Effect::Res(rr, n) if *rr == r => Some(*n as i32),
                _ => None,
            })
            .sum()
    }

    pub fn net_corn(&self) -> i32 {
        self.0
            .iter()
            .filter_map(|e| match e {
                Effect::Corn(n) => Some(*n as i32),
                _ => None,
            })
            .sum()
    }
}

impl fmt::Display for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("skip");
        }
        for (i, e) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}
