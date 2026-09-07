//! The 32 buildings, transcribed from Go `impl/buildings/age1.go` and `age2.go`.
//!
//! Ids are globally unique here: age 1 is 1..=14, age 2 is 15..=32. The Go
//! version numbered both decks from 1, so an age-1 card and an age-2 card could
//! share an id. That fed two things: `DealBuildings` treated an owned age-1 card
//! as a reason not to deal the same-numbered age-2 card, and Tikal's double
//! build excluded the wrong card from its second half.

use crate::effect::Effect;
use crate::ids::*;

/// How a building's benefit expands into concrete choices at generation time.
pub enum Payoff {
    /// Exactly one choice, always available.
    Fixed(&'static [Effect]),
    /// One free advance on a named track, then the tail. At level 3 the track's
    /// top payoff is enumerated instead of an advance.
    FreeTrack(Science, &'static [Effect]),
    /// `n` free advances on any tracks, then the tail.
    FreeAny(u8, &'static [Effect]),
    /// Build a second building, excluding this one, paying its cost; then one
    /// temple step of the player's choice.
    BuildAnother,
    /// The Uxmal corn-for-blocks exchange, then the tail.
    CornExchange(&'static [Effect]),
    /// The Uxmal mirror: any single action from another gear, then the tail.
    Mirror(&'static [Effect]),
}

pub struct BuildingDef {
    pub id: BuildingId,
    pub cost: Bundle,
    pub color: Color,
    pub payoff: Payoff,
}

pub const N_AGE1: usize = 14;
pub const N_AGE2: usize = 18;
pub const N_BUILDINGS: usize = N_AGE1 + N_AGE2;

/// Indexed by `id - 1`.
pub static BUILDINGS: [BuildingDef; N_BUILDINGS] = [
    // ---------------- Age 1 ----------------
    b(1, bundle(1, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::FreeWorker(1)])),
    b(2, bundle(1, 2, 0, 0), Color::Red, Payoff::Fixed(&[
        Effect::TempleStep(Temple::Brown, 1),
        Effect::TempleStep(Temple::Green, 1),
    ])),
    // Duplicate of #1.
    b(3, bundle(1, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::FreeWorker(1)])),
    b(4, bundle(2, 0, 0, 0), Color::Green, Payoff::FreeTrack(Science::Agriculture, &[])),
    b(5, bundle(4, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::WorkerDiscount(1)])),
    b(6, bundle(0, 0, 1, 0), Color::Blue, Payoff::FreeTrack(Science::Architecture, &[])),
    b(7, bundle(1, 0, 1, 0), Color::Red, Payoff::BuildAnother),
    b(8, bundle(2, 1, 0, 0), Color::Red, Payoff::Fixed(&[
        Effect::TempleStep(Temple::Brown, 1),
        Effect::TempleStep(Temple::Yellow, 1),
    ])),
    // The Go description said "1 G" but the body added corn; the body wins.
    b(9, bundle(1, 1, 0, 0), Color::Green,
      Payoff::FreeTrack(Science::Extraction, &[Effect::Corn(1)])),
    b(10, bundle(0, 1, 1, 0), Color::Blue,
      Payoff::FreeTrack(Science::Theology, &[Effect::TempleStep(Temple::Green, 1)])),
    // Duplicate of #5.
    b(11, bundle(4, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::WorkerDiscount(1)])),
    b(12, bundle(2, 1, 0, 0), Color::Green,
      Payoff::FreeTrack(Science::Extraction, &[Effect::Res(Resource::Gold, 1)])),
    b(13, bundle(3, 0, 0, 0), Color::Green,
      Payoff::FreeTrack(Science::Agriculture, &[Effect::Res(Resource::Stone, 1)])),
    // Duplicate of #1.
    b(14, bundle(1, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::FreeWorker(1)])),

    // ---------------- Age 2 ----------------
    b(15, bundle(0, 0, 2, 0), Color::Blue, Payoff::Fixed(&[
        Effect::TempleStep(Temple::Green, 1),
        Effect::TempleStep(Temple::Green, 1),
        Effect::Points(3),
    ])),
    b(16, bundle(0, 2, 0, 0), Color::Blue, Payoff::Fixed(&[
        Effect::TempleStep(Temple::Brown, 1),
        Effect::TempleStep(Temple::Brown, 1),
        Effect::Points(2),
    ])),
    // The Go card described "2 YT, 4 points" but awarded no points. Restored.
    b(17, bundle(0, 0, 3, 0), Color::Blue, Payoff::Fixed(&[
        Effect::TempleStep(Temple::Yellow, 1),
        Effect::TempleStep(Temple::Yellow, 1),
        Effect::Points(4),
    ])),
    b(18, bundle(0, 1, 2, 0), Color::Blue, Payoff::FreeAny(2, &[])),
    b(19, bundle(0, 2, 1, 0), Color::Blue, Payoff::FreeTrack(Science::Theology, &[
        Effect::TempleStep(Temple::Brown, 1),
        Effect::TempleStep(Temple::Green, 1),
    ])),
    b(20, bundle(3, 0, 0, 0), Color::Green,
      Payoff::FreeAny(1, &[Effect::Res(Resource::Stone, 1)])),
    b(21, bundle(1, 1, 1, 0), Color::Red, Payoff::Fixed(&[
        Effect::UnlockWorker,
        Effect::Points(6),
    ])),
    b(22, bundle(0, 0, 3, 0), Color::Red, Payoff::CornExchange(&[Effect::Points(6)])),
    b(23, bundle(1, 0, 2, 0), Color::Red, Payoff::Fixed(&[Effect::Points(8)])),
    b(24, bundle(2, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::FreeWorker(3)])),
    b(25, bundle(1, 2, 1, 0), Color::Red, Payoff::Fixed(&[
        Effect::TempleStep(Temple::Brown, 1),
        Effect::TempleStep(Temple::Yellow, 1),
        Effect::TempleStep(Temple::Green, 1),
        Effect::Points(3),
    ])),
    b(26, bundle(2, 2, 0, 0), Color::Green,
      Payoff::FreeAny(1, &[Effect::Res(Resource::Skull, 1)])),
    b(27, bundle(0, 1, 1, 0), Color::Blue,
      Payoff::FreeTrack(Science::Architecture, &[Effect::Points(3)])),
    // Duplicates of #24.
    b(28, bundle(2, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::FreeWorker(3)])),
    b(29, bundle(2, 0, 0, 0), Color::Yellow, Payoff::Fixed(&[Effect::FreeWorker(3)])),
    // Commented out in Go, so the card was face-up but unbuildable and jammed a
    // display slot for the rest of the game. Implemented.
    b(30, bundle(2, 1, 1, 0), Color::Red, Payoff::Mirror(&[Effect::Points(2)])),
    b(31, bundle(0, 0, 1, 0), Color::Green, Payoff::FreeAny(1, &[Effect::Corn(6)])),
    b(32, bundle(2, 1, 0, 0), Color::Green,
      Payoff::FreeAny(1, &[Effect::Res(Resource::Gold, 1)])),
];

const fn b(id: u8, cost: Bundle, color: Color, payoff: Payoff) -> BuildingDef {
    BuildingDef {
        id: BuildingId(id),
        cost,
        color,
        payoff,
    }
}

#[inline]
pub fn def(id: BuildingId) -> &'static BuildingDef {
    &BUILDINGS[(id.0 - 1) as usize]
}

/// Ids in each age's deck, before shuffling.
pub fn age1_ids() -> [u8; N_AGE1] {
    std::array::from_fn(|i| (i + 1) as u8)
}

pub fn age2_ids() -> [u8; N_AGE2] {
    std::array::from_fn(|i| (N_AGE1 + i + 1) as u8)
}
