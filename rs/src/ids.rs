//! Newtype indices and small closed enums.
//!
//! Everything here is `Copy` and index-shaped: no pointers into the game state,
//! which is what makes `GameState` a plain `Copy` struct.

use std::fmt;

pub const N_PLAYERS: usize = 4;
pub const WORKERS_PER_PLAYER: usize = 6;
pub const N_WORKERS: usize = N_PLAYERS * WORKERS_PER_PLAYER;
pub const STARTING_WORKERS: usize = 3;

/// Index into `GameState::players`. Distinct from `Color` on purpose: the Go
/// version used them interchangeably (`Players[p.Color]`), which only worked
/// because seating order happened to equal color order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PlayerId(pub u8);

impl PlayerId {
    pub const ALL: [PlayerId; N_PLAYERS] =
        [PlayerId(0), PlayerId(1), PlayerId(2), PlayerId(3)];

    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }

    /// Seat `n` places clockwise from this one.
    #[inline]
    pub fn next(self, n: usize) -> PlayerId {
        PlayerId(((self.0 as usize + n) % N_PLAYERS) as u8)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct WorkerId(pub u8);

impl WorkerId {
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }

    /// Workers are allocated in per-player blocks.
    #[inline]
    pub fn owner(self) -> PlayerId {
        PlayerId(self.0 / WORKERS_PER_PLAYER as u8)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Color {
    Red,
    Green,
    Blue,
    Yellow,
}

impl Color {
    pub const ALL: [Color; 4] = [Color::Red, Color::Green, Color::Blue, Color::Yellow];

    pub fn letter(self) -> char {
        match self {
            Color::Red => 'R',
            Color::Green => 'G',
            Color::Blue => 'B',
            Color::Yellow => 'Y',
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Color::Red => "R",
            Color::Green => "G",
            Color::Blue => "B",
            Color::Yellow => "Y",
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Resource {
    Wood,
    Stone,
    Gold,
    Skull,
}

impl Resource {
    /// The three *blocks*. Skulls are not blocks and can never be spent as one.
    pub const BLOCKS: [Resource; 3] = [Resource::Wood, Resource::Stone, Resource::Gold];
    pub const ALL: [Resource; 4] = [
        Resource::Wood,
        Resource::Stone,
        Resource::Gold,
        Resource::Skull,
    ];

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }

    pub fn letter(self) -> char {
        match self {
            Resource::Wood => 'W',
            Resource::Stone => 'S',
            Resource::Gold => 'G',
            Resource::Skull => 'C',
        }
    }

    /// End-of-game corn value of one unit (skulls score separately).
    pub fn corn_value(self) -> i32 {
        match self {
            Resource::Wood => 2,
            Resource::Stone => 3,
            Resource::Gold => 4,
            Resource::Skull => 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Science {
    Agriculture,
    Extraction,
    Architecture,
    Theology,
}

impl Science {
    pub const ALL: [Science; 4] = [
        Science::Agriculture,
        Science::Extraction,
        Science::Architecture,
        Science::Theology,
    ];

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }

    pub fn letter(self) -> char {
        match self {
            Science::Agriculture => 'A',
            Science::Extraction => 'R',
            Science::Architecture => 'C',
            Science::Theology => 'T',
        }
    }
}

/// Named by track colour, matching the Go source's `TempleDebug = "BYG"`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Temple {
    Brown,
    Yellow,
    Green,
}

impl Temple {
    pub const ALL: [Temple; 3] = [Temple::Brown, Temple::Yellow, Temple::Green];

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }

    pub fn letter(self) -> char {
        match self {
            Temple::Brown => 'B',
            Temple::Yellow => 'Y',
            Temple::Green => 'G',
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Gear {
    Palenque,
    Yaxchilan,
    Tikal,
    Uxmal,
    Chichen,
}

impl Gear {
    pub const ALL: [Gear; 5] = [
        Gear::Palenque,
        Gear::Yaxchilan,
        Gear::Tikal,
        Gear::Uxmal,
        Gear::Chichen,
    ];

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }

    /// Number of worker spaces: 8 on the small gears (0-7), 11 on Chichen
    /// Itza (0-10).
    ///
    /// Counting a physical copy gives 7 and 10, because space 0 carries no
    /// printed number -- it is the blank entry space. The printed numbers run
    /// 1..7 and 1..10, so a count of the *labels* is one short of the count of
    /// the spaces a worker can stand on. The rulebook settles it: the free
    /// choice action is "actions 6 and 7 on most gears and action 10 on Chichen
    /// Itza", and its extra-day example has a worker on space 6 blocking the
    /// privilege while one on space 7 does not, so 7 exists and is distinct
    /// from the blocking space.
    ///
    /// The gears have more *holes* than spaces (10 and 13) -- two of them fall
    /// in the dead arc where the gear meshes with the central calendar and the
    /// board prints no action there.
    pub fn size(self) -> u8 {
        match self {
            Gear::Chichen => 11,
            _ => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Gear::Palenque => "Palenque",
            Gear::Yaxchilan => "Yaxchilan",
            Gear::Tikal => "Tikal",
            Gear::Uxmal => "Uxmal",
            Gear::Chichen => "Chichen Itza",
        }
    }
}

/// A worker space on a gear, 0-based from the entry space.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Pos(pub u8);

impl Pos {
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BuildingId(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MonumentId(pub u8);

/// Which face of a Palenque jungle tile is taken.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum TileKind {
    Corn,
    Wood,
}

/// A cost or resource bundle, indexed by [`Resource`].
pub type Bundle = [i8; 4];

pub const EMPTY: Bundle = [0; 4];

pub const fn bundle(wood: i8, stone: i8, gold: i8, skull: i8) -> Bundle {
    [wood, stone, gold, skull]
}
