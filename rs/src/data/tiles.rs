//! The 21 starting wealth tiles.
//!
//! Each player is dealt four and keeps two, gaining everything depicted.
//!
//! Transcribed from a physical copy and numbered to match it. An audit pass
//! flagged these as incomplete, on the grounds that the rulebook's page-16
//! symbol glossary defines things no tile here uses: "you may construct a
//! building", "you may exchange at the market", "pay 1 corn and perform any
//! action", "advance 2 levels", "2 victory points", the worker feeding
//! discount, and the choose-your-own track and temple symbols.
//!
//! That was a false positive. The glossary is headed "Starting Wealth Tiles
//! **and Building Effects**" — it is shared between the two, and those symbols
//! belong to the buildings. Checked tile by tile against the components: no
//! starting tile carries a decision at all, so a plain effect list is the right
//! shape and every symbol used here is choice-free.
//!
//! Setup happens before anyone has research, so a free advance is always a
//! plain advance.

use crate::effect::Effect;
use crate::ids::*;

pub const N_TILES: usize = 21;

use Effect::*;
use Resource::{Gold, Skull, Stone, Wood};
use Science::{Agriculture, Architecture, Extraction, Theology};
use Temple::{Brown, Green, Yellow};

pub static TILES: [&[Effect]; N_TILES] = [
    /*  1 */ &[Corn(3), Res(Wood, 1), FreeWorker(1)],
    /*  2 */ &[Corn(6), Res(Wood, 1), Res(Stone, 1)],
    /*  3 */ &[Corn(2), Res(Wood, 2), TempleStep(Green, 1)],
    /*  4 */ &[UnlockWorker],
    /*  5 */ &[Corn(8), Res(Gold, 1)],
    /*  6 */ &[Corn(4), Res(Wood, 3)],
    /*  7 */ &[Corn(7), Res(Wood, 2)],
    /*  8 */ &[Corn(6), Res(Stone, 2)],
    /*  9 */ &[Corn(3), Res(Wood, 2), Res(Stone, 1)],
    /* 10 */ &[Res(Wood, 1), TempleStep(Green, 1), AdvanceResearch(Extraction)],
    /* 11 */ &[Res(Stone, 1), Res(Gold, 1), AdvanceResearch(Agriculture)],
    /* 12 */ &[Corn(4), Res(Wood, 1), AdvanceResearch(Extraction)],
    /* 13 */ &[Corn(5), Res(Gold, 1), TempleStep(Yellow, 1)],
    /* 14 */ &[Corn(5), Res(Stone, 1), TempleStep(Brown, 1)],
    /* 15 */ &[Corn(9), Res(Stone, 1)],
    /* 16 */ &[Corn(2), TempleStep(Brown, 1), AdvanceResearch(Architecture)],
    /* 17 */ &[Corn(3), TempleStep(Yellow, 1), AdvanceResearch(Agriculture)],
    /* 18 */ &[Corn(4), Res(Wood, 1), Res(Skull, 1)],
    /* 19 */ &[Corn(5), Res(Stone, 1), AdvanceResearch(Theology)],
    /* 20 */ &[Corn(3), Res(Gold, 1), AdvanceResearch(Architecture)],
    /* 21 */ &[Corn(2), Res(Wood, 2), AdvanceResearch(Theology)],
];

pub fn tile_ids() -> [u8; N_TILES] {
    std::array::from_fn(|i| i as u8)
}
