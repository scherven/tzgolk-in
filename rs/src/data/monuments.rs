//! The 13 monuments, transcribed from Go `impl/buildings/monument.go`.
//!
//! Scoring stays a function here rather than an `Effect`: it is a pure read-only
//! query run once at the end, so it never needs to be compared, hashed or
//! rolled back, and reads far better as code.
//!
//! Three of these panicked in Go:
//!   * #6  indexed `[0,0,0,6,12,18]` by available workers, which reaches 6.
//!   * #10 indexed `[0,6,5,4]` by `len(Players)`, which is always 4 -- so this
//!     monument crashed end-of-game scoring in every single game.
//!   * #13 read `CData.Full` over every Chichen space, but the mirror space at
//!     the top has no `CData`, so it dereferenced nil.

use crate::ids::*;
use crate::state::GameState;

pub struct MonumentDef {
    pub id: MonumentId,
    pub cost: Bundle,
    pub color: Color,
    pub score: fn(&GameState, PlayerId) -> i32,
}

pub const N_MONUMENTS: usize = 13;

fn count_buildings_of(g: &GameState, p: PlayerId, c: Color) -> i32 {
    g.players[p.idx()]
        .building_ids()
        .filter(|&id| crate::data::buildings::def(id).color == c)
        .count() as i32
}

fn count_monuments_of(g: &GameState, p: PlayerId, c: Color) -> i32 {
    g.players[p.idx()]
        .monument_ids()
        .filter(|&id| def(id).color == c)
        .count() as i32
}

pub static MONUMENTS: [MonumentDef; N_MONUMENTS] = [
    // 3 VP per step above the starting space on the best temple. Go used the
    // absolute step index with a floor of 2, paying 6 at the starting position.
    MonumentDef {
        id: MonumentId(1),
        cost: bundle(0, 3, 3, 0),
        color: Color::Blue,
        score: |g, p| {
            let highest = Temple::ALL
                .iter()
                .map(|&t| g.temple_pos(p, t) as i32)
                .max()
                .unwrap_or(0);
            (highest - crate::data::temples::STARTING_STEP as i32).max(0) * 3
        },
    },
    // 4 per green building and green monument (this card counts itself).
    MonumentDef {
        id: MonumentId(2),
        cost: bundle(2, 3, 1, 0),
        color: Color::Green,
        score: |g, p| {
            4 * (count_buildings_of(g, p, Color::Green) + count_monuments_of(g, p, Color::Green))
        },
    },
    // Sum of the point values of the three temple steps occupied.
    MonumentDef {
        id: MonumentId(3),
        cost: bundle(0, 4, 3, 0),
        color: Color::Blue,
        score: |g, p| {
            Temple::ALL
                .iter()
                .map(|&t| {
                    let step = g.temple_pos(p, t) as usize;
                    crate::data::temples::TEMPLES[t.idx()].points[step] as i32
                })
                .sum()
        },
    },
    MonumentDef {
        id: MonumentId(4),
        cost: bundle(3, 2, 1, 0),
        color: Color::Red,
        score: |g, p| {
            4 * (count_buildings_of(g, p, Color::Red) + count_monuments_of(g, p, Color::Red))
        },
    },
    MonumentDef {
        id: MonumentId(5),
        cost: bundle(0, 2, 3, 0),
        color: Color::Blue,
        score: |g, p| {
            4 * (count_buildings_of(g, p, Color::Blue) + count_monuments_of(g, p, Color::Blue))
        },
    },
    // Workers *in play* -- anything not still in the bank, so workers on gears
    // count too. Go counted only the ones in hand, and indexed a 6-entry table
    // by a count that reaches 6.
    MonumentDef {
        id: MonumentId(6),
        cost: bundle(3, 0, 3, 0),
        color: Color::Green,
        score: |g, p| {
            const TABLE: [i32; 7] = [0, 0, 0, 0, 6, 12, 18];
            TABLE[g.n_unlocked(p).min(6)]
        },
    },
    MonumentDef {
        id: MonumentId(7),
        cost: bundle(1, 1, 4, 0),
        color: Color::Red,
        score: |g, p| 4 * g.players[p.idx()].corn_tiles as i32,
    },
    MonumentDef {
        id: MonumentId(8),
        cost: bundle(1, 0, 4, 0),
        color: Color::Red,
        score: |g, p| 4 * g.players[p.idx()].wood_tiles as i32,
    },
    MonumentDef {
        id: MonumentId(9),
        cost: bundle(1, 3, 2, 0),
        color: Color::Red,
        score: |g, p| {
            let pl = &g.players[p.idx()];
            2 * (pl.n_buildings() as i32 + pl.n_monuments() as i32)
        },
    },
    // Per monument owned by anyone, scaled by player count. The Go lookup was
    // off by one and always went out of bounds at 4 players.
    MonumentDef {
        id: MonumentId(10),
        cost: bundle(2, 2, 2, 0),
        color: Color::Red,
        score: |g, _p| {
            let total: u32 = g.players.iter().map(|pl| pl.n_monuments()).sum();
            let per = match N_PLAYERS {
                2 => 6,
                3 => 5,
                _ => 4,
            };
            per * total as i32
        },
    },
    MonumentDef {
        id: MonumentId(11),
        cost: bundle(1, 1, 3, 0),
        color: Color::Red,
        score: |g, p| {
            const TABLE: [i32; 5] = [0, 9, 20, 33, 33];
            let maxed = g.research[p.idx()].iter().filter(|&&l| l == 3).count();
            TABLE[maxed]
        },
    },
    MonumentDef {
        id: MonumentId(12),
        cost: bundle(2, 1, 3, 0),
        color: Color::Red,
        score: |g, p| 3 * g.research[p.idx()].iter().map(|&l| l as i32).sum::<i32>(),
    },
    // Chichen spaces used this game. Only the nine skull spaces have a state to
    // read; the mirror space at the top has none.
    MonumentDef {
        id: MonumentId(13),
        cost: bundle(0, 0, 4, 1),
        color: Color::Red,
        score: |g, _p| 3 * g.chichen_filled.count_ones() as i32,
    },
];

#[inline]
pub fn def(id: MonumentId) -> &'static MonumentDef {
    &MONUMENTS[(id.0 - 1) as usize]
}

pub fn monument_ids() -> [u8; N_MONUMENTS] {
    std::array::from_fn(|i| (i + 1) as u8)
}
