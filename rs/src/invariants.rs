//! Structural invariants, checked after every turn by the fuzz harness.
//!
//! These are the properties the Go version had no way to state, let alone test:
//! it had zero test files across 4,462 lines, and its debugging loop was playing
//! a game by hand and diffing a 64KB log against the rulebook.

use crate::data::buildings::N_BUILDINGS;
use crate::ids::*;
use crate::state::{GameState, WorkerLoc, EMPTY_SPACE, LAST_DAY};

pub fn validate(g: &GameState) -> Result<(), String> {
    workers_consistent(g)?;
    temples_in_bounds(g)?;
    research_in_bounds(g)?;
    tiles_sane(g)?;
    cards_unique(g)?;
    calendar_sane(g)?;
    skulls_conserved(g)?;
    Ok(())
}

/// Every worker is in exactly one place, and the gears agree with the workers.
fn workers_consistent(g: &GameState) -> Result<(), String> {
    // Worker -> gear.
    for (i, loc) in g.workers.iter().enumerate() {
        let w = WorkerId(i as u8);
        if let WorkerLoc::OnGear { gear, pos } = *loc {
            if pos.0 >= gear.size() {
                return Err(format!("worker {i} at {}:{} past the end", gear.name(), pos.0));
            }
            match g.gears[gear.idx()].at(pos) {
                Some(occ) if occ == w => {}
                other => {
                    return Err(format!(
                        "worker {i} thinks it is at {}:{} but that space holds {other:?}",
                        gear.name(),
                        pos.0
                    ))
                }
            }
        }
    }

    // Gear -> worker.
    for gear in Gear::ALL {
        for pos in 0..gear.size() {
            let v = g.gears[gear.idx()].occ[pos as usize];
            if v == EMPTY_SPACE {
                continue;
            }
            match g.workers[v as usize] {
                WorkerLoc::OnGear { gear: gg, pos: pp } if gg == gear && pp.0 == pos => {}
                other => {
                    return Err(format!(
                        "{}:{} holds worker {v}, but that worker is {other:?}",
                        gear.name(),
                        pos
                    ))
                }
            }
        }
        // Nothing may sit above an empty space: workers enter at the lowest free
        // space, so occupancy is only ever a suffix-free prefix pattern broken
        // by rotation. Just check no duplicates.
        let mut seen = [false; N_WORKERS];
        for pos in 0..gear.size() {
            let v = g.gears[gear.idx()].occ[pos as usize];
            if v != EMPTY_SPACE {
                if seen[v as usize] {
                    return Err(format!("worker {v} occupies two spaces on {}", gear.name()));
                }
                seen[v as usize] = true;
            }
        }
    }

    // The first player space holds at most one worker, and it agrees.
    match g.first_player_space {
        Some(w) => {
            if g.workers[w.idx()] != WorkerLoc::FirstPlayerSpace {
                return Err(format!(
                    "first player space holds worker {}, which is {:?}",
                    w.0,
                    g.workers[w.idx()]
                ));
            }
        }
        None => {
            if let Some(i) = g
                .workers
                .iter()
                .position(|l| *l == WorkerLoc::FirstPlayerSpace)
            {
                return Err(format!(
                    "worker {i} is on the first player space, but the space is empty"
                ));
            }
        }
    }

    // Worker count per player never changes.
    for p in PlayerId::ALL {
        if GameState::worker_ids(p).count() != WORKERS_PER_PLAYER {
            return Err(format!("player {p:?} lost workers"));
        }
    }
    Ok(())
}

fn temples_in_bounds(g: &GameState) -> Result<(), String> {
    for t in Temple::ALL {
        let max = crate::data::temples::TEMPLES[t.idx()].steps - 1;
        for p in PlayerId::ALL {
            let s = g.temple_pos(p, t);
            if s > max {
                return Err(format!("{p:?} is at step {s} on {t:?}, max {max}"));
            }
        }
    }
    Ok(())
}

fn research_in_bounds(g: &GameState) -> Result<(), String> {
    for p in PlayerId::ALL {
        for s in Science::ALL {
            let l = g.level(p, s);
            if l > 3 {
                return Err(format!("{p:?} is at {s:?} level {l}"));
            }
        }
    }
    Ok(())
}

/// Palenque tiles are consumed and never replaced, so counts only fall, and a
/// space that started with four of each can never exceed that.
fn tiles_sane(g: &GameState) -> Result<(), String> {
    for i in 2..=5usize {
        let s = g.palenque[i];
        if s.corn > 4 || s.wood > 4 {
            return Err(format!("palenque {i} has {s:?}, above the starting supply"));
        }
    }
    Ok(())
}

/// No card exists in two places at once.
fn cards_unique(g: &GameState) -> Result<(), String> {
    let mut owner = [0u8; N_BUILDINGS + 1];
    for p in PlayerId::ALL {
        for id in g.players[p.idx()].building_ids() {
            if owner[id.0 as usize] != 0 {
                return Err(format!("building {} owned twice", id.0));
            }
            owner[id.0 as usize] = 1 + p.0;
        }
    }
    for id in g.face_up_buildings() {
        if owner[id.0 as usize] != 0 {
            return Err(format!(
                "building {} is face up and also owned by player {}",
                id.0,
                owner[id.0 as usize] - 1
            ));
        }
        owner[id.0 as usize] = 200;
    }

    let mut seen = [false; 16];
    for p in PlayerId::ALL {
        for id in g.players[p.idx()].monument_ids() {
            if seen[id.0 as usize] {
                return Err(format!("monument {} owned twice", id.0));
            }
            seen[id.0 as usize] = true;
        }
    }
    for id in g.face_up_monuments() {
        if seen[id.0 as usize] {
            return Err(format!("monument {} is face up and also owned", id.0));
        }
        seen[id.0 as usize] = true;
    }
    Ok(())
}

/// There are exactly 13 crystal skulls: in the bank, in someone's hand, or laid
/// on a Chichen Itza space. None are created and none are destroyed.
fn skulls_conserved(g: &GameState) -> Result<(), String> {
    let held: u32 = PlayerId::ALL
        .iter()
        .map(|&p| g.players[p.idx()].get(Resource::Skull) as u32)
        .sum();
    let on_board = g.chichen_filled.count_ones();
    let total = g.skulls_remaining as u32 + held + on_board;
    if total != crate::state::N_SKULLS as u32 {
        return Err(format!(
            "{total} crystal skulls accounted for, expected {} ({} in bank, {held} held, {on_board} placed)",
            crate::state::N_SKULLS,
            g.skulls_remaining
        ));
    }
    Ok(())
}

fn calendar_sane(g: &GameState) -> Result<(), String> {
    if g.day > LAST_DAY {
        return Err(format!("day {} is past the end of the calendar", g.day));
    }
    if g.age < 1 || g.age > 3 {
        return Err(format!("age {}", g.age));
    }
    if g.chichen_filled & !0b111_1111_1110 != 0 {
        return Err(format!(
            "chichen bitset {:b} marks a space that is not a skull space",
            g.chichen_filled
        ));
    }
    Ok(())
}
