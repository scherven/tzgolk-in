//! The replay record format, the shard writer, and the agent seam.
//!
//! It was previously `#[path]`-included into `selfplay.rs` and `arena.rs`,
//! which compiled it twice. It is now a proper module: `src/ffi.rs` needs
//! `decode_state` and `decode_phase` to hand the trainer the *real* encoder
//! rather than a numpy reimplementation of it.
//!
//! # What is in here
//!
//! * the record format and its codec, plus `schema_json` for the Python loader;
//! * `ShardWriter`, the durability policy and the `SIGINT` handler;
//! * `BatchQueue`, the inter-game evaluation batcher of `COMPUTE.md` §2.6;
//! * `Agent` and its implementations, including `SearchAgent`, the seam to
//!   `src/mcts.rs`;
//! * `AgentSpec`, which parses `--agent`/`--candidate` and hands each worker
//!   thread its own instance;
//! * `play_game`, the round driver both binaries share, and `Summary`, the
//!   statistics the arena reports with.
//!
//! # The format, in one paragraph
//!
//! A shard is a 64-byte header followed by fixed-size 512-byte records, little
//! endian throughout, no compression, no framing. Fixed-size records are the
//! whole design: `numpy.memmap` with a structured dtype reads a 4 GB shard in
//! microseconds and never parses anything, a shard truncated by a kill is still
//! valid up to the last whole record, and two shards concatenate. The price is
//! ~40 bytes per record of duplicated game-level trailer (final scores, `z_rel`,
//! `win_share`), which is 8% of the record and removes the need for the loader
//! to join records back to their game.
//!
//! Records store the **raw `GameState`**, not an encoded tensor, per
//! `LEARNING.md` §6.2: the encoder is still being written, `src/` has changed
//! four times this week, and storing tensors would mean discarding every game
//! whenever a feature is added. `GameState` does not derive `Serialize` and
//! `state.rs` is not mine to edit, so the codec below is written by hand. It is
//! explicit rather than a `transmute`, because `repr(Rust)` layout is
//! unspecified and this format has to survive a compiler upgrade.
//!
//! Every record carries `RULES_VERSION`. A rules change invalidates the learned
//! value function even when every tensor shape survives (`LEARNING.md` §8.4), so
//! without the stamp the whole buffer is suspect instead of just the records
//! before the change.
//!
//! # Self-description
//!
//! `src/state.rs` moved under this file while it was being written (Chichen
//! went 10 spaces -> 11), so nothing downstream may hard-code an offset. The
//! layout constants below are derived from `MAX_GEAR_SPACES` and friends, and
//! `schema_json()` emits them, so `train/replay.py` builds its dtype from a
//! `schema.json` written next to the shards rather than from a second copy of
//! this table. The state occupies a **fixed 320-byte slot** whatever its live
//! width, so growing the state never moves any other field.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::Rng;

use crate::eval::Ranking;
use crate::ids::*;
use crate::moves::{apply_move, sample_legal_move, Move};
use crate::mcts::{Mcts, MctsConfig, Priors, SearchResult};
use crate::phase::{Evaluation, Evaluator, HeuristicEvaluator, Phase, Step};
use crate::tree;
use crate::state::{Deck, GameState, GearState, Player, TileStack, WorkerLoc, MAX_GEAR_SPACES, N_DISPLAY};
use crate::RULES_VERSION;

// =======================================================================
// Layout constants
// =======================================================================

/// Bytes one `Player` takes: colour, corn, four resources, points, two tile
/// counts, free workers, discount, the first-player tile face, and the two
/// bitsets.
pub const PLAYER_BYTES: usize = 1 + 1 + 4 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 2; // 19

/// Live width of the encoded `GameState`. Derived, never typed by hand.
pub const STATE_BYTES: usize = PLAYER_BYTES * N_PLAYERS      // players
    + N_WORKERS * 2                                          // worker locations
    + 5 * MAX_GEAR_SPACES                                    // gear occupancy
    + 3 * N_PLAYERS                                          // temples
    + N_PLAYERS * 4                                          // research
    + 8 * 2                                                  // palenque tiles
    + 2                                                      // chichen_filled
    + 15 + 19 + 14                                           // three decks
    + N_DISPLAY * 2                                          // face-up rows
    + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1; // fps, corn, skulls, cur, fp, age, day, over

/// The state always occupies this much of a record, live bytes then zeros.
///
/// Fixing the slot is what makes every field after it immovable when the state
/// grows. `STATE_BYTES` is 293 today; when it passes 320 the assertion below
/// breaks the build, which is the moment to bump `FORMAT_VERSION` and the slot
/// together rather than the moment to discover silent corruption.
pub const STATE_SLOT: usize = 320;

const _: () = assert!(
    STATE_BYTES <= STATE_SLOT,
    "GameState outgrew its record slot; bump STATE_SLOT and FORMAT_VERSION"
);

/// Visit pairs kept per node. `LEARNING.md` §6.2 asks for the top 24; the tail
/// of a visit distribution is noise and costs 4 bytes an entry to keep.
pub const MAX_VISITS: usize = 24;

pub const RECORD_BYTES: usize = 512;
pub const HEADER_BYTES: usize = 64;
pub const FORMAT_VERSION: u16 = 1;
pub const MAGIC: [u8; 4] = *b"TZZR";

// Field offsets inside a record. Kept as consts so `schema_json` and the
// writer cannot disagree.
const O_STATE: usize = 0;
const O_GAME_ID: usize = 320;
const O_NODE: usize = 328;
const O_TURN: usize = 330;
const O_MOVER: usize = 331;
const O_PHASE_TAG: usize = 332;
const O_PHASE_ARGS: usize = 333;
const O_POLICY_KIND: usize = 338;
const O_FLAGS: usize = 339;
const O_N_EDGES: usize = 340;
const O_N_PAIRS: usize = 342;
const O_TOTAL_VISITS: usize = 344;
const O_VISITS: usize = 348;
const O_POLICY_WEIGHT: usize = 444;
const O_DAY: usize = 448;
const O_TEMPERATURE: usize = 450;
const O_FINAL_SCORES: usize = 452;
const O_Z_REL: usize = 460;
const O_WIN_SHARE: usize = 476;
const O_ROOT_VALUE: usize = 492;
const O_RULES_VERSION: usize = 508;

const _: () = assert!(O_RULES_VERSION + 4 == RECORD_BYTES);

/// `policy_kind`: what the `visits` array indexes, and therefore whether a
/// policy target can be reconstructed at all.
pub mod policy_kind {
    /// No usable policy target. The record is a value target only.
    ///
    /// This is what the one-ply agents below emit: their candidates come from
    /// `sample_legal_move`, so the indices are not reproducible at training
    /// time. Such records are still worth every byte — they are exactly the
    /// `(GameState, final scores)` pairs the value warm-start of
    /// `LEARNING.md` §6.7 asks for.
    pub const NONE: u8 = 0;
    /// `visits[i].0` indexes the search tree's edge enumeration at
    /// `(state, phase)`, which the loader regenerates. Requires the tree's edge
    /// order to be deterministic — see `SEARCH.md` §5b note 3.
    pub const TREE_EDGE: u8 = 1;
}

/// `flags` bits.
pub mod flags {
    /// This node began a turn (as opposed to a sub-decision inside one).
    pub const TURN_ROOT: u8 = 1 << 0;
    /// This node got the full simulation budget, not the reduced one from
    /// playout-cap randomisation (`LEARNING.md` §6.4).
    pub const FULL_BUDGET: u8 = 1 << 1;
}

// =======================================================================
// GameState codec
// =======================================================================

struct Cur<'a> {
    b: &'a mut [u8],
    at: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a mut [u8]) -> Self {
        Cur { b, at: 0 }
    }
    fn u8(&mut self, v: u8) {
        self.b[self.at] = v;
        self.at += 1;
    }
    fn u16(&mut self, v: u16) {
        self.b[self.at..self.at + 2].copy_from_slice(&v.to_le_bytes());
        self.at += 2;
    }
    fn i16(&mut self, v: i16) {
        self.b[self.at..self.at + 2].copy_from_slice(&v.to_le_bytes());
        self.at += 2;
    }
    fn u32(&mut self, v: u32) {
        self.b[self.at..self.at + 4].copy_from_slice(&v.to_le_bytes());
        self.at += 4;
    }
    fn bytes(&mut self, v: &[u8]) {
        self.b[self.at..self.at + v.len()].copy_from_slice(v);
        self.at += v.len();
    }
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Rd { b, at: 0 }
    }
    fn u8(&mut self) -> u8 {
        let v = self.b[self.at];
        self.at += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.b[self.at], self.b[self.at + 1]]);
        self.at += 2;
        v
    }
    fn i16(&mut self) -> i16 {
        self.u16() as i16
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes([
            self.b[self.at],
            self.b[self.at + 1],
            self.b[self.at + 2],
            self.b[self.at + 3],
        ]);
        self.at += 4;
        v
    }
}

const NO_WORKER: u8 = 0xFF;

fn color_idx(c: Color) -> u8 {
    match c {
        Color::Red => 0,
        Color::Green => 1,
        Color::Blue => 2,
        Color::Yellow => 3,
    }
}

/// Encode `g` into the first `STATE_BYTES` of `out`; the rest of the slot is
/// left as the caller found it (the record buffer starts zeroed).
///
/// Field by field rather than a `transmute` of the struct: `GameState` is
/// `repr(Rust)`, so its padding and field order are the compiler's business and
/// may change between releases. A replay buffer that a `rustc` upgrade can
/// silently reinterpret is not a replay buffer.
pub fn encode_state(g: &GameState, out: &mut [u8]) {
    let mut c = Cur::new(&mut out[..STATE_BYTES]);

    for p in &g.players {
        c.u8(color_idx(p.color));
        c.u8(p.corn);
        c.bytes(&p.res);
        c.i16(p.points);
        c.u8(p.corn_tiles);
        c.u8(p.wood_tiles);
        c.u8(p.free_workers);
        c.u8(p.worker_discount);
        c.u8(p.may_skip_day as u8);
        c.u32(p.buildings);
        c.u16(p.monuments);
    }

    // `gear * 16 + pos` rather than `gear * 10 + pos`: gears have grown once
    // already this week and a power-of-two stride survives them growing again.
    for w in &g.workers {
        match *w {
            WorkerLoc::Locked => {
                c.u8(0);
                c.u8(0);
            }
            WorkerLoc::Available => {
                c.u8(1);
                c.u8(0);
            }
            WorkerLoc::OnGear { gear, pos } => {
                c.u8(2);
                c.u8((gear.idx() as u8) * 16 + pos.0);
            }
            WorkerLoc::FirstPlayerSpace => {
                c.u8(3);
                c.u8(0);
            }
        }
    }

    for gear in &g.gears {
        c.bytes(&gear.occ);
    }
    for t in &g.temples {
        c.bytes(t);
    }
    for r in &g.research {
        c.bytes(r);
    }
    for t in &g.palenque {
        c.u8(t.corn);
        c.u8(t.wood);
    }
    c.u16(g.chichen_filled);

    c.bytes(&g.age1.ids);
    c.u8(g.age1.next);
    c.bytes(&g.age2.ids);
    c.u8(g.age2.next);
    c.bytes(&g.monument_deck.ids);
    c.u8(g.monument_deck.next);

    // Building and monument ids are 1-based (the bitsets use `id - 1`), so 0 is
    // a free sentinel for an empty slot.
    for s in &g.buildings_up {
        c.u8(s.map(|b| b.0).unwrap_or(0));
    }
    for s in &g.monuments_up {
        c.u8(s.map(|m| m.0).unwrap_or(0));
    }

    c.u8(g.first_player_space.map(|w| w.0).unwrap_or(NO_WORKER));
    c.u8(g.accumulated_corn);
    c.u8(g.skulls_remaining);
    c.u8(g.current.0);
    c.u8(g.first_player.0);
    c.u8(g.age);
    c.u8(g.day);
    c.u8(g.over as u8);

    debug_assert_eq!(c.at, STATE_BYTES);
}

/// Inverse of `encode_state`. Round-tripped by the tests at the bottom of this
/// file, which is the only thing keeping the two in step.
pub fn decode_state(b: &[u8]) -> GameState {
    let mut r = Rd::new(&b[..STATE_BYTES]);

    let players: [Player; N_PLAYERS] = std::array::from_fn(|_| {
        let color = Color::ALL[r.u8() as usize];
        let corn = r.u8();
        let res: [u8; 4] = std::array::from_fn(|_| r.u8());
        let points = r.i16();
        let corn_tiles = r.u8();
        let wood_tiles = r.u8();
        let free_workers = r.u8();
        let worker_discount = r.u8();
        let may_skip_day = r.u8() != 0;
        let buildings = r.u32();
        let monuments = r.u16();
        Player {
            color,
            corn,
            res,
            points,
            corn_tiles,
            wood_tiles,
            free_workers,
            worker_discount,
            may_skip_day,
            buildings,
            monuments,
        }
    });

    let workers: [WorkerLoc; N_WORKERS] = std::array::from_fn(|_| {
        let tag = r.u8();
        let arg = r.u8();
        match tag {
            0 => WorkerLoc::Locked,
            1 => WorkerLoc::Available,
            2 => WorkerLoc::OnGear {
                gear: Gear::ALL[(arg / 16) as usize],
                pos: Pos(arg % 16),
            },
            _ => WorkerLoc::FirstPlayerSpace,
        }
    });

    let gears: [GearState; 5] = std::array::from_fn(|_| GearState {
        occ: std::array::from_fn(|_| r.u8()),
    });
    let temples: [[u8; N_PLAYERS]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| r.u8()));
    let research: [[u8; 4]; N_PLAYERS] = std::array::from_fn(|_| std::array::from_fn(|_| r.u8()));
    let palenque: [TileStack; 8] = std::array::from_fn(|_| TileStack {
        corn: r.u8(),
        wood: r.u8(),
    });
    let chichen_filled = r.u16();

    let a1: [u8; 14] = std::array::from_fn(|_| r.u8());
    let mut age1 = Deck::new(a1);
    age1.next = r.u8();
    let a2: [u8; 18] = std::array::from_fn(|_| r.u8());
    let mut age2 = Deck::new(a2);
    age2.next = r.u8();
    let am: [u8; 13] = std::array::from_fn(|_| r.u8());
    let mut monument_deck = Deck::new(am);
    monument_deck.next = r.u8();

    let buildings_up: [Option<BuildingId>; N_DISPLAY] = std::array::from_fn(|_| {
        let v = r.u8();
        (v != 0).then_some(BuildingId(v))
    });
    let monuments_up: [Option<MonumentId>; N_DISPLAY] = std::array::from_fn(|_| {
        let v = r.u8();
        (v != 0).then_some(MonumentId(v))
    });

    let fps = r.u8();
    GameState {
        players,
        workers,
        gears,
        temples,
        research,
        palenque,
        chichen_filled,
        age1,
        age2,
        monument_deck,
        buildings_up,
        monuments_up,
        first_player_space: (fps != NO_WORKER).then_some(WorkerId(fps)),
        accumulated_corn: r.u8(),
        skulls_remaining: r.u8(),
        current: PlayerId(r.u8()),
        first_player: PlayerId(r.u8()),
        age: r.u8(),
        day: r.u8(),
        over: r.u8() != 0,
    }
}

// =======================================================================
// Phase codec
// =======================================================================

/// Five bytes of payload, keyed by `Phase::tag()`. Deliberately a flat byte
/// array rather than a union: the Python loader reads it as `u1[5]` and only
/// the heads that care look inside.
pub fn encode_phase(p: Phase) -> (u8, [u8; 5]) {
    let mut a = [0u8; 5];
    match p {
        Phase::Placing { n } => a[0] = n,
        Phase::Take { worker } => a[0] = worker.0,
        Phase::ExtraDay { claimer } => a[0] = claimer.0,
        Phase::DraftTile { dealt, kept } => {
            a[..4].copy_from_slice(&dealt);
            a[4] = kept;
        }
        _ => {}
    }
    (p.tag(), a)
}

pub fn decode_phase(tag: u8, a: [u8; 5]) -> Phase {
    match tag {
        0 => Phase::Beg,
        1 => Phase::Mode,
        2 => Phase::Placing { n: a[0] },
        3 => Phase::PickWorker,
        4 => Phase::Take {
            worker: WorkerId(a[0]),
        },
        5 => Phase::ExtraDay {
            claimer: PlayerId(a[0]),
        },
        6 => Phase::PityPlace,
        _ => Phase::DraftTile {
            dealt: [a[0], a[1], a[2], a[3]],
            kept: a[4],
        },
    }
}

// =======================================================================
// The record
// =======================================================================

/// One decision node on the played path, before the game's outcome is known.
///
/// The agent produces these; the driver holds a game's worth in memory and
/// backfills the trailer when the game ends. A game is ~350 nodes, so ~180 kB —
/// buffering it is free, and it is what makes "a killed run loses at most the
/// game in flight" true without any partial-game bookkeeping on disk.
#[derive(Clone, Debug)]
pub struct Node {
    pub state: GameState,
    /// Whose turn it is. `phase.mover(turn)` is whose decision this is.
    pub turn: PlayerId,
    pub phase: Phase,
    /// Workers already resolved this turn, which `tree::legal_steps` needs at
    /// `PickWorker` to decide whether `StopRetrieving` is legal.
    ///
    /// `(state, phase)` is *not* quite a complete search node: at `PickWorker`
    /// the edge set also depends on this counter. Without it a loader
    /// regenerating the enumeration would get a list one edge short and every
    /// stored index after the missing one would be wrong. It rides in
    /// `phase_args[0]`, which only `PickWorker` leaves free.
    pub done: u8,
    pub policy_kind: u8,
    pub n_edges: u16,
    pub total_visits: u32,
    /// `(edge index, visit count)`, best first. Truncated to `MAX_VISITS`.
    pub visits: Vec<(u16, u16)>,
    /// The agent's own value estimate at this node, in seat order. Logged so a
    /// drift between predicted and realised outcome is visible without a
    /// separate calibration run.
    pub root_value: [f32; N_PLAYERS],
    pub flags: u8,
    pub temperature: f32,
}

impl Node {
    /// A value-only node: no policy target, just "this position, that outcome".
    pub fn value_only(state: GameState, turn: PlayerId, phase: Phase, value: [f32; N_PLAYERS]) -> Self {
        Node {
            state,
            turn,
            phase,
            done: 0,
            policy_kind: policy_kind::NONE,
            n_edges: 0,
            total_visits: 0,
            visits: Vec::new(),
            root_value: value,
            flags: flags::TURN_ROOT,
            temperature: 0.0,
        }
    }
}

/// How the outcome of a game is squashed into a value target.
///
/// `LEARNING.md` §3.2: centred so max^n backups have one currency, bounded so
/// `c_puct` calibrates, and near-linear through the bulk of the score
/// distribution so search has a gradient to climb from round 1. `T = 25` is
/// roughly the observed spread of final scores around the table mean.
pub const Z_SCALE: f32 = 25.0;

pub fn z_rel(scores: [i16; N_PLAYERS]) -> [f32; N_PLAYERS] {
    let mean = scores.iter().map(|&s| s as f32).sum::<f32>() / N_PLAYERS as f32;
    std::array::from_fn(|i| ((scores[i] as f32 - mean) / Z_SCALE).tanh())
}

/// `1/|winners|` for each winner, 0 for everyone else. Reporting and the rank
/// head only — never the primary value target (`LEARNING.md` §3.2).
pub fn win_share(g: &GameState) -> [f32; N_PLAYERS] {
    let w = g.winners();
    let share = if w.is_empty() { 0.0 } else { 1.0 / w.len() as f32 };
    let mut out = [0.0f32; N_PLAYERS];
    for p in w {
        out[p.idx()] = share;
    }
    out
}

/// The policy-loss weight of `LEARNING.md` §6.3: a 4-visit distribution is
/// noise and an 800-visit one is a target, so weight by how much search backed
/// it.
pub fn policy_weight(total_visits: u32) -> f32 {
    const FULL: f32 = 800.0;
    ((1.0 + total_visits as f32).ln() / (1.0 + FULL).ln()).clamp(0.0, 1.0)
}

/// Serialise one node plus its game trailer into a fixed-size record.
pub fn write_record(
    out: &mut [u8; RECORD_BYTES],
    n: &Node,
    game_id: u64,
    node_idx: u16,
    scores: [i16; N_PLAYERS],
) {
    out.fill(0);
    encode_state(&n.state, &mut out[O_STATE..O_STATE + STATE_SLOT]);

    let put_u16 = |o: &mut [u8; RECORD_BYTES], at: usize, v: u16| {
        o[at..at + 2].copy_from_slice(&v.to_le_bytes())
    };
    let put_u32 = |o: &mut [u8; RECORD_BYTES], at: usize, v: u32| {
        o[at..at + 4].copy_from_slice(&v.to_le_bytes())
    };
    let put_f32 = |o: &mut [u8; RECORD_BYTES], at: usize, v: f32| {
        o[at..at + 4].copy_from_slice(&v.to_le_bytes())
    };

    out[O_GAME_ID..O_GAME_ID + 8].copy_from_slice(&game_id.to_le_bytes());
    put_u16(out, O_NODE, node_idx);
    out[O_TURN] = n.turn.0;
    out[O_MOVER] = n.phase.mover(n.turn).0;

    let (tag, mut args) = encode_phase(n.phase);
    // `PickWorker` is the one phase whose args are all free, and the one phase
    // whose edge set depends on something outside `Phase` — see `Node::done`.
    if matches!(n.phase, Phase::PickWorker) {
        args[0] = n.done;
    }
    out[O_PHASE_TAG] = tag;
    out[O_PHASE_ARGS..O_PHASE_ARGS + 5].copy_from_slice(&args);

    out[O_POLICY_KIND] = n.policy_kind;
    out[O_FLAGS] = n.flags;
    put_u16(out, O_N_EDGES, n.n_edges);

    let pairs = n.visits.len().min(MAX_VISITS);
    put_u16(out, O_N_PAIRS, pairs as u16);
    put_u32(out, O_TOTAL_VISITS, n.total_visits);
    for (i, &(idx, cnt)) in n.visits.iter().take(MAX_VISITS).enumerate() {
        put_u16(out, O_VISITS + i * 4, idx);
        put_u16(out, O_VISITS + i * 4 + 2, cnt);
    }

    put_f32(out, O_POLICY_WEIGHT, policy_weight(n.total_visits));
    put_u16(out, O_DAY, n.state.day as u16);
    out[O_TEMPERATURE] = (n.temperature * 100.0).clamp(0.0, 255.0) as u8;

    for i in 0..N_PLAYERS {
        out[O_FINAL_SCORES + i * 2..O_FINAL_SCORES + i * 2 + 2]
            .copy_from_slice(&scores[i].to_le_bytes());
    }
    let z = z_rel(scores);
    for i in 0..N_PLAYERS {
        put_f32(out, O_Z_REL + i * 4, z[i]);
    }
    // `win_share` is derived from the terminal state, which the driver knows
    // and this function does not; the driver patches it in afterwards via
    // `patch_win_share`. Keeping it out of the signature keeps the common
    // path (one game, one shared value) from recomputing `winners()` 350 times.
    for i in 0..N_PLAYERS {
        put_f32(out, O_ROOT_VALUE + i * 4, n.root_value[i]);
    }
    put_u32(out, O_RULES_VERSION, RULES_VERSION);
}

pub fn patch_win_share(out: &mut [u8; RECORD_BYTES], ws: [f32; N_PLAYERS]) {
    for i in 0..N_PLAYERS {
        out[O_WIN_SHARE + i * 4..O_WIN_SHARE + i * 4 + 4].copy_from_slice(&ws[i].to_le_bytes());
    }
}

/// Everything `read_record` hands back. Deliberately partial: the trainer reads
/// records in Python, and this is for the verifier and the tests.
pub struct ReadRecord {
    pub state: GameState,
    pub turn: PlayerId,
    pub phase: Phase,
    /// The retrieval counter, so `tree::legal_steps` can be regenerated exactly.
    pub done: u8,
    pub scores: [i16; N_PLAYERS],
    pub rules_version: u32,
}

/// Read back the parts of a record a Rust-side consumer (the verifier, the
/// batcher) needs. Deliberately partial: the trainer reads records in Python.
pub fn read_record(b: &[u8]) -> ReadRecord {
    let state = decode_state(&b[O_STATE..O_STATE + STATE_SLOT]);
    let turn = PlayerId(b[O_TURN]);
    let phase = decode_phase(
        b[O_PHASE_TAG],
        [
            b[O_PHASE_ARGS],
            b[O_PHASE_ARGS + 1],
            b[O_PHASE_ARGS + 2],
            b[O_PHASE_ARGS + 3],
            b[O_PHASE_ARGS + 4],
        ],
    );
    let scores: [i16; N_PLAYERS] = std::array::from_fn(|i| {
        i16::from_le_bytes([b[O_FINAL_SCORES + i * 2], b[O_FINAL_SCORES + i * 2 + 1]])
    });
    let rules = u32::from_le_bytes([
        b[O_RULES_VERSION],
        b[O_RULES_VERSION + 1],
        b[O_RULES_VERSION + 2],
        b[O_RULES_VERSION + 3],
    ]);
    ReadRecord {
        state,
        turn,
        phase,
        done: if matches!(phase, Phase::PickWorker) {
            b[O_PHASE_ARGS]
        } else {
            0
        },
        scores,
        rules_version: rules,
    }
}

// =======================================================================
// Shard IO
// =======================================================================

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A UTC stamp with no separators, for filenames that sort chronologically.
pub fn stamp(unix: u64) -> String {
    // Civil-from-days, Howard Hinnant's algorithm. Written out rather than
    // pulled in as a dependency because the crate list is fixed and this is
    // fifteen lines.
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        y,
        m,
        d,
        secs / 3600,
        (secs / 60) % 60,
        secs % 60
    )
}

fn write_header(f: &mut impl Write, generation: u32, producer: &str) -> std::io::Result<()> {
    let mut h = [0u8; HEADER_BYTES];
    h[0..4].copy_from_slice(&MAGIC);
    h[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    h[6..8].copy_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
    h[8..12].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
    h[12..16].copy_from_slice(&RULES_VERSION.to_le_bytes());
    h[16..18].copy_from_slice(&(STATE_BYTES as u16).to_le_bytes());
    h[18..20].copy_from_slice(&(N_PLAYERS as u16).to_le_bytes());
    h[20..22].copy_from_slice(&(MAX_VISITS as u16).to_le_bytes());
    // 22..24 flags, reserved
    h[24..32].copy_from_slice(&now_unix().to_le_bytes());
    h[32..36].copy_from_slice(&generation.to_le_bytes());
    let p = producer.as_bytes();
    let n = p.len().min(28);
    h[36..36 + n].copy_from_slice(&p[..n]);
    f.write_all(&h)
}

/// Appends whole games to a `.part` file and rolls it into a timestamped shard.
///
/// Two levels of durability, deliberately separated:
///
/// * **flush after every game.** Costs nothing (it is a `write(2)` into the
///   page cache) and means a `SIGINT`, a panic, or a `kill -9` loses nothing
///   that finished.
/// * **fsync every `fsync_games` games.** Costs a millisecond or two and bounds
///   what a power cut can take. The default of 8 is a deliberate trade: at
///   self-play rates a lost 8 games is a rounding error, and one fsync per game
///   at 100 games/s is not.
pub struct ShardWriter {
    dir: PathBuf,
    generation: u32,
    producer: String,
    part: PathBuf,
    w: BufWriter<File>,
    /// Sealed shards this run, newest last.
    pub sealed: Vec<String>,
    pub games_in_part: u64,
    pub records_in_part: u64,
    pub games_total: u64,
    pub records_total: u64,
    since_fsync: u64,
    fsync_games: u64,
    seq: u32,
}

impl ShardWriter {
    pub fn new(
        dir: &Path,
        generation: u32,
        producer: &str,
        fsync_games: u64,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let part = dir.join(format!("gen{generation:04}.part"));
        let mut w = BufWriter::new(File::create(&part)?);
        write_header(&mut w, generation, producer)?;
        w.flush()?;
        Ok(ShardWriter {
            dir: dir.to_path_buf(),
            generation,
            producer: producer.to_string(),
            part,
            w,
            sealed: Vec::new(),
            games_in_part: 0,
            records_in_part: 0,
            games_total: 0,
            records_total: 0,
            since_fsync: 0,
            fsync_games: fsync_games.max(1),
            seq: 0,
        })
    }

    /// Append one finished game. All-or-nothing at the game level: the buffer
    /// is written in one call and flushed before returning.
    pub fn append_game(&mut self, records: &[u8]) -> std::io::Result<()> {
        debug_assert_eq!(records.len() % RECORD_BYTES, 0);
        self.w.write_all(records)?;
        self.w.flush()?;
        self.games_in_part += 1;
        self.games_total += 1;
        let n = (records.len() / RECORD_BYTES) as u64;
        self.records_in_part += n;
        self.records_total += n;
        self.since_fsync += 1;
        if self.since_fsync >= self.fsync_games {
            self.w.get_ref().sync_data()?;
            self.since_fsync = 0;
        }
        Ok(())
    }

    /// Seal the current part into a timestamped shard and open a fresh one.
    ///
    /// Returns the sealed shard's filename, or `None` if it held no games (an
    /// empty shard is noise in a directory the trainer globs).
    pub fn roll(&mut self) -> std::io::Result<Option<String>> {
        self.w.flush()?;
        self.w.get_ref().sync_data()?;
        self.since_fsync = 0;

        let sealed = if self.games_in_part > 0 {
            let name = format!(
                "gen{:04}-{:03}-{}.tzr",
                self.generation,
                self.seq,
                stamp(now_unix())
            );
            std::fs::rename(&self.part, self.dir.join(&name))?;
            self.seq += 1;
            self.sealed.push(name.clone());
            Some(name)
        } else {
            std::fs::remove_file(&self.part).ok();
            None
        };

        self.games_in_part = 0;
        self.records_in_part = 0;
        let mut w = BufWriter::new(File::create(&self.part)?);
        write_header(&mut w, self.generation, &self.producer)?;
        w.flush()?;
        self.w = w;

        if let Some(name) = &sealed {
            // `latest` is a pointer, not a copy, and is rewritten by
            // temp-then-rename so a reader never observes it half-written
            // (`LEARNING.md` §6.9).
            atomic_write(&self.dir.join("latest"), name.as_bytes())?;
        }
        Ok(sealed)
    }

    /// Seal and stop. The part file is removed if empty.
    /// Delete the oldest sealed shards until the directory is under `keep_gb`.
    ///
    /// At the rates a batched self-play run reaches, replay data lands at
    /// 10-25 GB/day, so an unattended run needs this or it fills the disk. Shard
    /// names sort chronologically by construction, which is what makes the
    /// policy one pass. `latest`, `manifest.json` and `schema.json` are never
    /// touched, and the shard `latest` points at is always kept even if it
    /// alone exceeds the budget -- deleting the newest data to satisfy a
    /// retention limit would be perverse.
    pub fn prune(dir: &Path, keep_gb: f64) -> std::io::Result<(usize, u64)> {
        if !(keep_gb > 0.0) {
            return Ok((0, 0));
        }
        let budget = (keep_gb * 1e9) as u64;

        let newest = std::fs::read_to_string(dir.join("latest"))
            .ok()
            .map(|s| s.trim().to_string());

        let mut shards: Vec<(String, u64)> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if !name.ends_with(".tzr") {
                    return None;
                }
                let len = e.metadata().ok()?.len();
                Some((name, len))
            })
            .collect();
        shards.sort();

        let total: u64 = shards.iter().map(|(_, n)| *n).sum();
        if total <= budget {
            return Ok((0, 0));
        }

        let (mut removed, mut freed, mut have) = (0usize, 0u64, total);
        for (name, len) in shards {
            if have <= budget {
                break;
            }
            if Some(&name) == newest.as_ref() {
                continue;
            }
            if std::fs::remove_file(dir.join(&name)).is_ok() {
                have -= len;
                freed += len;
                removed += 1;
            }
        }
        Ok((removed, freed))
    }

    pub fn finish(mut self) -> std::io::Result<Vec<String>> {
        self.roll()?;
        std::fs::remove_file(&self.part).ok();
        Ok(self.sealed)
    }
}

/// Write to a sibling temp file and rename over the target, so the target is
/// never observed half-written. `rename` within a directory is atomic on APFS
/// and on NTFS via `MoveFileEx`, which is what `std::fs::rename` uses.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!(
        "tmp{}",
        std::process::id()
    ));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_data()?;
    }
    std::fs::rename(&tmp, path)
}

/// Read every record of a shard. For the verifier and for tests; the trainer
/// reads shards in Python.
pub fn read_shard(path: &Path) -> std::io::Result<(u32, Vec<[u8; RECORD_BYTES]>)> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() < HEADER_BYTES || buf[0..4] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: not a tzolkin replay shard", path.display()),
        ));
    }
    let rules = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    let body = &buf[HEADER_BYTES..];
    // Trailing partial record: a kill between `write_all` and the page cache
    // cannot produce one, but a truncated copy can. Drop it rather than fail.
    let n = body.len() / RECORD_BYTES;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut r = [0u8; RECORD_BYTES];
        r.copy_from_slice(&body[i * RECORD_BYTES..(i + 1) * RECORD_BYTES]);
        out.push(r);
    }
    Ok((rules, out))
}

// =======================================================================
// The schema, emitted so Python never re-types this table
// =======================================================================

/// The on-disk layout as JSON, for `train/replay.py` to build a numpy dtype
/// from.
///
/// `src/state.rs` grew Chichen from 10 spaces to 11 while this file was being
/// written. Anything downstream that had hard-coded an offset would now be
/// silently reading garbage. Emitting the schema next to the shards is the
/// cheap fix: there is one table, it lives here, and the loader reads it.
pub fn schema_json() -> String {
    let f = |name: &str, off: usize, dtype: &str, count: usize| {
        format!(r#"    {{"name": "{name}", "offset": {off}, "dtype": "{dtype}", "count": {count}}}"#)
    };
    let state = |name: &str, off: usize, dtype: &str, count: usize| {
        format!(r#"    {{"name": "{name}", "offset": {off}, "dtype": "{dtype}", "count": {count}}}"#)
    };

    // State sub-layout, computed the same way `encode_state` walks it.
    let mut o = 0usize;
    let mut sfields = Vec::new();
    sfields.push(state("players", o, "u1", PLAYER_BYTES * N_PLAYERS));
    o += PLAYER_BYTES * N_PLAYERS;
    sfields.push(state("workers", o, "u1", N_WORKERS * 2));
    o += N_WORKERS * 2;
    sfields.push(state("gears", o, "u1", 5 * MAX_GEAR_SPACES));
    o += 5 * MAX_GEAR_SPACES;
    sfields.push(state("temples", o, "u1", 3 * N_PLAYERS));
    o += 3 * N_PLAYERS;
    sfields.push(state("research", o, "u1", N_PLAYERS * 4));
    o += N_PLAYERS * 4;
    sfields.push(state("palenque", o, "u1", 16));
    o += 16;
    sfields.push(state("chichen_filled", o, "u2", 1));
    o += 2;
    sfields.push(state("age1_ids", o, "u1", 14));
    o += 14;
    sfields.push(state("age1_next", o, "u1", 1));
    o += 1;
    sfields.push(state("age2_ids", o, "u1", 18));
    o += 18;
    sfields.push(state("age2_next", o, "u1", 1));
    o += 1;
    sfields.push(state("monument_ids", o, "u1", 13));
    o += 13;
    sfields.push(state("monument_next", o, "u1", 1));
    o += 1;
    sfields.push(state("buildings_up", o, "u1", N_DISPLAY));
    o += N_DISPLAY;
    sfields.push(state("monuments_up", o, "u1", N_DISPLAY));
    o += N_DISPLAY;
    for n in [
        "first_player_space",
        "accumulated_corn",
        "skulls_remaining",
        "current",
        "first_player",
        "age",
        "day",
        "over",
    ] {
        sfields.push(state(n, o, "u1", 1));
        o += 1;
    }
    debug_assert_eq!(o, STATE_BYTES);

    let rfields = [
        f("state", O_STATE, "u1", STATE_SLOT),
        f("game_id", O_GAME_ID, "u8", 1),
        f("node", O_NODE, "u2", 1),
        f("turn", O_TURN, "u1", 1),
        f("mover", O_MOVER, "u1", 1),
        f("phase_tag", O_PHASE_TAG, "u1", 1),
        f("phase_args", O_PHASE_ARGS, "u1", 5),
        f("policy_kind", O_POLICY_KIND, "u1", 1),
        f("flags", O_FLAGS, "u1", 1),
        f("n_edges", O_N_EDGES, "u2", 1),
        f("n_visit_pairs", O_N_PAIRS, "u2", 1),
        f("total_visits", O_TOTAL_VISITS, "u4", 1),
        f("visits", O_VISITS, "u2", MAX_VISITS * 2),
        f("policy_weight", O_POLICY_WEIGHT, "f4", 1),
        f("day", O_DAY, "u2", 1),
        f("temperature_x100", O_TEMPERATURE, "u1", 1),
        f("final_scores", O_FINAL_SCORES, "i2", N_PLAYERS),
        f("z_rel", O_Z_REL, "f4", N_PLAYERS),
        f("win_share", O_WIN_SHARE, "f4", N_PLAYERS),
        f("root_value", O_ROOT_VALUE, "f4", N_PLAYERS),
        f("rules_version", O_RULES_VERSION, "u4", 1),
    ];

    // What the five `phase_args` bytes mean, per `phase_tag`. The loader needs
    // this to regenerate the edge enumeration, and `PickWorker`'s byte 0 — the
    // retrieval counter — is the one without which it cannot.
    let phase_args = r#"{
    "0": {"name": "Beg", "args": []},
    "1": {"name": "Mode", "args": []},
    "2": {"name": "Placing", "args": [{"byte": 0, "name": "n_placed"}]},
    "3": {"name": "PickWorker", "args": [{"byte": 0, "name": "retrieved_so_far"}]},
    "4": {"name": "Take", "args": [{"byte": 0, "name": "worker_id"}]},
    "5": {"name": "ExtraDay", "args": [{"byte": 0, "name": "claimer"}]},
    "6": {"name": "PityPlace", "args": []},
    "7": {"name": "DraftTile", "args": [{"byte": 0, "name": "dealt", "count": 4},
                                        {"byte": 4, "name": "kept"}]}
  }"#;

    format!(
        concat!(
            "{{\n",
            "  \"format_version\": {},\n",
            "  \"magic\": \"TZZR\",\n",
            "  \"endian\": \"little\",\n",
            "  \"header_bytes\": {},\n",
            "  \"record_bytes\": {},\n",
            "  \"state_slot\": {},\n",
            "  \"state_bytes\": {},\n",
            "  \"n_players\": {},\n",
            "  \"max_visits\": {},\n",
            "  \"rules_version\": {},\n",
            "  \"player_bytes\": {},\n",
            "  \"max_gear_spaces\": {},\n",
            "  \"phase_args\": {},\n",
            "  \"record_fields\": [\n{}\n  ],\n",
            "  \"state_fields\": [\n{}\n  ]\n",
            "}}\n"
        ),
        FORMAT_VERSION,
        HEADER_BYTES,
        RECORD_BYTES,
        STATE_SLOT,
        STATE_BYTES,
        N_PLAYERS,
        MAX_VISITS,
        RULES_VERSION,
        PLAYER_BYTES,
        MAX_GEAR_SPACES,
        phase_args,
        rfields.join(",\n"),
        sfields.join(",\n"),
    )
}

// =======================================================================
// Interruption
// =======================================================================

/// Ctrl-C without a new dependency.
///
/// The crate list is fixed (no `ctrlc`), so this declares the C runtime's
/// `signal` directly. It is available and behaves the same on macOS, Linux and
/// the MSVC CRT, and `SIGINT` is 2 on all three. The handler does one relaxed
/// store, which is the only thing that is genuinely safe inside a signal
/// handler; everything else — sealing shards, printing the summary — happens on
/// the main thread once the workers notice the flag.
///
/// A second Ctrl-C aborts. That is on purpose: the first one means "wind up",
/// and the user needs a way to say "no, now" without reaching for `kill`.
pub mod interrupt {
    use super::*;

    static STOP: AtomicBool = AtomicBool::new(false);

    #[cfg(unix)]
    const SIGINT: i32 = 2;
    #[cfg(windows)]
    const SIGINT: i32 = 2;

    extern "C" fn on_sigint(_sig: i32) {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::abort();
        }
    }

    extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }

    pub fn install() {
        // SAFETY: `signal` with a plain `extern "C" fn` handler is the C
        // standard library's documented interface. The handler touches only an
        // atomic.
        unsafe {
            signal(SIGINT, on_sigint);
        }
    }

    #[inline]
    pub fn stopping() -> bool {
        STOP.load(Ordering::Relaxed)
    }

    /// For tests and for a driver that wants to stop itself.
    pub fn request_stop() {
        STOP.store(true, Ordering::SeqCst);
    }
}

// =======================================================================
// Evaluator plumbing
// =======================================================================
//
// The batching itself lives in `src/net.rs`, not here. `net::BatchedEvaluator`
// is the queue-and-batcher-threads design of `COMPUTE.md` §2.6: N game threads
// each hold a `net::BatchHandle`, block inside the ordinary batch-1
// `Evaluator::evaluate`, and are woken by one broadcast per batch. `net.rs`
// deliberately removed `impl Evaluator for Net` so that a caller has to name
// which way in it wants — `BatchedEvaluator` or `Unbatched` — and cannot lose
// `COMPUTE.md` §0's 20.3x by accident.
//
// What this file owns is the other half of §2.6: the *driver* has to supply the
// concurrency that fills the batch. That is `AgentSpec::instance` below, one
// agent per game thread, and the thread-per-game loops in the two binaries.

// =======================================================================
// Inter-game evaluation batching
// =======================================================================

/// One position waiting to be scored.
///
/// The `GameState` is copied rather than borrowed. `GameState` is `Copy` and
/// ~300 bytes, so at the 350 k evaluations/s of `COMPUTE.md` §2.2 that is about
/// 100 MB/s of `memcpy` — measurable on a profile, invisible next to the
/// forward pass, and the alternative is handing a raw pointer across a thread
/// boundary and arguing about why it stays alive.
#[derive(Clone)]
pub struct EvalRequest {
    pub state: GameState,
    pub phase: Phase,
    pub turn: PlayerId,
    pub n_edges: usize,
}

/// Anything that can score many positions in one call.
///
/// This is the interface `COMPUTE.md` §2.2 is about: the same forward pass runs
/// at 5,491 evaluations/s at batch 1 and 111,469/s at batch 256 on one core.
/// `phase::Evaluator` is the batch-1 view and stays exactly as it is; this is
/// the batch-N view behind it.
pub trait BatchEvaluator: Send + Sync {
    /// Scores `qs`, appending one `Evaluation` per request, in order.
    fn evaluate_batch(&self, qs: &[EvalRequest], out: &mut Vec<Evaluation>);
    fn name(&self) -> String;
}

/// A `phase::Evaluator` with no batched fast path, batched by looping.
///
/// Correct but pointless for `HeuristicEvaluator`, whose cost is a few hundred
/// nanoseconds and which gains nothing from amortisation. It exists so the
/// queue can be exercised, and so a network backend can be swapped in without
/// the self-play driver knowing.
pub struct Serial<E>(pub E);

impl<E: Evaluator> BatchEvaluator for Serial<E> {
    fn evaluate_batch(&self, qs: &[EvalRequest], out: &mut Vec<Evaluation>) {
        for q in qs {
            out.push(self.0.evaluate(&q.state, q.phase, q.turn, q.n_edges));
        }
    }
    fn name(&self) -> String {
        self.0.name()
    }
}

/// The network's own batched path, which is the whole reason this file exists.
///
/// `net.rs`'s `impl Evaluator for Net` calls `evaluate_batch` with a slice of
/// one (`net.rs:1471`), which is the defect `COMPUTE.md` §0 item 1 names. This
/// is the same call with a full slice.
pub struct NetBatch {
    pub net: crate::net::Net,
    /// How the user named it. Taken from the spec rather than from the net so
    /// that this file does not depend on whether `net.rs` currently offers a
    /// `name()`, an `impl Evaluator for Net`, or neither.
    pub label: String,
}

impl BatchEvaluator for NetBatch {
    fn evaluate_batch(&self, qs: &[EvalRequest], out: &mut Vec<Evaluation>) {
        let queries: Vec<crate::net::Query<'_>> = qs
            .iter()
            .map(|q| crate::net::Query::new(&q.state, q.phase, q.turn, q.n_edges))
            .collect();
        self.net.evaluate_batch(&queries, out);
    }
    fn name(&self) -> String {
        self.label.clone()
    }
}

/// The batch-1 view of the same network, for a caller with no queue.
///
/// This goes through `Net::evaluate_batch` with a slice of one, which is
/// `COMPUTE.md` §0's 5,491 evals/s against 111,469 batched. Use it only where
/// there genuinely is one position in flight.
impl Evaluator for NetBatch {
    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        let mut out = Vec::with_capacity(1);
        self.net.evaluate_batch(
            &[crate::net::Query::new(state, phase, turn, n_edges)],
            &mut out,
        );
        out.pop().unwrap_or(Evaluation {
            priors: Vec::new(),
            value: [0.0; N_PLAYERS],
        })
    }
    fn name(&self) -> String {
        self.label.clone()
    }
}

struct Slot {
    req: Option<EvalRequest>,
    ans: Option<Evaluation>,
}

struct Inner {
    slots: Vec<Slot>,
    /// Tickets with a request waiting, oldest first.
    ///
    /// A `Vec` popped from the back would be LIFO, and with `--concurrency`
    /// above `--batch` the oldest requests could then sit unserved for as long
    /// as the load lasts. FIFO costs nothing here and bounds the tail.
    pending: VecDeque<usize>,
    /// Retired tickets, for reuse.
    free: Vec<usize>,
    stop: bool,
}

/// A queue that turns many blocking batch-1 callers into few batched calls.
///
/// The shape is `COMPUTE.md` §2.6's, which is KataGo's: N game threads, each
/// with its own tree, each blocking inside `Evaluator::evaluate`; one or two
/// batcher threads draining the queue and making one `evaluate_batch` call per
/// batch. Nothing in `mcts.rs` changes and nothing becomes a future.
///
/// **This costs no search quality** (`COMPUTE.md` §2.5). Because a batch is
/// assembled across *games* rather than across descents within one tree, each
/// tree has exactly one simulation in flight: it never selects against
/// statistics missing an in-flight result and never needs a virtual loss to
/// undo. The search a game gets here is bit-for-bit the search it would get
/// single-threaded from the same seed. What changes is latency per game, and
/// self-play does not care about that.
pub struct BatchQueue {
    inner: Mutex<Inner>,
    /// Batchers wait here for requests.
    work: Condvar,
    /// Workers wait here for their answer. One broadcast per batch rather than
    /// one signal per waiter (`COMPUTE.md` §2.6 note 4): at 350 k evals/s the
    /// futex traffic of the latter is a measurable tax.
    done: Condvar,
    ev: Arc<dyn BatchEvaluator>,
    max_batch: usize,
    /// How long a batcher will hold a short batch open hoping for more. Zero
    /// disables it.
    linger: Duration,
    batches: AtomicU64,
    queries: AtomicU64,
    wait_ns: AtomicU64,
    /// Time the batchers spent inside `evaluate_batch`. Against wall-clock time
    /// times the batcher count, this says whether the batchers are saturated or
    /// starved — which is the difference between "buy a GPU" and "raise
    /// --concurrency", and there is no other way to tell.
    busy_ns: AtomicU64,
    /// Realised batch sizes: `hist[k]` counts batches of size in
    /// `[2^k, 2^(k+1))`, so `hist[0]` is batch 1 — the number to watch.
    /// `COMPUTE.md` §2.6: "if the mean batch size is not close to the
    /// configured maximum, none of §5's numbers are happening, and it is the
    /// only symptom you will get."
    hist: [AtomicU64; 12],
}

/// What one batcher thread and the queue it drains measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchStats {
    pub batches: u64,
    pub queries: u64,
    pub mean_batch: f64,
    pub mean_wait_us: f64,
    pub busy_ns: u64,
    pub max_batch: usize,
    pub hist: [u64; 12],
}

impl BatchQueue {
    /// Build a queue. Spawn its batchers with [`BatchQueue::spawn_batchers`].
    pub fn new(ev: Arc<dyn BatchEvaluator>, max_batch: usize, linger: Duration) -> Arc<BatchQueue> {
        Arc::new(BatchQueue {
            inner: Mutex::new(Inner {
                slots: Vec::new(),
                pending: VecDeque::new(),
                free: Vec::new(),
                stop: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
            ev,
            max_batch: max_batch.max(1),
            linger,
            batches: AtomicU64::new(0),
            queries: AtomicU64::new(0),
            wait_ns: AtomicU64::new(0),
            busy_ns: AtomicU64::new(0),
            hist: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }

    /// Start `n` batcher threads.
    ///
    /// **These must not be members of the worker pool.** If every thread that
    /// could run the batcher is parked waiting for a batch, the pool deadlocks
    /// (`COMPUTE.md` §2.6 note 2). Plain `std::thread`, always.
    pub fn spawn_batchers(self: &Arc<Self>, n: usize) -> Vec<std::thread::JoinHandle<()>> {
        (0..n.max(1))
            .map(|i| {
                let q = Arc::clone(self);
                std::thread::Builder::new()
                    .name(format!("batcher-{i}"))
                    .spawn(move || q.batcher_loop())
                    .expect("cannot spawn batcher thread")
            })
            .collect()
    }

    /// A per-thread client. One handle per game thread; never share one.
    pub fn handle(self: &Arc<Self>) -> BatchHandle {
        let mut g = self.inner.lock().unwrap();
        let ticket = match g.free.pop() {
            Some(t) => t,
            None => {
                g.slots.push(Slot {
                    req: None,
                    ans: None,
                });
                g.slots.len() - 1
            }
        };
        BatchHandle {
            q: Arc::clone(self),
            ticket,
        }
    }

    pub fn evaluator_name(&self) -> String {
        self.ev.name()
    }

    /// Stop the batchers. Idempotent; safe to call with requests outstanding
    /// only after every handle has been dropped.
    pub fn shutdown(&self) {
        self.inner.lock().unwrap().stop = true;
        self.work.notify_all();
    }

    pub fn stats(&self) -> BatchStats {
        let batches = self.batches.load(Ordering::Relaxed);
        let queries = self.queries.load(Ordering::Relaxed);
        BatchStats {
            batches,
            queries,
            mean_batch: if batches == 0 {
                0.0
            } else {
                queries as f64 / batches as f64
            },
            mean_wait_us: if queries == 0 {
                0.0
            } else {
                self.wait_ns.load(Ordering::Relaxed) as f64 / queries as f64 / 1000.0
            },
            busy_ns: self.busy_ns.load(Ordering::Relaxed),
            max_batch: self.max_batch,
            hist: std::array::from_fn(|i| self.hist[i].load(Ordering::Relaxed)),
        }
    }

    fn submit(&self, ticket: usize, req: EvalRequest) -> Evaluation {
        let t0 = Instant::now();
        let mut g = self.inner.lock().unwrap();
        g.slots[ticket].req = Some(req);
        g.pending.push_back(ticket);
        self.work.notify_one();
        loop {
            if let Some(a) = g.slots[ticket].ans.take() {
                self.wait_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return a;
            }
            if g.stop {
                // The batchers are gone and nothing will ever answer. Returning
                // a neutral evaluation is wrong but bounded; blocking forever
                // is wrong and unbounded.
                g.slots[ticket].req = None;
                return Evaluation {
                    priors: Vec::new(),
                    value: [0.0; N_PLAYERS],
                };
            }
            g = self.done.wait(g).unwrap();
        }
    }

    fn retire(&self, ticket: usize) {
        let mut g = self.inner.lock().unwrap();
        g.slots[ticket].req = None;
        g.slots[ticket].ans = None;
        g.free.push(ticket);
    }

    fn batcher_loop(&self) {
        let mut tickets: Vec<usize> = Vec::with_capacity(self.max_batch);
        let mut reqs: Vec<EvalRequest> = Vec::with_capacity(self.max_batch);
        let mut out: Vec<Evaluation> = Vec::with_capacity(self.max_batch);

        loop {
            let mut g = self.inner.lock().unwrap();
            // Wait for anything at all. The timeout is only so `stop` is
            // noticed by a batcher that is parked on an empty queue.
            while g.pending.is_empty() {
                if g.stop {
                    return;
                }
                g = self
                    .work
                    .wait_timeout(g, Duration::from_millis(20))
                    .unwrap()
                    .0;
            }
            // Hold a short batch open briefly. A game thread that is between
            // simulations — backing up a value, applying a step — is about to
            // arrive, and waiting 200 us for it is far cheaper than running the
            // GEMM twice.
            if g.pending.len() < self.max_batch && !self.linger.is_zero() {
                let deadline = Instant::now() + self.linger;
                while g.pending.len() < self.max_batch {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        break;
                    };
                    let (ng, r) = self.work.wait_timeout(g, left).unwrap();
                    g = ng;
                    if r.timed_out() {
                        break;
                    }
                }
            }

            // The linger above released the lock, so another batcher may have
            // taken everything. An empty batch is not an error, but counting it
            // would swamp the size histogram with zeroes.
            let n = g.pending.len().min(self.max_batch);
            if n == 0 {
                continue;
            }
            tickets.clear();
            reqs.clear();
            for _ in 0..n {
                let t = g.pending.pop_front().expect("pending shrank under the lock");
                reqs.push(g.slots[t].req.take().expect("queued slot with no request"));
                tickets.push(t);
            }
            drop(g);

            out.clear();
            let t = Instant::now();
            self.ev.evaluate_batch(&reqs, &mut out);
            self.busy_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            debug_assert_eq!(out.len(), reqs.len(), "batch evaluator dropped requests");

            let mut g = self.inner.lock().unwrap();
            for (&t, a) in tickets.iter().zip(out.drain(..)) {
                g.slots[t].ans = Some(a);
            }
            drop(g);
            self.done.notify_all();

            self.batches.fetch_add(1, Ordering::Relaxed);
            self.queries.fetch_add(n as u64, Ordering::Relaxed);
            // `hist[k]` counts batches of size in `[2^k, 2^(k+1))`.
            let k = (usize::BITS - 1 - n.leading_zeros()) as usize;
            self.hist[k.min(11)].fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl BatchStats {
    /// One line for the end of a run. The mean batch size is the number that
    /// matters: if it is far below `max_batch`, the concurrency is not there.
    pub fn line(&self) -> String {
        format!(
            "batches {} of mean size {:.1} (max {}), mean queue wait {:.0} us, \
             {:.1} us per eval inside the net",
            self.batches,
            self.mean_batch,
            self.max_batch,
            self.mean_wait_us,
            if self.queries == 0 {
                0.0
            } else {
                self.busy_ns as f64 / self.queries as f64 / 1000.0
            }
        )
    }

    /// Fraction of the run each batcher thread spent inside `evaluate_batch`.
    ///
    /// Near 1.0 means the batchers are the bottleneck and only a faster
    /// evaluator (bigger batches, a GPU) helps. Well under 1.0 means they are
    /// starved and `--concurrency` is the knob.
    pub fn utilisation(&self, secs: f64, batchers: usize) -> f64 {
        if secs <= 0.0 || batchers == 0 {
            return 0.0;
        }
        self.busy_ns as f64 / 1e9 / secs / batchers as f64
    }

    /// The histogram `COMPUTE.md` §2.6 asks for, in `2^k ..= 2^(k+1)-1` buckets.
    pub fn histogram(&self) -> String {
        let total: u64 = self.hist.iter().sum();
        if total == 0 {
            return "  (no batches)".into();
        }
        let mut s = String::new();
        for (k, &c) in self.hist.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let lo = 1usize << k;
            let hi = if k == 11 { self.max_batch.max(lo) } else { (lo << 1) - 1 };
            let bar = "#".repeat(((c as f64 / total as f64) * 40.0).round() as usize);
            s.push_str(&format!("  {lo:>5}-{hi:<5} {c:>9}  {bar}\n"));
        }
        s
    }
}

/// One game thread's client on a [`BatchQueue`]. Implements `Evaluator`, so a
/// `Mcts` cannot tell it apart from a network it owns outright.
pub struct BatchHandle {
    q: Arc<BatchQueue>,
    ticket: usize,
}

impl Evaluator for BatchHandle {
    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        self.q.submit(
            self.ticket,
            EvalRequest {
                state: *state,
                phase,
                turn,
                n_edges,
            },
        )
    }

    fn name(&self) -> String {
        self.q.evaluator_name()
    }
}

impl Drop for BatchHandle {
    fn drop(&mut self) {
        self.q.retire(self.ticket);
    }
}

/// An `Evaluator` behind an `Arc`, so one `Mcts` type serves every backend.
///
/// `Mcts<E>` owns its evaluator by value and is generic over it, so a driver
/// that chooses its backend at run time needs one concrete `E`. This is it:
/// `Mcts<SharedEval>` is the only instantiation either binary uses.
#[derive(Clone)]
pub struct SharedEval(pub Arc<dyn Evaluator>);

impl Evaluator for SharedEval {
    fn evaluate(
        &self,
        state: &GameState,
        phase: Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> Evaluation {
        self.0.evaluate(state, phase, turn, n_edges)
    }
    fn name(&self) -> String {
        self.0.name()
    }
}

// =======================================================================
// Agents
// =======================================================================

/// What one turn of play produced.
pub struct TurnOutcome {
    pub mv: Move,
    /// Decision nodes on the played path. Empty is legal — an agent that has no
    /// policy to teach still moves.
    pub nodes: Vec<Node>,
}

/// A turn-level policy: everything the drivers need from a player.
///
/// # Instances are per-thread
///
/// `play_turn` takes `&self`, so a stateless agent (`RandomAgent`,
/// `GreedyAgent`) can be shared by every thread in the run. A searching one
/// cannot: it owns a mutable tree arena and a slot in the evaluation batch.
/// Both drivers therefore build one agent per worker thread through
/// [`AgentSpec::instance`], and anything genuinely shared — the network's
/// weights, the [`BatchQueue`] in front of them — lives behind an `Arc` in the
/// spec rather than in the agent.
///
/// # The seam to `src/mcts.rs`
///
/// [`SearchAgent`] is the adapter and it is closed. Two notes about its shape,
/// because both are places a plausible implementation would be wrong:
///
/// * It searches on a **copy** of the state and hands back a `Move`.
///   `Mcts` advances a position with `tree::apply_step`; `play_game` advances
///   it with `moves::apply_move`. Reconstructing the move with
///   `tree::move_from_path` + `tree::retag_workers` and letting `play_game`
///   apply it keeps self-play and the arena on exactly the `apply_move` path
///   the rules tests cover, and turns any divergence between the two
///   representations into an illegal move rather than a quietly different game.
/// * The edge index it records is **re-derived from `tree::legal_steps`**, not
///   taken from the tree. See [`search_node`].
pub trait Agent: Send + Sync {
    /// Play one turn for `p`. `None` means the player has no legal move and
    /// passes, which `Game::take_turn` also allows.
    fn play_turn(&self, g: &GameState, p: PlayerId, temp: f32, rng: &mut StdRng) -> Option<TurnOutcome>;

    /// Decide the extra calendar day. Returns the decision and, optionally, a
    /// record for it — it is a real node (`Phase::ExtraDay`) with a real head.
    fn extra_day(&self, g: &GameState, p: PlayerId, rng: &mut StdRng) -> (bool, Option<Node>) {
        let _ = g;
        let _ = p;
        (rng.gen_bool(0.5), None)
    }

    /// Which two of the four dealt starting tiles to keep, by tile id.
    ///
    /// Defaults to a uniform pick, which is what `Game::new` has always done
    /// and what `RandomAgent` should keep doing. Anything with an evaluator
    /// should override it with [`best_pair`]: the draft is worth real points —
    /// tile 4 is a free worker outright, tile 5 is eight corn and a gold — and
    /// it is the one decision in the game where a player can see the entire
    /// option set.
    fn draft(&self, g: &GameState, p: PlayerId, dealt: [u8; 4], rng: &mut StdRng) -> [u8; 2] {
        let _ = (g, p);
        let mut idx = [0usize, 1, 2, 3];
        for i in (1..4).rev() {
            idx.swap(i, rng.gen_range(0..=i));
        }
        [dealt[idx[0]], dealt[idx[1]]]
    }

    /// This agent's own ranking of the position's moves, best first.
    ///
    /// `None` from an agent that has no ranking to show — `RandomAgent` draws
    /// one move and never compares it to anything. Implementors that *do* rank
    /// should report the whole move count in `Ranking::total`, because the
    /// point of the display is to say what the shortlist was drawn from.
    ///
    /// This is an explanation hook, not part of play: `play_turn` must not be
    /// implemented in terms of it, since the two may cost very different
    /// amounts.
    fn ranked_moves(&self, g: &GameState, p: PlayerId, keep: usize) -> Option<Ranking> {
        let _ = (g, p, keep);
        None
    }

    fn name(&self) -> String;
}

/// The best two of the four dealt tiles, by trying all six pairs.
///
/// # Why this is not a search
///
/// Six pairs is the entire decision, so this is exhaustive on its own terms.
/// Searching *past* it would mean looking ahead through a board nobody has
/// touched — no worker is placed, no gear has turned — and what the tiles are
/// worth is almost entirely what they hand you: corn, blocks, a research level,
/// a temple step, a worker. `value` prices those directly, and one more ply
/// would price a guess about the first placement instead.
///
/// # Why order is part of the answer
///
/// The pair is returned in the order it should be applied. Two tiles that both
/// step the same temple can land differently depending on which goes first,
/// because the top step of a track is exclusive — so the pair is scored as an
/// ordered application and handed back that way.
pub fn best_pair<F>(g: &GameState, p: PlayerId, dealt: [u8; 4], value: F) -> [u8; 2]
where
    F: Fn(&GameState) -> f32,
{
    use crate::data::tiles::TILES;

    let apply = |st: &mut GameState, id: u8| {
        for e in TILES[id as usize] {
            e.apply(st, p);
        }
    };

    let mut best = [dealt[0], dealt[1]];
    let mut best_score = f32::NEG_INFINITY;
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                continue;
            }
            let pair = [dealt[i], dealt[j]];
            let mut probe = *g;
            apply(&mut probe, pair[0]);
            apply(&mut probe, pair[1]);
            let s = value(&probe);
            if s > best_score {
                best_score = s;
                best = pair;
            }
        }
    }
    best
}

/// Setup with each seat's starting-tile draft taken by its own agent.
///
/// The deal comes from `seed` alone, so a rotation block that reuses a seed
/// puts every agent in front of the same four tiles — the draft joins the
/// matched design rather than adding variance to it.
///
/// Seats draft in order, which is the model `tree.rs` already encodes for
/// `Phase::DraftTile`. It means a later seat sees where the earlier ones have
/// stepped on the temples; agents that score their own position rather than
/// their margin are unaffected by that, which is the other reason to score it
/// that way.
pub fn new_drafted_game(
    seed: u64,
    agents: &[&dyn Agent; N_PLAYERS],
    rng: &mut StdRng,
) -> GameState {
    let (mut game, deal) = crate::game::Game::new_undrafted(seed);
    for p in PlayerId::ALL {
        let kept = agents[p.idx()].draft(&game.state, p, deal[p.idx()], rng);
        debug_assert!(
            kept[0] != kept[1] && kept.iter().all(|k| deal[p.idx()].contains(k)),
            "{p:?} kept {kept:?}, which was not two of {:?}",
            deal[p.idx()]
        );
        game.keep_tiles(p, kept);
    }
    game.state
}

/// The rollout policy from `moves.rs`, wrapped as an agent.
///
/// It is the absolute floor of the anchor panel (`LEARNING.md` §6.8): an agent
/// that ever loses to this is broken, not weak. It emits a value-only record
/// per turn, which makes `selfplay --agent random` a literal implementation of
/// the value warm-start in §6.7.
pub struct RandomAgent {
    pub record: bool,
}

impl Agent for RandomAgent {
    fn play_turn(&self, g: &GameState, p: PlayerId, _temp: f32, rng: &mut StdRng) -> Option<TurnOutcome> {
        let mv = sample_legal_move(g, p, rng)?;
        let nodes = if self.record {
            vec![Node::value_only(*g, p, Phase::Mode, [0.0; N_PLAYERS])]
        } else {
            Vec::new()
        };
        Some(TurnOutcome { mv, nodes })
    }

    fn name(&self) -> String {
        "random".into()
    }
}

/// One-ply greedy over sampled candidate turns.
///
/// # Why sampled candidates rather than enumerated ones
///
/// `legal_moves` costs 8 ms and its widest measured node is 1.9 M moves
/// (`SEARCH.md` §1.1), so enumerating is not on the table.
/// `legal_moves_capped(k)` is affordable but takes the *first* `k` in traversal
/// order, which is systematically "place one worker low on Palenque" — a biased
/// slice, not a sample. `sample_legal_move` is the project's own rollout policy
/// and is deliberately shaped like real play (beg rarely, worker count
/// geometric), so `k` draws from it are a usable proposal distribution. `k`
/// draws cost ~1.2 µs each.
///
/// # What this is for
///
/// It is a **stand-in for MCTS**, not a rival to it: one ply, no opponent
/// model. Its purpose is that the arena, the self-play driver and the training
/// script can all be exercised end to end today, and that there is a real
/// strength ladder (`random` << `heuristic:8` << `heuristic:64`) to check the
/// arena's statistics against before any network exists.
///
/// Its records are `policy_kind::NONE`: the candidate indices come out of an
/// RNG and are not regenerable, so they teach the value heads and nothing else.
pub struct GreedyAgent<E: Evaluator> {
    pub ev: E,
    /// Where the turns it chooses between come from.
    pub cands: Candidates,
    pub record: bool,
}

/// Where a greedy agent's candidate turns come from.
///
/// `Sampled` is the original and the only affordable option for an expensive
/// evaluator. `All` is what `heuristic:full` selects: every legal move
/// generated and scored, streamed through `eval::rank_all_by` so that a node
/// with a six-figure action space costs time but not memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Candidates {
    /// `k` draws from the `moves::sample_legal_move` rollout policy.
    Sampled(usize),
    /// Every legal move, without exception.
    All,
}

impl Candidates {
    /// How many scored moves are worth keeping around for the record and for
    /// the temperature draw. The tail of a 20,000-move list is not a candidate
    /// under any temperature this game uses.
    fn keep(self) -> usize {
        match self {
            Candidates::Sampled(k) => k,
            Candidates::All => MAX_VISITS.max(32),
        }
    }
}

impl<E: Evaluator> GreedyAgent<E> {
    /// Value of `g` to every seat, as the evaluator sees it.
    fn value(&self, g: &GameState, p: PlayerId) -> [f32; N_PLAYERS] {
        let Evaluation { value, .. } = self.ev.evaluate(g, Phase::Mode, p, 0);
        value
    }

    /// Successor state of playing `m`, including the building refill that
    /// `Game::play` does — evaluating the pre-refill row would score a display
    /// no player ever sees.
    fn successor(&self, g: &GameState, p: PlayerId, m: &Move) -> GameState {
        let mut probe = *g;
        apply_move(&mut probe, p, m);
        probe.refill_buildings();
        probe
    }
}

impl<E: Evaluator> GreedyAgent<E> {
    /// The candidate turns and their scores, best first for `All` and in draw
    /// order for `Sampled`.
    fn shortlist(&self, g: &GameState, p: PlayerId, rng: &mut StdRng) -> (Vec<Move>, Vec<f32>) {
        match self.cands {
            Candidates::Sampled(k) => {
                let mut cands: Vec<Move> = Vec::with_capacity(k);
                for _ in 0..k {
                    let Some(m) = sample_legal_move(g, p, rng) else {
                        break;
                    };
                    if !cands.contains(&m) {
                        cands.push(m);
                    }
                }
                let scores = cands
                    .iter()
                    .map(|m| self.value(&self.successor(g, p, m), p)[p.idx()])
                    .collect();
                (cands, scores)
            }
            Candidates::All => {
                // Bounded: this is driving a game, and the widest turns a strong
                // player reaches are wide enough to stall one for minutes.
                let r = crate::eval::rank_all_within(
                    g,
                    p,
                    self.cands.keep(),
                    Some(crate::eval::FULL_BUDGET),
                    |s| self.value(s, p)[p.idx()],
                );
                r.moves.into_iter().unzip()
            }
        }
    }
}

impl<E: Evaluator> Agent for GreedyAgent<E> {
    fn play_turn(&self, g: &GameState, p: PlayerId, temp: f32, rng: &mut StdRng) -> Option<TurnOutcome> {
        let (cands, scores) = self.shortlist(g, p, rng);
        if cands.is_empty() {
            return None;
        }

        let pick = choose(&scores, temp, rng);
        let mv = cands[pick].clone();

        let nodes = if self.record {
            let root_value = self.value(g, p);
            // Visits are the softmax mass, scaled to look like a visit count so
            // `policy_weight` behaves. `policy_kind::NONE` says not to train on
            // them; they are here so a human reading a shard can see what the
            // agent was considering.
            let w = softmax(&scores, temp.max(0.05));
            let mut order: Vec<usize> = (0..w.len()).collect();
            order.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
            let scale = cands.len() as f32 * 8.0;
            let visits = order
                .iter()
                .take(MAX_VISITS)
                .map(|&i| (i as u16, (w[i] * scale) as u16))
                .collect();
            vec![Node {
                state: *g,
                turn: p,
                phase: Phase::Mode,
                done: 0,
                policy_kind: policy_kind::NONE,
                n_edges: cands.len() as u16,
                total_visits: 0,
                visits,
                root_value,
                flags: flags::TURN_ROOT,
                temperature: temp,
            }]
        } else {
            Vec::new()
        };

        Some(TurnOutcome { mv, nodes })
    }

    fn extra_day(&self, g: &GameState, p: PlayerId, _rng: &mut StdRng) -> (bool, Option<Node>) {
        // A genuine two-way node: roll the calendar forward both ways and
        // compare, rather than flipping a coin the way `Game::end_round` still
        // does. Note the decision is taken *before* the advance and is worth
        // two days rather than one, per `GameState::spend_extra_day`.
        let root_value = self.value(g, p);

        let mut one = *g;
        one.advance_days(1);
        let decline = self.value(&one, p)[p.idx()];

        let mut two = *g;
        two.spend_extra_day(p);
        two.advance_days(2);
        let take = self.value(&two, p)[p.idx()];

        let node = self.record.then(|| Node {
            state: *g,
            turn: p,
            phase: Phase::ExtraDay { claimer: p },
            done: 0,
            policy_kind: policy_kind::NONE,
            n_edges: 2,
            total_visits: 0,
            visits: vec![(if take > decline { 1 } else { 0 }, 1)],
            root_value,
            flags: 0,
            temperature: 0.0,
        });
        (take > decline, node)
    }

    fn draft(&self, g: &GameState, p: PlayerId, dealt: [u8; 4], _rng: &mut StdRng) -> [u8; 2] {
        best_pair(g, p, dealt, |s| self.value(s, p)[p.idx()])
    }

    fn ranked_moves(&self, g: &GameState, p: PlayerId, keep: usize) -> Option<Ranking> {
        // Always exhaustive, whatever `cands` says: the panel's whole claim is
        // that the shortlist came out of the entire move space, and a `Sampled`
        // agent's own draws would not support it. What stays the agent's is the
        // *evaluator* doing the scoring.
        let mut r = crate::eval::rank_all_within(
            g,
            p,
            keep,
            Some(crate::eval::FULL_BUDGET),
            |s| self.value(s, p)[p.idx()],
        );
        let seen = if r.exhaustive {
            format!("all {} moves scored", r.total)
        } else {
            format!("{} moves scored before the budget ran out", r.total)
        };
        r.note = match self.cands {
            Candidates::All => format!("{} · {seen}", self.ev.name()),
            Candidates::Sampled(k) => {
                format!("{} · {seen} (it plays from {k} sampled)", self.ev.name())
            }
        };
        Some(r)
    }

    fn name(&self) -> String {
        match self.cands {
            Candidates::Sampled(k) => format!("{}:{}", self.ev.name(), k),
            Candidates::All => format!("{}:full", self.ev.name()),
        }
    }
}

fn softmax(scores: &[f32], temp: f32) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut w: Vec<f32> = scores.iter().map(|&s| ((s - max) / temp).exp()).collect();
    let sum: f32 = w.iter().sum();
    if sum > 0.0 {
        for x in &mut w {
            *x /= sum;
        }
    }
    w
}

/// Argmax at `temp == 0`, softmax sample above it. Evaluation games use 0
/// throughout (`SEARCH.md` §3.8); self-play anneals.
fn choose(scores: &[f32], temp: f32, rng: &mut StdRng) -> usize {
    if temp <= 0.0 {
        let mut best = 0;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        return best;
    }
    let w = softmax(scores, temp);
    let mut r: f32 = rng.gen();
    for (i, &p) in w.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    w.len() - 1
}

// -----------------------------------------------------------------------
// The MCTS seam
// -----------------------------------------------------------------------

/// Default full simulation budget for a search agent.
///
/// `LEARNING.md` §6.4 said 800. `COMPUTE.md` §7 says to spend the batching
/// surplus on depth rather than on games and to take it to 3,200: games/hour
/// falls 4x while every policy target gets four times the search behind it.
/// At the rates in `COMPUTE.md` §5 the loop would otherwise be discarding most
/// of what self-play produces, so this is the trade that costs nothing.
pub const DEFAULT_SIMS: u32 = 3200;

/// Share of turns that get the full budget; the rest get `full / 8`.
///
/// Playout-cap randomisation, `LEARNING.md` §6.3/§6.4. Only full-budget turns
/// are written as records, so this is also the write rate.
pub const FULL_BUDGET_SHARE: f64 = 0.25;

/// A tree-search player: `Mcts` from `src/mcts.rs` wrapped as an `Agent`.
///
/// # One tree per instance, never a shared one
///
/// `Agent::play_turn` takes `&self` and `Mcts::search_at` takes `&mut self`, so
/// something has to give. A `Mutex<Mcts>` shared between the game threads would
/// serialise the entire run — with 256 concurrent games that is the opposite of
/// the point. So the driver asks [`AgentSpec::instance`] for **one agent per
/// worker thread**, each with its own tree arena and its own slot in the
/// evaluation batch, and this mutex is uncontended: it is locked once per
/// sub-decision by the only thread that can reach it.
pub struct SearchAgent {
    mcts: Mutex<Mcts<SharedEval>>,
    full_sims: u32,
    fast_sims: u32,
    /// Playout-cap randomisation. `1.0` for evaluation games, which want every
    /// turn searched to the same depth and write nothing.
    full_share: f64,
    record: bool,
    label: String,
}

/// Whether a searching agent is generating training data or answering a
/// question, which is **not** the same as whether it hands back decision nodes.
///
/// Conflating the two cost the TUI both halves of this enum: it asks for
/// `record: true` to populate its search panel, and used to get 0.25 Dirichlet
/// noise scrambling its root priors *and* playout-cap randomisation running
/// seven turns in eight at a eighth of the budget. A human watching a game
/// wants the policy the search actually believes, at the budget they asked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exploration {
    /// `SEARCH.md` §3.7 Dirichlet noise on the root turn, and `LEARNING.md`
    /// §6.3 playout-cap randomisation. Both exist to diversify a replay buffer.
    SelfPlay,
    /// The search's own policy, every turn at the full budget.
    Off,
}

impl SearchAgent {
    /// Self-play when `record`, evaluation otherwise — the historical shape,
    /// kept because it is what every caller but the TUI wants.
    pub fn new(ev: Arc<dyn Evaluator>, full_sims: u32, record: bool, seed: u64) -> SearchAgent {
        let explore = if record {
            Exploration::SelfPlay
        } else {
            Exploration::Off
        };
        SearchAgent::with_config(ev, full_sims, record, explore, MctsConfig::default(), seed)
    }

    pub fn with_config(
        ev: Arc<dyn Evaluator>,
        full_sims: u32,
        record: bool,
        explore: Exploration,
        cfg: MctsConfig,
        seed: u64,
    ) -> SearchAgent {
        let mut cfg = MctsConfig { seed, ..cfg };
        if explore == Exploration::Off {
            cfg.dirichlet_eps = 0.0;
        }
        let label = mcts_label(full_sims, &ev.name(), &cfg);
        SearchAgent {
            mcts: Mutex::new(Mcts::new(SharedEval(ev), cfg)),
            full_sims: full_sims.max(1),
            fast_sims: (full_sims / 8).max(1),
            full_share: if explore == Exploration::SelfPlay {
                FULL_BUDGET_SHARE
            } else {
                1.0
            },
            record,
            label,
        }
    }
}

impl SearchAgent {
    /// The configuration the search is actually running.
    ///
    /// Exposed so a test can check that `record` no longer drags exploration in
    /// with it, rather than inferring it from play — Dirichlet noise is seeded,
    /// so a noised search and a clean one are both deterministic and a
    /// behavioural test cannot tell them apart.
    pub fn mcts_config(&self) -> MctsConfig {
        *self.mcts.lock().expect("mcts poisoned").config()
    }

    /// Share of turns given the full simulation budget. 1.0 unless self-play
    /// exploration is on.
    pub fn full_share(&self) -> f64 {
        self.full_share
    }
}

/// The agent's name, with every knob that differs from `MctsConfig::default()`
/// spelled out.
///
/// Derived from the config rather than from the spec string on purpose:
/// `minimax_label` was written the other way, silently ignored the harness
/// flags, and an arena run racing two variants printed the *same* name on both
/// sides — a result file nobody can attribute. A label read off the config
/// cannot drift from what the agent is actually doing.
pub fn mcts_label(sims: u32, evaluator: &str, cfg: &MctsConfig) -> String {
    let d = MctsConfig::default();
    let mut s = format!("mcts{sims}/{evaluator}");
    let mut f = |x: String| s.push_str(&x);
    if cfg.priors != d.priors {
        f(match cfg.priors {
            Priors::Evaluator => ":pri=eval".into(),
            Priors::OnePly => ":pri=1ply".into(),
        });
    }
    if cfg.priors == Priors::OnePly {
        // Only meaningful under a one-ply prior, but then always shown: they
        // are the two knobs a sweep moves and an unlabelled sweep is a wasted
        // one.
        f(format!(":pt={}", cfg.prior_temp));
        f(format!(":pmin={}", cfg.prior_min_edges));
    }
    if cfg.c_puct_init != d.c_puct_init {
        f(format!(":cp={}", cfg.c_puct_init));
    }
    if cfg.c_puct_base != d.c_puct_base {
        f(format!(":cpb={}", cfg.c_puct_base));
    }
    if cfg.fpu_reduction != d.fpu_reduction {
        f(format!(":fpu={}", cfg.fpu_reduction));
    }
    if cfg.max_edges != d.max_edges {
        f(format!(":k={}", cfg.max_edges));
    }
    if cfg.widen_c != d.widen_c {
        f(format!(":wc={}", cfg.widen_c));
    }
    if cfg.widen_alpha != d.widen_alpha {
        f(format!(":wa={}", cfg.widen_alpha));
    }
    if cfg.widen_cap != d.widen_cap {
        f(format!(":wcap={}", cfg.widen_cap));
    }
    if cfg.tree_reuse != d.tree_reuse {
        f(":noreuse".into());
    }
    if cfg.virtual_loss != d.virtual_loss {
        f(format!(":vl={}", cfg.virtual_loss));
    }
    s
}

/// Turn one searched sub-decision into a record.
///
/// # Why this re-derives the edge list
///
/// `record.rs`'s `policy_kind::TREE_EDGE` promises that the index stored in
/// `visits` is regenerable at training time from `(state, phase, turn, done)`
/// alone. `SearchResult::visits` is indexed by position in `Mcts`'s own edge
/// array — and `mcts.rs` **sorts that array by prior and truncates it** once a
/// node is wider than `max_edges` (32). Those priors come from the network
/// being trained, so that ordering is not reproducible by a loader and changes
/// every generation.
///
/// `SearchResult::visits` carries the `Step` itself, not only a count, so the
/// fix is local: look each step up in `tree::legal_steps`, which is exactly the
/// enumeration the loader will regenerate, and store *that* index. One
/// `legal_steps` call per recorded node, on the 25% of turns that are written.
fn search_node(
    state: &GameState,
    turn: PlayerId,
    phase: Phase,
    done: u8,
    r: &SearchResult,
    temp: f32,
    root: bool,
) -> Option<Node> {
    let total: u32 = r.visits.iter().map(|(_, n)| n).sum();
    if total == 0 {
        return None;
    }
    let legal = tree::legal_steps(state, phase, turn, done);
    let mut pairs: Vec<(u16, u16)> = Vec::with_capacity(r.visits.len());
    for (step, n) in &r.visits {
        if *n == 0 {
            continue;
        }
        let Some(i) = legal.iter().position(|s| s == step) else {
            // The tree offered an edge the generator does not, which would make
            // the whole record unreadable. Drop the node rather than write a
            // target that indexes nothing.
            debug_assert!(false, "searched step {step:?} is not in legal_steps at {phase:?}");
            return None;
        };
        pairs.push((i as u16, (*n).min(u16::MAX as u32) as u16));
    }
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs.truncate(MAX_VISITS);
    Some(Node {
        state: *state,
        turn,
        phase,
        done,
        policy_kind: policy_kind::TREE_EDGE,
        n_edges: legal.len().min(u16::MAX as usize) as u16,
        total_visits: total,
        visits: pairs,
        root_value: r.root_value,
        flags: if root { flags::TURN_ROOT } else { 0 } | flags::FULL_BUDGET,
        temperature: temp,
    })
}

/// A turn is a bounded chain of sub-decisions; this is well past the longest
/// one (six placements, or seven retrievals with their `Take`s).
const MAX_SUB_DECISIONS: usize = 64;

impl Agent for SearchAgent {
    fn play_turn(
        &self,
        g: &GameState,
        p: PlayerId,
        temp: f32,
        rng: &mut StdRng,
    ) -> Option<TurnOutcome> {
        let full = self.full_share >= 1.0 || rng.gen_bool(self.full_share);
        let sims = if full { self.full_sims } else { self.fast_sims };

        let mut m = self.mcts.lock().unwrap();
        m.config_mut().temperature = temp;

        // The search runs on a copy. `play_game` owns the real state and
        // applies the reconstructed `Move` to it, so that self-play and the
        // arena go through exactly the same `apply_move` the rules tests cover.
        let mut probe = *g;
        let mut at = (Phase::Beg, p, 0u8);
        let mut path: Vec<Step> = Vec::new();
        let mut nodes: Vec<Node> = Vec::new();

        for i in 0..MAX_SUB_DECISIONS {
            let (phase, turn, done) = at;
            let before = probe;
            let r = m.search_at(&probe, phase, turn, done, sims);
            // `sims == 0` is a forced node: one legal edge, collapsed without an
            // evaluation. There is no distribution to teach.
            if self.record && full && r.sims > 0 {
                if let Some(n) = search_node(&before, turn, phase, done, &r, temp, i == 0) {
                    nodes.push(n);
                }
            }
            let step = r.step.clone();
            path.push(step.clone());
            let t = tree::apply_step(&mut probe, phase, turn, done, &step);
            match t.next() {
                None => break,
                Some(next) => {
                    if t.committed() {
                        break;
                    }
                    at = next;
                }
            }
        }
        drop(m);

        let mut mv = tree::move_from_path(&path)?;
        // A bare path does not know which physical worker went where; the
        // engine's `Move` does.
        tree::retag_workers(g, p, &mut mv);
        Some(TurnOutcome { mv, nodes })
    }

    /// The search's own ranking of **whole turns**, best first by visit share.
    ///
    /// # Why the score is a percentage and not points
    ///
    /// Everything else in this panel is `eval::margin` points, and a minimax
    /// score genuinely is one. An MCTS score is not: the quantity the search
    /// produces over turns is a visit distribution, and its backed-up value is
    /// on §4.5's `z_rel` scale, which is a `tanh` of a *difference* of points
    /// and would read as a plausible-looking number meaning something else. So
    /// the column is the share of finished simulations that played this turn,
    /// in percent, and the note says so.
    ///
    /// # Why `exhaustive` is false
    ///
    /// The panel's standing claim is that the shortlist was drawn from the
    /// entire move space. This root cannot support it. A turn is a chain of ~8
    /// sub-decisions; the tree holds only the paths its simulations walked, so
    /// the candidate set is bounded by the simulation count and shaped by the
    /// search's own priors. `total` is therefore the number of complete turns
    /// the tree *reached*, not the number that are legal, and the note is
    /// explicit about which.
    fn ranked_moves(&self, g: &GameState, p: PlayerId, keep: usize) -> Option<Ranking> {
        let mut m = self.mcts.lock().unwrap();
        let saved = m.config().temperature;
        m.config_mut().temperature = 0.0;
        // `keep * 4` before deduplication: retrieval orderings that commute are
        // distinct paths in the tree and the same `Move` to a reader, so a top
        // ten drawn from exactly ten lines routinely shows five moves twice.
        let r = m.ranked_turns(g, p, self.full_sims, keep.max(1) * 4);
        m.config_mut().temperature = saved;
        let (found, committed, all_open, nodes) = (r.found, r.committed, r.all_edges_open, r.nodes);
        let lines = r.lines;
        drop(m);

        let denom = committed.max(1) as f32;
        let mut moves: Vec<(Move, f32)> = Vec::with_capacity(lines.len());
        for line in &lines {
            let Some(mut mv) = tree::move_from_path(&line.steps) else {
                continue;
            };
            tree::retag_workers(g, p, &mut mv);
            match moves.iter_mut().find(|(x, _)| x.same_effect(&mv)) {
                // Two spellings of one turn are one entry, and the share is the
                // sum: the search split its visits between them but a reader is
                // being told how much of it went to *this turn*.
                Some((_, share)) => *share += 100.0 * line.visits as f32 / denom,
                None => moves.push((mv, 100.0 * line.visits as f32 / denom)),
            }
        }
        moves.sort_by(|a, b| b.1.total_cmp(&a.1));
        let distinct = moves.len();
        moves.truncate(keep);

        let reach = if all_open {
            "every legal edge open"
        } else {
            "some edges still closed by the cap or widening"
        };
        Some(Ranking {
            moves,
            total: found,
            distinct,
            // Not a claim about the move list: see the doc comment.
            exhaustive: false,
            note: format!(
                "{} · visit share (%) of {committed} finished sims over {found} turns \
                 the tree reached, {nodes} nodes, {reach} — NOT the whole move space",
                self.label
            ),
        })
    }

    fn extra_day(&self, g: &GameState, p: PlayerId, _rng: &mut StdRng) -> (bool, Option<Node>) {
        let phase = Phase::ExtraDay { claimer: p };
        let mut m = self.mcts.lock().unwrap();
        // A two-edge node whose consequences run to the end of the game. Always
        // at full depth: it is one node per round, not one per turn.
        m.config_mut().temperature = 0.0;
        let r = m.search_at(g, phase, p, 0, self.full_sims);
        let take = matches!(r.step, Step::ExtraDay(true));
        let node = if self.record && r.sims > 0 {
            search_node(g, p, phase, 0, &r, 0.0, false)
        } else {
            None
        };
        (take, node)
    }

    fn name(&self) -> String {
        self.label.clone()
    }
}

// =======================================================================
// Minimax
// =======================================================================

/// Paranoid minimax with alpha-beta pruning, from [`crate::search`].
///
/// The strongest player here that needs no training run: it enumerates the
/// whole move list at its own turn, and looks ahead through the opponents'
/// replies and the round boundary — including the food day the greedy agent
/// cannot see coming.
///
/// # Why the `Mutex`
///
/// `Agent::play_turn` takes `&self` and `Search::search` takes `&mut self`,
/// because the search owns a transposition table it fills as it goes. Both
/// drivers already build one agent per worker thread through
/// [`AgentSpec::instance`], so this lock is uncontended in every intended use;
/// it is here to satisfy `Send + Sync`, not to share a search between threads.
/// Keeping the table across turns of one game is worth real nodes, which is why
/// the search is not simply rebuilt per call.
pub struct MinimaxAgent {
    inner: Mutex<crate::search::Search>,
    label: String,
    record: bool,
}

/// How a minimax spec prints. Carries the opponent model, because paranoid and
/// greedy are different players and a run log that cannot tell them apart is
/// not a run log.
pub fn minimax_label(cfg: &crate::search::Config) -> String {
    let mut s = format!(
        "minimax:d{}:{}ms:w{}:{}",
        cfg.max_depth,
        cfg.budget.as_millis(),
        cfg.width_at(0),
        match cfg.opponents {
            crate::search::Opponents::Paranoid => "paranoid",
            crate::search::Opponents::Greedy => "greedy",
        }
    );
    // AB-HARNESS (temporary): the flags have to show, or an arena run that
    // races two variants prints the same name on both sides and the progress
    // file cannot say afterwards which one won.
    let d = crate::search::Config::default();
    if cfg.cache != d.cache {
        s.push_str(if cfg.cache { ":cache" } else { ":nocache" });
    }
    if cfg.own_width != d.own_width {
        s.push_str(":ownwidth");
    }
    if cfg.keep_tt != d.keep_tt {
        s.push_str(":keeptt");
    }
    if cfg.beam_first != d.beam_first {
        s.push_str(if cfg.beam_first { ":beamfirst" } else { ":scorefirst" });
    }
    if cfg.opp_width != d.opp_width {
        s.push_str(&format!(":oppw{}", cfg.opp_width));
    }
    if cfg.cap_per_width != d.cap_per_width {
        match cfg.cap_per_width {
            Some(k) => s.push_str(&format!(":capw{k}")),
            None => s.push_str(":nocapw"),
        }
    }
    if cfg.interior_cap != d.interior_cap {
        s.push_str(&format!(":cap{}", cfg.interior_cap));
    }
    if cfg.keep != d.keep {
        s.push_str(&format!(":keep{}", cfg.keep));
    }
    // The `w{}` above already reports the root width, and the WIDTH field
    // derives the whole taper from it -- so spell the taper out only when it is
    // something the `w=` flag set and `w{}` therefore cannot imply.
    let implied: Vec<usize> = std::iter::once(cfg.width_at(0))
        .chain(
            d.widths
                .iter()
                .skip(1)
                .map(|&x| (x * cfg.width_at(0) / d.widths[0]).max(2)),
        )
        .collect();
    if cfg.widths != implied {
        s.push_str(&format!(
            ":w[{}]",
            cfg.widths
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(".")
        ));
    }
    s
}

impl MinimaxAgent {
    pub fn new(cfg: crate::search::Config, record: bool) -> Self {
        let label = minimax_label(&cfg);
        MinimaxAgent {
            inner: Mutex::new(crate::search::Search::new(cfg)),
            label,
            record,
        }
    }

    fn ranking(&self, g: &GameState, p: PlayerId, keep: usize) -> Ranking {
        let mut s = self.inner.lock().expect("minimax search poisoned");
        let want = s.cfg.keep.max(keep);
        let saved = std::mem::replace(&mut s.cfg.keep, want);
        let r = s.search(g, p);
        s.cfg.keep = saved;
        r
    }
}

impl Agent for MinimaxAgent {
    fn play_turn(&self, g: &GameState, p: PlayerId, temp: f32, rng: &mut StdRng) -> Option<TurnOutcome> {
        let r = self.ranking(g, p, MAX_VISITS);
        if r.moves.is_empty() {
            return None;
        }

        let scores: Vec<f32> = r.moves.iter().map(|(_, s)| *s).collect();
        let pick = choose(&scores, temp, rng);
        let mv = r.moves[pick].0.clone();

        let nodes = if self.record {
            // `policy_kind::NONE`: these are minimax values over a beam, not a
            // visit distribution, so they are not a policy target. They are
            // recorded so a shard can be read back and the agent's choice
            // explained.
            let w = softmax(&scores, temp.max(0.05));
            let mut order: Vec<usize> = (0..w.len()).collect();
            order.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
            let scale = r.moves.len() as f32 * 8.0;
            let visits = order
                .iter()
                .take(MAX_VISITS)
                .map(|&i| (i as u16, (w[i] * scale) as u16))
                .collect();
            let root_value = std::array::from_fn(|i| {
                (crate::eval::heuristic(g, PlayerId(i as u8)) / 25.0).tanh()
            });
            vec![Node {
                state: *g,
                turn: p,
                phase: Phase::Mode,
                done: 0,
                policy_kind: policy_kind::NONE,
                n_edges: r.moves.len() as u16,
                total_visits: 0,
                visits,
                root_value,
                flags: flags::TURN_ROOT,
                temperature: temp,
            }]
        } else {
            Vec::new()
        };

        Some(TurnOutcome { mv, nodes })
    }

    fn extra_day(&self, g: &GameState, p: PlayerId, _rng: &mut StdRng) -> (bool, Option<Node>) {
        // The same rule the search itself assumes at every round boundary it
        // steps over, so the agent cannot contradict its own model of the game.
        (crate::search::prefers_extra_day(g, p), None)
    }

    fn draft(&self, g: &GameState, p: PlayerId, dealt: [u8; 4], _rng: &mut StdRng) -> [u8; 2] {
        // Its own estimate, not its margin: at setup the opponents to a later
        // seat have drafted and the ones to an earlier seat have not, so a
        // comparative score would be measuring the seat.
        best_pair(g, p, dealt, |s| crate::eval::heuristic(s, p))
    }

    fn ranked_moves(&self, g: &GameState, p: PlayerId, keep: usize) -> Option<Ranking> {
        Some(self.ranking(g, p, keep))
    }

    fn name(&self) -> String {
        self.label.clone()
    }
}

// =======================================================================
// Agent specs on the command line
// =======================================================================

/// A parsed agent spec, and the factory that builds one instance per thread.
///
/// # Why a factory and not a `Box<dyn Agent>`
///
/// `Agent::play_turn` takes `&self`, so one boxed agent can serve every thread
/// — which is what both binaries used to do. A search agent cannot: it owns a
/// tree arena that `Mcts::search_at` mutates, and it owns a slot in the
/// evaluation batch. Sharing either between 256 concurrent games would
/// serialise the run.
///
/// So the driver parses once and calls [`AgentSpec::instance`] once per worker
/// thread. Everything genuinely shared — the network's weights, the batch queue
/// in front of them — lives in the spec behind an `Arc`; everything per-thread
/// is built fresh.
///
/// # Grammar
///
/// * `random` — the `sample_legal_move` rollout policy.
/// * `heuristic` / `heuristic:K` — one-ply greedy over K sampled candidate
///   turns against `phase::HeuristicEvaluator`. K defaults to 32.
/// * `heuristic:full` — one ply, but over **every** legal move rather than a
///   sample of them. No cap and no traversal-order bias; see
///   [`Candidates::All`].
/// * `greedy:K:EVAL` / `greedy:full:EVAL` — the same two, over any evaluator.
/// * `minimax` / `minimax:DEPTH[:MS[:WIDTH[:MODEL]]]` — alpha-beta from
///   `src/search.rs`. `DEPTH` is in turns (4 is one full round), `MS` is the
///   deepening budget per turn, `WIDTH` the root beam, `MODEL` is `paranoid`
///   (opponents minimise your margin; prunes) or `greedy` (opponents play their
///   own best move; cannot prune but is far cheaper, so it buys depth). Its
///   first ply is always exhaustive whatever the width; the width bounds what
///   gets *deepened*.
/// * `mcts:SIMS` / `mcts:SIMS:EVAL` / `mcts:SIMS[:EVAL]:FLAGS` — tree search
///   from `src/mcts.rs`. `EVAL` defaults to `heuristic`. `FLAGS` is a
///   comma-separated `key=value` list over [`MctsConfig`], recognised by shape
///   so that an evaluator path is never mistaken for one:
///   `pri=eval|1ply`, `ptemp=`, `pmin=`, `cp=`, `cpb=`, `fpu=`, `k=`, `wc=`,
///   `wa=`, `wcap=`, `vl=`, `reuse`, `noreuse`. Every one of them appears in
///   the agent's name — see [`mcts_label`].
/// * `net-random` / `net-random:small` / `net-random:main` — an untrained
///   network. Not a player: it exists so the batched pipeline can be measured
///   at its real cost before any weights are trained.
/// * a path ending `.safetensors` or `.tzw` — `mcts:3200:PATH`.
pub struct AgentSpec {
    spec: String,
    name: String,
    record: bool,
    kind: SpecKind,
    backend: Backend,
    queue: Option<Arc<BatchQueue>>,
    /// Self-play exploration, separately from `record`. See [`Exploration`].
    explore: Exploration,
    /// Instances get distinct MCTS seeds, or every thread searches identically.
    next_seed: AtomicU64,
}

enum SpecKind {
    Random,
    Greedy { cands: Candidates },
    Search { sims: u32, cfg: MctsConfig },
    Minimax { cfg: crate::search::Config },
}

enum Backend {
    Heuristic,
    Net(Arc<NetBatch>),
}

impl Backend {
    /// The evaluator one game thread should use.
    ///
    /// With a pool, this is a `net::BatchHandle`: it blocks inside `evaluate`
    /// while a batcher assembles a full forward pass across the other games in
    /// flight. Without one it is `net::Unbatched`, which `net.rs` names
    /// explicitly so that giving up the 20x is a thing you type.
    fn evaluator(&self) -> Arc<dyn Evaluator> {
        match self {
            Backend::Heuristic => Arc::new(HeuristicEvaluator),
            Backend::Net(n) => Arc::clone(n) as Arc<dyn Evaluator>,
        }
    }
    fn batch_evaluator(&self) -> Arc<dyn BatchEvaluator> {
        match self {
            Backend::Heuristic => Arc::new(Serial(HeuristicEvaluator)),
            Backend::Net(n) => Arc::clone(n) as Arc<dyn BatchEvaluator>,
        }
    }
    fn name(&self) -> String {
        match self {
            Backend::Heuristic => "heuristic".into(),
            Backend::Net(n) => n.label.clone(),
        }
    }
    /// Whether a batcher pool in front of this pays for itself.
    ///
    /// `COMPUTE.md` §2.2 measures 20.3x for the network. `HeuristicEvaluator`
    /// costs a few hundred nanoseconds and gains nothing from amortisation, so
    /// queueing it would only add a condvar round trip per leaf.
    fn batchable(&self) -> bool {
        matches!(self, Backend::Net(_))
    }
}

/// `K` draws from the rollout policy, or `full` to score every legal move.
///
/// Sampling is the cheap default; `full` removes the ceiling that sampling
/// imposes. Eight times the draws still only sees a sliver of a six-figure move
/// list, so `heuristic:256` gains little over `heuristic:32` — the way past that
/// is to stop sampling, not to sample harder.
fn parse_candidates(rest: Option<&str>, what: &str) -> Result<Candidates, String> {
    match rest {
        None => Ok(Candidates::Sampled(32)),
        Some("full") | Some("all") => Ok(Candidates::All),
        Some(v) => {
            let k: usize = v
                .parse()
                .map_err(|_| format!("{what}:K needs a number or `full`, got {v:?}"))?;
            if k == 0 {
                return Err(format!("{what}:K needs K >= 1"));
            }
            Ok(Candidates::Sampled(k))
        }
    }
}

/// `DEPTH[:MS[:WIDTH[:MODEL]]]`, each field defaulting to
/// `search::Config::default`. `MODEL` is `paranoid` or `greedy`.
///
/// An empty field takes the default, so `minimax:6:::greedy` sets only the
/// depth and the opponent model.
fn parse_minimax(rest: Option<&str>) -> Result<crate::search::Config, String> {
    let mut cfg = crate::search::Config::default();
    let Some(rest) = rest else {
        return Ok(cfg);
    };

    let mut parts = rest.split(':');
    let field = |v: Option<&str>, what: &str| -> Result<Option<u64>, String> {
        match v {
            None | Some("") => Ok(None),
            Some(v) => v
                .parse::<u64>()
                .map(Some)
                .map_err(|_| format!("minimax {what} needs a number, got {v:?}")),
        }
    };

    if let Some(d) = field(parts.next(), "DEPTH")? {
        if d == 0 {
            return Err("minimax:DEPTH needs DEPTH >= 1".into());
        }
        cfg.max_depth = d.min(255) as u8;
    }
    if let Some(ms) = field(parts.next(), "MS")? {
        cfg.budget = Duration::from_millis(ms);
    }
    if let Some(w) = field(parts.next(), "WIDTH")? {
        if w == 0 {
            return Err("minimax WIDTH needs WIDTH >= 1".into());
        }
        // The root beam widens; the plies below it keep their taper, scaled to
        // the new root so a wider root does not blow the tree up quadratically.
        let w = w as usize;
        let base = crate::search::Config::default().widths;
        cfg.widths = std::iter::once(w)
            .chain(
                base.iter()
                    .skip(1)
                    .map(|&x| (x * w / base[0]).max(2)),
            )
            .collect();
    }
    match parts.next() {
        None | Some("") => {}
        Some("paranoid") => cfg.opponents = crate::search::Opponents::Paranoid,
        Some("greedy") => cfg.opponents = crate::search::Opponents::Greedy,
        Some(other) => {
            return Err(format!(
                "minimax MODEL is `paranoid` or `greedy`, got {other:?}"
            ))
        }
    }
    // AB-HARNESS (temporary): a comma-separated flag list, so two search
    // variants can be raced in one arena process under identical machine load.
    // Delete once the measurements are banked.
    if let Some(flags) = parts.next() {
        for f in flags.split(',').filter(|f| !f.is_empty()) {
            match f {
                "nocache" => cfg.cache = false,
                "cache" => cfg.cache = true,
                "ownwidth" => cfg.own_width = true,
                "keeptt" => cfg.keep_tt = true,
                "beamfirst" => cfg.beam_first = true,
                "scorefirst" => cfg.beam_first = false,
                "nocapw" => cfg.cap_per_width = None,
                f if f.starts_with("oppw=") => {
                    cfg.opp_width = f[5..]
                        .parse::<usize>()
                        .map_err(|_| "minimax oppw= needs a number".to_string())?
                        .max(1);
                }
                f if f.starts_with("capw=") => {
                    cfg.cap_per_width = f[5..].parse::<usize>().ok().filter(|k| *k > 0);
                    if cfg.cap_per_width.is_none() {
                        return Err("minimax capw= needs a positive number".into());
                    }
                }
                f if f.starts_with("cap=") => {
                    cfg.interior_cap = f[4..]
                        .parse::<usize>()
                        .map_err(|_| "minimax cap= needs a number".to_string())?
                        .max(1);
                }
                f if f.starts_with("keep=") => {
                    cfg.keep = f[5..]
                        .parse::<usize>()
                        .map_err(|_| "minimax keep= needs a number".to_string())?
                        .max(1);
                }
                f if f.starts_with("w=") => {
                    cfg.widths = f[2..]
                        .split('.')
                        .map(|x| x.parse::<usize>().unwrap_or(1).max(1))
                        .collect();
                    if cfg.widths.is_empty() {
                        return Err("minimax w= needs at least one width".into());
                    }
                }
                other => return Err(format!("minimax flag {other:?} unknown")),
            }
        }
    }
    if let Some(extra) = parts.next() {
        return Err(format!(
            "minimax takes minimax:DEPTH[:MS[:WIDTH[:MODEL]]]; did not expect {extra:?}"
        ));
    }
    Ok(cfg)
}

/// The `FLAGS` field of `mcts:SIMS[:EVAL[:FLAGS]]` — a comma-separated
/// `key=value` list over [`MctsConfig`].
///
/// It exists so two variants can be raced inside **one** arena process under
/// identical machine load, which is the only comparison worth anything while
/// `eval.rs` and `moves.rs` are being edited by other agents: legal moves per
/// position moved 184 -> 644 mid-session during the last run, so two numbers
/// from two processes are not comparable even an hour apart.
fn parse_mcts_flags(flags: &str) -> Result<MctsConfig, String> {
    let mut cfg = MctsConfig::default();
    for f in flags.split(',').filter(|f| !f.is_empty()) {
        let (k, v) = f.split_once('=').unwrap_or((f, ""));
        let num = |what: &str| -> Result<f32, String> {
            v.parse::<f32>()
                .map_err(|_| format!("mcts {what}= needs a number, got {v:?}"))
        };
        let int = |what: &str| -> Result<usize, String> {
            v.parse::<usize>()
                .map_err(|_| format!("mcts {what}= needs a number, got {v:?}"))
        };
        match k {
            "priors" | "pri" => {
                cfg.priors = match v {
                    "eval" | "evaluator" | "uniform" => Priors::Evaluator,
                    "1ply" | "oneply" | "heuristic" => Priors::OnePly,
                    other => return Err(format!("mcts priors= is eval or 1ply, got {other:?}")),
                }
            }
            "ptemp" | "pt" => cfg.prior_temp = num("ptemp")?,
            "pmin" => cfg.prior_min_edges = int("pmin")?.max(2),
            "cpuct" | "cp" => cfg.c_puct_init = num("cpuct")?,
            "cpbase" | "cpb" => cfg.c_puct_base = num("cpbase")?.max(1.0),
            "fpu" => cfg.fpu_reduction = num("fpu")?,
            "k" | "maxedges" => cfg.max_edges = int("k")?.max(1),
            "wc" | "widenc" => cfg.widen_c = num("wc")?,
            "wa" | "widenalpha" => cfg.widen_alpha = num("wa")?,
            "wcap" => cfg.widen_cap = int("wcap")?.max(1),
            "vl" => cfg.virtual_loss = int("vl")? as u32,
            "reuse" => cfg.tree_reuse = true,
            "noreuse" => cfg.tree_reuse = false,
            other => return Err(format!("mcts flag {other:?} unknown")),
        }
    }
    Ok(cfg)
}

fn parse_backend(spec: &str) -> Result<Backend, String> {
    let wrap = |net: crate::net::Net| {
        Backend::Net(Arc::new(NetBatch {
            net,
            label: spec.to_string(),
        }))
    };
    match spec {
        "heuristic" => Ok(Backend::Heuristic),
        _ if spec.starts_with("net-random") => {
            let arch = match spec.split_once(':').map(|(_, a)| a) {
                None | Some("small") => crate::net::Arch::SMALL,
                Some("main") => crate::net::Arch::MAIN,
                Some(other) => return Err(format!("net-random:{other:?}; try small or main")),
            };
            Ok(wrap(crate::net::Net::random(arch, 0)))
        }
        path if path.ends_with(".safetensors") || path.ends_with(".tzw") => {
            let net =
                crate::net::Net::load(path).map_err(|e| format!("cannot load {path:?}: {e}"))?;
            Ok(wrap(net))
        }
        other => Err(format!(
            "unknown evaluator {other:?}; try `heuristic`, `net-random`, or a \n\
             path to a .safetensors checkpoint"
        )),
    }
}

impl AgentSpec {
    /// The search configuration behind a `minimax:...` spec, for tests that
    /// need to check a flag reached it rather than trusting the parser.
    pub fn minimax_config(&self) -> Option<&crate::search::Config> {
        match &self.kind {
            SpecKind::Minimax { cfg } => Some(cfg),
            _ => None,
        }
    }

    pub fn parse(spec: &str, record: bool) -> Result<AgentSpec, String> {
        let (kind, backend) = Self::parse_parts(spec)?;
        let name = match &kind {
            SpecKind::Random => "random".to_string(),
            SpecKind::Greedy { cands } => match cands {
                Candidates::Sampled(k) => format!("{}:{k}", backend.name()),
                Candidates::All => format!("{}:full", backend.name()),
            },
            SpecKind::Search { sims, cfg } => mcts_label(*sims, &backend.name(), cfg),
            SpecKind::Minimax { cfg } => minimax_label(cfg),
        };
        Ok(AgentSpec {
            spec: spec.to_string(),
            name,
            record,
            kind,
            backend,
            queue: None,
            // Recording is what self-play does, so it is the right default —
            // but it is only a default now, and the TUI turns it off.
            explore: if record {
                Exploration::SelfPlay
            } else {
                Exploration::Off
            },
            next_seed: AtomicU64::new(0x51ED_5EED),
        })
    }

    fn parse_parts(spec: &str) -> Result<(SpecKind, Backend), String> {
        let (head, rest) = match spec.split_once(':') {
            Some((h, r)) => (h, Some(r)),
            None => (spec, None),
        };
        let num = |v: &str, what: &str| -> Result<u64, String> {
            v.parse::<u64>()
                .map_err(|_| format!("{what} needs a number, got {v:?}"))
        };
        match head {
            "random" => Ok((SpecKind::Random, Backend::Heuristic)),
            "heuristic" => Ok((
                SpecKind::Greedy {
                    cands: parse_candidates(rest, "heuristic")?,
                },
                Backend::Heuristic,
            )),
            "greedy" => {
                let rest = rest.ok_or("greedy needs greedy:K:EVAL or greedy:full:EVAL")?;
                let (k, ev) = rest
                    .split_once(':')
                    .ok_or("greedy needs greedy:K:EVAL or greedy:full:EVAL")?;
                Ok((
                    SpecKind::Greedy {
                        cands: parse_candidates(Some(k), "greedy")?,
                    },
                    parse_backend(ev)?,
                ))
            }
            "minimax" => Ok((
                SpecKind::Minimax {
                    cfg: parse_minimax(rest)?,
                },
                Backend::Heuristic,
            )),
            "mcts" => {
                let rest = rest.ok_or("mcts needs mcts:SIMS or mcts:SIMS:EVAL")?;
                let (sims, tail) = match rest.split_once(':') {
                    Some((s, e)) => (s, e),
                    None => (rest, "heuristic"),
                };
                // The evaluator may itself be a path, which has no colons but
                // does have dots and slashes, so the flag field is recognised
                // by *shape* rather than by position: every comma-separated
                // item has to be a `key=value` or one of the two bare flags.
                // Getting this wrong reads `mcts:64:noreuse` as an evaluator
                // called "noreuse" and says so, which is the failure mode to
                // want.
                let is_flags = |f: &str| {
                    !f.is_empty()
                        && f.split(',')
                            .all(|x| x.contains('=') || matches!(x, "noreuse" | "reuse"))
                };
                let (ev, flags) = match tail.rsplit_once(':') {
                    Some((e, f)) if is_flags(f) => (e, f),
                    _ if is_flags(tail) => ("heuristic", tail),
                    _ => (tail, ""),
                };
                let sims = num(sims, "mcts:SIMS")? as u32;
                if sims == 0 {
                    return Err("mcts:SIMS needs SIMS >= 1".into());
                }
                Ok((
                    SpecKind::Search {
                        sims,
                        cfg: parse_mcts_flags(flags)?,
                    },
                    parse_backend(ev)?,
                ))
            }
            "net-random" => Ok((
                SpecKind::Search {
                    sims: DEFAULT_SIMS,
                    cfg: MctsConfig::default(),
                },
                parse_backend(spec)?,
            )),
            // A bare checkpoint path, checked *after* the named heads. Testing
            // the suffix first meant `mcts:800:ckpt.safetensors` was read as a
            // path called "mcts:800:ckpt.safetensors" and could never load.
            _ if spec.ends_with(".safetensors") || spec.ends_with(".tzw") => Ok((
                SpecKind::Search {
                    sims: DEFAULT_SIMS,
                    cfg: MctsConfig::default(),
                },
                parse_backend(spec)?,
            )),
            _ => Err(format!(
                "unknown agent {spec:?}; try `random`, `heuristic:64`, `heuristic:full`, \n\
                 `minimax:4`, `mcts:800`, `mcts:800:net-random`, or a path to a checkpoint"
            )),
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn spec(&self) -> &str {
        &self.spec
    }

    /// True when this agent searches, and therefore wants many concurrent games
    /// rather than one per core (`COMPUTE.md` §2.1).
    pub fn searches(&self) -> bool {
        matches!(self.kind, SpecKind::Search { .. })
    }

    /// The search configuration behind an `mcts:...` spec, for tests that check
    /// a flag reached it rather than trusting the parser.
    pub fn mcts_config(&self) -> Option<&MctsConfig> {
        match &self.kind {
            SpecKind::Search { cfg, .. } => Some(cfg),
            _ => None,
        }
    }

    /// Turn self-play exploration on or off independently of `record`.
    ///
    /// The TUI wants both: decision nodes for its search panel, and the policy
    /// the search actually believes rather than one with 0.25 Dirichlet noise
    /// mixed into it. Before this the two came as a pair.
    pub fn set_exploration(&mut self, explore: Exploration) {
        self.explore = explore;
    }

    /// True when a batch queue in front of the evaluator pays for itself.
    pub fn batchable(&self) -> bool {
        self.searches() && self.backend.batchable()
    }

    /// Start a batcher pool in front of the network.
    ///
    /// Call before any `instance`: an instance built earlier holds an evaluator
    /// that does not know about the pool. Returns false when the backend is not
    /// a network, in which case there is nothing worth batching.
    pub fn enable_batching(&mut self, max_batch: usize, linger: Duration, threads: usize) -> bool {
        if !self.backend.batchable() {
            return false;
        }
        let q = BatchQueue::new(self.backend.batch_evaluator(), max_batch, linger);
        q.spawn_batchers(threads);
        self.queue = Some(q);
        true
    }

    /// What the batcher actually saw, or `None` if there is no pool.
    ///
    /// `COMPUTE.md` §2.6: if the mean batch size is not close to the configured
    /// maximum, none of §5's throughput is happening, and this is the only
    /// symptom you will get.
    pub fn batch_stats(&self) -> Option<BatchStats> {
        self.queue.as_ref().map(|q| q.stats())
    }

    /// Stop the batchers. Call once every instance has been dropped.
    pub fn shutdown(&self) {
        if let Some(q) = &self.queue {
            q.shutdown();
        }
    }

    /// One agent for one worker thread. Never share the result between threads.
    pub fn instance(&self) -> Box<dyn Agent> {
        let seed = self.next_seed.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
        match &self.kind {
            SpecKind::Random => Box::new(RandomAgent {
                record: self.record,
            }),
            SpecKind::Greedy { cands } => Box::new(GreedyAgent {
                ev: SharedEval(self.backend.evaluator()),
                cands: *cands,
                record: self.record,
            }),
            SpecKind::Search { sims, cfg } => {
                let ev: Arc<dyn Evaluator> = match &self.queue {
                    Some(q) => Arc::new(q.handle()),
                    None => self.backend.evaluator(),
                };
                Box::new(SearchAgent::with_config(
                    ev,
                    *sims,
                    self.record,
                    self.explore,
                    *cfg,
                    seed,
                ))
            }
            SpecKind::Minimax { cfg } => Box::new(MinimaxAgent::new(cfg.clone(), self.record)),
        }
    }
}

/// Back-compatible single-instance parse, for callers with one thread.
///
/// Both binaries use [`AgentSpec`] directly; this stays because it is a
/// one-liner and because the tests and any future one-shot tool want it.
/// Reject any `--flag` the caller does not know.
///
/// Without this a mistyped flag is silently ignored and the tool runs its
/// defaults, which is indistinguishable from success. `arena --a X --b Y`
/// (the flags are `--candidate` and `--baseline`) reported a clean, confident
/// benchmark of `heuristic:32` against `random` and called it a trained net.
/// `1h 04m 12s`, `04m 12s`, `12s` -- whichever is the shortest that is honest.
pub fn hms(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}h {m:02}m {sec:02}s")
    } else if m > 0 {
        format!("{m}m {sec:02}s")
    } else {
        format!("{sec}s")
    }
}

/// Group digits, because eight-figure evaluation counts are unreadable raw.
pub fn thousands(n: u64) -> String {
    let d = n.to_string();
    let mut out = String::with_capacity(d.len() + d.len() / 3);
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn reject_unknown_flags(known: &[&str]) -> Result<(), String> {
    for a in std::env::args().skip(1) {
        if a.starts_with("--") && !known.contains(&a.as_str()) {
            let near: Vec<&str> = known
                .iter()
                .copied()
                .filter(|k| {
                    let (a, k) = (a.trim_start_matches('-'), k.trim_start_matches('-'));
                    k.starts_with(a) || a.starts_with(k)
                })
                .collect();
            return Err(if near.is_empty() {
                format!("unknown option {a}; --help lists them")
            } else {
                format!("unknown option {a}; did you mean {}?", near.join(" or "))
            });
        }
    }
    Ok(())
}

pub fn parse_agent(spec: &str, record: bool) -> Result<Box<dyn Agent>, String> {
    Ok(AgentSpec::parse(spec, record)?.instance())
}

/// An agent that hands back its decision nodes but plays as if it were being
/// evaluated: no Dirichlet noise, no playout-cap randomisation.
///
/// This is what a viewer wants — `bin/tui` and `bin/uidump` — and it is not
/// what `parse_agent(spec, true)` gives, because `record` used to imply both.
pub fn parse_analysis_agent(spec: &str) -> Result<Box<dyn Agent>, String> {
    let mut spec = AgentSpec::parse(spec, true)?;
    spec.set_exploration(Exploration::Off);
    Ok(spec.instance())
}

// =======================================================================
// The game driver, shared by self-play and the arena
// =======================================================================

pub struct GameConfig {
    /// Move-selection temperature by round. `SEARCH.md` §3.8 and
    /// `LEARNING.md` §6.4: 1.0 early, annealing to ~0 late. Evaluation games
    /// pass all zeros.
    pub temperature: fn(u8) -> f32,
    /// Run `check_move` and `invariants::validate` after every turn. Slow;
    /// worth it for the first few thousand games of a new pipeline.
    pub check: bool,
}

impl GameConfig {
    pub fn evaluation() -> Self {
        GameConfig {
            temperature: |_| 0.0,
            check: false,
        }
    }
    pub fn selfplay() -> Self {
        GameConfig {
            temperature: selfplay_temperature,
            check: false,
        }
    }
}

/// `LEARNING.md` §6.4, adapted to a 27-round calendar: 1.0 for rounds 1-9,
/// 0.5 for 10-18, 0.1 after. Not "the first 30 moves" — move counts do not
/// transfer from chess.
pub fn selfplay_temperature(day: u8) -> f32 {
    match day {
        0..=8 => 1.0,
        9..=17 => 0.5,
        _ => 0.1,
    }
}

pub struct GameResult {
    pub scores: [i16; N_PLAYERS],
    pub winners: Vec<PlayerId>,
    pub win_share: [f32; N_PLAYERS],
    pub z_rel: [f32; N_PLAYERS],
    pub days: u8,
    pub nodes: Vec<Node>,
    /// Set if the game hit the round guard, which means a rules bug.
    pub aborted: bool,
}

/// Play one complete game with a possibly different agent in each seat.
///
/// This deliberately does not use `Game::play_round`: that method hard-codes
/// random moves and coin-flips the extra day. The round flow below mirrors
/// `Game::end_round` exactly — turns in seat order from `first_player`, then
/// `resolve_first_player`, then the extra-day decision taken *before* the
/// advance, then one `advance_days(1 or 2)` — so any divergence is a bug here,
/// not a design difference. That ordering matters: a food day passed over must
/// resolve on the round landed on, which is why the two days go in one call.
pub fn play_game(
    seed: u64,
    agents: &[&dyn Agent; N_PLAYERS],
    cfg: &GameConfig,
    rng: &mut StdRng,
) -> GameResult {
    use crate::invariants::validate;
    use crate::moves::check_move;

    // Setup, with each seat drafting its own starting tiles. `Game::new` would
    // draft them at random, which is two tiles a game of pure luck in a
    // measurement whose whole design is about removing exactly that.
    let mut state = new_drafted_game(seed, agents, rng);
    let mut nodes: Vec<Node> = Vec::new();
    let mut aborted = false;
    let mut guard = 0u32;

    while !state.over {
        state.current = state.first_player;
        for _ in 0..N_PLAYERS {
            let p = state.current;
            let temp = (cfg.temperature)(state.day);
            if let Some(out) = agents[p.idx()].play_turn(&state, p, temp, rng) {
                if cfg.check {
                    if let Err(e) = check_move(&state, p, &out.mv) {
                        eprintln!("seed {seed} day {}: illegal move: {e}", state.day);
                        aborted = true;
                    }
                }
                nodes.extend(out.nodes);
                apply_move(&mut state, p, &out.mv);
                state.refill_buildings();
                if cfg.check {
                    if let Err(e) = validate(&state) {
                        eprintln!("seed {seed} day {}: invariant: {e}", state.day);
                        aborted = true;
                    }
                }
            }
            state.current = state.current.next(1);
        }

        let claimer = state.resolve_first_player();
        let mut days = 1u8;
        if let Some(p) = claimer {
            if state.may_take_extra_day(p) {
                let (take, node) = agents[p.idx()].extra_day(&state, p, rng);
                if let Some(n) = node {
                    nodes.push(n);
                }
                if take {
                    state.spend_extra_day(p);
                    days = 2;
                }
            }
        }
        state.advance_days(days);

        guard += 1;
        if guard > 200 {
            aborted = true;
            break;
        }
    }

    let scores = state.scores();
    GameResult {
        winners: state.winners(),
        win_share: win_share(&state),
        z_rel: z_rel(scores),
        scores,
        days: state.day,
        nodes,
        aborted,
    }
}

// =======================================================================
// Statistics
// =======================================================================

/// Sample mean, sample standard deviation, and a two-sided 95% confidence
/// interval half-width for the mean.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    /// Half-width of the 95% CI: report `mean ± ci`.
    pub ci: f64,
}

impl Summary {
    pub fn of(xs: &[f64]) -> Summary {
        let n = xs.len();
        if n == 0 {
            return Summary::default();
        }
        let mean = xs.iter().sum::<f64>() / n as f64;
        if n == 1 {
            return Summary {
                n,
                mean,
                sd: f64::NAN,
                ci: f64::INFINITY,
            };
        }
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let sd = var.sqrt();
        let se = sd / (n as f64).sqrt();
        Summary {
            n,
            mean,
            sd,
            ci: t_crit_95(n as f64 - 1.0) * se,
        }
    }

    /// Two-sided p-value for `mean == null`, from the t statistic under a
    /// normal approximation. Reported as a sanity check, not as a decision
    /// rule — see the note about repeated looks in `docs/TRAINING.md`.
    pub fn p_value(&self, null: f64) -> f64 {
        if self.n < 2 || self.sd == 0.0 || !self.sd.is_finite() {
            return f64::NAN;
        }
        let se = self.sd / (self.n as f64).sqrt();
        let t = (self.mean - null) / se;
        2.0 * (1.0 - normal_cdf(t.abs()))
    }
}

/// Two-sided 95% t critical value, Cornish-Fisher expansion about z = 1.96.
/// Within 0.001 of the exact value for df >= 6, which is far tighter than the
/// clustering assumptions underneath it.
pub fn t_crit_95(df: f64) -> f64 {
    if df < 1.0 {
        return f64::INFINITY;
    }
    let z = 1.959_963_985_f64;
    let (z2, z3) = (z * z, z * z * z);
    let z5 = z3 * z2;
    let z7 = z5 * z2;
    let v = df;
    z + (z3 + z) / (4.0 * v)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * v * v)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * v * v * v)
}

/// Abramowitz & Stegun 7.1.26 for `erf`, folded into the normal CDF. Absolute
/// error under 1.5e-7, which is well past what a few hundred games can say.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

// =======================================================================
// Tests
// =======================================================================
//
// These live here rather than in `tests/rules.rs`, which is the engine's.
// Because this module is `#[path]`-included by two binaries, `cargo test` runs
// them once per binary — duplicated work, but the alternative is not running
// them at all.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;

    /// The one property the whole replay buffer rests on: a state that goes
    /// through the codec comes back `==` to itself. `GameState` derives `Eq`
    /// over every field, so this is not a partial check.
    #[test]
    fn state_codec_round_trips() {
        for seed in 0..12u64 {
            let mut g = Game::new(seed);
            for _ in 0..7 {
                if g.state.over {
                    break;
                }
                g.play_round();
                let mut buf = [0u8; STATE_SLOT];
                encode_state(&g.state, &mut buf);
                assert_eq!(decode_state(&buf), g.state, "seed {seed}");
            }
            // Terminal states too: `over`, the emptied decks and the end-game
            // score adjustment are all in the codec's path.
            g.run_sampled();
            let mut buf = [0u8; STATE_SLOT];
            encode_state(&g.state, &mut buf);
            assert_eq!(decode_state(&buf), g.state, "seed {seed} terminal");
        }
    }

    #[test]
    fn phase_codec_round_trips() {
        let all = [
            Phase::Beg,
            Phase::Mode,
            Phase::Placing { n: 3 },
            Phase::PickWorker,
            Phase::Take {
                worker: WorkerId(17),
            },
            Phase::ExtraDay {
                claimer: PlayerId(2),
            },
            Phase::PityPlace,
            Phase::DraftTile {
                dealt: [4, 9, 1, 20],
                kept: 1,
            },
        ];
        for p in all {
            let (t, a) = encode_phase(p);
            assert_eq!(decode_phase(t, a), p, "{p:?}");
            assert_eq!(t, p.tag());
        }
    }

    #[test]
    fn record_round_trips() {
        let g = Game::new(7).state;
        let node = Node {
            state: g,
            turn: PlayerId(1),
            phase: Phase::Placing { n: 2 },
            done: 0,
            policy_kind: policy_kind::TREE_EDGE,
            n_edges: 7,
            total_visits: 800,
            visits: vec![(3, 500), (0, 200), (6, 100)],
            root_value: [0.1, -0.2, 0.05, 0.05],
            flags: flags::TURN_ROOT | flags::FULL_BUDGET,
            temperature: 1.0,
        };
        let scores = [80i16, 71, 95, 60];
        let mut buf = [0u8; RECORD_BYTES];
        write_record(&mut buf, &node, 12345, 4, scores);
        let r = read_record(&buf);
        let (st, turn, ph, sc, rules) = (r.state, r.turn, r.phase, r.scores, r.rules_version);
        // `done` rides in phase_args[0] and only `PickWorker` uses it.
        {
            let mut pick = node.clone();
            pick.phase = Phase::PickWorker;
            pick.done = 3;
            let mut b2 = [0u8; RECORD_BYTES];
            write_record(&mut b2, &pick, 1, 0, [0; N_PLAYERS]);
            let back = read_record(&b2);
            assert_eq!(back.phase, Phase::PickWorker);
            assert_eq!(back.done, 3);
        }
        assert_eq!(st, g);
        assert_eq!(turn, PlayerId(1));
        assert_eq!(ph, Phase::Placing { n: 2 });
        assert_eq!(sc, scores);
        assert_eq!(rules, RULES_VERSION);
        assert_eq!(
            u32::from_le_bytes(buf[O_TOTAL_VISITS..O_TOTAL_VISITS + 4].try_into().unwrap()),
            800
        );
    }

    /// `z_rel` must be centred, or a max^n backup has two currencies.
    #[test]
    fn z_rel_is_centred_and_bounded() {
        for scores in [[80i16, 71, 95, 60], [0, 0, 0, 0], [-70, 40, 10, 5]] {
            let z = z_rel(scores);
            assert!(z.iter().all(|v| v.abs() < 1.0));
            // tanh is not linear, so the sum is only near zero; what matters is
            // that the ordering and the sign structure survive.
            let mean = z.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 0.35, "{z:?}");
            let best = scores.iter().enumerate().max_by_key(|(_, s)| **s).unwrap().0;
            let zbest = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(z[best], zbest);
        }
    }

    #[test]
    fn t_crit_matches_tables() {
        for (df, want) in [(5.0, 2.571), (10.0, 2.228), (30.0, 2.042), (120.0, 1.980)] {
            let got = t_crit_95(df);
            assert!((got - want).abs() < 0.01, "df {df}: got {got}, want {want}");
        }
    }

    #[test]
    fn summary_ci_covers() {
        let xs: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
        let s = Summary::of(&xs);
        assert_eq!(s.n, 100);
        assert!(s.ci > 0.0 && s.ci.is_finite());
        assert!((s.mean - 3.0).abs() < 0.5);
    }

    #[test]
    fn a_greedy_agent_beats_a_random_one() {
        // The whole apparatus in one test: if one ply of the heuristic does not
        // beat the rollout policy over a handful of games, either the driver or
        // the evaluator is wired up wrong and every arena number is noise.
        let good = GreedyAgent {
            ev: HeuristicEvaluator,
            cands: Candidates::Sampled(24),
            record: false,
        };
        let bad = RandomAgent { record: false };
        let cfg = GameConfig::evaluation();
        let mut margin = 0.0f64;
        for seed in 0..6u64 {
            let agents: [&dyn Agent; N_PLAYERS] = [&good, &bad, &bad, &bad];
            let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(seed);
            let r = play_game(seed, &agents, &cfg, &mut rng);
            assert!(!r.aborted, "seed {seed} did not terminate");
            let mean = r.scores.iter().map(|&s| s as f64).sum::<f64>() / 4.0;
            margin += r.scores[0] as f64 - mean;
        }
        assert!(margin / 6.0 > 5.0, "mean centred margin only {}", margin / 6.0);
    }

    #[test]
    fn every_record_carries_the_rules_version() {
        // The stamp is what turns "a rules change makes the whole buffer
        // suspect" into "it invalidates the records before generation N".
        // `write_record` writes it unconditionally; this is the regression test
        // that keeps it that way.
        let g = crate::Game::new(9).state;
        let n = Node::value_only(g, PlayerId(0), Phase::Mode, [0.0; N_PLAYERS]);
        let mut buf = [0u8; RECORD_BYTES];
        for i in 0..3u16 {
            write_record(&mut buf, &n, 7, i, [10, 20, 30, 40]);
            assert_eq!(read_record(&buf).rules_version, RULES_VERSION);
            assert_eq!(
                u32::from_le_bytes(buf[O_RULES_VERSION..].try_into().unwrap()),
                RULES_VERSION
            );
        }
    }

    #[test]
    fn agent_specs_parse_or_say_why() {
        for spec in ["random", "heuristic", "heuristic:8", "mcts:4", "mcts:4:heuristic"] {
            let a = AgentSpec::parse(spec, false).unwrap_or_else(|e| panic!("{spec}: {e}"));
            assert!(!a.name().is_empty());
        }
        assert!(AgentSpec::parse("mcts:4", false).unwrap().searches());
        assert!(!AgentSpec::parse("heuristic:8", false).unwrap().searches());
        // A heuristic evaluator gains nothing from batching, so the driver must
        // not spend 256 threads and a condvar round trip on it.
        assert!(!AgentSpec::parse("mcts:4:heuristic", false).unwrap().batchable());
        for bad in ["nonsense", "heuristic:0", "mcts:0", "mcts", "greedy:4"] {
            assert!(AgentSpec::parse(bad, false).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn a_search_agent_plays_legal_moves_and_records_regenerable_targets() {
        use crate::moves::check_move;
        use rand::SeedableRng;
        let spec = AgentSpec::parse("mcts:24:heuristic", true).unwrap();
        let agent = spec.instance();
        let mut rng = StdRng::seed_from_u64(4);
        let mut state = crate::Game::new(11).state;

        // Playout-cap randomisation writes only the 25% of turns that get the
        // full budget, so a dozen turns is not enough to be sure of seeing one.
        let mut checked = 0;
        let mut turns = 0;
        for _ in 0..60 {
            turns += 1;
            let p = state.current;
            let Some(out) = agent.play_turn(&state, p, 1.0, &mut rng) else {
                break;
            };
            check_move(&state, p, &out.mv).unwrap_or_else(|e| panic!("illegal move: {e}"));
            for n in &out.nodes {
                assert_eq!(n.policy_kind, policy_kind::TREE_EDGE);
                // The whole contract of `TREE_EDGE`: the stored index must be
                // regenerable from `(state, phase)` by a loader that has no
                // access to the priors the tree sorted its edges by.
                let legal = tree::legal_steps(&n.state, n.phase, n.turn, n.done);
                assert_eq!(n.n_edges as usize, legal.len());
                for &(idx, cnt) in &n.visits {
                    assert!((idx as usize) < legal.len(), "edge index out of range");
                    assert!(cnt > 0);
                }
                assert!(n.visits.len() <= MAX_VISITS);
                checked += 1;
            }
            apply_move(&mut state, p, &out.mv);
            state.refill_buildings();
            state.current = state.current.next(1);
        }
        assert!(checked > 0, "a searching agent recorded nothing in {turns} turns");
    }

    #[test]
    fn the_batch_queue_answers_every_request_from_many_threads() {
        // The property that matters is not throughput but that a parked worker
        // is always woken with *its own* answer. 24 threads x 40 requests, each
        // carrying a distinguishable payload.
        struct Echo;
        impl BatchEvaluator for Echo {
            fn evaluate_batch(&self, qs: &[EvalRequest], out: &mut Vec<Evaluation>) {
                for q in qs {
                    out.push(Evaluation {
                        priors: vec![1.0; q.n_edges],
                        value: [q.n_edges as f32, 0.0, 0.0, 0.0],
                    });
                }
            }
            fn name(&self) -> String {
                "echo".into()
            }
        }
        let q = BatchQueue::new(Arc::new(Echo), 16, Duration::from_micros(50));
        q.spawn_batchers(2);
        let g = crate::Game::new(3).state;
        std::thread::scope(|s| {
            for t in 0..24usize {
                let q = &q;
                let g = &g;
                s.spawn(move || {
                    let h = q.handle();
                    for i in 0..40usize {
                        let n = 1 + (t * 40 + i) % 29;
                        let e = h.evaluate(g, Phase::Mode, PlayerId(0), n);
                        assert_eq!(e.priors.len(), n);
                        assert_eq!(e.value[0], n as f32, "a worker got someone else's answer");
                    }
                });
            }
        });
        let st = q.stats();
        assert_eq!(st.queries, 24 * 40);
        assert!(st.batches > 0);
        assert!(st.mean_batch >= 1.0);
        q.shutdown();
    }

    #[test]
    fn batching_does_not_change_what_the_search_sees() {
        // `COMPUTE.md` §2.5's claim, as a test: with one descent per tree the
        // batch is assembled across games, so a batched evaluator returns
        // exactly what an unbatched one would and the search is identical.
        let g = crate::Game::new(21).state;
        let plain = HeuristicEvaluator;
        let q = BatchQueue::new(
            Arc::new(Serial(HeuristicEvaluator)),
            8,
            Duration::from_micros(0),
        );
        q.spawn_batchers(1);
        let h = q.handle();
        for phase in [Phase::Beg, Phase::Mode, Phase::PickWorker] {
            for n in [1usize, 5, 17] {
                let a = plain.evaluate(&g, phase, PlayerId(1), n);
                let b = h.evaluate(&g, phase, PlayerId(1), n);
                assert_eq!(a.priors, b.priors);
                assert_eq!(a.value, b.value);
            }
        }
        drop(h);
        q.shutdown();
    }

    /// What the batching is worth on this machine, measured rather than
    /// assumed. `COMPUTE.md` §2.2 predicts 5,491 evals/s at batch 1 against
    /// 111,469 at batch 256, on one core.
    ///
    ///     cargo test --release --bin selfplay -- --ignored --nocapture batching_is_worth
    #[test]
    #[ignore = "a benchmark, not a test"]
    fn batching_is_worth_measuring() {
        use rand::SeedableRng;
        use std::time::Instant;
        let net = crate::net::Net::random(crate::net::Arch::SMALL, 0);
        println!("gemm backend: {}", net.backend());
        let g = crate::Game::new(5).state;

        // A `Take` node too. `net.rs` says an unsupplied `Query::candidates`
        // costs a `choices_for_worker` re-derivation of ~35 us — "more than two
        // forward passes" — and `phase::Evaluator::evaluate` has no way to pass
        // one, so every `Take` evaluation the search makes pays it.
        let mut take = g;
        let mut rng = StdRng::seed_from_u64(3);
        let mut take_at = None;
        'outer: for _ in 0..8 {
            for i in 0..N_PLAYERS {
                let p = PlayerId(i as u8);
                if let Some(m) = sample_legal_move(&take, p, &mut rng) {
                    apply_move(&mut take, p, &m);
                    take.refill_buildings();
                }
                if let Some(w) = take.on_board(p).next() {
                    let n = tree::take_candidates(&take, p, w).len();
                    if n > 1 {
                        take_at = Some((p, w, n));
                        break 'outer;
                    }
                }
            }
        }

        for b in [1usize, 8, 32, 64, 128, 256] {
            if let Some((p, w, n)) = take_at {
                let qs: Vec<crate::net::Query<'_>> = (0..b)
                    .map(|_| {
                        crate::net::Query::new(&take, Phase::Take { worker: w }, p, n)
                    })
                    .collect();
                let mut out = Vec::with_capacity(b);
                for _ in 0..2 {
                    out.clear();
                    net.evaluate_batch(&qs, &mut out);
                }
                let reps = (2048 / b).max(2);
                let t = Instant::now();
                for _ in 0..reps {
                    out.clear();
                    net.evaluate_batch(&qs, &mut out);
                }
                println!(
                    "  Take({n} candidates) batch {b:>4}: {:>9.0} evals/s",
                    (reps * b) as f64 / t.elapsed().as_secs_f64()
                );
            }
            let qs: Vec<crate::net::Query<'_>> = (0..b)
                .map(|_| crate::net::Query::new(&g, Phase::Placing { n: 0 }, PlayerId(0), 6))
                .collect();
            let mut out = Vec::with_capacity(b);
            // warm up
            for _ in 0..3 {
                out.clear();
                net.evaluate_batch(&qs, &mut out);
            }
            let reps = (4096 / b).max(2);
            let t = Instant::now();
            for _ in 0..reps {
                out.clear();
                net.evaluate_batch(&qs, &mut out);
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "  Placing        batch {b:>4}: {:>9.0} evals/s   ({:.1} us per call)",
                (reps * b) as f64 / secs,
                secs / reps as f64 * 1e6
            );
        }
    }

    #[test]
    fn schema_is_well_formed_and_agrees_with_the_writer() {
        let s = schema_json();
        let v: serde_json::Value = serde_json::from_str(&s).expect("schema is valid JSON");
        assert_eq!(v["record_bytes"], RECORD_BYTES);
        assert_eq!(v["state_bytes"], STATE_BYTES);
        let fields = v["state_fields"].as_array().unwrap();
        let last = fields.last().unwrap();
        assert_eq!(
            last["offset"].as_u64().unwrap() + last["count"].as_u64().unwrap(),
            STATE_BYTES as u64
        );
    }
}
