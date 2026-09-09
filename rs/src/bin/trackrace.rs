//! SCRATCH — the per-track research-shape race. `docs/FINDINGS-track-shape.md`.
//!
//! `bin/arena` cannot answer "is this evaluator better than that one": both of
//! its agent specs resolve to whatever `eval::heuristic` is compiled into the
//! binary, so in a patched binary the candidate and the baseline are the same
//! agent and the centred score is 0 by construction.
//!
//! `bin/evalab` was built for exactly this job and cannot be used: its
//! parameterised copy of the evaluator is pinned at `6a99f2f` and `src/eval.rs`
//! has moved a long way since (`RESEARCH_SCALE` 0.5 -> 0.05, `ACTION_VALUE`
//! 1.2 -> 0.2, `CORN_INCOME_PER_ROUND` 1.9 -> 0.0, new `BOARD_SCALE` /
//! `TEMPLE_SCALE` / `GEAR_SCALE` / `hungry` / the temple `ceiling`), so
//! `V::HEAD` no longer reproduces `eval::heuristic`. And it would only move the
//! **leaf value**: `mcts.rs` calls `crate::eval::heuristic` *directly* for the
//! default prior (`Priors::OnePly`, line ~2269) and for the default edge
//! ordering (`EdgeOrder::Gradient`, `Gradient::new`, lines ~953/957), neither
//! of which goes through the injected `Evaluator`. Edge ordering is precisely
//! where "which track does it pick" is decided, so an injected evaluator would
//! systematically under-measure the thing under test.
//!
//! # How two evaluators live in one process
//!
//! `<scratch>/ts/patches/harness.patch` (unapplied; applied only into a
//! pristine `git archive` copy of the tree) makes the arm a property of the
//! **seat**: a thread-local `[u8; 4]` inside `eval.rs`, read by
//! `engine_value`. Every `eval::heuristic` call site passes the seat whose
//! position is being estimated — `one_ply` passes the mover, `Gradient::new`
//! passes the mover, `HeuristicEvaluator` estimates all four and centres them —
//! so a seat mask is exactly "these two agents are playing each other".
//!
//! This file compiles and runs **without** that patch, where only `--arm head`
//! is available; that is the null run and the platform check. With the patch it
//! is built as
//!
//!     RUSTFLAGS="--cfg trackarm" cargo build --release --bin trackrace
//!
//! # Design
//!
//! `bin/arena --mode solo`'s, exactly: one rotation **block** is one seed
//! played four times with the armed seat rotating through all four positions,
//! three HEAD seats opposite it. The block, not the game, is the independent
//! unit; null centred score 0; **null win rate 0.25**.
//!
//! # Why its own driver
//!
//! `record::play_game` returns a `GameResult` with no final state, and R16's
//! prediction is behavioural: research *uptake per track*, not score. So this
//! file carries its own copy of the round flow — the same copy `bin/rlab` needs
//! and for the same reason — and `--verify` plays the same seeds through both
//! drivers and compares scores. Any disagreement is a bug here.
//!
//!     trackrace --table                    print every arm's twelve constants
//!     trackrace --verify 8                 this driver against record::play_game
//!     trackrace --arm both --blocks 400 --out FILE
//!
//! Progress is appended to JSONL as each block completes, so a Ctrl-C loses
//! nothing and `--resume` picks up where it stopped. A `.lock` file beside
//! `--out` refuses a second writer, which is the failure `FINDINGS-eval.md`
//! F52a paid for.
#![allow(unexpected_cfgs)]

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use tzolkin::ids::*;
use tzolkin::moves::apply_move;
use tzolkin::record::{self, interrupt, Agent, AgentSpec, GameConfig, Summary};


// =======================================================================
// THE TABLE. Generated from `<scratch>/ts/gen.py`, which is the single source
// of truth; `<scratch>/ts/tables.py` diffs this against `harness.patch` and
// against every landing patch, so a typo in one cannot survive.
//
// head    src/eval.rs as committed.                     sums 1.55/1.95/1.95/2.00 = 7.45
// redist  FINDINGS-research-value.md R16, verbatim.      sums 1.08/0.65/3.40/2.31 = 7.44
// mirror  the exact reflection of redist about head:
//         sum_mirror[t] = 2*sum_head[t] - sum_redist[t]. sums 2.02/3.25/0.50/1.69 = 7.46
// =======================================================================

/// Arm ids. Index 0 must stay `head`.
const ARM_NAMES: [&str; 6] = ["head", "redist", "nostep2", "both", "mirror", "step2x"];

/// `research_step_value`'s twelve constants, per arm, `[track][level - 1]`,
/// tracks in `Science::ALL` order (Agriculture, Extraction, Architecture,
/// Theology).
const ARM_STEPS: [[[f32; 3]; 4]; 6] = [
    // head     head
    [[0.35, 0.45, 0.75], [0.55, 0.65, 0.75], [0.30, 0.85, 0.80], [0.45, 0.65, 0.90]],
    // redist   redist
    [[0.25, 0.31, 0.52], [0.18, 0.22, 0.25], [0.52, 1.48, 1.40], [0.52, 0.75, 1.04]],
    // nostep2  head
    [[0.35, 0.45, 0.75], [0.55, 0.65, 0.75], [0.30, 0.85, 0.80], [0.45, 0.65, 0.90]],
    // both     redist
    [[0.25, 0.31, 0.52], [0.18, 0.22, 0.25], [0.52, 1.48, 1.40], [0.52, 0.75, 1.04]],
    // mirror   mirror
    [[0.46, 0.59, 0.97], [0.92, 1.08, 1.25], [0.08, 0.21, 0.21], [0.38, 0.55, 0.76]],
    // step2x   head
    [[0.35, 0.45, 0.75], [0.55, 0.65, 0.75], [0.30, 0.85, 0.80], [0.45, 0.65, 0.90]],
];

/// The `lvl == 2 && horizon > 0.15` step, per arm.
const ARM_L2_BONUS: [f32; 6] = [0.4, 0.4, 0.0, 0.0, 0.4, 0.8];

// =======================================================================
// The seam. Present at HEAD as a stub that accepts only arm 0.
// =======================================================================

#[cfg(trackarm)]
const HARNESS: bool = true;
#[cfg(not(trackarm))]
const HARNESS: bool = false;

#[cfg(trackarm)]
fn set_arms(mask: [u8; N_PLAYERS]) {
    tzolkin::eval::set_arms(mask);
}

/// Without the harness there is one evaluator and it is HEAD's, so any arm but
/// 0 would be silently measured as the null. Refuse instead.
#[cfg(not(trackarm))]
fn set_arms(mask: [u8; N_PLAYERS]) {
    assert!(
        mask.iter().all(|&a| a == 0),
        "built without --cfg trackarm: only --arm head is available, and asking \
         for any other arm here would measure HEAD and label it as the arm"
    );
}

/// Cross-check the table this file carries against the one compiled into
/// `eval.rs`. Without the harness there is nothing to check against.
#[cfg(trackarm)]
fn check_tables() -> Result<(), String> {
    for (i, name) in ARM_NAMES.iter().enumerate() {
        let theirs = tzolkin::eval::arm_steps(i);
        if theirs != ARM_STEPS[i] {
            return Err(format!("arm {name}: eval.rs table {theirs:?} != {:?}", ARM_STEPS[i]));
        }
        let b = tzolkin::eval::arm_l2_bonus(i);
        if b != ARM_L2_BONUS[i] {
            return Err(format!("arm {name}: eval.rs bonus {b} != {}", ARM_L2_BONUS[i]));
        }
    }
    Ok(())
}
#[cfg(not(trackarm))]
fn check_tables() -> Result<(), String> {
    Ok(())
}

fn arm_id(name: &str) -> Result<u8, String> {
    ARM_NAMES
        .iter()
        .position(|&a| a == name)
        .map(|i| i as u8)
        .ok_or_else(|| format!("unknown --arm {name:?}; try one of {ARM_NAMES:?}"))
}

// =======================================================================
// One game, with the state kept
// =======================================================================

/// The day the early/late split is taken at. `FINDINGS-eval.md` F52b's `early`
/// column and `FINDINGS-research-value.md`'s forcing windows both cut here: it
/// is the first `POINT_DAY`, and 81% of the extra research a raised
/// `RESEARCH_SCALE` buys is in place by it.
const SPLIT_DAY: u8 = 14;

/// Never advanced.
const NEVER: u8 = 255;

struct Trace {
    scores: [i16; N_PLAYERS],
    win_share: [f32; N_PLAYERS],
    days: u8,
    aborted: bool,
    /// Final level, per seat per track.
    lv: [[u8; 4]; N_PLAYERS],
    /// Level at `SPLIT_DAY`, per seat per track.
    lv14: [[u8; 4]; N_PLAYERS],
    /// Day of the first advance on that track, `NEVER` if none.
    first: [[u8; 4]; N_PLAYERS],
    /// Buildings held at the end — Architecture's whole mechanism is builds
    /// (R12: a granted C3 seat builds 9.14 a game against a baseline 2.71), so
    /// this is the cheapest confirmation that a shift toward C is real.
    buildings: [u8; N_PLAYERS],
}

fn play_one(
    seed: u64,
    agents: &[&dyn Agent; N_PLAYERS],
    cfg: &GameConfig,
    rng: &mut StdRng,
) -> Trace {
    let mut state = record::new_drafted_game(seed, agents, rng);
    let mut lv14 = state.research;
    let mut split_taken = false;
    let mut first = [[NEVER; 4]; N_PLAYERS];
    let mut aborted = false;
    let mut guard = 0u32;

    while !state.over {
        if !split_taken && state.day >= SPLIT_DAY {
            lv14 = state.research;
            split_taken = true;
        }
        state.current = state.first_player;
        for _ in 0..N_PLAYERS {
            let p = state.current;
            let temp = (cfg.temperature)(state.day);
            if let Some(out) = agents[p.idx()].play_turn(&state, p, temp, rng) {
                let before = state.research[p.idx()];
                let day = state.day;
                apply_move(&mut state, p, &out.mv);
                state.refill_buildings();
                let after = state.research[p.idx()];
                for s in 0..4 {
                    if after[s] > before[s] && first[p.idx()][s] == NEVER {
                        first[p.idx()][s] = day;
                    }
                }
            }
            state.current = state.current.next(1);
        }

        let claimer = state.resolve_first_player();
        let mut days = 1u8;
        if let Some(p) = claimer {
            if state.may_take_extra_day(p) {
                let (take, _) = agents[p.idx()].extra_day(&state, p, rng);
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
    if !split_taken {
        lv14 = state.research;
    }

    let scores = state.scores();
    let winners = state.winners();
    let share = if winners.is_empty() {
        0.0
    } else {
        1.0 / winners.len() as f32
    };
    Trace {
        scores,
        win_share: std::array::from_fn(|i| {
            if winners.contains(&PlayerId(i as u8)) {
                share
            } else {
                0.0
            }
        }),
        days: state.day,
        aborted,
        lv: state.research,
        lv14,
        first,
        buildings: std::array::from_fn(|i| state.players[i].n_buildings() as u8),
    }
}

// =======================================================================
// One rotation block
// =======================================================================

/// What one game contributes, from the armed seat's point of view. The four
/// score fields are `bin/arena`'s, computed the same way, so the two files'
/// JSONL is directly comparable.
#[derive(Clone, Copy, Default)]
struct Row {
    centred: f64,
    vs_base: f64,
    margin: f64,
    win: f64,
    cand_score: f64,
    base_score: f64,
    days: f64,
    aborted: f64,
    /// Armed seat's final levels, per track.
    c_lv: [f64; 4],
    /// Mean over the three HEAD seats.
    b_lv: [f64; 4],
    c_lv14: [f64; 4],
    b_lv14: [f64; 4],
    /// Fraction of the armed seat's tracks that reached level 3.
    c_max: [f64; 4],
    b_max: [f64; 4],
    /// Armed seat started this track at all (0/1) — the denominator behind
    /// "day of first advance".
    c_any: [f64; 4],
    /// Sum of first-advance days over tracks the armed seat did start; divide
    /// by `c_any` in the analysis, never here (a block with no advance would
    /// otherwise contribute a fabricated day).
    c_firstsum: [f64; 4],
    c_bld: f64,
    b_bld: f64,
    /// Did the armed seat reach level >= 2 on *any* track this game? The
    /// redistribution is fitted to **maxed** tracks and a track reaches level 3
    /// in 1.9% of player-games (`eval.rs`'s own note, F52b), so a null is only
    /// a refutation if the games actually exercised the deep half of the table.
    /// This is the denominator of that argument and the analysis refuses to
    /// call a refutation without it.
    c_reach2: f64,
    c_reach3: f64,
    b_reach2: f64,
}

fn block(
    seed: u64,
    arm: u8,
    agents: &[Box<dyn Agent>; N_PLAYERS],
    cfg: &GameConfig,
) -> (u64, Vec<Row>) {
    let mut rows = Vec::with_capacity(N_PLAYERS);
    for c in 0..N_PLAYERS {
        // The armed seat is `c`; every other seat is arm 0 (HEAD).
        let mut mask = [0u8; N_PLAYERS];
        mask[c] = arm;
        set_arms(mask);

        let refs: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|s| agents[s].as_ref());
        // `bin/arena`'s offset, so a block here and a block there with the same
        // seed are the same four games.
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9) ^ c as u64);
        let t = play_one(seed, &refs, cfg, &mut rng);

        let sc: [f64; N_PLAYERS] = std::array::from_fn(|s| t.scores[s] as f64);
        let table_mean = sc.iter().sum::<f64>() / N_PLAYERS as f64;
        let others: Vec<usize> = (0..N_PLAYERS).filter(|&s| s != c).collect();
        let base_score = others.iter().map(|&s| sc[s]).sum::<f64>() / others.len() as f64;
        let best_base = others.iter().map(|&s| sc[s]).fold(f64::NEG_INFINITY, f64::max);
        let mean_b = |f: &dyn Fn(usize) -> f64| -> f64 {
            others.iter().map(|&s| f(s)).sum::<f64>() / others.len() as f64
        };

        rows.push(Row {
            centred: sc[c] - table_mean,
            vs_base: sc[c] - base_score,
            margin: sc[c] - best_base,
            win: t.win_share[c] as f64,
            cand_score: sc[c],
            base_score,
            days: t.days as f64,
            aborted: t.aborted as u8 as f64,
            c_lv: std::array::from_fn(|k| t.lv[c][k] as f64),
            b_lv: std::array::from_fn(|k| mean_b(&|s| t.lv[s][k] as f64)),
            c_lv14: std::array::from_fn(|k| t.lv14[c][k] as f64),
            b_lv14: std::array::from_fn(|k| mean_b(&|s| t.lv14[s][k] as f64)),
            c_max: std::array::from_fn(|k| (t.lv[c][k] >= 3) as u8 as f64),
            b_max: std::array::from_fn(|k| mean_b(&|s| (t.lv[s][k] >= 3) as u8 as f64)),
            c_any: std::array::from_fn(|k| (t.first[c][k] != NEVER) as u8 as f64),
            c_firstsum: std::array::from_fn(|k| {
                if t.first[c][k] == NEVER {
                    0.0
                } else {
                    t.first[c][k] as f64
                }
            }),
            c_bld: t.buildings[c] as f64,
            b_bld: mean_b(&|s| t.buildings[s] as f64),
            c_reach2: t.lv[c].iter().any(|&l| l >= 2) as u8 as f64,
            c_reach3: t.lv[c].iter().any(|&l| l >= 3) as u8 as f64,
            b_reach2: mean_b(&|s| t.lv[s].iter().any(|&l| l >= 2) as u8 as f64),
        });
    }
    // Leave the thread as we found it.
    set_arms([0; N_PLAYERS]);
    (seed, rows)
}

fn mean_of(rows: &[Row], f: impl Fn(&Row) -> f64) -> f64 {
    rows.iter().map(&f).sum::<f64>() / rows.len() as f64
}

fn arr(rows: &[Row], f: impl Fn(&Row) -> [f64; 4]) -> String {
    let v: Vec<String> = (0..4)
        .map(|k| format!("{:.4}", rows.iter().map(|r| f(r)[k]).sum::<f64>() / rows.len() as f64))
        .collect();
    format!("[{}]", v.join(","))
}

fn block_json(seed: u64, arm: u8, agent: &str, rows: &[Row]) -> String {
    let m = |f: fn(&Row) -> f64| mean_of(rows, f);
    format!(
        concat!(
            r#"{{"seed":{},"arm":"{}","agent":"{}","games":{},"#,
            r#""centred":{:.4},"vs_base":{:.4},"margin":{:.4},"win":{:.4},"#,
            r#""cand":{:.3},"base":{:.3},"days":{:.2},"aborted":{},"#,
            r#""c_lv":{},"b_lv":{},"c_lv14":{},"b_lv14":{},"#,
            r#""c_max":{},"b_max":{},"c_any":{},"c_firstsum":{},"#,
            r#""c_bld":{:.3},"b_bld":{:.3},"c_reach2":{:.4},"c_reach3":{:.4},"b_reach2":{:.4}}}"#
        ),
        seed,
        ARM_NAMES[arm as usize],
        agent,
        rows.len(),
        m(|r| r.centred),
        m(|r| r.vs_base),
        m(|r| r.margin),
        m(|r| r.win),
        m(|r| r.cand_score),
        m(|r| r.base_score),
        m(|r| r.days),
        rows.iter().filter(|r| r.aborted > 0.0).count(),
        arr(rows, |r| r.c_lv),
        arr(rows, |r| r.b_lv),
        arr(rows, |r| r.c_lv14),
        arr(rows, |r| r.b_lv14),
        arr(rows, |r| r.c_max),
        arr(rows, |r| r.b_max),
        arr(rows, |r| r.c_any),
        arr(rows, |r| r.c_firstsum),
        m(|r| r.c_bld),
        m(|r| r.b_bld),
        m(|r| r.c_reach2),
        m(|r| r.c_reach3),
        m(|r| r.b_reach2),
    )
}

// =======================================================================
// One writer per --out file
// =======================================================================

/// A pid lock beside the output. `FINDINGS-eval.md` F52a: a `pkill` on a driver
/// left its `xargs` alive and two processes appended to one file, which is a
/// silent doubling of half the blocks.
struct Lock(PathBuf);

impl Lock {
    fn take(out: &Path) -> Result<Lock, String> {
        let p = out.with_extension("lock");
        if let Ok(s) = std::fs::read_to_string(&p) {
            let pid: i32 = s.trim().parse().unwrap_or(0);
            let alive = pid > 0 && unsafe { libc_kill(pid) };
            if alive {
                return Err(format!(
                    "{} is held by pid {pid}, which is alive. One writer per --out file.",
                    p.display()
                ));
            }
            eprintln!("trackrace: {} held by dead pid {pid}; taking it over", p.display());
        }
        std::fs::write(&p, format!("{}\n", std::process::id()))
            .map_err(|e| format!("cannot write {}: {e}", p.display()))?;
        Ok(Lock(p))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `kill(pid, 0)` — true if the process exists.
unsafe fn libc_kill(pid: i32) -> bool {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    kill(pid, 0) == 0
}

fn seeds_on_file(out: &Path) -> Vec<u64> {
    let Ok(s) = std::fs::read_to_string(out) else {
        return Vec::new();
    };
    s.lines()
        .filter_map(|l| {
            let at = l.find("\"seed\":")? + 7;
            let rest = &l[at..];
            let end = rest.find(|c: char| !c.is_ascii_digit())?;
            rest[..end].parse().ok()
        })
        .collect()
}

// =======================================================================
// main
// =======================================================================

fn help() {
    println!(
        "trackrace — the per-track research-shape race. docs/FINDINGS-track-shape.md

  --arm NAME        one of {ARM_NAMES:?}  (default head)
  --agent SPEC      the agent in every seat (default mcts:2048:heuristic:cp=0.06)
  --blocks N        rotation blocks; 4 games each (default 400)
  --seed S          first block seed (default 4000000)
  --concurrency K   blocks in flight (default 4 — this machine is shared)
  --out FILE        JSONL, one line per block, appended as blocks finish
  --resume          skip seeds already on file
  --quiet           no heartbeat

  --table           print every arm's twelve constants and exit
  --probe           eval::heuristic on one position under every arm (no games)
  --verify N        play N seeds through this driver and record::play_game
  --help

Built {} the harness (--cfg trackarm): {}",
        if HARNESS { "WITH" } else { "WITHOUT" },
        if HARNESS {
            "every arm available"
        } else {
            "only --arm head; any other arm is refused"
        }
    );
}

fn print_table() {
    println!("# arm ARMNAME TRACK L1 L2 L3     (Science::ALL order)");
    println!("# arm ARMNAME bonus B           (the lvl==2 && horizon>0.15 step)");
    let tracks = ["Agriculture", "Extraction", "Architecture", "Theology"];
    for (i, name) in ARM_NAMES.iter().enumerate() {
        for (k, t) in tracks.iter().enumerate() {
            let s = ARM_STEPS[i][k];
            println!("arm {name} {t} {:.2} {:.2} {:.2}", s[0], s[1], s[2]);
        }
        let sum: f32 = ARM_STEPS[i].iter().flatten().sum();
        println!("arm {name} bonus {:.2}", ARM_L2_BONUS[i]);
        println!("arm {name} total {sum:.2}");
    }
    match check_tables() {
        Ok(()) => println!(
            "# table check: {}",
            if HARNESS {
                "OK — identical to the table compiled into eval.rs"
            } else {
                "skipped — built without --cfg trackarm, there is no eval.rs table to check"
            }
        ),
        Err(e) => {
            eprintln!("# table check FAILED: {e}");
            std::process::exit(2);
        }
    }
}

/// Game-free proof that the arms are not no-ops, and that arm 0 is HEAD.
///
/// Builds one fresh position, writes a research row into seat 0, and prints
/// `eval::heuristic` for that seat under every arm. No games, no agents, no
/// wall-clock deadline — so it reproduces exactly on a loaded machine, which
/// `--verify` on `heuristic:full` does not.
fn probe(seed: u64) {
    let base = tzolkin::Game::new(seed).state;
    let rows: [([u8; 4], &str); 7] = [
        ([0, 0, 0, 0], "empty"),
        ([1, 0, 0, 0], "A1"),
        ([0, 1, 0, 0], "R1  Extraction 1"),
        ([0, 0, 1, 0], "C1  Architecture 1"),
        ([0, 0, 2, 0], "C2  the lvl==2 step is live here"),
        ([0, 0, 3, 0], "C3  Architecture maxed (+23.52 causal)"),
        ([0, 3, 0, 0], "R3  Extraction maxed (+4.51 causal)"),
    ];
    print!("  {:<40}", "research row (seat 0)");
    for a in ARM_NAMES {
        print!(" {a:>9}");
    }
    println!();
    for (row, what) in rows {
        let mut g = base;
        g.research[0] = row;
        print!("  {:?} {:<33}", row, what);
        let mut head = 0.0f32;
        for (i, _) in ARM_NAMES.iter().enumerate() {
            let mut mask = [0u8; N_PLAYERS];
            mask[0] = i as u8;
            set_arms(mask);
            let v = tzolkin::eval::heuristic(&g, PlayerId(0));
            if i == 0 {
                head = v;
                print!(" {v:9.4}");
            } else {
                print!(" {:+9.4}", v - head);
            }
        }
        println!();
    }
    set_arms([0; N_PLAYERS]);
    println!("  (arm 0 is the absolute value; the rest are differences from it)");
    if !HARNESS {
        println!("  built without --cfg trackarm: every column above is arm 0, so the \
                  differences are all zero by construction.");
    }
}

fn verify(spec: &AgentSpec, seed0: u64, n: u64) {
    let cfg = GameConfig::evaluation();
    let mut bad = 0;
    for i in 0..n {
        let seed = seed0 + i;
        let agents: [Box<dyn Agent>; N_PLAYERS] = std::array::from_fn(|_| spec.instance());
        let refs: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|s| agents[s].as_ref());
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
        let mine = play_one(seed, &refs, &cfg, &mut rng);

        let agents2: [Box<dyn Agent>; N_PLAYERS] = std::array::from_fn(|_| spec.instance());
        let refs2: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|s| agents2[s].as_ref());
        let mut rng2 = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
        let theirs = record::play_game(seed, &refs2, &cfg, &mut rng2);

        // Printed for every seed, agreeing or not: two binaries' `--verify`
        // output diffed against each other is the check that arm 0 is
        // bit-identical to HEAD.
        println!("seed {seed}: {:?}", mine.scores);
        if mine.scores != theirs.scores {
            bad += 1;
            println!("seed {seed}: DISAGREE mine {:?} theirs {:?}", mine.scores, theirs.scores);
        }
    }
    println!("--verify: {n} seeds, {bad} disagreements");
    if bad > 0 {
        eprintln!(
            "note: `heuristic:full` is NOT reproducible under load -- \
             eval::rank_all_within carries a 2,500 ms wall-clock budget \
             (eval::FULL_BUDGET) and falls back to sampled top-ups when it \
             trips. FINDINGS-research-value.md R17.5 records the same thing. \
             Verify with a sampled agent (heuristic:8) instead."
        );
    }
    if bad > 0 {
        std::process::exit(1);
    }
}

fn main() {
    if let Err(e) = record::reject_unknown_flags(&[
        "--agent", "--arm", "--blocks", "--concurrency", "--help", "--out", "--quiet", "--resume",
        "--probe", "--seed", "--table", "--verify",
    ]) {
        eprintln!("trackrace: {e}");
        std::process::exit(2);
    }
    let argv: Vec<String> = std::env::args().collect();
    let get = |n: &str| -> Option<String> {
        argv.iter().position(|a| a == n).and_then(|i| argv.get(i + 1)).cloned()
    };
    let has = |n: &str| argv.iter().any(|a| a == n);

    if has("--help") || has("-h") {
        help();
        return;
    }
    if has("--table") {
        print_table();
        return;
    }
    if has("--probe") {
        if let Err(e) = check_tables() {
            eprintln!("trackrace: {e}");
            std::process::exit(2);
        }
        probe(get("--seed").and_then(|v| v.parse().ok()).unwrap_or(4_000_000));
        return;
    }
    if let Err(e) = check_tables() {
        eprintln!("trackrace: {e}");
        std::process::exit(2);
    }

    let agent = get("--agent").unwrap_or_else(|| "mcts:2048:heuristic:cp=0.06".into());
    let spec = match AgentSpec::parse(&agent, false) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("trackrace: --agent {agent:?}: {e}");
            std::process::exit(2);
        }
    };

    if let Some(n) = get("--verify").and_then(|v| v.parse::<u64>().ok()) {
        verify(&spec, get("--seed").and_then(|v| v.parse().ok()).unwrap_or(4_000_000), n);
        return;
    }

    let arm = match arm_id(&get("--arm").unwrap_or_else(|| "head".into())) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("trackrace: {e}");
            std::process::exit(2);
        }
    };
    if arm != 0 && !HARNESS {
        eprintln!(
            "trackrace: --arm {} needs a binary built with RUSTFLAGS=\"--cfg trackarm\" \
             from a tree with harness.patch applied. Refusing to measure HEAD and \
             label it as the arm.",
            ARM_NAMES[arm as usize]
        );
        std::process::exit(2);
    }
    // A net leaf blends `eval::heuristic` back in on a *batcher* thread
    // (`record::NetBatch::mix`), which has no arm mask, so the blend would be
    // silently computed at HEAD. See `docs/FINDINGS-track-shape.md` §5.
    let net_backend =
        agent.contains(".safetensors") || agent.contains(".tzw") || agent.contains("net-random");
    if net_backend && arm != 0 {
        eprintln!(
            "trackrace: --agent {agent:?} is not the heuristic backend. A net leaf \
             mixes eval::heuristic in on a batcher thread with no arm mask, so the \
             arm would not reach it. Refused."
        );
        std::process::exit(2);
    }

    let blocks: u64 = get("--blocks").and_then(|v| v.parse().ok()).unwrap_or(400);
    let seed0: u64 = get("--seed").and_then(|v| v.parse().ok()).unwrap_or(4_000_000);
    let conc: usize = get("--concurrency").and_then(|v| v.parse().ok()).unwrap_or(4);
    let quiet = has("--quiet");
    let out = PathBuf::from(
        get("--out").unwrap_or_else(|| format!("track-{}.jsonl", ARM_NAMES[arm as usize])),
    );

    let _lock = match Lock::take(&out) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("trackrace: {e}");
            std::process::exit(2);
        }
    };

    let done: Vec<u64> = if has("--resume") { seeds_on_file(&out) } else { Vec::new() };
    let todo: Vec<u64> = (0..blocks)
        .map(|i| seed0 + i)
        .filter(|s| !done.contains(s))
        .collect();

    eprintln!(
        "trackrace: arm {} ({}), agent {}, {} blocks ({} on file), concurrency {}, -> {}",
        ARM_NAMES[arm as usize],
        if HARNESS { "harness" } else { "no harness" },
        spec.name(),
        todo.len(),
        done.len(),
        conc,
        out.display()
    );

    interrupt::install();
    let file = Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", out.display())),
    );
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(conc)
        .build()
        .expect("thread pool");
    let cfg = GameConfig::evaluation();
    let n_done = std::sync::atomic::AtomicU64::new(0);

    pool.install(|| {
        todo.par_iter().for_each(|&seed| {
            if interrupt::stopping() {
                return;
            }
            let agents: [Box<dyn Agent>; N_PLAYERS] = std::array::from_fn(|_| spec.instance());
            let (seed, rows) = block(seed, arm, &agents, &cfg);
            let line = block_json(seed, arm, &spec.name(), &rows);
            let mut f = file.lock().expect("progress file");
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
            let k = n_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if !quiet && k % 10 == 0 {
                eprintln!(
                    "  {k}/{} blocks, {:.1} s/block wall",
                    todo.len(),
                    started.elapsed().as_secs_f64() / k as f64
                );
            }
        })
    });

    let all: Vec<String> = std::fs::read_to_string(&out)
        .unwrap_or_default()
        .lines()
        .map(|s| s.to_string())
        .collect();
    let centred: Vec<f64> = all
        .iter()
        .filter_map(|l| {
            let at = l.find("\"centred\":")? + 10;
            let rest = &l[at..];
            let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))?;
            rest[..end].parse().ok()
        })
        .collect();
    let s = Summary::of(&centred);
    eprintln!(
        "trackrace: {} blocks on file, centred {:+.4} +- {:.4}, {:.0} s wall. \
         Read the interval with <scratch>/ts/an.py, not by eye.",
        s.n,
        s.mean,
        s.ci,
        started.elapsed().as_secs_f64()
    );
}
