//! Invariant fuzz: play seeded games to completion with random legal moves and
//! assert the state stays coherent after every turn.
//!
//! Progress is checkpointed after every game and failures are appended and
//! flushed as they happen, so interrupting the run never loses what it found.
//! Re-run with `--start <n>` (or `--resume`) to continue.
//!
//!     cargo run --release --bin fuzz -- --games 10000

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::legal_moves;

const CHECKPOINT: &str = "fuzz-progress.txt";
const FAILURES: &str = "fuzz-failures.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, default: u64| -> u64 {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };

    let games = flag("--games", 1000);
    let mut start = flag("--start", 0);
    if args.iter().any(|a| a == "--resume") {
        if let Ok(s) = std::fs::read_to_string(CHECKPOINT) {
            if let Some(n) = s.trim().split_whitespace().last().and_then(|v| v.parse().ok()) {
                start = n;
                println!("resuming from seed {start}");
            }
        }
    }

    let mut failures = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(FAILURES)
            .expect("open failures file"),
    );

    let t0 = Instant::now();
    let mut n_fail = 0usize;
    let mut total_days = 0u64;
    let mut total_moves = 0u64;
    let mut max_moves = 0usize;
    let mut peak = 0usize;
    let mut peak_seed = 0u64;
    let mut score_lo = i16::MAX;
    let mut score_hi = i16::MIN;
    let mut completed = 0u64;

    for seed in start..start + games {
        let mut g = Game::new(seed);
        g.trace = true;

        // Sample the branching factor on the opening turn of each game.
        let opening = legal_moves(&g.state, PlayerId(0)).len();
        max_moves = max_moves.max(opening);
        total_moves += opening as u64;

        match g.run_checked() {
            Ok(()) => {
                completed += 1;
                if g.max_branching > peak {
                    peak = g.max_branching;
                    peak_seed = seed;
                }
                total_days += g.state.day as u64;
                for s in g.scores() {
                    score_lo = score_lo.min(s);
                    score_hi = score_hi.max(s);
                }
            }
            Err(e) => {
                n_fail += 1;
                writeln!(failures, "=== seed {seed}: {e}").unwrap();
                for line in g.log.iter().rev().take(25).rev() {
                    writeln!(failures, "    {line}").unwrap();
                }
                failures.flush().unwrap();
                eprintln!("seed {seed}: {e}");
                if n_fail >= 20 {
                    eprintln!("stopping after 20 failures");
                    checkpoint(seed + 1);
                    break;
                }
            }
        }

        checkpoint(seed + 1);
    }

    let secs = t0.elapsed().as_secs_f64();
    println!("\n{completed} games completed, {n_fail} failed, in {secs:.1}s");
    if completed > 0 {
        println!(
            "  {:.0} games/s, mean length {:.1} days, scores {score_lo}..{score_hi}",
            completed as f64 / secs,
            total_days as f64 / completed as f64,
        );
    }
    println!(
        "  opening branching: mean {:.0}, max {max_moves}",
        total_moves as f64 / games as f64
    );
    println!("  peak branching at any turn: {peak} (seed {peak_seed})");
    if n_fail > 0 {
        println!("  failures written to {FAILURES}");
        std::process::exit(1);
    }
}

fn checkpoint(next_seed: u64) {
    if let Ok(f) = File::create(Path::new(CHECKPOINT)) {
        let mut w = BufWriter::new(f);
        let _ = writeln!(w, "next_seed {next_seed}");
        let _ = w.flush();
    }
}
