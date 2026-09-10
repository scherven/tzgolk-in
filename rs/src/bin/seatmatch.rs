//! Play the same four-agent lineup many times and record every seat's score.
//!
//! `arena` deliberately rotates its candidate through every seating so that
//! seat advantage cancels inside a rotation block. That is the right thing for
//! "is this agent stronger", and the wrong thing for "what does each seat
//! score", which is the question this binary answers: the lineup is **fixed**,
//! one game per seed, and the four raw scores go to disk.
//!
//! Seat advantage is therefore *not* controlled here. Seat 0 moves first and
//! the corn surcharge on higher spaces means seat 0 is not seat 3, so two
//! copies of one agent in seats 0 and 1 will not score the same as two copies
//! in seats 2 and 3. Run the mirrored lineup and compare if you want the agent
//! effect on its own; `tools/seatmatch.py --mirror` does exactly that.
//!
//!     seatmatch --seat 0 A --seat 1 A --seat 2 B --seat 3 B \
//!               --games 10000 --out games.jsonl --resume
//!
//! One line per completed game, flushed as it lands, so the file is the source
//! of truth for both the summary and the resume.

use rand::SeedableRng;
use std::io::Write;
use std::time::Duration;
use tzolkin::record::{interrupt, play_game, Agent, AgentSpec, GameConfig};
use tzolkin::ids::N_PLAYERS;

const SEAT: [&str; N_PLAYERS] = ["R", "G", "B", "Y"];

const USAGE: &str = "\
seatmatch -- one fixed lineup, every seat's score

USAGE
  seatmatch --seat 0 SPEC --seat 1 SPEC --seat 2 SPEC --seat 3 SPEC [options]

OPTIONS
  --seat N SPEC    put SPEC in seat N (0=R 1=G 2=B 3=Y). All four are
                   required. Same spec language as arena, selfplay and tui --
                   and specs contain commas (`,tilt=agri`), which is why this
                   is repeated rather than one comma-separated list.
  --games N        games to play                          [400]
  --seed N         first game seed                        [3000000]
  --concurrency N  games in flight; with a network agent this is also the
                   reachable batch                        [128 searching]
  --batch N        max evaluations per forward pass        [min(256, conc)]
  --batchers N     inference threads                       [cores/5, <= 4]
  --linger-us N    how long a short batch waits for more            [200]
  --no-batch       one evaluation per forward pass
  --out PATH       progress JSONL                     [seatmatch.jsonl]
  --resume         skip seeds already in --out
  --quiet          no per-game progress line
";

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        print!("{USAGE}");
        return;
    }
    let flag = |name: &str| -> Option<String> {
        argv.iter().position(|a| a == name).and_then(|i| argv.get(i + 1).cloned())
    };
    let num = |name: &str, d: usize| -> usize {
        flag(name).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let present = |name: &str| argv.iter().any(|a| a == name);

    let mut parts: [Option<String>; N_PLAYERS] = Default::default();
    let mut i = 1;
    while i < argv.len() {
        if argv[i] == "--seat" {
            let Some(n) = argv.get(i + 1).and_then(|v| v.parse::<usize>().ok()) else {
                eprintln!("seatmatch: --seat wants a seat 0..3, got {:?}", argv.get(i + 1));
                std::process::exit(2);
            };
            let Some(spec) = argv.get(i + 2) else {
                eprintln!("seatmatch: --seat {n} wants a spec after it");
                std::process::exit(2);
            };
            if n >= N_PLAYERS {
                eprintln!("seatmatch: --seat wants a seat 0..3, got {n}");
                std::process::exit(2);
            }
            parts[n] = Some(spec.clone());
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut specs: Vec<AgentSpec> = Vec::new();
    for (n, part) in parts.iter().enumerate() {
        let Some(s) = part else {
            eprintln!("seatmatch: no agent for seat {n} ({})\n\n{USAGE}", SEAT[n]);
            std::process::exit(2);
        };
        match AgentSpec::parse(s, false) {
            Ok(a) => specs.push(a),
            Err(e) => {
                eprintln!("seatmatch: seat {n} ({s}): {e}");
                std::process::exit(2);
            }
        }
    }

    let games = num("--games", 400);
    let seed0 = flag("--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(3_000_000);
    let out = flag("--out").unwrap_or_else(|| "seatmatch.jsonl".into());
    let quiet = present("--quiet");

    let searching = specs.iter().any(|s| s.searches());
    let concurrency = match num("--concurrency", 0) {
        0 if searching => 128,
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        n => n,
    };
    let batch = match num("--batch", 0) {
        0 => concurrency.min(256),
        n => n,
    };
    let batchers = match num("--batchers", 0) {
        0 => (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
            .div_ceil(5)
            .clamp(1, 4),
        n => n,
    };
    let linger = num("--linger-us", 200);

    // Every distinct spec gets its own queue; two seats holding the same spec
    // still hold two instances, because a search agent owns a mutable tree.
    let batching = if present("--no-batch") {
        false
    } else {
        specs
            .iter_mut()
            .filter(|s| s.batchable())
            .fold(false, |acc, s| s.enable_batching(batch, Duration::from_micros(linger as u64), batchers) || acc)
    };
    let _ = rayon::ThreadPoolBuilder::new().num_threads(concurrency).build_global();

    interrupt::install();

    let already: std::collections::HashSet<u64> = if present("--resume") {
        read_seeds(&out)
    } else {
        Default::default()
    };
    let todo: Vec<u64> = (0..games as u64)
        .map(|i| seed0 + i)
        .filter(|s| !already.contains(s))
        .collect();

    println!("seatmatch");
    for (i, s) in specs.iter().enumerate() {
        println!("  seat {i} ({}) : {}", SEAT[i], s.name());
    }
    println!(
        "  compute   : {concurrency} games in flight, {}",
        if batching {
            format!("batch <= {batch} over {batchers} inference threads")
        } else {
            "unbatched evaluator".into()
        }
    );
    println!(
        "  plan      : {} games ({} already on file)",
        todo.len(),
        already.len()
    );
    println!("  progress  : {out}");

    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .unwrap_or_else(|e| {
            eprintln!("seatmatch: cannot open {out}: {e}");
            std::process::exit(2);
        });
    let progress = std::sync::Mutex::new(f);
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let total = todo.len();
    let started = std::time::Instant::now();
    let cfg = GameConfig::evaluation();

    use rayon::prelude::*;
    todo.par_iter().for_each(|&seed| {
        if interrupt::stopping() {
            return;
        }
        // One instance per game per seat: a search agent owns a mutable tree
        // arena, and sharing it would serialise the run.
        let held: Vec<Box<dyn Agent>> = specs.iter().map(|s| s.instance()).collect();
        let agents: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|i| held[i].as_ref());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
        let r = play_game(seed, &agents, &cfg, &mut rng);

        // Append and flush before anything else: everything after this point
        // can be lost without losing the game.
        {
            let line = format!(
                "{{\"seed\":{seed},\"scores\":[{},{},{},{}],\"win\":[{:.3},{:.3},{:.3},{:.3}],\
                 \"days\":{},\"aborted\":{}}}",
                r.scores[0], r.scores[1], r.scores[2], r.scores[3],
                r.win_share[0], r.win_share[1], r.win_share[2], r.win_share[3],
                r.days, r.aborted
            );
            let mut f = progress.lock().unwrap();
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }

        let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if !quiet && (n % 8 == 0 || n == total) {
            let per = started.elapsed().as_secs_f64() / n as f64;
            let eta = per * (total - n) as f64;
            eprint!(
                "\r  {n}/{total} games  {:.2} games/s  eta {}      ",
                1.0 / per.max(1e-9),
                hms(eta)
            );
            let _ = std::io::stderr().flush();
        }
    });
    if !quiet {
        eprintln!();
    }
    if interrupt::stopping() {
        println!("stopped early; {} games are on file", read_seeds(&out).len());
    }
}

fn hms(s: f64) -> String {
    let s = s.max(0.0) as u64;
    format!("{}h {:02}m {:02}s", s / 3600, (s % 3600) / 60, s % 60)
}

fn read_seeds(path: &str) -> std::collections::HashSet<u64> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Default::default();
    };
    text.lines()
        .filter_map(|l| l.split_once("\"seed\":"))
        .filter_map(|(_, rest)| {
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            rest[..end].parse().ok()
        })
        .collect()
}
