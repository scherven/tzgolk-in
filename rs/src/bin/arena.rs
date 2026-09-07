//! The benchmark. Run this every few hours to see whether training is working.
//!
//!     cargo run --release --bin arena -- --candidate heuristic:64 --baseline heuristic:8 --games 400
//!
//! It plays a candidate against a baseline over N four-player games and reports
//! mean centred score, points against a baseline seat, margin against the best
//! rival, and win rate — each with a confidence interval. **An evaluation with no error bar cannot decide whether generation
//! 40 beats generation 39**, so every number here comes with one and the
//! summary says outright whether the difference is distinguishable from zero.
//!
//! # Seating, and what the win rate therefore means
//!
//! This is not a two-player game, and the two things that make four-player
//! evaluation hard are both handled by the seating scheme rather than by the
//! statistics.
//!
//! **Turn order is worth real points.** The first-player marker and the corn
//! surcharge on higher spaces mean seat 0 is not seat 3. Measuring an agent in
//! one seat measures the seat. So the unit of work is a **rotation block**: the
//! same game seed played once with the candidate in each of the four seats,
//! baselines in the rest. Every block contains exactly one candidate-game per
//! seat, so seat advantage cancels *within* the unit that the statistics then
//! treat as independent.
//!
//! **Games are not independent of each other.** The four games in a block share
//! a seed, which means the same shuffled decks, the same monument row and the
//! same dealt starting tiles. That is deliberate — it is a matched design and
//! it removes a large chunk of shuffle variance — but it also means those four
//! games are correlated, and pretending otherwise would shrink the interval by
//! up to a factor of two. So **the block, not the game, is the independent
//! unit**: n in every interval below is the number of completed blocks.
//!
//! Two modes, both of which `LEARNING.md` §6.8 asks for:
//!
//! * `--mode solo` (default) — one candidate against three baselines, rotated
//!   through all four seats. 4 games per block. **A candidate exactly as strong
//!   as the baseline wins 25% of games, not 50%.** This is the mode that
//!   exposes an agent which has only learned to play against copies of itself.
//! * `--mode pairs` — two candidates against two baselines, over all six
//!   distinct seatings of `{C,C,B,B}`. 6 games per block, null win rate 50%,
//!   and the candidate win rate is the share of games *some* candidate won.
//!
//! # Which number to look at
//!
//! **Mean centred score**, `score[candidate] - mean(all four scores)`. Null is
//! 0 in both modes. With four players and a few hundred games, win counts are
//! far noisier than margins (`LEARNING.md` §3.2 and §6.8 make the same argument
//! for the training target and for the measurement). Win rate is reported
//! because it is the number a human wants, but the centred score is the number
//! that will move first.
//!
//! # Interruption
//!
//! Blocks are appended to a JSONL progress file and flushed as they complete,
//! and `--resume` skips blocks already on file. Ctrl-C finishes the blocks in
//! flight, prints the full summary over everything on disk, and exits; a second
//! Ctrl-C aborts. Nothing is ever lost and a half-finished run still answers
//! the question, just with a wider interval.

#[allow(dead_code)]
use tzolkin::record;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rayon::prelude::*;

use record::{interrupt, play_game, Agent, AgentSpec, GameConfig, Summary};
use tzolkin::ids::*;
use tzolkin::RULES_VERSION;

// =======================================================================
// One game's worth of measurement
// =======================================================================

/// What one game contributes, from the candidate side's point of view.
#[derive(Clone, Copy, Debug)]
struct Outcome {
    /// `score[candidate] - mean(table)`. The headline metric.
    centred: f64,
    /// `score[candidate] - mean(score[baseline seats])`. The interpretable
    /// margin, in points, and its null is 0 for equal agents.
    vs_base: f64,
    /// `score[candidate] - max(score[baseline seats])`.
    ///
    /// Descriptive only: **its null is not zero**. One draw against the maximum
    /// of three is negative even for identical agents (about -11 points in this
    /// game), so read it across runs, never against zero.
    margin: f64,
    /// `1/|winners|` summed over candidate seats. Ties split.
    win: f64,
    cand_score: f64,
    base_score: f64,
    days: u8,
    aborted: bool,
}

/// One rotation block: the same seed played once per seating.
#[derive(Clone, Debug)]
struct Block {
    seed: u64,
    games: Vec<Outcome>,
}

impl Block {
    fn mean(&self, f: impl Fn(&Outcome) -> f64) -> f64 {
        self.games.iter().map(&f).sum::<f64>() / self.games.len() as f64
    }
    fn to_json(&self) -> String {
        let g = |f: fn(&Outcome) -> f64| self.mean(f);
        format!(
            r#"{{"seed":{},"games":{},"centred":{:.4},"vs_base":{:.4},"margin":{:.4},"win":{:.4},"cand":{:.3},"base":{:.3},"days":{:.2},"aborted":{}}}"#,
            self.seed,
            self.games.len(),
            g(|o| o.centred),
            g(|o| o.vs_base),
            g(|o| o.margin),
            g(|o| o.win),
            g(|o| o.cand_score),
            g(|o| o.base_score),
            self.mean(|o| o.days as f64),
            self.games.iter().filter(|o| o.aborted).count(),
        )
    }
}

/// Parse back a progress line. Only the fields the summary needs; the rest of
/// the line is there for a human and for a spreadsheet.
fn block_from_json(line: &str) -> Option<Block> {
    let num = |key: &str| -> Option<f64> {
        let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
        let rest = &line[at..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    let seed = num("seed")? as u64;
    let n = num("games")? as usize;
    // A replayed block is stored as its mean; reconstruct it as `n` identical
    // games so the block mean is preserved exactly and n_blocks is right.
    let o = Outcome {
        centred: num("centred")?,
        vs_base: num("vs_base").unwrap_or(f64::NAN),
        margin: num("margin")?,
        win: num("win")?,
        cand_score: num("cand")?,
        base_score: num("base")?,
        // Rounds and aborts survive a resume too, or the "mean game length"
        // line — the cheapest rules-bug detector in the report — would read as
        // zero for every block that came off disk.
        days: num("days").unwrap_or(0.0).round() as u8,
        aborted: num("aborted").unwrap_or(0.0) > 0.0,
    };
    Some(Block {
        seed,
        games: vec![o; n.max(1)],
    })
}

// =======================================================================
// Seatings
// =======================================================================

/// `true` at index i means seat i holds the candidate.
fn seatings(mode: Mode) -> Vec<[bool; N_PLAYERS]> {
    match mode {
        // One candidate, rotated through every seat.
        Mode::Solo => (0..N_PLAYERS)
            .map(|c| std::array::from_fn(|i| i == c))
            .collect(),
        // Every distinct way to seat two candidates among four seats: C(4,2)=6.
        // All six, in equal numbers, or the match measures seats.
        Mode::Pairs => {
            let mut v = Vec::new();
            for a in 0..N_PLAYERS {
                for b in a + 1..N_PLAYERS {
                    v.push(std::array::from_fn(|i| i == a || i == b));
                }
            }
            v
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Solo,
    Pairs,
}

impl Mode {
    fn null_win_rate(self) -> f64 {
        match self {
            // Four players, one of them the candidate: an equal agent takes a
            // quarter of the wins.
            Mode::Solo => 0.25,
            // Two of four seats are the candidate's.
            Mode::Pairs => 0.50,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Mode::Solo => "solo (1 candidate vs 3 baselines, all 4 seats)",
            Mode::Pairs => "pairs (2 candidates vs 2 baselines, all 6 seatings)",
        }
    }
}

/// Play one rotation block.
///
/// `cand` and `base` are this worker's own agent instances. A search agent owns
/// a tree arena and a slot in the evaluation batch, neither of which can be
/// shared, so the caller builds them per task rather than once per run
/// (`AgentSpec::instance`).
fn play_block(
    seed: u64,
    mode: Mode,
    cand: &dyn Agent,
    base: &dyn Agent,
    cfg: &GameConfig,
) -> Block {
    let mut games = Vec::new();
    for (i, seats) in seatings(mode).into_iter().enumerate() {
        let agents: [&dyn Agent; N_PLAYERS] =
            std::array::from_fn(|s| if seats[s] { cand } else { base });
        // The seed fixes the setup; the play RNG is offset per seating so two
        // seatings of the same deal are not the same game.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9) ^ i as u64);
        let r = play_game(seed, &agents, cfg, &mut rng);

        let scores: [f64; N_PLAYERS] = std::array::from_fn(|s| r.scores[s] as f64);
        let table_mean = scores.iter().sum::<f64>() / N_PLAYERS as f64;
        let n_cand = seats.iter().filter(|&&b| b).count() as f64;

        let cand_score = (0..N_PLAYERS)
            .filter(|&s| seats[s])
            .map(|s| scores[s])
            .sum::<f64>()
            / n_cand;
        let base_score = (0..N_PLAYERS)
            .filter(|&s| !seats[s])
            .map(|s| scores[s])
            .sum::<f64>()
            / (N_PLAYERS as f64 - n_cand);
        let best_base = (0..N_PLAYERS)
            .filter(|&s| !seats[s])
            .map(|s| scores[s])
            .fold(f64::NEG_INFINITY, f64::max);
        let win = (0..N_PLAYERS)
            .filter(|&s| seats[s])
            .map(|s| r.win_share[s] as f64)
            .sum::<f64>();

        games.push(Outcome {
            centred: cand_score - table_mean,
            vs_base: cand_score - base_score,
            margin: cand_score - best_base,
            win,
            cand_score,
            base_score,
            days: r.days,
            aborted: r.aborted,
        });
    }
    Block { seed, games }
}

// =======================================================================
// main
// =======================================================================

struct Args {
    candidate: String,
    baseline: String,
    games: usize,
    mode: Mode,
    /// Blocks in flight. With a network agent this is also the reachable
    /// evaluation batch, so it should exceed the core count (`COMPUTE.md` §2.1).
    concurrency: usize,
    batch: usize,
    batchers: usize,
    linger: Duration,
    no_batch: bool,
    out: PathBuf,
    resume: bool,
    seed0: u64,
    quiet: bool,
    /// Seconds between heartbeat lines.
    heartbeat: u64,
}

fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn parse_args() -> Result<Args, String> {
    // A mistyped flag must not be silently ignored: the tool would run its
    // defaults and report a confident answer to a question nobody asked.
    record::reject_unknown_flags(&["--heartbeat", "--baseline", "--batch", "--batchers", "--candidate", "--concurrency", "--games", "--help", "--linger-us", "--mode", "--no-batch", "--out", "--quiet", "--resume", "--seed", "--threads"])?;

    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        std::process::exit(0);
    }
    let get = |name: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let num = |name: &str, d: usize| -> usize {
        get(name).and_then(|v| v.parse().ok()).unwrap_or(d)
    };

    let mode = match get("--mode").as_deref() {
        None | Some("solo") => Mode::Solo,
        Some("pairs") => Mode::Pairs,
        Some(other) => return Err(format!("unknown --mode {other:?}; try solo or pairs")),
    };
    Ok(Args {
        candidate: get("--candidate").unwrap_or_else(|| "heuristic:32".into()),
        baseline: get("--baseline").unwrap_or_else(|| "random".into()),
        games: num("--games", 400),
        mode,
        concurrency: get("--concurrency")
            .or_else(|| get("--threads"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        batch: num("--batch", 0),
        batchers: num("--batchers", 0),
        linger: Duration::from_micros(
            get("--linger-us").and_then(|v| v.parse().ok()).unwrap_or(200),
        ),
        no_batch: argv.iter().any(|a| a == "--no-batch"),
        out: get("--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("arena-progress.jsonl")),
        resume: argv.iter().any(|a| a == "--resume"),
        seed0: get("--seed").and_then(|v| v.parse().ok()).unwrap_or(1_000_000),
        quiet: argv.iter().any(|a| a == "--quiet"),
        heartbeat: get("--heartbeat").and_then(|v| v.parse().ok()).unwrap_or(30),
    })
}

fn print_help() {
    println!(
        "\
arena -- how strong is this agent, with an error bar

USAGE
  cargo run --release --bin arena -- [options]

OPTIONS
  --candidate SPEC   agent under test          [heuristic:32]
  --baseline  SPEC   agent to measure against  [random]
  --games N          approximate games to play [400]
  --mode solo|pairs  seating scheme            [solo]
  --out PATH         progress JSONL            [arena-progress.jsonl]
  --resume           skip blocks already in --out
  --seed N           first block seed          [1000000]
  --quiet            no per-block progress line
  --heartbeat SECS   progress line interval                    [30]

CONCURRENCY   (docs/COMPUTE.md section 2)
  --concurrency N    blocks in flight; with a network agent this is also the
                     reachable evaluation batch [128 when searching, else cores]
  --threads N        old name for --concurrency
  --batch N          max evaluations per forward pass  [min(256, concurrency)]
  --batchers N       inference threads, outside the block pool [cores/5, <= 4]
  --linger-us N      how long a short batch waits for more      [200]
  --no-batch         one evaluation per forward pass

AGENT SPECS  (identical in selfplay; see src/record.rs AgentSpec)
  random             the sample_legal_move rollout policy
  heuristic[:K]      one-ply greedy over K sampled turns; K is the strength knob
  heuristic:full     one ply over EVERY legal move, not a sample of them
  minimax[:D[:MS[:W]]]  paranoid alpha-beta, D turns deep, MS ms of deepening
                     per turn, W root moves deepened. The first ply is
                     exhaustive whatever W is. Budget it: `minimax:4:120` plays
                     a game in seconds, the 600 ms default takes minutes.
  greedy:K:EVAL      one-ply greedy over any evaluator
  mcts:SIMS[:EVAL]   tree search; EVAL defaults to heuristic
  net-random[:small|main]   an untrained net, for timing the pipeline
  PATH.safetensors   a checkpoint; same as mcts:3200:PATH

EXAMPLES
  # is the pipeline wired up at all?  This must be a large positive number.
  arena --candidate heuristic:32 --baseline random --games 200

  # does seeing every move beat sampling 32 of them?
  arena --candidate heuristic:full --baseline heuristic:32 --games 120

  # does looking ahead beat one ply?
  arena --candidate minimax:4:120 --baseline heuristic:full --games 120

  # does search beat one ply?  the first thing to check after mcts lands.
  arena --candidate mcts:400 --baseline heuristic:32 --games 120

  # the every-few-hours question: is generation 40 better than 39?
  arena --candidate ckpt/gen0040.safetensors --baseline ckpt/gen0039.safetensors \\
        --games 600 --mode solo --out eval/g40-vs-g39.jsonl --resume"
    );
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("arena: {e}");
            std::process::exit(2);
        }
    };

    let mut cand = match AgentSpec::parse(&args.candidate, false) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("arena: --candidate: {e}");
            std::process::exit(2);
        }
    };
    let mut base = match AgentSpec::parse(&args.baseline, false) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("arena: --baseline: {e}");
            std::process::exit(2);
        }
    };

    interrupt::install();

    let searching = cand.searches() || base.searches();
    // An arena block is four or six whole games played back to back, so blocks
    // in flight is what fills the batch here. It is lower than self-play's 512
    // because each block holds several games' worth of tree at once.
    let concurrency = if args.concurrency > 0 {
        args.concurrency
    } else if searching {
        128
    } else {
        cores()
    };
    let batch = if args.batch > 0 {
        args.batch
    } else {
        concurrency.min(256)
    };
    let batchers = if args.batchers > 0 {
        args.batchers
    } else {
        cores().div_ceil(5).clamp(1, 4)
    };
    // The two sides may be different networks, so each gets its own queue. The
    // batchers are `std::thread`s outside the rayon pool: if they were rayon
    // tasks and every worker were parked waiting for a batch, nothing would
    // ever assemble one (`COMPUTE.md` §2.6 note 2).
    let batching = !args.no_batch
        && [&mut cand, &mut base]
            .into_iter()
            .filter(|s| s.batchable())
            .fold(false, |acc, s| {
                s.enable_batching(batch, args.linger, batchers) || acc
            });
    // Blocks in flight is the rayon pool size, and for a searching agent it is
    // deliberately larger than the core count: that is where the batch comes
    // from. Most of those threads are parked inside `Evaluator::evaluate`.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build_global();

    let per_block = seatings(args.mode).len();
    let want_blocks = args.games.div_ceil(per_block).max(1);

    // Resume: read what is already on file, and skip those seeds. The progress
    // file is the source of truth for both the summary and the resume, so a run
    // that is stopped and restarted five times reports the same interval as one
    // that ran straight through.
    let mut done: Vec<Block> = if args.resume {
        read_progress(&args.out)
    } else {
        Vec::new()
    };
    let already: std::collections::HashSet<u64> = done.iter().map(|b| b.seed).collect();
    let todo: Vec<u64> = (0..want_blocks as u64)
        .map(|i| args.seed0 + i)
        .filter(|s| !already.contains(s))
        .collect();

    println!("arena  rules v{RULES_VERSION}");
    println!("  candidate : {}", cand.name());
    println!("  baseline  : {}", base.name());
    println!("  mode      : {}", args.mode.label());
    if batching {
        println!(
            "  compute   : {concurrency} blocks in flight, batch <= {batch} over {batchers} \
             inference threads"
        );
    } else {
        println!("  compute   : {concurrency} blocks in flight, unbatched evaluator");
    }
    println!(
        "  plan      : {} blocks x {} games = {} games ({} already on file)",
        todo.len(),
        per_block,
        todo.len() * per_block,
        done.len() * per_block
    );
    println!("  progress  : {}", args.out.display());
    println!();

    let progress = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.out)
            .unwrap_or_else(|e| {
                eprintln!("arena: cannot open {}: {e}", args.out.display());
                std::process::exit(1);
            }),
    );
    let collected: Mutex<Vec<Block>> = Mutex::new(Vec::new());
    let cfg = GameConfig::evaluation();
    let t0 = Instant::now();
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let total = todo.len();

    // Parallel across blocks, not across the games inside one: a block is the
    // unit of statistical independence and also a convenient unit of work
    // (four full games, hundreds of milliseconds), so the mutex below is
    // contended once per block and never measurably.
    // A heartbeat, for the same reason self-play has one: a block is four full
    // games, and with a searching agent at a real budget that is minutes. The
    // block-completion line says nothing in between, which is indistinguishable
    // from a hang. Evaluations move continuously, so they are what to report.
    let hb_stop = std::sync::atomic::AtomicBool::new(false);

    std::thread::scope(|scope| {
        if !args.quiet {
            scope.spawn(|| {
                let every = Duration::from_secs(args.heartbeat.max(1));
                let (mut last_evals, mut last_at) = (0u64, Instant::now());
                while !hb_stop.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(250));
                    if hb_stop.load(std::sync::atomic::Ordering::Relaxed) || last_at.elapsed() < every {
                        continue;
                    }
                    let n = counter.load(std::sync::atomic::Ordering::Relaxed);
                    let secs = t0.elapsed().as_secs_f64();

                    let (evals, rate, batch) = match cand.batch_stats() {
                        Some(st) => {
                            let d = st.queries.saturating_sub(last_evals);
                            last_evals = st.queries;
                            (
                                st.queries,
                                d as f64 / last_at.elapsed().as_secs_f64().max(1e-9),
                                format!("  batch {:.0}", st.mean_batch),
                            )
                        }
                        None => (0, 0.0, String::new()),
                    };
                    last_at = Instant::now();

                    let eta = if n > 0 {
                        let left = (total - n) as f64 * (secs / n as f64);
                        format!("  eta {}", record::hms(left))
                    } else {
                        "  eta --".to_string()
                    };

                    eprint!(
                        "\r  [{}] {n}/{total} blocks  {} evals ({:.0}/s){batch}{eta}      ",
                        record::hms(secs),
                        record::thousands(evals),
                        rate,
                    );
                    let _ = std::io::stderr().flush();
                }
            });
        }

    todo.par_iter().for_each(|&seed| {
        if interrupt::stopping() {
            return;
        }
        // One instance per task. A search agent owns a mutable tree arena and
        // a slot in the evaluation batch; sharing either across blocks would
        // serialise the run.
        let (ca, ba) = (cand.instance(), base.instance());
        let b = play_block(seed, args.mode, ca.as_ref(), ba.as_ref(), &cfg);

        // Append and flush before anything else. Everything after this point
        // can be lost without losing the block.
        {
            let mut f = progress.lock().unwrap();
            let _ = writeln!(f, "{}", b.to_json());
            let _ = f.flush();
        }
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if !args.quiet {
            let running = collected.lock().unwrap();
            let so_far: Vec<f64> = running
                .iter()
                .chain(std::iter::once(&b))
                .map(|b| b.mean(|o| o.centred))
                .collect();
            let s = Summary::of(&so_far);
            eprint!(
                "\r  {n}/{total} blocks   centred {:+.2} +/- {:.2}   {:.1} blocks/s     ",
                s.mean,
                s.ci,
                n as f64 / t0.elapsed().as_secs_f64().max(1e-9)
            );
            let _ = std::io::stderr().flush();
        }
        collected.lock().unwrap().push(b);
    });

        // `par_iter().for_each` blocks until every block is done, so signalling
        // here is safe -- unlike a `scope.spawn` loop, which returns at once.
        hb_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    if !args.quiet {
        eprintln!();
    }

    cand.shutdown();
    base.shutdown();
    done.extend(collected.into_inner().unwrap());
    let interrupted = interrupt::stopping();
    if interrupted {
        println!("\ninterrupted -- reporting the {} blocks that finished\n", done.len());
    }
    report(&done, args.mode, &cand.name(), &base.name(), t0.elapsed().as_secs_f64());
    // `COMPUTE.md` §2.6: the realised batch size is the only symptom you get
    // when the concurrency is not producing one.
    for (who, s) in [("candidate", cand.batch_stats()), ("baseline", base.batch_stats())] {
        if let Some(s) = s {
            println!("  {who} batching: {}", s.line());
        }
    }

    if interrupted {
        std::process::exit(130);
    }
}

fn read_progress(path: &Path) -> Vec<Block> {
    let Ok(s) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    s.lines().filter_map(block_from_json).collect()
}

// =======================================================================
// The report
// =======================================================================

fn report(blocks: &[Block], mode: Mode, cand: &str, base: &str, secs: f64) {
    if blocks.is_empty() {
        println!("no blocks completed; nothing to report");
        return;
    }
    let games: usize = blocks.iter().map(|b| b.games.len()).sum();
    let aborted: usize = blocks
        .iter()
        .flat_map(|b| &b.games)
        .filter(|o| o.aborted)
        .count();

    // Every summary is over *block* means. See the module header: the four (or
    // six) games in a block share a seed and are correlated, so treating each
    // game as independent would understate the interval by up to ~2x.
    let centred = Summary::of(&per_block(blocks, |o| o.centred));
    let vs_base = Summary::of(&per_block(blocks, |o| o.vs_base));
    let margin = Summary::of(&per_block(blocks, |o| o.margin));
    let win = Summary::of(&per_block(blocks, |o| o.win));
    let cand_s = Summary::of(&per_block(blocks, |o| o.cand_score));
    let base_s = Summary::of(&per_block(blocks, |o| o.base_score));

    let null_win = mode.null_win_rate();

    println!("=================================================================");
    println!(" {cand}   vs   {base}");
    println!(" {} blocks / {games} games in {secs:.0}s   ({:.1} games/s)", blocks.len(), games as f64 / secs.max(1e-9));
    println!(" {}", mode.label());
    println!("=================================================================");
    println!();
    println!("  metric                  value        95% CI              null");
    println!("  ----------------------------------------------------------------");
    row("mean centred score", centred, Some(0.0));
    row("points vs baseline", vs_base, Some(0.0));
    row("margin vs best rival", margin, None);
    row("win rate", win, Some(null_win));
    println!("  ----------------------------------------------------------------");
    println!(
        "  candidate mean score  {:>8.2}                        baseline {:>7.2}",
        cand_s.mean, base_s.mean
    );
    // A full game is 27 rounds. Anything materially short of that means games
    // are ending early, which is a rules bug and makes every number above
    // meaningless -- so it is reported next to them, not buried.
    let days: f64 = blocks
        .iter()
        .flat_map(|b| &b.games)
        .map(|o| o.days as f64)
        .sum::<f64>()
        / games.max(1) as f64;
    if days > 0.0 {
        println!("  mean game length      {days:>8.1} rounds (a complete game is 27)");
    }
    if aborted > 0 {
        println!("  !! {aborted} games hit the 200-round guard -- that is a rules bug");
    }
    println!();

    // The verdict, spelled out. This is the line the user reads at 3am.
    let verdict = if centred.mean - centred.ci > 0.0 {
        format!(
            "STRONGER. The candidate's centred score is {:+.2} and the interval \n  ({:+.2}, {:+.2}) excludes zero.",
            centred.mean,
            centred.mean - centred.ci,
            centred.mean + centred.ci
        )
    } else if centred.mean + centred.ci < 0.0 {
        format!(
            "WEAKER. The candidate's centred score is {:+.2} and the interval \n  ({:+.2}, {:+.2}) is entirely below zero.",
            centred.mean,
            centred.mean - centred.ci,
            centred.mean + centred.ci
        )
    } else {
        // How many blocks would it take to resolve the observed effect? This is
        // the actionable number when the answer is "cannot tell": either play
        // that many, or accept that the difference is small.
        let need = if centred.mean.abs() > 1e-9 && centred.sd.is_finite() {
            let z = 1.96 * centred.sd / centred.mean.abs();
            (z * z).ceil() as usize
        } else {
            usize::MAX
        };
        let more = if need == usize::MAX || need > 1_000_000 {
            "no plausible number of games would resolve an effect this small".to_string()
        } else {
            format!(
                "~{need} blocks ({} games) would resolve an effect of this size",
                need * blocks[0].games.len()
            )
        };
        format!(
            "NOT DISTINGUISHABLE. Centred score {:+.2}, interval ({:+.2}, {:+.2}) \n  straddles zero. {more}.",
            centred.mean,
            centred.mean - centred.ci,
            centred.mean + centred.ci
        )
    };
    println!("  {verdict}");
    println!();
    match mode {
        Mode::Solo => println!(
            "  Read the win rate against a null of 25%, not 50%: one seat of four is \n  the candidate's, so an equally strong agent wins a quarter of the games."
        ),
        Mode::Pairs => println!(
            "  Win rate is the share of games won by *either* candidate seat, so its \n  null is 50%: two of the four seats are the candidate's."
        ),
    }
    println!(
        "  `margin vs best rival` has no null column on purpose: it is one score \n  against the maximum of {}, which is negative even between identical agents \n  (about {:.0} points here). Compare it across runs, never against zero.",
        match mode { Mode::Solo => "three", Mode::Pairs => "two" },
        margin.mean,
    );
    println!(
        "  n in every interval is {} blocks, not {games} games: games sharing a seed \n  are correlated and a per-game interval would be up to 2x too narrow.",
        blocks.len()
    );
    let p = centred.p_value(0.0);
    if p.is_finite() {
        println!("  Two-sided p for centred score = 0: {p:.4} (one look; see docs/TRAINING.md).");
    }
    println!();
}

fn per_block(blocks: &[Block], f: impl Fn(&Outcome) -> f64 + Copy) -> Vec<f64> {
    blocks.iter().map(|b| b.mean(f)).collect()
}

/// One metric row. `null` is `None` for a metric whose null value is not zero
/// and not analytically known — `margin vs best rival` is one draw against the
/// maximum of three, which is about -11 points even between identical agents.
/// Printing a `+`/`-` verdict against a null that does not exist would be
/// worse than printing nothing, so such rows get no marker and no null column.
fn row(name: &str, s: Summary, null: Option<f64>) {
    let lo = s.mean - s.ci;
    let hi = s.mean + s.ci;
    let (mark, col) = match null {
        Some(n) if lo > n => ("  +", format!("{n:>5.2}")),
        Some(n) if hi < n => ("  -", format!("{n:>5.2}")),
        Some(n) => ("  .", format!("{n:>5.2}")),
        None => ("   ", "  n/a".to_string()),
    };
    println!(
        "  {name:<22}{:>8.3}   ({:>+7.3}, {:>+7.3}){mark}   {col}",
        s.mean, lo, hi
    );
}
