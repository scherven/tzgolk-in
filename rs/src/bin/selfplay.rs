//! Generate self-play games and persist them as training records.
//!
//!     cargo run --release --bin selfplay -- --agent heuristic:32 --games 20000 --out replay
//!
//! One generation's worth of data. Records are appended to a `.part` shard as
//! each game finishes and the part is sealed into a timestamped `.tzr` on a
//! wall-clock interval, so a run killed at any moment loses at most the games
//! that were mid-flight — never one that had already ended.
//!
//! # What gets written
//!
//! The format lives in `src/record.rs` and is described field by field in
//! `docs/TRAINING.md`. In one line: a 64-byte header then fixed 512-byte
//! records, each carrying the **raw `GameState`** (not an encoded tensor — the
//! encoder is still being written and `LEARNING.md` §6.2 wants old games to
//! survive encoder changes), the phase, the visit distribution, the final
//! per-player scores, `z_rel`, and `tzolkin::RULES_VERSION`.
//!
//! The rules stamp is the cheapest insurance in the project. A rules change
//! invalidates the learned value function even when every tensor shape survives
//! (`LEARNING.md` §8.4); with the stamp, such a change turns "the whole buffer
//! is suspect" into "the records before generation N are".
//!
//! # Concurrency: games in flight, not threads
//!
//! This driver runs **one OS thread per concurrent game**, defaulting to 256
//! of them for a searching agent, and puts a batching evaluator behind the
//! `Evaluator` trait (`src/record.rs`, `BatchQueue`). It used to run
//! `seeds.par_iter()`, one game per rayon task, which caps the achievable
//! evaluation batch at the core count.
//!
//! `COMPUTE.md` §2.2 measures the same forward pass at **5,491 evaluations/s at
//! batch 1 and 111,469/s at batch 256** — 20.3x, on one core, with no new
//! hardware. The batch has to come from somewhere, and §2.1's answer is that it
//! comes from the number of *games* in flight, which is a free parameter
//! bounded only by memory (~0.5-1.4 MB of tree per game, so 256 games is
//! ~360 MB). 256 mostly-parked OS threads is the intended design, not a smell;
//! it is what KataGo's "far larger than the number of cores" configuration
//! means.
//!
//! Crucially this costs **no search quality** (`COMPUTE.md` §2.5): one descent
//! per tree means no virtual loss, no stale statistics, and a search that is
//! bit-identical to a sequential one. What it costs is latency per game, which
//! self-play does not care about.
//!
//! Watch `--batch` in the summary. If the realised mean batch size is far below
//! the configured maximum, none of the throughput above is happening, and it is
//! the only symptom you will get.
//!
//! # A note on value-only records
//!
//! The one-ply agents (`random`, `heuristic:K`) draw their candidate moves from
//! an RNG rather than from a tree, so their indices are not regenerable at
//! training time. Those records are written with `policy_kind = NONE`, meaning
//! "value target only". That is not a consolation prize: `LEARNING.md` §6.7
//! asks for exactly this data — positions paired with the final scores of a
//! plausible continuation — as the value warm-start, and calls it the
//! highest-leverage two hours in the plan. Running `selfplay --agent
//! heuristic:32` overnight *is* the warm-start.
//!
//! A search agent (`mcts:SIMS:EVAL`, or a checkpoint path) writes
//! `policy_kind = TREE_EDGE` records from the 25% of turns that get the full
//! playout budget, which are the real policy targets.

#[allow(dead_code)]
use tzolkin::record;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::SeedableRng;

use record::{
    atomic_write, interrupt, now_unix, patch_win_share, play_game, read_shard, schema_json, stamp,
    write_record, Agent, AgentSpec, GameConfig, Node, ShardWriter, RECORD_BYTES,
};
use tzolkin::ids::*;
use tzolkin::RULES_VERSION;

struct Args {
    agent: String,
    games: u64,
    out: PathBuf,
    generation: u32,
    /// Games in flight. One OS thread each.
    concurrency: usize,
    /// Maximum evaluations per forward pass.
    batch: usize,
    /// Batcher threads. These live outside the game pool.
    batchers: usize,
    /// How long a batcher holds a short batch open for more arrivals.
    linger: Duration,
    no_batch: bool,
    seed0: u64,
    snapshot: Duration,
    fsync_games: u64,
    keep_gb: f64,
    /// Seconds between heartbeat lines.
    heartbeat: u64,
    check: bool,
    quiet: bool,
}

fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Games in flight, when the user did not say.
///
/// `COMPUTE.md` §2.1: batch size is limited by memory and by nothing else. The
/// sweep in `docs/TRAINING.md` §2.1 was still improving at 2,048 games in
/// flight on this machine — 512 is the point where the return flattens and the
/// tree memory at the default 3,200 simulations is still under a gigabyte.
///
/// For an agent that does not search there is no batch to fill and no reason to
/// run more games than cores.
const DEFAULT_CONCURRENCY: usize = 512;

fn default_concurrency(searches: bool) -> usize {
    if searches {
        DEFAULT_CONCURRENCY
    } else {
        cores()
    }
}

/// Batcher threads, when the user did not say.
///
/// Measured, not derived: on this 14-core Mac the whole-machine rate peaks near
/// three, and more batchers only fragment the queue into batches too small to
/// amortise anything (`docs/TRAINING.md` §2.1 has the sweep). `COMPUTE.md` §2.2
/// suggested eight; eight measures 8% *slower* than three here, because its
/// figure is per-core inference throughput and does not account for the game
/// threads competing for the same cores.
fn default_batchers() -> usize {
    cores().div_ceil(5).clamp(1, 4)
}

fn parse_args() -> Result<Args, String> {
    // A mistyped flag must not be silently ignored: the tool would run its
    // defaults and report a confident answer to a question nobody asked.
    record::reject_unknown_flags(&["--agent", "--batch", "--batchers", "--check", "--concurrency", "--fsync-games", "--games", "--gen", "--heartbeat", "--help", "--keep-gb", "--linger-us", "--no-batch", "--out", "--print-schema", "--quiet", "--seed", "--snapshot-hours", "--threads", "--verify"])?;

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
    let hours: f64 = get("--snapshot-hours")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);
    if hours <= 0.0 {
        return Err("--snapshot-hours must be positive".into());
    }
    // `--threads` is the old name and meant the same thing: how many games run
    // at once. Kept so the commands in docs/TRAINING.md still work.
    let concurrency = get("--concurrency")
        .or_else(|| get("--threads"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    Ok(Args {
        agent: get("--agent").unwrap_or_else(|| "heuristic:32".into()),
        games: get("--games").and_then(|v| v.parse().ok()).unwrap_or(1000),
        out: get("--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("replay")),
        generation: get("--gen").and_then(|v| v.parse().ok()).unwrap_or(0),
        concurrency,
        batch: get("--batch").and_then(|v| v.parse().ok()).unwrap_or(0),
        batchers: get("--batchers").and_then(|v| v.parse().ok()).unwrap_or(0),
        linger: Duration::from_micros(
            get("--linger-us").and_then(|v| v.parse().ok()).unwrap_or(200),
        ),
        no_batch: argv.iter().any(|a| a == "--no-batch"),
        seed0: get("--seed").and_then(|v| v.parse().ok()).unwrap_or(0),
        snapshot: Duration::from_secs_f64(hours * 3600.0),
        fsync_games: get("--fsync-games").and_then(|v| v.parse().ok()).unwrap_or(8),
        keep_gb: get("--keep-gb").and_then(|v| v.parse().ok()).unwrap_or(0.0),
        heartbeat: get("--heartbeat").and_then(|v| v.parse().ok()).unwrap_or(30),
        check: argv.iter().any(|a| a == "--check"),
        quiet: argv.iter().any(|a| a == "--quiet"),
    })
}

fn print_help() {
    println!(
        "\
selfplay -- generate training records

USAGE
  cargo run --release --bin selfplay -- [options]
  cargo run --release --bin selfplay -- --print-schema
  cargo run --release --bin selfplay -- --verify replay/gen0000-000-....tzr

OPTIONS
  --agent SPEC         who plays; all four seats     [heuristic:32]
  --games N            games to generate             [1000]
  --out DIR            shard directory               [replay]
  --gen N              generation number, in names   [0]
  --seed N             first game seed               [0]
  --snapshot-hours H   seal a shard this often       [3]
  --fsync-games N      fsync every N games           [8]
  --keep-gb N          drop oldest shards past N GB   [off]
  --heartbeat SECS     progress line interval          [30]
  --check              check_move + validate every turn (slow)
  --quiet              no progress line

CONCURRENCY   (docs/COMPUTE.md section 2)
  --concurrency N      games in flight, one thread each
                       [512 when searching, else one per core]
  --threads N          old name for --concurrency
  --batch N            max evaluations per forward pass  [min(256, N games)]
  --batchers N         inference threads, outside the game pool  [cores/5, <= 4]
  --linger-us N        how long a short batch waits for more     [200]
  --no-batch           one evaluation per forward pass (the old behaviour)

AGENT SPECS
  random               the sample_legal_move rollout policy
  heuristic[:K]        one-ply greedy over K sampled turns  [K=32]
  greedy:K:EVAL        one-ply greedy over any evaluator
  mcts:SIMS[:EVAL]     tree search; EVAL defaults to heuristic
  net-random[:small|main]   an untrained net, for measuring the pipeline
  PATH.safetensors     a checkpoint; same as mcts:3200:PATH

OUTPUT
  DIR/gen0000-000-<UTC>.tzr    sealed shards, chronologically sortable
  DIR/gen0000.part             in flight; sealed on interval and at exit
  DIR/latest                   one line: filename of the newest sealed shard
  DIR/manifest.json            run provenance, rewritten atomically
  DIR/schema.json              the record layout, for train/replay.py

NOTES
  Set VECLIB_MAXIMUM_THREADS=1 before running with a network backend, or
  Accelerate spawns its own pool and fights the batchers for the same cores."
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--print-schema") {
        print!("{}", schema_json());
        return;
    }
    if let Some(i) = argv.iter().position(|a| a == "--verify") {
        let path = argv.get(i + 1).map(PathBuf::from).unwrap_or_default();
        std::process::exit(verify(&path));
    }

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("selfplay: {e}");
            std::process::exit(2);
        }
    };
    let mut spec = match AgentSpec::parse(&args.agent, true) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("selfplay: --agent: {e}");
            std::process::exit(2);
        }
    };
    interrupt::install();

    let concurrency = if args.concurrency > 0 {
        args.concurrency
    } else {
        default_concurrency(spec.searches())
    }
    .clamp(1, args.games.max(1) as usize);

    // The batcher threads are deliberately `std::thread`s outside the game
    // pool. If every thread that could run a batcher is parked waiting for a
    // batch, nothing ever assembles one (`COMPUTE.md` §2.6 note 2).
    let batchers = if args.batchers > 0 {
        args.batchers
    } else {
        default_batchers()
    };
    // `COMPUTE.md` §1.4: the GEMM saturates at 32 and everything above that
    // amortises fixed cost. Measured here, 256 beats 128 by ~14%.
    let batch = if args.batch > 0 {
        args.batch
    } else {
        concurrency.min(256)
    };
    let batching = spec.batchable()
        && !args.no_batch
        && spec.enable_batching(batch, args.linger, batchers);
    if batching && std::env::var_os("VECLIB_MAXIMUM_THREADS").is_none() {
        eprintln!(
            "selfplay: warning: VECLIB_MAXIMUM_THREADS is unset. Accelerate will spawn its\n\
             own pool and fight the {batchers} batcher threads for the same cores. Set it to 1."
        );
    }

    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("selfplay: cannot create {}: {e}", args.out.display());
        std::process::exit(1);
    }
    // The schema goes next to the shards, so `train/replay.py` never carries a
    // second copy of the layout table. `src/state.rs` grew a gear space this
    // week; anything with a hard-coded offset would now be reading garbage.
    if let Err(e) = atomic_write(&args.out.join("schema.json"), schema_json().as_bytes()) {
        eprintln!("selfplay: cannot write schema.json: {e}");
        std::process::exit(1);
    }

    let agent_name = spec.name();
    let producer = format!("selfplay/{agent_name}");
    let writer = match ShardWriter::new(&args.out, args.generation, &producer, args.fsync_games) {
        Ok(w) => Mutex::new(w),
        Err(e) => {
            eprintln!("selfplay: cannot open shard: {e}");
            std::process::exit(1);
        }
    };

    println!("selfplay  rules v{RULES_VERSION}  generation {}", args.generation);
    println!("  agent    : {agent_name}");
    println!("  games    : {}", args.games);
    println!("  out      : {}", args.out.display());
    if batching {
        println!(
            "  compute  : {concurrency} games in flight, batch <= {batch} over {batchers} \
             inference threads, {} us linger",
            args.linger.as_micros()
        );
    } else {
        println!(
            "  compute  : {concurrency} games in flight, unbatched evaluator{}",
            if args.no_batch { " (--no-batch)" } else { "" }
        );
    }
    println!(
        "  snapshot : every {:.1} h, fsync every {} games",
        args.snapshot.as_secs_f64() / 3600.0,
        args.fsync_games
    );
    println!();

    let cfg = GameConfig {
        temperature: record::selfplay_temperature,
        check: args.check,
    };
    let t0 = Instant::now();
    let last_roll = Mutex::new(Instant::now());
    let next = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let records = AtomicU64::new(0);
    let aborted = AtomicU64::new(0);
    let score_sum = AtomicU64::new(0); // as (score + 1000) to stay unsigned
    let score_n = AtomicU64::new(0);

    // One OS thread per concurrent game, each with its own agent instance —
    // its own tree arena and its own slot in the evaluation batch. Most of them
    // are parked inside `Evaluator::evaluate` at any instant; that is the point.
    // A heartbeat, because the completion-driven line above says nothing until
    // a game finishes -- and with hundreds of games in flight they advance in
    // lockstep, so the first hour of a searching run prints not one character
    // and looks exactly like a hang.
    //
    // Evaluations are the thing to report: they move continuously whatever the
    // game count is doing, so a rising number is proof of progress and a static
    // one is proof of a stall.
    let hb_stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        if !args.quiet {
            scope.spawn(|| {
                let every = Duration::from_secs(args.heartbeat.max(1));
                let mut last_evals = 0u64;
                let mut last_at = Instant::now();
                while !hb_stop.load(Ordering::Relaxed) {
                    // Wake often so shutdown is prompt; report on the interval.
                    std::thread::sleep(Duration::from_millis(250));
                    if hb_stop.load(Ordering::Relaxed) || last_at.elapsed() < every {
                        continue;
                    }
                    let now = Instant::now();
                    let secs = t0.elapsed().as_secs_f64();
                    let n = done.load(Ordering::Relaxed);

                    // `BatchStats::queries` is the evaluation count; `batches`
                    // is how many forward passes carried them.
                    let (evals, rate, batch) = match spec.batch_stats() {
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
                    last_at = now;

                    // Estimate from evaluations once any game has landed; the
                    // per-game cost is only knowable after the first finishes.
                    let eta = if n > 0 {
                        let per = secs / n as f64;
                        let left = (args.games.saturating_sub(n)) as f64 * per;
                        format!("  eta {}", record::hms(left))
                    } else {
                        "  eta --".to_string()
                    };

                    eprint!(
                        "\r  [{}] {n}/{} games  {} evals ({:.0}/s){batch}{eta}      ",
                        record::hms(secs),
                        args.games,
                        record::thousands(evals),
                        rate,
                    );
                    let _ = std::io::stderr().flush();
                }
            });
        }

        let mut workers = Vec::with_capacity(concurrency);
        for w in 0..concurrency {
            let spec = &spec;
            let args = &args;
            let cfg = &cfg;
            let writer = &writer;
            let last_roll = &last_roll;
            let (next, done, records, aborted, score_sum, score_n) =
                (&next, &done, &records, &aborted, &score_sum, &score_n);
            let h = std::thread::Builder::new()
                .name(format!("game-{w}"))
                .stack_size(4 << 20)
                .spawn_scoped(scope, move || {
                    let agent = spec.instance();
                    let agents: [&dyn Agent; N_PLAYERS] = [agent.as_ref(); N_PLAYERS];
                    loop {
                        if interrupt::stopping() {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= args.games {
                            break;
                        }
                        let seed = args.seed0 + i;
                        let mut rng =
                            rand::rngs::StdRng::seed_from_u64(seed ^ 0x5DEE_CE66_D1CE_u64);
                        let r = play_game(seed, &agents, cfg, &mut rng);
                        if r.aborted {
                            aborted.fetch_add(1, Ordering::Relaxed);
                            // A game that did not terminate is a rules bug, and
                            // its scores are meaningless. Poisoning the buffer
                            // with it would be worse than losing it.
                            continue;
                        }

                        let bytes = serialise_game(seed, &r.nodes, r.scores, r.win_share);
                        let n_rec = (bytes.len() / RECORD_BYTES) as u64;
                        {
                            let mut wr = writer.lock().unwrap();
                            if let Err(e) = wr.append_game(&bytes) {
                                eprintln!("\nselfplay: write failed: {e}");
                                interrupt::request_stop();
                                break;
                            }
                            // Roll on the wall clock, inside the same lock so
                            // two threads cannot seal the same part twice.
                            let mut lr = last_roll.lock().unwrap();
                            if lr.elapsed() >= args.snapshot {
                                match wr.roll() {
                                    Ok(Some(name)) => {
                                        eprintln!("\n  sealed {name}");
                                        match record::ShardWriter::prune(
                                            args.out.as_ref(),
                                            args.keep_gb,
                                        ) {
                                            Ok((n, freed)) if n > 0 => eprintln!(
                                                "  pruned {n} shard(s), freed {:.1} GB",
                                                freed as f64 / 1e9
                                            ),
                                            Err(e) => eprintln!("  prune failed: {e}"),
                                            _ => {}
                                        }
                                        let _ =
                                            write_manifest(&args.out, args, &spec.name(), &wr);
                                    }
                                    Ok(None) => {}
                                    Err(e) => eprintln!("\nselfplay: roll failed: {e}"),
                                }
                                *lr = Instant::now();
                            }
                        }

                        records.fetch_add(n_rec, Ordering::Relaxed);
                        for s in r.scores {
                            score_sum.fetch_add((s as i64 + 1000) as u64, Ordering::Relaxed);
                            score_n.fetch_add(1, Ordering::Relaxed);
                        }
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if !args.quiet && n % 8 == 0 {
                            let secs = t0.elapsed().as_secs_f64();
                            let sn = score_n.load(Ordering::Relaxed).max(1);
                            let mean =
                                score_sum.load(Ordering::Relaxed) as f64 / sn as f64 - 1000.0;
                            let bs = match spec.batch_stats() {
                                Some(s) => format!("  batch {:.0}", s.mean_batch),
                                None => String::new(),
                            };
                            eprint!(
                                "\r  {n}/{} games  {:.2} games/s  {} records  mean score \
                                 {mean:+.1}{bs}     ",
                                args.games,
                                n as f64 / secs.max(1e-9),
                                records.load(Ordering::Relaxed)
                            );
                            let _ = std::io::stderr().flush();
                        }
                    }
                })
                .expect("cannot spawn game thread");
            workers.push(h);
        }

        // Join the games *before* stopping the heartbeat: `scope.spawn` returns
        // at once, so signalling here without joining would end the heartbeat a
        // quarter of a second into the run.
        for h in workers {
            let _ = h.join();
        }
        hb_stop.store(true, Ordering::Relaxed);
    });
    if !args.quiet {
        eprintln!();
    }
    let mut w = writer.into_inner().unwrap();
    match w.roll() {
        Ok(Some(name)) => println!("  sealed {name}"),
        Ok(None) => {}
        Err(e) => eprintln!("selfplay: final roll failed: {e}"),
    }
    let _ = write_manifest(&args.out, &args, &agent_name, &w);
    let games_total = w.games_total;
    let records_total = w.records_total;
    let sealed = w.finish().unwrap_or_default();

    let secs = t0.elapsed().as_secs_f64();
    let sn = score_n.load(Ordering::Relaxed).max(1);
    println!();
    println!(
        "{games_total} games, {records_total} records, {} shards, in {secs:.0}s ({:.2} games/s)",
        sealed.len(),
        games_total as f64 / secs.max(1e-9)
    );
    println!(
        "  mean final score {:+.1}, {:.0} records/game, {:.1} MB",
        score_sum.load(Ordering::Relaxed) as f64 / sn as f64 - 1000.0,
        records_total as f64 / games_total.max(1) as f64,
        (records_total * RECORD_BYTES as u64) as f64 / 1e6,
    );
    // `COMPUTE.md` §2.6: "instrument the batch... if the mean batch size is not
    // close to the configured maximum, none of §5's numbers are happening, and
    // it is the only symptom you will get."
    if let Some(s) = spec.batch_stats() {
        println!("  {}", s.line());
        println!(
            "  {:.0} evaluations/s over the whole run; batcher utilisation {:.0}%",
            s.queries as f64 / secs.max(1e-9),
            s.utilisation(secs, batchers) * 100.0
        );
        // `COMPUTE.md` §1.4 puts the GEMM's saturation point at 32; below that
        // the batching is not doing its job. A mean well under `--batch` but
        // comfortably over 32 is normal and not worth a warning.
        if s.mean_batch < 32.0 {
            println!(
                "  !! mean batch {:.1} is below the GEMM's saturation point of 32, so most of\n     \
                 the batching is being wasted. Raise --concurrency, or lower --batchers so\n     \
                 fewer of them split the same queue.",
                s.mean_batch
            );
        }
        print!("{}", s.histogram());
    }
    let ab = aborted.load(Ordering::Relaxed);
    if ab > 0 {
        println!("  !! {ab} games hit the 200-round guard and were discarded -- rules bug");
    }
    if interrupt::stopping() {
        println!("  interrupted; everything above is on disk");
        std::process::exit(130);
    }
}

/// Lay a finished game out as records.
///
/// The trailer — final scores, `z_rel`, `win_share` — is the same for every
/// record in the game, which is why the whole game is buffered in memory and
/// written in one call. A game is ~350 nodes, so ~180 kB; buffering it is what
/// makes "an interrupted run loses at most the game in flight" true without any
/// partial-game bookkeeping on disk.
fn serialise_game(
    seed: u64,
    nodes: &[Node],
    scores: [i16; N_PLAYERS],
    win_share: [f32; N_PLAYERS],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(nodes.len() * RECORD_BYTES);
    let mut buf = [0u8; RECORD_BYTES];
    for (i, n) in nodes.iter().enumerate() {
        write_record(&mut buf, n, seed, i as u16, scores);
        patch_win_share(&mut buf, win_share);
        out.extend_from_slice(&buf);
    }
    out
}

/// Run provenance. Rewritten atomically after every roll so it is never
/// observed half-written (`LEARNING.md` §6.9), and it records the git commit so
/// that when the rules move again the buffer can be truncated rather than
/// thrown away (§8.4).
fn write_manifest(dir: &Path, args: &Args, agent: &str, w: &ShardWriter) -> std::io::Result<()> {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let shards = w
        .sealed
        .iter()
        .map(|s| format!("    {:?}", s))
        .collect::<Vec<_>>()
        .join(",\n");
    let json = format!(
        "{{\n  \"generation\": {},\n  \"rules_version\": {},\n  \"agent\": {:?},\n  \
         \"commit\": {:?},\n  \"updated\": {:?},\n  \"games\": {},\n  \"records\": {},\n  \
         \"record_bytes\": {},\n  \"shards\": [\n{}\n  ]\n}}\n",
        args.generation,
        RULES_VERSION,
        agent,
        commit,
        stamp(now_unix()),
        w.games_total,
        w.records_total,
        RECORD_BYTES,
        shards,
    );
    atomic_write(&dir.join("manifest.json"), json.as_bytes())
}

/// Read a shard back through the same codec that wrote it and report what is in
/// it. Cheap insurance: run it once after the first generation and again after
/// any change to `src/state.rs`.
fn verify(path: &Path) -> i32 {
    let (rules, recs) = match read_shard(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("selfplay --verify: {e}");
            return 1;
        }
    };
    if recs.is_empty() {
        println!("{}: header only, no records", path.display());
        return 0;
    }
    let mut games = std::collections::HashSet::new();
    let mut bad = 0usize;
    let mut phases = [0usize; 8];
    let mut score_sum = 0i64;
    for r in &recs {
        let rec = record::read_record(r);
        let (state, phase, scores) = (rec.state, rec.phase, rec.scores);
        if rec.rules_version != rules {
            bad += 1;
        }
        // Round-tripping the state through the codec is the check that matters:
        // if `state.rs` has changed shape since the shard was written, this is
        // where it shows up rather than in a silently mistrained net.
        let mut buf = [0u8; record::STATE_SLOT];
        record::encode_state(&state, &mut buf);
        if record::decode_state(&buf) != state {
            bad += 1;
        }
        phases[phase.tag() as usize] += 1;
        score_sum += scores.iter().map(|&s| s as i64).sum::<i64>();
        games.insert(u64::from_le_bytes(r[320..328].try_into().unwrap()));
    }
    println!("{}", path.display());
    println!("  rules version : {rules} (current {RULES_VERSION})");
    println!("  records       : {}", recs.len());
    println!("  games         : {}", games.len());
    println!(
        "  records/game  : {:.1}",
        recs.len() as f64 / games.len().max(1) as f64
    );
    println!(
        "  mean score    : {:+.1}",
        score_sum as f64 / (recs.len() * N_PLAYERS) as f64
    );
    println!("  phase tags    : {phases:?}");
    if rules != RULES_VERSION {
        println!("  !! stale rules version -- the loader should drop this shard");
    }
    if bad > 0 {
        println!("  !! {bad} records failed the codec round trip");
        return 1;
    }
    println!("  OK");
    0
}
