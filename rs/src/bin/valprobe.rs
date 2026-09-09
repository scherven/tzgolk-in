//! `valprobe` -- how well does an evaluator's value head predict the outcome
//! stored in a replay shard?
//!
//! The arena answers "is this agent stronger", which is the question that
//! matters and also the expensive one. This answers the cheap upstream
//! question: on the positions the buffer actually contains, does a candidate
//! evaluator's `value` track the realised `z_rel` better than `eval::heuristic`
//! does? A net that cannot beat the heuristic *here* -- on the very
//! distribution it was fitted to -- cannot beat it inside the search either,
//! and finding that out costs seconds instead of core-hours.
//!
//! Read the numbers with the caveat stamped on them: `z_rel` is the outcome of
//! the *generating* agent's continuation, not of the search that will use the
//! evaluator. A value head that fits this perfectly has learned "how a game
//! between four `heuristic:32` agents ends", which is not the same function as
//! "how a game between four `mcts:8192` agents ends".
//!
//! ```text
//! valprobe SHARD.tzr N [EVAL ...]        EVAL: `heuristic` (default) or a path
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use tzolkin::ids::{PlayerId, N_PLAYERS};
use tzolkin::net::Net;
use tzolkin::phase::{Evaluator, HeuristicEvaluator};
use tzolkin::record::{read_record, HEADER_BYTES, RECORD_BYTES};

/// `record.rs` writes the agent's own estimate here, seat order.
const O_ROOT_VALUE: usize = 492;
const O_DAY: usize = 448;

/// `(1 - blend) * net + blend * eval::heuristic`, matching `NetBatch::blend`.
struct Blended {
    net: Net,
    blend: f32,
}

impl Evaluator for Blended {
    fn evaluate(
        &self,
        state: &tzolkin::state::GameState,
        phase: tzolkin::phase::Phase,
        turn: PlayerId,
        n_edges: usize,
    ) -> tzolkin::phase::Evaluation {
        let mut e = self.net.evaluate(state, phase, turn, n_edges);
        if self.blend > 0.0 {
            let h = HeuristicEvaluator.evaluate(state, phase, turn, 0);
            let (a, b) = (1.0 - self.blend, self.blend);
            for i in 0..N_PLAYERS {
                e.value[i] = a * e.value[i] + b * h.value[i];
            }
        }
        e
    }
    fn name(&self) -> String {
        format!("blend{}", self.blend)
    }
}

struct Acc {
    name: String,
    /// Sum of squared error against `z_rel`, and the count, over every seat.
    se: f64,
    n: f64,
    /// Pearson accumulators over (prediction, target) across all seats.
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
    /// Does the evaluator's best seat match the game's actual best seat?
    top1: f64,
    games: f64,
    /// Per-day-bucket squared error: 0-8, 9-17, 18-26.
    bucket_se: [f64; 3],
    bucket_n: [f64; 3],
}

impl Acc {
    fn new(name: impl Into<String>) -> Acc {
        Acc {
            name: name.into(),
            se: 0.0,
            n: 0.0,
            sx: 0.0,
            sy: 0.0,
            sxx: 0.0,
            syy: 0.0,
            sxy: 0.0,
            top1: 0.0,
            games: 0.0,
            bucket_se: [0.0; 3],
            bucket_n: [0.0; 3],
        }
    }

    fn add(&mut self, pred: &[f32; N_PLAYERS], target: &[f32; N_PLAYERS], day: usize) {
        let b = (day / 9).min(2);
        for i in 0..N_PLAYERS {
            let (x, y) = (pred[i] as f64, target[i] as f64);
            let d = x - y;
            self.se += d * d;
            self.n += 1.0;
            self.sx += x;
            self.sy += y;
            self.sxx += x * x;
            self.syy += y * y;
            self.sxy += x * y;
            self.bucket_se[b] += d * d;
            self.bucket_n[b] += 1.0;
        }
        let am = |v: &[f32; N_PLAYERS]| {
            (0..N_PLAYERS).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
        };
        if am(pred) == am(target) {
            self.top1 += 1.0;
        }
        self.games += 1.0;
    }

    fn report(&self) -> String {
        let rmse = (self.se / self.n).sqrt();
        let cov = self.sxy / self.n - (self.sx / self.n) * (self.sy / self.n);
        let vx = (self.sxx / self.n - (self.sx / self.n).powi(2)).max(0.0);
        let vy = (self.syy / self.n - (self.sy / self.n).powi(2)).max(0.0);
        let r = if vx > 0.0 && vy > 0.0 {
            cov / (vx.sqrt() * vy.sqrt())
        } else {
            f64::NAN
        };
        let b: Vec<String> = (0..3)
            .map(|i| {
                if self.bucket_n[i] > 0.0 {
                    format!("{:.4}", (self.bucket_se[i] / self.bucket_n[i]).sqrt())
                } else {
                    "  -   ".into()
                }
            })
            .collect();
        // Slope of the target regressed on the prediction. >1 means the
        // evaluator is *compressed*: it is right about the ordering and
        // under-confident about the size, which a search reads as a flatter
        // landscape at fixed `c_puct`.
        let slope = if vx > 0.0 { cov / vx } else { f64::NAN };
        format!(
            "  {:<38} rmse {:.4}  r {:+.4}  top1 {:.3}  sd {:.3} (tgt {:.3}) slope {:.2}  by day {} {} {}",
            self.name,
            rmse,
            r,
            self.top1 / self.games,
            vx.sqrt(),
            vy.sqrt(),
            slope,
            b[0],
            b[1],
            b[2],
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" {
        eprintln!("valprobe SHARD.tzr [N] [EVAL ...]   EVAL: heuristic | PATH.safetensors");
        std::process::exit(2);
    }
    let path = &args[0];
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let specs: Vec<String> = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        vec!["heuristic".into()]
    };

    let mut evals: Vec<(String, Box<dyn Evaluator>)> = Vec::new();
    for s in &specs {
        if s == "heuristic" {
            evals.push(("heuristic (eval.rs, today)".into(), Box::new(HeuristicEvaluator)));
            continue;
        }
        // `PATH@0.35` is the same spelling `parse_backend` takes: mix
        // `eval::heuristic` into the value at that weight. Sweeping the knob
        // here costs seconds; sweeping it in the arena costs core-hours, and
        // `@1.0` must reproduce the `heuristic` row exactly, which is the
        // control on the arithmetic.
        let (p, blend) = match s.rsplit_once('@') {
            Some((p, l)) if l.parse::<f32>().is_ok() => (p, l.parse::<f32>().unwrap()),
            _ => (s.as_str(), 0.0),
        };
        match Net::load(p) {
            Ok(net) => evals.push((s.clone(), Box::new(Blended { net, blend }))),
            Err(e) => {
                eprintln!("cannot load {p}: {e}");
                std::process::exit(1);
            }
        }
    }

    let mut f = File::open(path).unwrap_or_else(|e| {
        eprintln!("cannot open {path}: {e}");
        std::process::exit(1)
    });
    let size = f.metadata().unwrap().len() as usize;
    let total = (size - HEADER_BYTES) / RECORD_BYTES;
    // `VALPROBE_FROM=0.98` probes only the tail `train.py --holdout` reserves.
    // Records are written a whole game at a time, so a contiguous tail is a set
    // of whole games no optimiser step ever drew from -- which is the
    // difference between measuring generalisation and measuring memorisation.
    let from: f64 = std::env::var("VALPROBE_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let start = ((total as f64) * from) as usize;
    let span = total - start;
    let stride = (span / n.max(1)).max(1);
    println!(
        "{path}\n  {total} records, probing [{start}, {total}) every {stride} -> {} probed",
        span / stride
    );

    let mut accs: Vec<Acc> = evals.iter().map(|(n, _)| Acc::new(n.clone())).collect();
    let mut stored = Acc::new("root_value stored in the record");

    let mut buf = [0u8; RECORD_BYTES];
    let mut probed = 0usize;
    let mut i = start;
    while i < total {
        f.seek(SeekFrom::Start((HEADER_BYTES + i * RECORD_BYTES) as u64))
            .unwrap();
        if f.read_exact(&mut buf).is_err() {
            break;
        }
        i += stride;

        let r = read_record(&buf);
        let mean = r.scores.iter().map(|&s| s as f32).sum::<f32>() / N_PLAYERS as f32;
        let target: [f32; N_PLAYERS] =
            std::array::from_fn(|k| ((r.scores[k] as f32 - mean) / 25.0).tanh());
        let day = u16::from_le_bytes([buf[O_DAY], buf[O_DAY + 1]]) as usize;

        let rv: [f32; N_PLAYERS] = std::array::from_fn(|k| {
            f32::from_le_bytes(
                buf[O_ROOT_VALUE + k * 4..O_ROOT_VALUE + k * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        });
        stored.add(&rv, &target, day);

        for (a, (_, e)) in accs.iter_mut().zip(evals.iter()) {
            let ev = e.evaluate(&r.state, r.phase, PlayerId(r.turn.0), 0);
            a.add(&ev.value, &target, day);
        }
        probed += 1;
    }

    println!("  probed {probed}\n");
    println!("  {:<38} {}", "", "rmse against z_rel, lower better");
    for a in &accs {
        println!("{}", a.report());
    }
    println!("{}", stored.report());
    println!(
        "\n  Day buckets are 0-8 / 9-17 / 18-26. `top1` is the share of positions\n  \
         where the evaluator's best-placed seat is the one that actually won."
    );

    if std::env::var("VALPROBE_PHASES").is_ok() {
        phase_sweep(path, &evals, start, total, stride.max(1) * 20);
    }
}

/// How much does an evaluator's value move when only the *phase* changes?
///
/// The buffer holds two phase tags of eight (N3). A search evaluates all eight.
/// `eval::heuristic` ignores the phase entirely, so its answer is 0 by
/// construction; a net's answer is the size of the extrapolation it is being
/// asked to make on every `Placing`, `Take` and `PickWorker` leaf.
fn phase_sweep(
    path: &str,
    evals: &[(String, Box<dyn Evaluator>)],
    start: usize,
    total: usize,
    stride: usize,
) {
    use tzolkin::phase::Phase;
    let probes = [
        ("Beg", Phase::Beg),
        ("Mode", Phase::Mode),
        ("Placing{0}", Phase::Placing { n: 0 }),
        ("Placing{2}", Phase::Placing { n: 2 }),
        ("PickWorker", Phase::PickWorker),
        ("Take{w0}", Phase::Take { worker: tzolkin::ids::WorkerId(0) }),
        ("PityPlace", Phase::PityPlace),
    ];
    let mut f = File::open(path).unwrap();
    let mut buf = [0u8; RECORD_BYTES];
    let mut sums = vec![vec![0.0f64; probes.len()]; evals.len()];
    let mut base = vec![0.0f64; evals.len()];
    let mut n = 0.0f64;
    let mut i = start;
    while i < total {
        f.seek(SeekFrom::Start((HEADER_BYTES + i * RECORD_BYTES) as u64))
            .unwrap();
        if f.read_exact(&mut buf).is_err() {
            break;
        }
        i += stride;
        let r = read_record(&buf);
        if r.phase.tag() != 1 {
            continue;
        }
        for (e, (_, ev)) in evals.iter().enumerate() {
            let m = ev.evaluate(&r.state, Phase::Mode, r.turn, 0).value;
            base[e] += m[r.turn.0 as usize] as f64;
            for (k, (_, ph)) in probes.iter().enumerate() {
                let v = ev.evaluate(&r.state, *ph, r.turn, 0).value;
                sums[e][k] += (v[r.turn.0 as usize] - m[r.turn.0 as usize]).abs() as f64;
            }
        }
        n += 1.0;
    }
    println!("\n  Phase sensitivity: mean |value(phase) - value(Mode)| for the turn holder,");
    println!("  over {n} turn-root positions. Only Mode and ExtraDay are in the buffer.");
    print!("  {:<38}", "");
    for (name, _) in &probes {
        print!(" {:>11}", name);
    }
    println!();
    for (e, (name, _)) in evals.iter().enumerate() {
        let short: String = name.rsplit('/').next().unwrap_or(name).chars().take(37).collect();
        print!("  {:<38}", short);
        for k in 0..probes.len() {
            print!(" {:>11.4}", sums[e][k] / n);
        }
        println!();
    }
}
