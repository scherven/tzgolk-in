//! The bench for `src/plan.rs`: fit the phase weight schedule, then measure it.
//!
//!     cargo run --release --bin planlab -- duel --sched fitted --blocks 500
//!     cargo run --release --bin planlab -- fit --games 400
//!     cargo run --release --bin planlab -- probe
//!
//! # Why this is not `bin/arena.rs`
//!
//! It is the same experiment, run on agents the arena cannot name: an
//! `AgentSpec` string cannot construct a `plan::PlanEvaluator` with a
//! particular focus weight, and `record.rs` belongs to someone else. So the
//! design is copied and the agents are built here.
//!
//! # The design, and why it has to be this one
//!
//! Copied from `arena.rs`'s module docs, which argue it at length. In short:
//!
//! * **Turn order is worth real points.** Seat 0 gets the first-player marker
//!   and pays no corn surcharge on the low spaces. Measuring an agent in one
//!   seat measures the seat. So the unit of work is a **rotation block**: one
//!   seed, played four times, with the candidate in each seat in turn and the
//!   baseline in the other three. Seat advantage cancels inside the block.
//! * **The four games in a block are not independent.** They share a seed, so
//!   the same decks, the same monument row and the same dealt starting tiles —
//!   which is the point, it is a matched design — but treating them as four
//!   independent observations would shrink the interval by up to a factor of
//!   two. So **the block is the unit**: `n` in every interval below is the
//!   number of blocks, and each block contributes the mean of its four games.
//! * **Mean centred score**, `score[candidate] - mean(all four)`. Null is 0.
//!   With four players, win counts are far noisier than margins.
//!
//! Scale for reading the output: `heuristic:full` beats `heuristic:32` by
//! +4.76 centred, and the last structural change to `eval.rs` (charging a
//! worker for the rounds it waits) was worth +2.02 (+1.21..+2.83, 500 blocks).
//! **Never call an effect smaller than its interval an improvement.**
//!
//! # Interruption
//!
//! Blocks append to a JSONL as they finish and `--resume` skips seeds already
//! on file, so a run that is stopped, or that is still going when the answer is
//! wanted, still answers the question with a wider interval. Ctrl-C stops
//! taking new blocks and prints the summary over everything on disk.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rand::SeedableRng;
use rayon::prelude::*;

use tzolkin::eval::{components, Components};
use tzolkin::ids::*;
use tzolkin::mcts::{Mcts, MctsConfig, PriorBias, Priors};
use tzolkin::phase::{Evaluator, HeuristicEvaluator, Phase, Step};
use tzolkin::plan::{
    self, BiasWeights, DraftMode, OnePlyBias, PlanAgent, PlanBias, PlanEvaluator, PlanWeights,
    temple_target, PriceSource, PricedBias, Schedule,
};
use tzolkin::record::{
    self, flags, interrupt, play_game, Agent, Candidates, GameConfig, GreedyAgent, Summary,
    TurnOutcome,
};
use tzolkin::state::{GameState, LAST_DAY};
use tzolkin::tree;

// =======================================================================
// Agents
// =======================================================================

/// How many sampled turns each side chooses between.
///
/// 32 is the arena's own default rung and is where the published reference
/// numbers were measured, so a result here is comparable with them. Both sides
/// always get the same `k`: this measures an evaluator, not a budget.
const DEFAULT_K: usize = 32;

fn baseline(k: usize) -> GreedyAgent<HeuristicEvaluator> {
    GreedyAgent {
        ev: HeuristicEvaluator,
        cands: Candidates::Sampled(k),
        record: false,
    }
}

// =======================================================================
// The MCTS harness
// =======================================================================

/// A four-arm `Mcts` agent: the evaluator and the prior bias are chosen
/// independently, which is the whole design of the experiment.
///
/// # Why this is not `record::SearchAgent`
///
/// `SearchAgent` never calls `Mcts::set_bias`, so the `PriorBias` seam
/// `mcts.rs` documents as "where `src/plan.rs` plugs in" is unreachable through
/// it. This is `SearchAgent::play_turn` copied, plus the one line that installs
/// the bias.
///
/// It also fixes the draft. `SearchAgent` inherits `Agent`'s default, which is
/// a uniform pick, and a random draft on both sides is symmetric but adds deal
/// variance to every block for nothing. Both sides here take `best_pair` over
/// `eval::heuristic`, so the draft is identical whatever arm is running and
/// cannot carry the effect.
struct MctsAgent {
    mcts: std::sync::Mutex<Mcts<record::SharedEval>>,
    sims: u32,
    label: String,
}

/// A turn is 5-8 sub-decisions; the guard is `record.rs`'s.
const MAX_SUB_DECISIONS: usize = 64;

impl MctsAgent {
    fn new(
        ev: std::sync::Arc<dyn Evaluator>,
        bias: Option<std::sync::Arc<dyn PriorBias>>,
        sims: u32,
        cfg: MctsConfig,
        seed: u64,
    ) -> MctsAgent {
        // Evaluation games, not self-play: no root noise, every turn at the
        // full budget. Anything else and the two arms differ by their noise
        // draws as much as by their priors.
        let cfg = MctsConfig {
            seed,
            dirichlet_eps: 0.0,
            ..cfg
        };
        let label = format!(
            "mcts{sims}/{}{}",
            ev.name(),
            match (&bias, cfg.priors) {
                (Some(b), Priors::OnePly) => format!("+{}+1ply", b.name()),
                (Some(b), _) => format!("+{}", b.name()),
                (None, Priors::OnePly) => "+1ply".into(),
                (None, _) => String::new(),
            }
        );
        let mut m = Mcts::new(record::SharedEval(ev), cfg);
        m.set_bias(bias);
        MctsAgent {
            mcts: std::sync::Mutex::new(m),
            sims: sims.max(1),
            label,
        }
    }
}

impl Agent for MctsAgent {
    fn play_turn(
        &self,
        g: &GameState,
        p: PlayerId,
        temp: f32,
        _rng: &mut rand::rngs::StdRng,
    ) -> Option<TurnOutcome> {
        let mut m = self.mcts.lock().unwrap();
        m.config_mut().temperature = temp;
        let mut probe = *g;
        let mut at = (Phase::Beg, p, 0u8);
        let mut path: Vec<Step> = Vec::new();
        for _ in 0..MAX_SUB_DECISIONS {
            let (phase, turn, done) = at;
            let r = m.search_at(&probe, phase, turn, done, self.sims);
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
        tree::retag_workers(g, p, &mut mv);
        Some(TurnOutcome { mv, nodes: Vec::new() })
    }

    fn extra_day(
        &self,
        g: &GameState,
        p: PlayerId,
        _rng: &mut rand::rngs::StdRng,
    ) -> (bool, Option<record::Node>) {
        let mut m = self.mcts.lock().unwrap();
        m.config_mut().temperature = 0.0;
        let r = m.search_at(g, Phase::ExtraDay { claimer: p }, p, 0, self.sims);
        (matches!(r.step, Step::ExtraDay(true)), None)
    }

    fn draft(
        &self,
        g: &GameState,
        p: PlayerId,
        dealt: [u8; 4],
        _rng: &mut rand::rngs::StdRng,
    ) -> [u8; 2] {
        // Deliberately *not* the candidate's own evaluator: the draft is a
        // separate question with its own `DraftMode` sweep, and letting it move
        // between arms here would put a draft effect inside a priors number.
        record::best_pair(g, p, dealt, |s| tzolkin::eval::heuristic(s, p))
    }

    fn name(&self) -> String {
        self.label.clone()
    }
}

// =======================================================================
// One block
// =======================================================================

#[derive(Clone, Copy, Debug)]
struct Outcome {
    centred: f64,
    vs_base: f64,
    win: f64,
    cand_score: f64,
    days: f64,
}

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
        format!(
            r#"{{"seed":{},"games":{},"centred":{:.4},"vs_base":{:.4},"win":{:.4},"cand":{:.3},"days":{:.2}}}"#,
            self.seed,
            self.games.len(),
            self.mean(|o| o.centred),
            self.mean(|o| o.vs_base),
            self.mean(|o| o.win),
            self.mean(|o| o.cand_score),
            self.mean(|o| o.days),
        )
    }
}

fn block_from_json(line: &str) -> Option<Block> {
    let num = |key: &str| -> Option<f64> {
        let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
        let rest = &line[at..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    let o = Outcome {
        centred: num("centred")?,
        vs_base: num("vs_base")?,
        win: num("win")?,
        cand_score: num("cand")?,
        days: num("days").unwrap_or(0.0),
    };
    // Stored as the block mean; rebuilt as `n` identical games so the block
    // mean survives a resume exactly and the block count stays right.
    Some(Block {
        seed: num("seed")? as u64,
        games: vec![o; (num("games")? as usize).max(1)],
    })
}

/// One rotation block: the same seed, the candidate in each of the four seats.
fn play_block(seed: u64, cand: &dyn Agent, base: &dyn Agent, cfg: &GameConfig) -> Block {
    let mut games = Vec::new();
    for c in 0..N_PLAYERS {
        let agents: [&dyn Agent; N_PLAYERS] =
            std::array::from_fn(|s| if s == c { cand } else { base });
        // The seed fixes the deal; the play RNG is offset per seating so the
        // four games of a block are not one game played four times.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9) ^ c as u64);
        let r = play_game(seed, &agents, cfg, &mut rng);

        let scores: [f64; N_PLAYERS] = std::array::from_fn(|s| r.scores[s] as f64);
        let table_mean = scores.iter().sum::<f64>() / N_PLAYERS as f64;
        let base_mean =
            (scores.iter().sum::<f64>() - scores[c]) / (N_PLAYERS as f64 - 1.0);
        games.push(Outcome {
            centred: scores[c] - table_mean,
            vs_base: scores[c] - base_mean,
            win: r.win_share[c] as f64,
            cand_score: scores[c],
            days: r.days as f64,
        });
    }
    Block { seed, games }
}

// =======================================================================
// duel
// =======================================================================

fn read_progress(path: &Path) -> Vec<Block> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(block_from_json)
        .collect()
}

fn report(label: &str, blocks: &[Block]) {
    if blocks.is_empty() {
        println!("{label}: no blocks");
        return;
    }
    let col = |f: fn(&Outcome) -> f64| -> Vec<f64> { blocks.iter().map(|b| b.mean(f)).collect() };
    let centred = Summary::of(&col(|o| o.centred));
    let vs_base = Summary::of(&col(|o| o.vs_base));
    let win = Summary::of(&col(|o| o.win));
    let score = Summary::of(&col(|o| o.cand_score));
    let days = Summary::of(&col(|o| o.days));

    println!("\n=== {label} ===");
    println!(
        "  blocks {}  ({} games)                        [the block is the independent unit]",
        centred.n,
        centred.n * blocks[0].games.len()
    );
    println!(
        "  centred score   {:+7.2}  (95% CI {:+.2}..{:+.2})   <- the number. null 0",
        centred.mean,
        centred.mean - centred.ci,
        centred.mean + centred.ci
    );
    println!(
        "  vs baseline     {:+7.2}  (95% CI {:+.2}..{:+.2})",
        vs_base.mean,
        vs_base.mean - vs_base.ci,
        vs_base.mean + vs_base.ci
    );
    println!(
        "  win rate         {:6.3}  (95% CI {:.3}..{:.3})   null 0.25",
        win.mean,
        win.mean - win.ci,
        win.mean + win.ci
    );
    println!(
        "  candidate score  {:6.2}      mean game length {:.2} days",
        score.mean, days.mean
    );
    let verdict = if centred.mean.abs() <= centred.ci {
        "indistinguishable from the baseline at this sample size"
    } else if centred.mean > 0.0 {
        "better than the baseline"
    } else {
        "WORSE than the baseline"
    };
    println!("  verdict: {verdict}  (p = {:.4})", centred.p_value(0.0));
}

fn duel(args: &Args) {
    let label = args.label();
    let cfg = GameConfig::evaluation();

    let mut done: Vec<Block> = if args.resume {
        read_progress(&args.out)
    } else {
        Vec::new()
    };
    let already: std::collections::HashSet<u64> = done.iter().map(|b| b.seed).collect();
    let todo: Vec<u64> = (0..args.blocks as u64)
        .map(|i| args.seed0 + i)
        .filter(|s| !already.contains(s))
        .collect();

    println!("planlab duel");
    if args.sims > 0 {
        println!("  match     : {label}");
    } else {
        println!("  candidate : {label} (greedy:{})", args.k);
        println!("  baseline  : heuristic:{}", args.k);
    }
    println!(
        "  plan      : {} blocks x 4 games ({} already on file)",
        todo.len(),
        done.len()
    );
    println!("  progress  : {}", args.out.display());
    println!("  schedule  :\n{}", args.sched().to_source());

    let progress = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.out)
            .expect("cannot open progress file"),
    );
    let collected: Mutex<Vec<Block>> = Mutex::new(Vec::new());
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let t0 = std::time::Instant::now();

    todo.par_iter().for_each(|&seed| {
        if interrupt::stopping() {
            return;
        }
        // Fresh agent instances per task. They are stateless here, but the
        // arena builds them per task because a searching agent owns a tree, and
        // copying that discipline keeps this harness swappable for one.
        let (cand, base) = args.agents(seed);
        let b = play_block(seed, cand.as_ref(), base.as_ref(), &cfg);
        {
            let mut f = progress.lock().unwrap();
            let _ = writeln!(f, "{}", b.to_json());
            let _ = f.flush();
        }
        collected.lock().unwrap().push(b);
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 25 == 0 {
            let per = t0.elapsed().as_secs_f64() / n as f64;
            eprintln!(
                "  {n}/{} blocks  {:.2}s/block  {:.0}s left",
                todo.len(),
                per,
                per * (todo.len() - n) as f64
            );
        }
    });

    done.extend(collected.into_inner().unwrap());
    report(&label, &done);
}

/// Paired difference between two finished duels, joined on block seed.
///
/// Both runs measure a candidate against the same three baselines on the same
/// seeds, so the block seed is a matched pair and the difference of the two
/// centred scores has most of the deal variance removed. Comparing the two
/// unpaired intervals instead throws that away and can easily call a real 1-2
/// point difference "overlapping".
///
/// Null is 0: `a` is better than `b` only if the whole interval is above it.
fn compare(a: &Path, b: &Path) {
    let ma: std::collections::HashMap<u64, f64> = read_progress(a)
        .iter()
        .map(|x| (x.seed, x.mean(|o| o.centred)))
        .collect();
    let pairs: Vec<f64> = read_progress(b)
        .iter()
        .filter_map(|x| ma.get(&x.seed).map(|&v| v - x.mean(|o| o.centred)))
        .collect();
    let s = Summary::of(&pairs);
    println!(
        "{} - {}: {:+.2} (95% CI {:+.2}..{:+.2}), {} matched blocks, p = {:.4}",
        a.display(),
        b.display(),
        s.mean,
        s.mean - s.ci,
        s.mean + s.ci,
        s.n,
        s.p_value(0.0),
    );
}

// =======================================================================
// fit
// =======================================================================

/// One recorded turn root, already centred across the four seats of the same
/// position.
///
/// Centring is the whole methodology. Two thirds of the variance in a raw
/// component is the calendar — `engine` is mostly `rounds_left` — and the
/// calendar says nothing about who wins. What survives centring is exactly the
/// quantity a search is ranking, so a coefficient here says what the term is
/// worth as a *discriminator* rather than as a level.
struct Row {
    day: u8,
    /// `terms[seat][k]`, centred across seats.
    x: [[f64; 8]; N_PLAYERS],
    /// Final score, centred across seats.
    y: [f64; N_PLAYERS],
}

/// The hat basis for the schedule's linear interpolation: `B[k](day)` is 1 at
/// knot `k`, 0 at every other knot, linear in between and summing to 1.
///
/// Fitting in this basis rather than in day buckets means the fit optimises the
/// weights the evaluator will actually use, instead of producing bucket means
/// that then have to be reconciled with the interpolation.
fn hat(day: u8) -> [f64; plan::KNOTS.len()] {
    let mut b = [0.0; plan::KNOTS.len()];
    let d = day.min(LAST_DAY);
    let mut k = 0;
    while k + 2 < plan::KNOTS.len() && d >= plan::KNOTS[k + 1] {
        k += 1;
    }
    let (a, c) = (plan::KNOTS[k], plan::KNOTS[k + 1]);
    let t = ((d.saturating_sub(a)) as f64 / (c - a) as f64).clamp(0.0, 1.0);
    b[k] = 1.0 - t;
    b[k + 1] = t;
    b
}

/// Solve `(A + lambda I) u = b` by Gaussian elimination with partial pivoting.
/// 30 unknowns, so nothing cleverer is warranted.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        for r in i + 1..n {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        if a[i][i].abs() < 1e-12 {
            continue;
        }
        for r in i + 1..n {
            let f = a[r][i] / a[i][i];
            if f == 0.0 {
                continue;
            }
            for c in i..n {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-12 {
            continue;
        }
        let mut s = b[i];
        for c in i + 1..n {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    x
}

/// The six terms that are fitted. `banked` and `liquidation` are exact and are
/// pinned at 1.0 — see `plan.rs`'s module docs for why, and note that pinning
/// them is also what makes `held` identifiable, since `held` and `liquidation`
/// are near-linear in the same block count.
const FREE: [usize; 6] = [
    plan::HELD,
    plan::TEMPLE,
    plan::ENGINE,
    plan::BOARD,
    plan::MONUMENT,
    plan::STARVATION,
];

fn fit(args: &Args) {
    let ev = args.evaluator();
    let cfg = GameConfig::evaluation();

    println!("planlab fit");
    println!("  policy  : {} (greedy:{})", args.label(), args.k);
    println!("  games   : {}", args.games);
    println!(
        "  note    : the fit is on-policy for *this* agent. Weights fitted on\n\
         \x20           one policy and used by another are off-distribution, so a\n\
         \x20           fitted schedule wants one refit round before it is trusted."
    );

    let rows: Vec<Row> = (0..args.games as u64)
        .into_par_iter()
        .flat_map(|seed| {
            if interrupt::stopping() {
                return Vec::new();
            }
            // All four seats on the same policy: the fit wants the state
            // distribution this agent produces, not a mixture of two.
            let a = GreedyAgent {
                ev: ev(),
                cands: Candidates::Sampled(args.k),
                record: true,
            };
            let agents: [&dyn Agent; N_PLAYERS] = [&a, &a, &a, &a];
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x9114_7);
            let r = play_game(args.seed0 + seed, &agents, &cfg, &mut rng);
            let out: [f64; N_PLAYERS] = std::array::from_fn(|i| r.scores[i] as f64);
            let ybar = out.iter().sum::<f64>() / N_PLAYERS as f64;
            r.nodes
                .iter()
                .filter(|n| n.flags & flags::TURN_ROOT != 0 && !n.state.over)
                .map(|n| {
                    let parts: [Components; N_PLAYERS] =
                        std::array::from_fn(|i| components(&n.state, PlayerId(i as u8)));
                    let t: [[f64; 8]; N_PLAYERS] =
                        std::array::from_fn(|i| parts[i].terms().map(|v| v as f64));
                    let tbar: [f64; 8] = std::array::from_fn(|k| {
                        (0..N_PLAYERS).map(|i| t[i][k]).sum::<f64>() / N_PLAYERS as f64
                    });
                    Row {
                        day: n.state.day,
                        x: std::array::from_fn(|i| std::array::from_fn(|k| t[i][k] - tbar[k])),
                        y: std::array::from_fn(|i| out[i] - ybar),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    println!("  rows    : {} positions", rows.len());
    if rows.is_empty() {
        return;
    }

    // Design: one column per (knot, free term), the column being
    // `hat_k(day) * centred_term`. The pinned terms enter as a fixed offset.
    // Ridge shrinks toward **1.0**, not toward 0: the prior is that `eval` is
    // roughly right and the fit is being asked where it is wrong, so a term the
    // data cannot speak to keeps its face value instead of vanishing.
    let ncol = plan::KNOTS.len() * FREE.len();
    let mut ata = vec![vec![0.0f64; ncol]; ncol];
    let mut aty = vec![0.0f64; ncol];
    let mut n_obs = 0usize;
    let mut col = vec![0.0f64; ncol];

    for r in &rows {
        let b = hat(r.day);
        for s in 0..N_PLAYERS {
            for (j, &t) in FREE.iter().enumerate() {
                for k in 0..plan::KNOTS.len() {
                    col[k * FREE.len() + j] = b[k] * r.x[s][t];
                }
            }
            // The residual after the pinned terms and the identity prior on the
            // free ones: what is left for `u = w - 1` to explain.
            let mut resid = r.y[s] - r.x[s][plan::BANKED] - r.x[s][plan::LIQUIDATION];
            for &t in FREE.iter() {
                resid -= r.x[s][t];
            }
            for i in 0..ncol {
                if col[i] == 0.0 {
                    continue;
                }
                aty[i] += col[i] * resid;
                for j in 0..ncol {
                    if col[j] != 0.0 {
                        ata[i][j] += col[i] * col[j];
                    }
                }
            }
            n_obs += 1;
        }
    }
    // Scaled with the sample so the prior's strength does not depend on how
    // many games were played.
    let lambda = 0.02 * n_obs as f64;
    for i in 0..ncol {
        ata[i][i] += lambda;
    }
    let u = solve(ata, aty);

    let mut s = Schedule::IDENTITY;
    for k in 0..plan::KNOTS.len() {
        for (j, &t) in FREE.iter().enumerate() {
            s.w[k][t] = 1.0 + u[k * FREE.len() + j] as f32;
        }
    }

    println!("\n-- raw fit (before clamping) --");
    println!("{}", s.to_source());
    println!("\n-- sanitised: pinned exact terms, clamped to [0, {MAX}] --", MAX = plan::MAX_W);
    println!("{}", s.sanitise().to_source());
    println!(
        "\nterm order: {:?}",
        Components::NAMES
    );
}

// =======================================================================
// probe
// =======================================================================

/// What the two layers actually do to a position, printed for a handful of
/// real games. Cheap sanity: a term schedule that reads well in a table can
/// still be doing nothing, or everything, on the board.
fn probe(args: &Args) {
    let base = baseline(args.k);
    let cfg = GameConfig::evaluation();
    let agents: [&dyn Agent; N_PLAYERS] = [&base, &base, &base, &base];
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let a = GreedyAgent {
        ev: HeuristicEvaluator,
        cands: Candidates::Sampled(args.k),
        record: true,
    };
    let rec: [&dyn Agent; N_PLAYERS] = [&a, &a, &a, &a];
    let _ = agents;
    let r = play_game(args.seed0, &rec, &cfg, &mut rng);

    let ev = args.evaluator()();
    let flat = PlanEvaluator::identity();

    println!(
        "{:>4}  {:>7}  {:>7}  {:>7}  {:>6}  {:>6}  {:>6}  {}",
        "day", "eval", "phased", "delta", "reach", "focus", "food", "leading line"
    );
    let mut last = 255u8;
    for n in r
        .nodes
        .iter()
        .filter(|n| n.flags & flags::TURN_ROOT != 0 && !n.state.over)
    {
        if n.state.day == last {
            continue;
        }
        last = n.state.day;
        let g: &GameState = &n.state;
        let p = n.turn;
        let a = flat.raw(g, p);
        let b = ev.raw(g, p);
        let lead = plan::Line::ALL
            .iter()
            .map(|&l| (l, plan::line_progress(g, p, l)))
            .fold((plan::Line::Skulls, f32::NEG_INFINITY), |acc, x| {
                if x.1 > acc.1 {
                    x
                } else {
                    acc
                }
            });
        println!(
            "{:>4}  {:>7.1}  {:>7.1}  {:>+7.1}  {:>6.2}  {:>6.2}  {:>6}  {:?} {:.1}",
            g.day,
            a,
            b,
            b - a,
            plan::conversion_reach(g, p),
            plan::focus(g, p, 1.0),
            plan::days_to_food(g).map(|d| d as i32).unwrap_or(-1),
            lead.0,
            lead.1,
        );
    }
}

/// What each of the 21 starting tiles is worth to the day-0 evaluator, under
/// the identity weights and under the schedule.
///
/// The draft is the one decision in the game taken with no board at all, so it
/// is the sharpest place to see what the weights did: `engine` at 0 on day 0
/// means a free worker earns no credit at all for the actions it will take, and
/// this table says whether that is survivable.
fn tiles(args: &Args) {
    use tzolkin::data::tiles::TILES;
    let g = tzolkin::game::Game::new(1).state;
    let p = PlayerId(0);
    let flat = PlanEvaluator::identity();
    let ev = args.evaluator()();

    let delta = |e: &PlanEvaluator, id: usize| -> f32 {
        let mut probe = g;
        for eff in TILES[id] {
            eff.apply(&mut probe, p);
        }
        e.raw(&probe, p) - e.raw(&g, p)
    };
    let mut rows: Vec<(usize, f32, f32, f32)> = (0..TILES.len())
        .map(|i| {
            let (a, b) = (delta(&flat, i), delta(&ev, i));
            (i, a, b, plan::draft_value(&{
                let mut probe = g;
                for eff in TILES[i] {
                    eff.apply(&mut probe, p);
                }
                probe
            }, p) - plan::draft_value(&g, p))
        })
        .collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "{:>5}  {:>7}  {:>7}  {:>7}   effects",
        "tile", "eval", "phased", "prior"
    );
    for (i, a, b, c) in rows {
        println!("{:>5}  {:>7.2}  {:>7.2}  {:>7.2}   {:?}", i + 1, a, b, c, TILES[i]);
    }
}

// =======================================================================
// priors: does the bias reach `edge.prior`, and what does it cost
// =======================================================================

/// Every node on the played path of one game, with its real edge list.
///
/// The population is nodes *on the path*, not nodes the search expands. It
/// under-samples the deep, narrow ones a descent creates and over-samples the
/// wide root-adjacent ones — which is the conservative direction for a cost
/// number, because the wide nodes are exactly where a per-edge prior hurts.
fn walk_nodes(seed0: u64, games: usize, cap: usize) -> Vec<(GameState, Phase, PlayerId, Vec<Step>)> {
    let mut out = Vec::new();
    for i in 0..games {
        walk_one(seed0 + i as u64, cap, &mut out);
        if out.len() >= cap {
            break;
        }
    }
    out
}

fn walk_one(seed: u64, cap: usize, out: &mut Vec<(GameState, Phase, PlayerId, Vec<Step>)>) {
    // A recorded greedy game gives the turn roots; the sub-decision chain under
    // each of them is walked here. `record: true` is what makes `play_game`
    // hand back the nodes at all.
    let a = GreedyAgent {
        ev: HeuristicEvaluator,
        cands: Candidates::Sampled(16),
        record: true,
    };
    let agents: [&dyn Agent; N_PLAYERS] = [&a, &a, &a, &a];
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let r = play_game(seed, &agents, &GameConfig::evaluation(), &mut rng);

    for n in r.nodes.iter().filter(|n| !n.state.over) {
        let mut probe = n.state;
        let mut at = (n.phase, n.turn, n.done);
        for _ in 0..32 {
            let (phase, turn, done) = at;
            let steps = tree::legal_steps(&probe, phase, turn, done);
            if steps.is_empty() {
                break;
            }
            if steps.len() > 1 {
                out.push((probe, phase, phase.mover(turn), steps.clone()));
                if out.len() >= cap {
                    return;
                }
            }
            // Which continuation this takes only has to be legal: it is a node
            // sampler, not a player.
            let step = steps[0].clone();
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
    }
}

fn priors_probe(args: &Args) {
    let plan_bias = PlanBias {
        w: BiasWeights {
            k: if args.bias == 0.0 { 4.0 } else { args.bias },
            place: args.bias_place,
            blind: args.bias_blind,
            ..BiasWeights::OFF
        },
    };
    let nodes = walk_nodes(args.seed0, args.games.max(1), 200_000);

    // ---- what the bias says, node by node ----------------------------------
    let mut by_phase: std::collections::BTreeMap<&'static str, (usize, usize, usize, f64)> =
        Default::default();
    let mut scratch = Vec::new();
    for (g, phase, mover, steps) in &nodes {
        scratch.clear();
        scratch.resize(steps.len(), 1.0f32);
        plan_bias.bias(g, *phase, *mover, steps, &mut scratch);
        let opinionated = scratch.iter().filter(|&&x| x != 1.0).count();
        let name = match phase {
            Phase::Beg => "Beg",
            Phase::Mode => "Mode",
            Phase::Placing { .. } => "Placing",
            Phase::PickWorker => "PickWorker",
            Phase::Take { .. } => "Take",
            Phase::ExtraDay { .. } => "ExtraDay",
            Phase::PityPlace => "Pity",
            Phase::DraftTile { .. } => "Draft",
        };
        let e = by_phase.entry(name).or_default();
        e.0 += 1;
        e.1 += steps.len();
        e.2 += if opinionated > 0 { 1 } else { 0 };
        e.3 += opinionated as f64;
    }
    println!("\n=== what the bias touches ({} nodes, seed {}) ===", nodes.len(), args.seed0);
    println!("  {:<11} {:>7} {:>9} {:>10} {:>12}", "phase", "nodes", "mean w", "w/ opinion", "edges moved");
    for (name, (n, w, op, moved)) in &by_phase {
        println!(
            "  {:<11} {:>7} {:>9.1} {:>9.1}% {:>11.1}%",
            name,
            n,
            *w as f64 / *n as f64,
            100.0 * *op as f64 / *n as f64,
            100.0 * moved / *w as f64
        );
    }

    // ---- does it reach `edge.prior`? ---------------------------------------
    //
    // `Edge::prior` is private, so this asks the search instead. At `sims = 1`
    // every Q is the same FPU constant, so PUCT reduces to argmax of the prior
    // and the single visit lands on the highest-prior edge. Run the same node
    // twice, once with the bias and once without, and a changed choice is proof
    // the multiplier reached the edge -- there is no other path by which it
    // could.
    let wide: Vec<_> = nodes
        .iter()
        .filter(|(_, p, _, s)| matches!(p, Phase::Take { .. }) && s.len() >= 8)
        .take(3000)
        .collect();
    let mut changed = 0usize;
    let mut agreed = 0usize;
    let mut leaderward = 0usize;
    for (g, phase, mover, steps) in &wide {
        let mut plain = Mcts::new(HeuristicEvaluator, MctsConfig { dirichlet_eps: 0.0, ..MctsConfig::default() });
        let a = plain.search_at(g, *phase, *mover, 0, 1).step;
        let mut biased = Mcts::new(HeuristicEvaluator, MctsConfig { dirichlet_eps: 0.0, ..MctsConfig::default() });
        biased.set_bias(Some(std::sync::Arc::new(PlanBias { w: plan_bias.w })));
        let b = biased.search_at(g, *phase, *mover, 0, 1).step;
        scratch.clear();
        scratch.resize(steps.len(), 1.0f32);
        plan_bias.bias(g, *phase, *mover, steps, &mut scratch);
        let up = scratch.iter().copied().fold(0.0f32, f32::max);
        let picked_up = steps
            .iter()
            .position(|x| *x == b)
            .map(|i| scratch[i] >= up)
            .unwrap_or(false);
        if a == b {
            agreed += 1;
        } else {
            changed += 1;
        }
        if picked_up && up > 1.0 {
            leaderward += 1;
        }
    }
    println!("\n=== does the bias reach `edge.prior`? ({} wide Take nodes) ===", wide.len());
    println!("  first simulation goes elsewhere with the bias on : {changed} / {}", changed + agreed);
    println!("  and lands on a top-weighted edge                 : {leaderward}");

    // ---- what naming the temple actually changes ---------------------------
    //
    // The claim under test is that `Line::Temples` folding `max` over three
    // tracks throws away the one fact a prior needs. If naming the temple never
    // moved the top-weighted edge, the axis would be inert whatever a duel said.
    let named_bias = PlanBias {
        w: BiasWeights { name_temple: true, ..plan_bias.w },
    };
    let (mut nodes_with_target, mut nodes_named, mut top_moved) = (0usize, 0usize, 0usize);
    let (mut flat_top, mut named_top) = (0usize, 0usize);
    let mut other = Vec::new();
    for (g, phase, mover, steps) in &nodes {
        if temple_target(g, *mover).is_some() {
            nodes_with_target += 1;
        }
        scratch.clear();
        scratch.resize(steps.len(), 1.0f32);
        plan_bias.bias(g, *phase, *mover, steps, &mut scratch);
        other.clear();
        other.resize(steps.len(), 1.0f32);
        named_bias.bias(g, *phase, *mover, steps, &mut other);
        if scratch != other {
            nodes_named += 1;
        }
        let arg = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
        };
        if steps.len() > 1 && arg(&scratch) != arg(&other) {
            top_moved += 1;
        }
    }
    println!("\n=== what naming the temple changes ({} nodes) ===", nodes.len());
    println!("  nodes where `temple_target` names one : {nodes_with_target} ({:.1}%)",
        100.0 * nodes_with_target as f64 / nodes.len() as f64);
    println!("  nodes where the weights differ        : {nodes_named} ({:.1}%)",
        100.0 * nodes_named as f64 / nodes.len() as f64);
    println!("  nodes where the top-weighted edge moves: {top_moved} ({:.1}%)",
        100.0 * top_moved as f64 / nodes.len() as f64);

    // Same question for the priced arms, on the wide nodes where truncation
    // makes the ordering load-bearing.
    let pflat = PricedBias { min_edges: 0, ..PricedBias::new(PriceSource::PlanFlat, args.priced_w) };
    let pnamed = PricedBias { min_edges: 0, ..PricedBias::new(PriceSource::PlanNamed, args.priced_w) };
    let pgrad = PricedBias { min_edges: 0, ..PricedBias::new(PriceSource::Gradient, 0.0) };
    for (label, a, b) in [
        ("named vs gradient", &pgrad, &pnamed),
        ("named vs flat    ", &pflat, &pnamed),
    ] {
        let (mut n, mut moved) = (0usize, 0usize);
        for (g, phase, mover, steps) in &wide {
            scratch.clear();
            scratch.resize(steps.len(), 1.0f32);
            a.bias(g, *phase, *mover, steps, &mut scratch);
            other.clear();
            other.resize(steps.len(), 1.0f32);
            b.bias(g, *phase, *mover, steps, &mut other);
            let arg = |v: &[f32]| {
                v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i)
            };
            n += 1;
            if arg(&scratch) != arg(&other) {
                moved += 1;
            }
        }
        println!("  PricedBias {label}: top edge moves at {moved} / {n} wide Take nodes");
    }
    let _ = (&mut flat_top, &mut named_top);

    // ---- does `EdgeOrder::Gradient` outrank the plan? -----------------------
    //
    // `Mcts::node_for` calls `select_edges` **before** `priors_for`, so at a
    // node past `widen_cap` the search has already priced every edge with
    // `mcts::Gradient` and deleted the tail by the time a `PriorBias` is
    // consulted. A plan can re-rank the survivors; it cannot rescue an edge the
    // gradient dropped. That matters most for exactly the claim this file is
    // testing: a temple step that overtakes nobody moves `temple_points` by
    // zero, so the gradient is at its least informative on the climb a plan is
    // most committed to.
    let cap = MctsConfig::default().widen_cap.max(MctsConfig::default().max_edges);
    let (mut past_cap, mut plan_top_dropped, mut temple_dropped, mut temple_kept) =
        (0usize, 0usize, 0usize, 0usize);
    for (g, phase, mover, steps) in &nodes {
        if steps.len() <= cap {
            continue;
        }
        past_cap += 1;
        let grad = tzolkin::mcts::Gradient::new(g, *mover);
        let mut order: Vec<usize> = (0..steps.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            grad.step(&steps[b]).total_cmp(&grad.step(&steps[a])).then(a.cmp(&b))
        });
        let kept: std::collections::HashSet<usize> = order[..cap].iter().copied().collect();
        scratch.clear();
        scratch.resize(steps.len(), 1.0f32);
        named_bias.bias(g, *phase, *mover, steps, &mut scratch);
        // Not "the argmax survived": `PlanBias` hands every promoted edge the
        // same multiplier, so an argmax over it is a tie broken by position and
        // says nothing. The question that has an answer is whether the cut
        // leaves the plan *anything* to promote.
        let up = scratch.iter().copied().fold(0.0f32, f32::max);
        if up > 1.0 {
            let promoted: Vec<usize> =
                (0..steps.len()).filter(|&i| scratch[i] >= up).collect();
            if !promoted.is_empty() && !promoted.iter().any(|i| kept.contains(i)) {
                plan_top_dropped += 1;
            }
        }
        // And specifically: does the survivor set still contain a step on the
        // temple the plan named?
        if let Some(t) = temple_target(g, *mover) {
            let on_target = |i: &usize| match &steps[*i] {
                Step::Take(c) => c.0.iter().any(|e| {
                    matches!(e, tzolkin::Effect::TempleStep(x, n) if *x == t && *n > 0)
                }),
                _ => false,
            };
            let any = (0..steps.len()).any(|i| on_target(&i));
            if any {
                if kept.iter().any(on_target) {
                    temple_kept += 1;
                } else {
                    temple_dropped += 1;
                }
            }
        }
    }
    println!("\n=== `EdgeOrder::Gradient` vs the plan (cap {cap}) ===");
    println!("  nodes past the cap                          : {past_cap}");
    if past_cap > 0 {
        println!("  every edge the plan promotes is deleted first     : {plan_top_dropped} ({:.1}%)",
            100.0 * plan_top_dropped as f64 / past_cap as f64);
        println!("  a step on the named temple existed and survived   : {temple_kept}");
        println!("  a step on the named temple existed and was deleted: {temple_dropped}");
    }

    // ---- cost --------------------------------------------------------------
    let one_ply = OnePlyBias {
        ev: args.evaluator()(),
        temp: 4.0,
    };
    let plan_ev = args.evaluator()();
    let bench = |name: &str, f: &mut dyn FnMut(&(GameState, Phase, PlayerId, Vec<Step>))| {
        let t0 = std::time::Instant::now();
        let mut reps = 0usize;
        let mut edges = 0usize;
        while t0.elapsed().as_secs_f64() < 1.5 {
            for n in &nodes {
                f(n);
                edges += n.3.len();
            }
            reps += 1;
        }
        let per_node = t0.elapsed().as_secs_f64() * 1e6 / (reps * nodes.len()) as f64;
        let per_edge = t0.elapsed().as_secs_f64() * 1e9 / edges as f64;
        println!("  {name:<34} {per_node:>8.2} us/node  {per_edge:>8.1} ns/edge");
    };
    println!("\n=== cost of one expansion ===");
    let mut buf: Vec<f32> = Vec::new();
    // Both value heads, because the number that matters is the *difference*:
    // `node_for` calls one of these exactly once per expansion whichever arm is
    // running, so the plan's value head costs what it costs over this line and
    // not what it costs absolutely.
    bench("HeuristicEvaluator::evaluate", &mut |(g, p, m, st)| {
        std::hint::black_box(HeuristicEvaluator.evaluate(g, *p, *m, st.len()));
    });
    bench("PlanEvaluator::evaluate (the value)", &mut |(g, p, m, st)| {
        std::hint::black_box(plan_ev.evaluate(g, *p, *m, st.len()));
    });
    bench("PlanBias::bias (the prior)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        plan_bias.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench("OnePlyBias::bias (the baseline)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        one_ply.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    // Both `PricedBias` arms, because the interesting split is that the plan's
    // temple term is free and the *gradient underneath it* is not.
    let grad_bias = PricedBias { min_edges: 0, ..PricedBias::new(PriceSource::Gradient, 0.0) };
    bench("PricedBias::bias (gradient)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        grad_bias.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench("PricedBias::bias (plan-named temple)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        pnamed.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench("plan::temple_target (the naming)", &mut |(g, _p, m, _st)| {
        std::hint::black_box(tzolkin::plan::temple_target(g, *m));
    });

    // Every arm again over `Take` nodes only. `PricedBias` abstains on every
    // other phase -- `EffectPrice` reads a `Choice` and nothing else has one --
    // so averaging it over a walk that is 71% `Mode` and `PickWorker` nodes
    // divides its cost by the nodes it declined to look at. This is the table
    // to quote.
    let take_only: Vec<_> = nodes
        .iter()
        .filter(|(_, p, _, _)| matches!(p, Phase::Take { .. }))
        .cloned()
        .collect();
    let bench2 = |name: &str, f: &mut dyn FnMut(&(GameState, Phase, PlayerId, Vec<Step>))| {
        let t0 = std::time::Instant::now();
        let (mut reps, mut edges) = (0usize, 0usize);
        while t0.elapsed().as_secs_f64() < 1.5 {
            for n in &take_only {
                f(n);
                edges += n.3.len();
            }
            reps += 1;
        }
        println!(
            "  {name:<34} {:>8.2} us/node  {:>8.1} ns/edge",
            t0.elapsed().as_secs_f64() * 1e6 / (reps * take_only.len()) as f64,
            t0.elapsed().as_secs_f64() * 1e9 / edges as f64
        );
    };
    println!("\n=== cost on `Take` nodes only ({} of {}) ===", take_only.len(), nodes.len());
    bench2("PlanBias::bias", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        plan_bias.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench2("PlanBias::bias (+named temple)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        named_bias.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    let shuf_bias = PlanBias {
        w: BiasWeights { shuffled: true, ..plan_bias.w },
    };
    bench2("PlanBias::bias (+shuffled control)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        shuf_bias.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench2("PricedBias::bias (gradient)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        grad_bias.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench2("PricedBias::bias (plan-named)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        pnamed.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    // The same 17 `eval::heuristic` probes `EdgeOrder::Gradient` pays -- but
    // the search caches them per mover per sub-decision and a `PriorBias` is
    // handed a node, so through this seam they are paid again at every node.
    bench2("  of which: the 17 gradient probes", &mut |(g, _p, m, _st)| {
        std::hint::black_box(tzolkin::mcts::Gradient::new(g, *m));
    });
    bench2("  of which: EffectPrice::choice", &mut |(_g, _p, _m, st)| {
        let price = tzolkin::options::EffectPrice::default();
        for step in st {
            if let Step::Take(c) = step {
                std::hint::black_box(price.choice(c));
            }
        }
    });
    bench2("OnePlyBias::bias (the baseline)", &mut |(g, p, m, st)| {
        buf.clear();
        buf.resize(st.len(), 1.0);
        one_ply.bias(g, *p, *m, st, &mut buf);
        std::hint::black_box(&buf);
    });
    bench("tree::legal_steps (for scale)", &mut |(g, p, m, _)| {
        std::hint::black_box(tree::legal_steps(g, *p, *m, 0));
    });
    let widths: Vec<usize> = nodes.iter().map(|n| n.3.len()).collect();
    let mut sorted = widths.clone();
    sorted.sort_unstable();
    println!(
        "  node width: mean {:.1}  median {}  p99 {}  max {}",
        widths.iter().sum::<usize>() as f64 / widths.len() as f64,
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 99 / 100],
        sorted[sorted.len() - 1]
    );
}

// =======================================================================
// args
// =======================================================================

/// Component index by name, for the ablation flags. `None` for "no ablation".
fn term_index(name: &str) -> Option<usize> {
    Components::NAMES.iter().position(|&n| n == name)
}

struct Args {
    cmd: String,
    sched: String,
    /// How far to move from the identity schedule toward the fitted one.
    shrink: f32,
    /// Strip the calendar shape, keeping each term's average level.
    flat: bool,
    /// Override these two terms with a flat weight. Negative means "leave".
    engine: f32,
    board: f32,
    /// Leave every term but this one at its identity weight.
    only: String,
    /// Pin this one term back to its identity weight.
    drop: String,
    focus: f32,
    reach: bool,
    draft: String,
    /// Simulations per sub-decision. Zero keeps the greedy harness, which is
    /// the arm that isolates the *value* head with no priors in the picture.
    sims: u32,
    /// `PlanBias` strength. Zero installs no bias at all, not a bias of one:
    /// the identity bias must be bit-identical to the unbiased search and an
    /// absent `Option` is the only way to be sure of that.
    bias: f32,
    bias_place: bool,
    /// Credit a temple step only on the temple `plan::temple_target` names.
    bias_named: bool,
    /// `PricedBias` source: "grad", "flat" or "named". Empty leaves it off.
    priced: String,
    /// Points per step of climb the plan adds to its temple in `PricedBias`.
    priced_w: f32,
    /// `PricedBias` softmax temperature, in points. The strength knob for the
    /// priced arms, and the counterpart of `--bias K` for `PlanBias`: without
    /// sweeping it, "the gradient prior is worth less than the plan prior" is
    /// as likely to be a statement about this number as about either prior.
    priced_temp: f32,
    /// `PricedBias::min_edges`. The priced arms abstain below it and `PlanBias`
    /// has no such gate, so leaving it fixed while comparing the two would be
    /// comparing the *set of nodes touched* as much as the prior.
    priced_min: usize,
    /// Run the plan-blind control: same edges, same strength, no leader test.
    bias_blind: bool,
    /// The harder control: same multipliers, attached to shuffled edges.
    bias_shuffled: bool,
    /// Use `OnePlyBias` at this temperature instead of `PlanBias`. Negative
    /// leaves it off.
    ///
    /// **It probes with `--sched`, so pass `--sched identity` for the
    /// baseline.** Only under `Schedule::IDENTITY` is this `mcts::Priors::OnePly`;
    /// under the default `fitted` it is a one-ply probe over a value head that
    /// is 25 points worse -- which measured *five points better as a prior*
    /// (+6.55 against +1.40), so the difference is not a rounding error.
    bias_1ply: f32,
    /// Which value head the candidate carries under MCTS. The baseline is
    /// always `heuristic`.
    cand_ev: String,
    /// `MctsConfig::priors` for both sides.
    priors: String,
    k: usize,
    blocks: usize,
    games: usize,
    seed0: u64,
    out: PathBuf,
    resume: bool,
}

impl Args {
    /// The `MctsConfig` both sides run under. Identical across arms by
    /// construction: this measures an evaluator and a prior, not a budget.
    fn mcts_cfg(&self) -> MctsConfig {
        MctsConfig {
            priors: if self.priors == "1ply" {
                Priors::OnePly
            } else {
                Priors::Evaluator
            },
            ..MctsConfig::default()
        }
    }

    /// The candidate's prior steer, or `None`.
    ///
    /// `None` rather than a zero-strength bias: `Mcts::priors_for` skips the
    /// scratch buffer and the multiply entirely when the `Option` is empty, so
    /// this is the only construction that is *provably* the unbiased search.
    fn bias(&self) -> Option<std::sync::Arc<dyn PriorBias>> {
        if !self.priced.is_empty() {
            let source = match self.priced.as_str() {
                "grad" => PriceSource::Gradient,
                "flat" => PriceSource::PlanFlat,
                _ => PriceSource::PlanNamed,
            };
            return Some(std::sync::Arc::new(PricedBias {
                temp: self.priced_temp,
                min_edges: self.priced_min,
                ..PricedBias::new(source, self.priced_w)
            }));
        }
        if self.bias_1ply >= 0.0 {
            return Some(std::sync::Arc::new(OnePlyBias {
                ev: self.evaluator()(),
                temp: self.bias_1ply,
            }));
        }
        if self.bias == 0.0 {
            return None;
        }
        Some(std::sync::Arc::new(PlanBias {
            w: BiasWeights {
                k: self.bias,
                place: self.bias_place,
                name_temple: self.bias_named,
                blind: self.bias_blind,
                shuffled: self.bias_shuffled,
                ..BiasWeights::OFF
            },
        }))
    }

    /// The candidate and the baseline for one block.
    ///
    /// Fresh per block: an `Mcts` owns an arena and a transposition index, and
    /// sharing one across the four seatings of a block would let seat 0's tree
    /// answer seat 3's question.
    fn agents(&self, seed: u64) -> (Box<dyn Agent>, Box<dyn Agent>) {
        if self.sims == 0 {
            return (
                Box::new(PlanAgent::new(self.evaluator()(), self.k)),
                Box::new(baseline(self.k)),
            );
        }
        let cand_ev: std::sync::Arc<dyn Evaluator> = if self.cand_ev == "heuristic" {
            std::sync::Arc::new(HeuristicEvaluator)
        } else {
            std::sync::Arc::new(self.evaluator()())
        };
        let cfg = self.mcts_cfg();
        (
            Box::new(MctsAgent::new(cand_ev, self.bias(), self.sims, cfg, seed)),
            Box::new(MctsAgent::new(
                std::sync::Arc::new(HeuristicEvaluator),
                None,
                self.sims,
                cfg,
                seed,
            )),
        )
    }

    /// The schedule, with the optional single-term ablations applied.
    ///
    /// `--only T` leaves every term but `T` at its identity weight and
    /// `--drop T` does the reverse. Between them they say which term of the
    /// schedule carried the effect, which one pooled number cannot.
    fn sched(&self) -> Schedule {
        let mut s = match self.sched.as_str() {
            "identity" => Schedule::IDENTITY,
            "half" => plan::HALF,
            "refit" => plan::REFIT.shrunk(self.shrink),
            _ => plan::FITTED.shrunk(self.shrink),
        };
        if let Some(t) = term_index(&self.only) {
            let mut base = Schedule::IDENTITY;
            for k in 0..plan::KNOTS.len() {
                base.w[k][t] = s.w[k][t];
            }
            s = base;
        }
        if self.flat {
            s = s.flattened();
        }
        if self.engine >= 0.0 {
            s = s.with(plan::ENGINE, self.engine);
        }
        if self.board >= 0.0 {
            s = s.with(plan::BOARD, self.board);
        }
        if let Some(t) = term_index(&self.drop) {
            for k in 0..plan::KNOTS.len() {
                s.w[k][t] = 1.0;
            }
        }
        s
    }
    /// A constructor rather than a value: each block builds its own agents.
    fn evaluator(&self) -> impl Fn() -> PlanEvaluator + '_ {
        move || PlanEvaluator {
            sched: self.sched(),
            plan: PlanWeights {
                focus: self.focus,
                reach: self.reach,
            },
            draft: match self.draft.as_str() {
                "prior" => DraftMode::Prior,
                "random" => DraftMode::Random,
                "unweighted" => DraftMode::Unweighted,
                _ => DraftMode::Board,
            },
            label: "plan",
        }
    }
    fn label(&self) -> String {
        if self.sims > 0 {
            let (c, b) = self.agents(0);
            return format!("{}  vs  {}", c.name(), b.name());
        }
        format!(
            "sched={}@{}{}{}{}{}{} focus={} reach={} draft={}",
            self.sched,
            self.shrink,
            if self.flat { " flat" } else { "" },
            if self.engine >= 0.0 { format!(" engine={}", self.engine) } else { String::new() },
            if self.board >= 0.0 { format!(" board={}", self.board) } else { String::new() },
            if self.only.is_empty() { String::new() } else { format!(" only={}", self.only) },
            if self.drop.is_empty() { String::new() } else { format!(" drop={}", self.drop) },
            self.focus,
            self.reach,
            self.draft
        )
    }
}

fn parse() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let num = |name: &str, d: usize| get(name).and_then(|v| v.parse().ok()).unwrap_or(d);
    let cmd = argv
        .get(1)
        .cloned()
        .unwrap_or_else(|| "duel".into());
    if cmd == "--help" || cmd == "-h" {
        println!(
            "planlab -- fit and measure the phase/plan layer\n\
             \n\
             SUBCOMMANDS\n\
               duel     rotation-block match against heuristic:K   [default]\n\
               fit      refit the weight schedule from self-play\n\
               probe    print what the layer does to one game\n\
               priors   what the prior bias touches, and what it costs\n\
               tiles    what each starting tile is worth on day 0\n\
               compare  paired difference between two duel progress files\n\
             \n\
             OPTIONS\n\
               --sched identity|half|fitted|refit  weight schedule  [fitted]\n\
               --shrink T                how far from identity toward fitted [1]\n\
               --flat                    strip the calendar shape, keep the level\n\
               --engine W / --board W    force a flat weight on those terms\n\
               --only TERM               only this term deviates from identity\n\
               --drop TERM               pin this term back to identity\n\
               --focus F                 plan concentration weight  [0]\n\
               --reach                   gate `held` on conversion reach\n\
               --draft board|unweighted|prior|random  who drafts  [board]\n\
             \n\
             MCTS ARMS (both sides get the same simulation count)\n\
               --sims N                  simulations per sub-decision; 0 = greedy [0]\n\
               --cand-ev plan|heuristic  candidate's value head          [plan]\n\
               --bias K                  PlanBias strength, nats/lead    [0]\n\
               --bias-place              also steer Step::Place, by gear\n\
               --bias-blind              CONTROL: same edges, no leader test\n\
               --bias-shuffled           CONTROL: same weights, shuffled edges\n\
               --bias-1ply T             OnePlyBias at temperature T instead\n\
               --priced grad|flat|named  gradient prior, temple axis from the plan\n\
               --priced-w W / --priced-temp T   plan weight / softmax temp  [4/4]\n\
               --priced-min N            narrowest node the priced arms price [8]\n\
               --priors eval|1ply        MctsConfig::priors, both sides  [eval]\n\
               --k K                     sampled turns per side     [32]\n\
               --blocks N                rotation blocks (duel)     [200]\n\
               --games N                 games (fit)                [400]\n\
               --seed N                  first seed                 [3000000]\n\
               --out PATH                progress JSONL\n\
               --resume                  skip seeds already on file"
        );
        std::process::exit(0);
    }
    let sched = get("--sched").unwrap_or_else(|| "fitted".into());
    let shrink = get("--shrink").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let flat = argv.iter().any(|a| a == "--flat");
    let engine = get("--engine").and_then(|v| v.parse().ok()).unwrap_or(-1.0);
    let board = get("--board").and_then(|v| v.parse().ok()).unwrap_or(-1.0);
    let only = get("--only").unwrap_or_default();
    let drop = get("--drop").unwrap_or_default();
    for (flag, v) in [("--only", &only), ("--drop", &drop)] {
        if !v.is_empty() && term_index(v).is_none() {
            eprintln!("planlab: {flag} {v:?}: not a term; try one of {:?}", Components::NAMES);
            std::process::exit(2);
        }
    }
    let focus = get("--focus").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let reach = argv.iter().any(|a| a == "--reach");
    let draft = get("--draft").unwrap_or_else(|| "board".into());
    let sims = get("--sims").and_then(|v| v.parse().ok()).unwrap_or(0u32);
    let bias = get("--bias").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let bias_place = argv.iter().any(|a| a == "--bias-place");
    let bias_blind = argv.iter().any(|a| a == "--bias-blind");
    let bias_1ply = get("--bias-1ply").and_then(|v| v.parse().ok()).unwrap_or(-1.0);
    let cand_ev = get("--cand-ev").unwrap_or_else(|| "plan".into());
    let priors = get("--priors").unwrap_or_else(|| "eval".into());
    let out = get("--out").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(format!(
            "planlab-{sched}{shrink}{}{}{}{}{}-f{focus}-r{}-d{}-s{sims}-x{bias}.jsonl",
            if flat { "-flat" } else { "" },
            if engine >= 0.0 { format!("-e{engine}") } else { String::new() },
            if board >= 0.0 { format!("-b{board}") } else { String::new() },
            if only.is_empty() { String::new() } else { format!("-only{only}") },
            if drop.is_empty() { String::new() } else { format!("-drop{drop}") },
            reach as u8,
            draft
        ))
    });
    Args {
        cmd,
        sched,
        shrink,
        flat,
        engine,
        board,
        only,
        drop,
        focus,
        reach,
        draft,
        sims,
        bias,
        bias_place,
        bias_named: argv.iter().any(|a| a == "--bias-named"),
        priced: get("--priced").unwrap_or_default(),
        priced_w: get("--priced-w").and_then(|v| v.parse().ok()).unwrap_or(4.0),
        priced_temp: get("--priced-temp").and_then(|v| v.parse().ok()).unwrap_or(4.0),
        priced_min: num("--priced-min", 8),
        bias_blind,
        bias_shuffled: argv.iter().any(|a| a == "--bias-shuffled"),
        bias_1ply,
        cand_ev,
        priors,
        k: num("--k", DEFAULT_K),
        blocks: num("--blocks", 200),
        games: num("--games", 400),
        seed0: get("--seed").and_then(|v| v.parse().ok()).unwrap_or(3_000_000),
        out,
        resume: argv.iter().any(|a| a == "--resume"),
    }
}

fn main() {
    let args = parse();
    interrupt::install();
    let _ = record::reject_unknown_flags(&[
        "--sched", "--shrink", "--flat", "--engine", "--board", "--only", "--drop", "--focus", "--reach", "--draft", "--k", "--blocks",
        "--games", "--seed", "--out", "--resume", "--help",
        "--sims", "--bias", "--bias-place", "--bias-blind", "--bias-shuffled", "--bias-1ply",
        "--cand-ev", "--priors",
        "--bias-named", "--priced", "--priced-w", "--priced-temp", "--priced-min",
    ]);
    match args.cmd.as_str() {
        "compare" => {
            let argv: Vec<String> = std::env::args().collect();
            let files: Vec<PathBuf> = argv[2..]
                .iter()
                .filter(|a| a.ends_with(".jsonl"))
                .map(PathBuf::from)
                .collect();
            if files.len() != 2 {
                eprintln!("planlab compare A.jsonl B.jsonl");
                std::process::exit(2);
            }
            compare(&files[0], &files[1]);
        }
        "fit" => fit(&args),
        "priors" => priors_probe(&args),
        "tiles" => tiles(&args),
        "probe" => probe(&args),
        _ => duel(&args),
    }
}
