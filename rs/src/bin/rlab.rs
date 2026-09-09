//! `rlab` — what is research actually worth, measured by **forcing** it.
//!
//! Every existing measurement of research in this repository goes through
//! `src/eval.rs`: sweep `RESEARCH_SCALE`, watch uptake move, watch the score
//! fall (`docs/FINDINGS-eval.md` F50-F57). That answers "does the evaluator's
//! research term want raising", which is not the same question as "is early
//! research good in this game". A term whose *shape* is wrong loses points when
//! its *scale* is raised, and the two are indistinguishable from a scale sweep.
//!
//! This binary takes the evaluator out of the loop. One seat is **compelled**
//! to pursue a stated goal on a stated schedule; the other three play the same
//! base agent normally; the seat rotates through all four positions on a shared
//! seed, exactly as `bin/arena` does. The number that comes back is the causal
//! cost or benefit of the behaviour, with no term to be mis-priced.
//!
//! # The forcing is lexicographic, not a bonus
//!
//! A "big bonus on research" is just `RESEARCH_SCALE` again. Instead the
//! candidate's move is chosen by
//!
//! 1. **progress toward the goal**, capped at what the goal asks for;
//! 2. **workers standing on Tikal**, capped, as a tie-break when no move makes
//!    progress — this is the setup half, because research is cashed on
//!    *retrieval* and a worker has to be standing there first;
//! 3. the base evaluator's own score.
//!
//! Key 3 is the whole of the decision whenever the goal is met or the window is
//! closed, so **with `--force none` the candidate is bit-identical to the
//! baseline** and the null is exactly +0.0000. That is checked by `--force
//! none` runs and it is the control every number here is read against.
//!
//! # The matched control is the point
//!
//! Forcing *anything* on a greedy agent costs points, so "forcing research
//! costs N" is uninterpretable on its own. `--force temple=K@a-b` and
//! `--force build=K@a-b` compel the same seat, through the same gear (Tikal),
//! over the same window, to buy temple steps or buildings instead. Tikal 1/3
//! are research, Tikal 2/4 are buildings, Tikal 5 is two temple steps: the same
//! worker-turns, the same corn, three different payoffs. The difference between
//! those arms is what "research is a bad use of a Tikal action" means.
//!
//! # Base agent
//!
//! Default `heuristic:full`, and that default matters: at `temperature = 0` a
//! `Candidates::All` greedy agent draws no random numbers, so a game is a pure
//! function of its seed and the *only* difference between the arms is the
//! candidate's own moves. `heuristic:64` samples, would consume the shared RNG,
//! and would desynchronise every downstream draw for all four seats.
//!
//! # Modes
//!
//! * (default) forced-arm rotation blocks -> JSONL, one row per block.
//! * `--census` — no forcing; per-turn placement-depth accounting, which is the
//!   corn-as-actions question. For every turn it records workers in hand, corn
//!   held, how many workers the corn could pay for, and how many more it could
//!   pay for with +1..+6 corn. That converts corn into *actions* directly,
//!   without pricing it.
//! * `--verify` — plays N seeds through this file's driver and through
//!   `record::play_game` and asserts the scores agree. The driver here is a
//!   copy of `play_game` with instrumentation; this is what keeps the copy
//!   honest.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use tzolkin::eval;
use tzolkin::ids::*;
use tzolkin::moves::{apply_move, Move};
use tzolkin::phase::{Evaluator, HeuristicEvaluator, Phase};
use tzolkin::record::{self, Agent, AgentSpec, GameConfig, Summary, TurnOutcome};
use tzolkin::state::{GameState, LAST_DAY};

// =======================================================================
// Goals and schedules
// =======================================================================

#[derive(Clone, Debug, PartialEq)]
enum Goal {
    /// Nothing forced. The candidate is the base agent.
    None,
    /// Reach at least level `l` on each listed track.
    Res(Vec<(Science, u8)>),
    /// Accumulate `k` temple steps above the starting row.
    Temple(u32),
    /// Construct `k` buildings.
    Build(u32),
}

#[derive(Clone, Debug)]
struct Plan {
    goal: Goal,
    /// Force only on days `from ..= by`, inclusive.
    from: u8,
    by: u8,
    /// Demand the whole goal from day `from`, instead of pacing it.
    asap: bool,
    /// How many own workers the setup key wants standing on Tikal.
    tikal: u32,
    label: String,
}

fn science_of(c: char) -> Option<Science> {
    match c.to_ascii_uppercase() {
        'A' => Some(Science::Agriculture),
        'R' => Some(Science::Extraction),
        'C' => Some(Science::Architecture),
        'T' => Some(Science::Theology),
        _ => None,
    }
}

impl Plan {
    /// `none` | `res=A3[,R1]@from-by` | `temple=K@from-by` | `build=K@from-by`
    fn parse(s: &str, asap: bool, tikal: u32) -> Result<Plan, String> {
        let label = format!("{s}{}{}", if asap { "!" } else { "" },
            if tikal != 1 { format!("/t{tikal}") } else { String::new() });
        if s == "none" {
            return Ok(Plan {
                goal: Goal::None,
                from: 0,
                by: 0,
                asap,
                tikal,
                label,
            });
        }
        let (body, window) = s
            .split_once('@')
            .ok_or_else(|| format!("--force {s:?}: need KIND@FROM-BY, e.g. res=A3@0-12"))?;
        let (from, by) = window
            .split_once('-')
            .ok_or_else(|| format!("--force {s:?}: window is FROM-BY, e.g. 0-12"))?;
        let from: u8 = from.parse().map_err(|_| format!("bad window start {from:?}"))?;
        let by: u8 = by.parse().map_err(|_| format!("bad window end {by:?}"))?;
        let (kind, arg) = body
            .split_once('=')
            .ok_or_else(|| format!("--force {s:?}: need res=/temple=/build="))?;
        let goal = match kind {
            "res" => {
                let mut v = Vec::new();
                for item in arg.split(',') {
                    let mut ch = item.chars();
                    let t = ch
                        .next()
                        .and_then(science_of)
                        .ok_or_else(|| format!("bad track in {item:?}; use A R C T"))?;
                    let l: u8 = ch
                        .as_str()
                        .parse()
                        .map_err(|_| format!("bad level in {item:?}"))?;
                    if !(1..=3).contains(&l) {
                        return Err(format!("level in {item:?} must be 1..3"));
                    }
                    v.push((t, l));
                }
                Goal::Res(v)
            }
            "temple" => Goal::Temple(arg.parse().map_err(|_| "temple=K needs a number")?),
            "build" => Goal::Build(arg.parse().map_err(|_| "build=K needs a number")?),
            other => return Err(format!("unknown goal kind {other:?}")),
        };
        Ok(Plan {
            goal,
            from,
            by,
            asap,
            tikal,
            label,
        })
    }

    /// Progress toward the goal, already capped at what the goal asks for, so
    /// that overshooting is never rewarded.
    fn progress(&self, g: &GameState, p: PlayerId) -> u32 {
        match &self.goal {
            Goal::None => 0,
            Goal::Res(v) => v
                .iter()
                .map(|&(t, l)| (g.research[p.idx()][t.idx()] as u32).min(l as u32))
                .sum(),
            Goal::Temple(k) => temple_steps(g, p).min(*k),
            Goal::Build(k) => g.players[p.idx()].n_buildings().min(*k),
        }
    }

    fn target(&self) -> u32 {
        match &self.goal {
            Goal::None => 0,
            Goal::Res(v) => v.iter().map(|&(_, l)| l as u32).sum(),
            Goal::Temple(k) | Goal::Build(k) => *k,
        }
    }

    /// The setup key: own workers standing on Tikal, capped at `tikal`.
    ///
    /// Every goal in this file is bought at Tikal, which is what makes the arms
    /// comparable. The cap is 1 by default and that matters: pinning *two* of a
    /// three-worker opening onto one gear is not "commit to research", it is a
    /// handcuff, and it measures the handcuff.
    fn setup(&self, g: &GameState, p: PlayerId, owed: u32) -> u32 {
        if owed == 0 {
            return 0;
        }
        tikal_workers(g, p).min(self.tikal).min(owed)
    }

    /// Progress the schedule demands by day `d`.
    ///
    /// `asap` is the harsh reading — everything, immediately, for the whole
    /// window. The default is a *pace*: the goal is spread evenly over the
    /// window, so "max Agriculture by day 12" asks for level 1 by day 4 and
    /// level 2 by day 8 and leaves the agent free in between. That is the
    /// schedule a human would describe, and the difference between the two is
    /// itself a measurement — see the `asap` rows.
    fn required(&self, d: u8) -> u32 {
        let t = self.target();
        if self.asap {
            return t;
        }
        if d < self.from {
            return 0;
        }
        let span = (self.by.saturating_sub(self.from) as u32) + 1;
        let elapsed = (d - self.from) as u32 + 1;
        (t * elapsed.min(span)) / span
    }

    fn active(&self, g: &GameState, p: PlayerId) -> bool {
        self.goal != Goal::None
            && g.day >= self.from
            && g.day <= self.by
            && self.progress(g, p) < self.required(g.day)
    }
}

/// Temple steps above the starting row, summed over the three temples.
fn temple_steps(g: &GameState, p: PlayerId) -> u32 {
    let base = tzolkin::data::temples::STARTING_STEP as i32;
    Temple::ALL
        .iter()
        .map(|&t| (g.temple_pos(p, t) as i32 - base).max(0) as u32)
        .sum()
}

fn tikal_workers(g: &GameState, p: PlayerId) -> u32 {
    GameState::worker_ids(p)
        .filter(|&w| {
            matches!(g.workers[w.idx()].on_board(), Some((Gear::Tikal, _)))
        })
        .count() as u32
}

// =======================================================================
// The forcing wrapper
// =======================================================================

/// Weights for the lexicographic keys. `value` lives in `(-1, 1)` because
/// `HeuristicEvaluator` squashes with `tanh`, so these separate cleanly.
const W_PROGRESS: f32 = 1.0e6;
const W_SETUP: f32 = 1.0e3;

/// Matches `Candidates::keep()` for `Candidates::All`, so that an unforced turn
/// is scored by exactly the shortlist `heuristic:full` would build.
const KEEP: usize = 32;

struct Forced {
    inner: Box<dyn Agent>,
    plan: Plan,
    /// Turns on which the forcing actually changed the ranking key.
    hits: AtomicU32,
    /// Turns the window was open and something was owed.
    open: AtomicU32,
}

impl Forced {
    fn value(&self, s: &GameState, p: PlayerId) -> f32 {
        HeuristicEvaluator
            .evaluate(s, Phase::Mode, p, 0)
            .value[p.idx()]
    }
}

impl Agent for Forced {
    fn play_turn(
        &self,
        g: &GameState,
        p: PlayerId,
        temp: f32,
        rng: &mut StdRng,
    ) -> Option<TurnOutcome> {
        if !self.plan.active(g, p) {
            return self.inner.play_turn(g, p, temp, rng);
        }
        self.open.fetch_add(1, Ordering::Relaxed);

        let p0 = self.plan.progress(g, p);
        let owed = self.plan.required(g.day).saturating_sub(p0);
        let s0 = self.plan.setup(g, p, owed);

        let r = eval::rank_all_within(g, p, KEEP, Some(eval::FULL_BUDGET), |s| {
            let gain = self.plan.progress(s, p).saturating_sub(p0) as f32;
            let setup = self.plan.setup(s, p, owed) as f32;
            W_PROGRESS * gain + W_SETUP * setup + self.value(s, p)
        });
        let (mv, sc) = r.moves.first()?.clone();
        // "Changed the key" means the winner was chosen for progress or for
        // getting a worker onto Tikal, not merely re-ranked at the third key.
        if sc >= W_PROGRESS || sc >= W_SETUP * (s0 as f32 + 1.0) {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        Some(TurnOutcome {
            mv,
            nodes: Vec::new(),
        })
    }

    fn extra_day(
        &self,
        g: &GameState,
        p: PlayerId,
        rng: &mut StdRng,
    ) -> (bool, Option<record::Node>) {
        self.inner.extra_day(g, p, rng)
    }

    fn draft(&self, g: &GameState, p: PlayerId, dealt: [u8; 4], rng: &mut StdRng) -> [u8; 2] {
        self.inner.draft(g, p, dealt, rng)
    }

    fn name(&self) -> String {
        format!("{}+force[{}]", self.inner.name(), self.plan.label)
    }
}

// =======================================================================
// The driver — `record::play_game` plus instrumentation
// =======================================================================

/// Everything one game hands back. The extra fields over `GameResult` are the
/// point: research levels at the end and at the half-way mark, the monument row
/// that was dealt, and what the seat spent on placement.
struct Trace {
    scores: [i16; N_PLAYERS],
    win_share: [f32; N_PLAYERS],
    days: u8,
    aborted: bool,
    research_end: [[u8; 4]; N_PLAYERS],
    research_mid: [[u8; 4]; N_PLAYERS],
    monuments_dealt: [Option<u8>; 6],
    monuments_held: [u32; N_PLAYERS],
    buildings_held: [u32; N_PLAYERS],
    corn_to_place: [u32; N_PLAYERS],
    workers_placed: [u32; N_PLAYERS],
    /// Turns that begged: a temple step *down* to buy three corn. If forcing
    /// pushes a seat into begging, the cost of the forcing is mostly temple
    /// track, not research, and this is how that shows up.
    begs: [u32; N_PLAYERS],
    /// Sum of the three temple positions at the end.
    temples: [u32; N_PLAYERS],
    /// Corn held at the start of each own turn, summed over the game. A corn
    /// *wealth integral*: the cheapest observable that says whether a track
    /// bought with corn actually delivered corn.
    corn_hold: [u32; N_PLAYERS],
    turns: [u32; N_PLAYERS],
    census: Vec<CensusRow>,
}

/// One turn's placement-depth accounting.
#[derive(Clone, Copy)]
struct CensusRow {
    day: u8,
    seat: u8,
    hand: u32,
    corn: u32,
    /// Workers the corn on hand can pay to place, given the free spaces.
    depth: u32,
    /// The same with +1 .. +6 corn.
    depth_plus: [u32; 6],
    /// Workers actually placed this turn (0 for a retrieval).
    placed: u32,
}

/// The half-way snapshot day. `POINT_DAYS` is `[14, 27]`; F52b's "early"
/// column is the level sum as of day 14 and this reproduces it.
const MID_DAY: u8 = 14;

/// What `--gift` hands over. `None` in the option means corn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GiftKind {
    Res(Resource),
    /// One step up whichever temple the evaluator rates highest.
    Temple,
}

fn play_traced(
    seed: u64,
    agents: &[&dyn Agent; N_PLAYERS],
    cfg: &GameConfig,
    rng: &mut StdRng,
    census: bool,
    // `(seat, resource, units, every)` — a standing subsidy, for pricing corn
    // and blocks in points without going through the evaluator.
    gift: Option<(PlayerId, Option<GiftKind>, u8, u32)>,
    // `(seat, levels)` — research handed over at day 0 for **free**: no Tikal
    // action, no corn, no worker-turn. This is the other half of the forcing
    // experiment. `--force` measures `benefit − price`; this measures `benefit`
    // alone, so the difference is the price the board charges.
    grant: Option<(PlayerId, [u8; 4], u8)>,
) -> Trace {
    let mut state = record::new_drafted_game(seed, agents, rng);
    let mut granted = false;
    if let Some((who, levels, day)) = grant {
        if day == 0 {
            for (s, &l) in levels.iter().enumerate() {
                state.research[who.idx()][s] = state.research[who.idx()][s].max(l);
            }
            granted = true;
        }
    }
    let monuments_dealt = std::array::from_fn(|i| state.monuments_up[i].map(|m| m.0));
    let mut research_mid = state.research;
    let mut corn_to_place = [0u32; N_PLAYERS];
    let mut workers_placed = [0u32; N_PLAYERS];
    let mut begs = [0u32; N_PLAYERS];
    let mut corn_hold = [0u32; N_PLAYERS];
    let mut turns = [0u32; N_PLAYERS];
    let mut rows: Vec<CensusRow> = Vec::new();
    let mut aborted = false;
    let mut guard = 0u32;

    while !state.over {
        if let Some((who, levels, day)) = grant {
            if !granted && state.day >= day {
                for (s, &l) in levels.iter().enumerate() {
                    state.research[who.idx()][s] = state.research[who.idx()][s].max(l);
                }
                granted = true;
            }
        }
        if state.day <= MID_DAY {
            research_mid = state.research;
        }
        state.current = state.first_player;
        for _ in 0..N_PLAYERS {
            let p = state.current;
            let temp = (cfg.temperature)(state.day);
            let pre = if census { Some(depth_probe(&state, p)) } else { None };
            if let Some((who, res, num, den)) = gift {
                if who == p && turns[p.idx()] % den == 0 {
                    match res {
                        // A free temple step, taken on the temple the evaluator
                        // likes best. This is the currency the forced arms are
                        // actually spending, so it needs a causal price measured
                        // the same way corn and wood were.
                        Some(GiftKind::Temple) => {
                            for _ in 0..num {
                                let mut best: Option<(f32, Temple)> = None;
                                for t in Temple::ALL {
                                    if state.can_temple_step(p, t, 1) {
                                        state.temple_step(p, t, 1);
                                        let v = eval::heuristic(&state, p);
                                        state.temple_step(p, t, -1);
                                        if best.map_or(true, |(b, _)| v > b) {
                                            best = Some((v, t));
                                        }
                                    }
                                }
                                if let Some((_, t)) = best {
                                    state.temple_step(p, t, 1);
                                }
                            }
                        }
                        Some(GiftKind::Res(r)) => {
                            let pl = &mut state.players[p.idx()];
                            pl.res[r as usize] = pl.res[r as usize].saturating_add(num)
                        }
                        None => {
                            let pl = &mut state.players[p.idx()];
                            pl.corn = pl.corn.saturating_add(num)
                        }
                    }
                }
            }
            corn_hold[p.idx()] += state.players[p.idx()].corn as u32;
            if let Some(out) = agents[p.idx()].play_turn(&state, p, temp, rng) {
                turns[p.idx()] += 1;
                if out.mv.beg.is_some() {
                    begs[p.idx()] += 1;
                }
                corn_to_place[p.idx()] += out.mv.corn_cost as u32;
                let placed = match &out.mv.kind {
                    tzolkin::moves::MoveKind::Place(v) => v.len() as u32,
                    _ => 0,
                };
                workers_placed[p.idx()] += placed;
                if let Some((hand, corn, depth, depth_plus)) = pre {
                    rows.push(CensusRow {
                        day: state.day,
                        seat: p.0,
                        hand,
                        corn,
                        depth,
                        depth_plus,
                        placed,
                    });
                }
                apply_move(&mut state, p, &out.mv);
                state.refill_buildings();
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

    let research_end = state.research;
    let temples: [u32; N_PLAYERS] = std::array::from_fn(|i| {
        Temple::ALL
            .iter()
            .map(|&t| state.temple_pos(PlayerId(i as u8), t) as u32)
            .sum()
    });
    let monuments_held = std::array::from_fn(|i| state.players[i].n_monuments());
    let buildings_held = std::array::from_fn(|i| state.players[i].n_buildings());
    let scores = state.scores();
    let winners = state.winners();
    let share = if winners.is_empty() {
        0.0
    } else {
        1.0 / winners.len() as f32
    };
    let win_share =
        std::array::from_fn(|i| if winners.contains(&PlayerId(i as u8)) { share } else { 0.0 });

    Trace {
        scores,
        win_share,
        days: state.day,
        aborted,
        research_end,
        research_mid,
        monuments_dealt,
        monuments_held,
        buildings_held,
        corn_to_place,
        workers_placed,
        begs,
        temples,
        corn_hold,
        turns,
        census: rows,
    }
}

/// How many workers `p`'s corn could pay to place right now, and how many with
/// 1..6 more corn.
///
/// The cost rule is exact and comes from `moves::place_rec`: the `d`-th worker
/// placed (0-based) costs `d + space_index`. So the cheapest way to place `k`
/// workers is the `k` cheapest free spaces plus `k(k-1)/2`. Free spaces are
/// enumerated per gear, ascending, plus the first-player space at index 0 if it
/// is empty. Begging is ignored — it is a separate, once-per-turn option and
/// this is a question about the corn a player *holds*.
fn depth_probe(g: &GameState, p: PlayerId) -> (u32, u32, u32, [u32; 6]) {
    let hand = g.available(p).count() as u32;
    let corn = g.players[p.idx()].corn as u32;

    let mut spots: Vec<u32> = Vec::with_capacity(16);
    for gear in Gear::ALL {
        for i in 0..gear.size() {
            if g.gears[gear.idx()].is_free(Pos(i)) {
                spots.push(i as u32);
            }
        }
    }
    if g.first_player_space.is_none() {
        spots.push(0);
    }
    spots.sort_unstable();

    let afford = |budget: u32| -> u32 {
        let mut cost = 0u32;
        let mut k = 0u32;
        for (d, &s) in spots.iter().enumerate() {
            if k >= hand {
                break;
            }
            let step = d as u32 + s;
            if cost + step > budget {
                break;
            }
            cost += step;
            k += 1;
        }
        k
    };

    let depth = afford(corn);
    let depth_plus = std::array::from_fn(|i| afford(corn + i as u32 + 1));
    (hand, corn, depth, depth_plus)
}

// =======================================================================
// Rotation blocks
// =======================================================================

#[derive(Clone, Copy)]
struct Outcome {
    centred: f64,
    vs_base: f64,
    win: f64,
    cand_score: f64,
    base_score: f64,
    /// Candidate's research level sum, 0..=12.
    lv: f64,
    lv_base: f64,
    /// The same as of day 14.
    lv_mid: f64,
    lv_mid_base: f64,
    maxed: f64,
    mons: f64,
    mons_base: f64,
    builds: f64,
    builds_base: f64,
    corn_place: f64,
    corn_place_base: f64,
    placed: f64,
    placed_base: f64,
    begs: f64,
    begs_base: f64,
    temples: f64,
    temples_base: f64,
    corn_hold: f64,
    corn_hold_base: f64,
    days: f64,
    aborted: bool,
}

struct Block {
    seed: u64,
    mon11: bool,
    mon12: bool,
    hits: f64,
    open: f64,
    games: Vec<Outcome>,
}

impl Block {
    fn mean(&self, f: impl Fn(&Outcome) -> f64) -> f64 {
        self.games.iter().map(&f).sum::<f64>() / self.games.len() as f64
    }
    fn to_json(&self) -> String {
        let g = |f: fn(&Outcome) -> f64| self.mean(f);
        format!(
            concat!(
                r#"{{"seed":{},"games":{},"mon11":{},"mon12":{},"hits":{:.2},"open":{:.2},"#,
                r#""centred":{:.4},"vs_base":{:.4},"win":{:.4},"cand":{:.3},"base":{:.3},"#,
                r#""lv":{:.3},"lv_base":{:.3},"lv_mid":{:.3},"lv_mid_base":{:.3},"maxed":{:.3},"#,
                r#""mons":{:.3},"mons_base":{:.3},"builds":{:.3},"builds_base":{:.3},"#,
                r#""corn_place":{:.2},"corn_place_base":{:.2},"#,
                r#""placed":{:.2},"placed_base":{:.2},"begs":{:.2},"begs_base":{:.2},"#,
                r#""temples":{:.2},"temples_base":{:.2},"corn_hold":{:.2},"corn_hold_base":{:.2},"#,
                r#""days":{:.2},"aborted":{}}}"#
            ),
            self.seed,
            self.games.len(),
            self.mon11,
            self.mon12,
            self.hits,
            self.open,
            g(|o| o.centred),
            g(|o| o.vs_base),
            g(|o| o.win),
            g(|o| o.cand_score),
            g(|o| o.base_score),
            g(|o| o.lv),
            g(|o| o.lv_base),
            g(|o| o.lv_mid),
            g(|o| o.lv_mid_base),
            g(|o| o.maxed),
            g(|o| o.mons),
            g(|o| o.mons_base),
            g(|o| o.builds),
            g(|o| o.builds_base),
            g(|o| o.corn_place),
            g(|o| o.corn_place_base),
            g(|o| o.placed),
            g(|o| o.placed_base),
            g(|o| o.begs),
            g(|o| o.begs_base),
            g(|o| o.temples),
            g(|o| o.temples_base),
            g(|o| o.corn_hold),
            g(|o| o.corn_hold_base),
            g(|o| o.days),
            self.games.iter().filter(|o| o.aborted).count(),
        )
    }
}

fn lv_sum(r: &[u8; 4]) -> f64 {
    r.iter().map(|&x| x as f64).sum()
}

fn play_block(
    seed: u64,
    plan: &Plan,
    base_spec: &AgentSpec,
    cfg: &GameConfig,
    gift: Option<(Option<GiftKind>, u8, u32)>,
    grant: Option<([u8; 4], u8)>,
) -> Block {
    let base = base_spec.instance();
    let cand = Forced {
        inner: base_spec.instance(),
        plan: plan.clone(),
        hits: AtomicU32::new(0),
        open: AtomicU32::new(0),
    };
    let mut games = Vec::new();
    let mut mon11 = false;
    let mut mon12 = false;

    for c in 0..N_PLAYERS {
        let agents: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|s| {
            if s == c {
                &cand as &dyn Agent
            } else {
                base.as_ref()
            }
        });
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9) ^ c as u64);
        let g = gift.map(|(r, n, d)| (PlayerId(c as u8), r, n, d));
        let gr = grant.map(|(lv, d)| (PlayerId(c as u8), lv, d));
        let t = play_traced(seed, &agents, cfg, &mut rng, false, g, gr);

        mon11 = t.monuments_dealt.iter().any(|m| *m == Some(11));
        mon12 = t.monuments_dealt.iter().any(|m| *m == Some(12));

        let scores: [f64; N_PLAYERS] = std::array::from_fn(|s| t.scores[s] as f64);
        let table_mean = scores.iter().sum::<f64>() / N_PLAYERS as f64;
        let cand_score = scores[c];
        let base_score =
            (scores.iter().sum::<f64>() - cand_score) / (N_PLAYERS as f64 - 1.0);
        let others = |f: &dyn Fn(usize) -> f64| -> f64 {
            (0..N_PLAYERS).filter(|&s| s != c).map(f).sum::<f64>() / (N_PLAYERS as f64 - 1.0)
        };

        games.push(Outcome {
            centred: cand_score - table_mean,
            vs_base: cand_score - base_score,
            win: t.win_share[c] as f64,
            cand_score,
            base_score,
            lv: lv_sum(&t.research_end[c]),
            lv_base: others(&|s| lv_sum(&t.research_end[s])),
            lv_mid: lv_sum(&t.research_mid[c]),
            lv_mid_base: others(&|s| lv_sum(&t.research_mid[s])),
            maxed: t.research_end[c].iter().filter(|&&x| x >= 3).count() as f64,
            mons: t.monuments_held[c] as f64,
            mons_base: others(&|s| t.monuments_held[s] as f64),
            builds: t.buildings_held[c] as f64,
            builds_base: others(&|s| t.buildings_held[s] as f64),
            corn_place: t.corn_to_place[c] as f64,
            corn_place_base: others(&|s| t.corn_to_place[s] as f64),
            placed: t.workers_placed[c] as f64,
            placed_base: others(&|s| t.workers_placed[s] as f64),
            begs: t.begs[c] as f64,
            begs_base: others(&|s| t.begs[s] as f64),
            temples: t.temples[c] as f64,
            temples_base: others(&|s| t.temples[s] as f64),
            corn_hold: t.corn_hold[c] as f64,
            corn_hold_base: others(&|s| t.corn_hold[s] as f64),
            days: t.days as f64,
            aborted: t.aborted,
        });
    }

    Block {
        seed,
        mon11,
        mon12,
        hits: cand.hits.load(Ordering::Relaxed) as f64 / N_PLAYERS as f64,
        open: cand.open.load(Ordering::Relaxed) as f64 / N_PLAYERS as f64,
        games,
    }
}

// =======================================================================
// main
// =======================================================================

fn arg(name: &str) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
        if let Some(v) = a.strip_prefix(&format!("{name}=")) {
            return Some(v.to_string());
        }
    }
    None
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

const HELP: &str = "\
rlab — the causal value of research, measured by forcing it.

  --agent SPEC      base agent for every seat (default heuristic:full)
  --force PLAN      none | res=A3@0-12 | res=A1,R1@0-12 | temple=6@0-12 | build=2@0-12
  --asap            demand the whole goal from day `from` (default: pace it)
  --tikal N         workers the setup key wants standing on Tikal (default 1)
  --blocks N        rotation blocks (4 games each)
  --seed S          first seed (default 700000)
  --seeds FILE      take the block seeds from FILE (whitespace separated)
  --scan N          print `seed mon11 mon12 row` for N seeds; plays no games
  --jobs J          blocks in flight (default 3 — the machine is shared)
  --out FILE        JSONL, one row per block; appended
  --resume          skip seeds already in --out
  --gift RES:N[/D]  subsidise the candidate N units of corn/wood/stone/gold/temple
                    every D of its own turns; prices a resource in points
  --grant A3[,R1][@D]
                    hand the candidate those levels on day D (default 0), free:
                    no Tikal action, no corn, no worker-turn. `--force` measures
                    benefit-minus-price; this measures benefit alone, and
                    sweeping D measures the shape of `uses` directly.
  --pricecurve N    eval delta for every currency, days 0-6
  --evalcurve N     what `eval::heuristic` thinks --grant is worth, by day
  --census N        no forcing: per-turn placement-depth rows for N seeds
  --verify N        check this file's driver against record::play_game
";

fn main() {
    if flag("--help") || std::env::args().len() == 1 {
        println!("{HELP}");
        return;
    }
    let spec_s = arg("--agent").unwrap_or_else(|| "heuristic:full".to_string());
    let spec = match AgentSpec::parse(&spec_s, false) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("--agent: {e}");
            std::process::exit(2);
        }
    };
    let seed0: u64 = arg("--seed").and_then(|s| s.parse().ok()).unwrap_or(700_000);

    // `--scan N` prints `seed mon11 mon12` for N seeds without playing a
    // single game: the monument row is fixed by the seed alone, through
    // `Game::new_undrafted`. That turns question (b) from an underpowered
    // split of a general run into a *selected* experiment — a row with both
    // #11 and #12 is C(11,4)/C(13,6) = 19.2% of deals, so conditioning by
    // selection is five times cheaper than conditioning after the fact.
    if let Some(n) = arg("--scan").and_then(|s| s.parse::<u64>().ok()) {
        for i in 0..n {
            let seed = seed0 + i;
            let (game, _) = tzolkin::game::Game::new_undrafted(seed);
            let up: Vec<u8> = game
                .state
                .monuments_up
                .iter()
                .filter_map(|m| m.map(|x| x.0))
                .collect();
            println!(
                "{seed} {} {} {:?}",
                up.contains(&11) as u8,
                up.contains(&12) as u8,
                up
            );
        }
        return;
    }
    if let Some(n) = arg("--verify").and_then(|s| s.parse::<u64>().ok()) {
        verify(&spec, seed0, n);
        return;
    }
    if let Some(n) = arg("--pricecurve").and_then(|s| s.parse::<u64>().ok()) {
        price_curve(&spec, seed0, n);
        return;
    }
    if let Some(n) = arg("--evalcurve").and_then(|s| s.parse::<u64>().ok()) {
        eval_curve(&spec, seed0, n, arg("--grant").unwrap_or_else(|| "A3".into()));
        return;
    }
    if let Some(n) = arg("--census").and_then(|s| s.parse::<u64>().ok()) {
        census(&spec, seed0, n, arg("--out").map(PathBuf::from));
        return;
    }

    let plan = match Plan::parse(
        &arg("--force").unwrap_or_else(|| "none".to_string()),
        flag("--asap"),
        arg("--tikal").and_then(|s| s.parse().ok()).unwrap_or(1),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    // `--gift corn:1` or `--gift wood:1/4` — units every `den` own turns, to
    // the candidate seat only. This is a cheat arm on purpose: it prices a
    // resource in points directly, which is the only way to check
    // `CORN_PER_POINT` without asking the evaluator what it thinks corn is
    // worth.
    let gift: Option<(Option<GiftKind>, u8, u32)> = arg("--gift").map(|g| {
        let (what, rate) = g.split_once(':').expect("--gift RES:N[/D]");
        let (n, d) = match rate.split_once('/') {
            Some((n, d)) => (n.parse().expect("N"), d.parse().expect("D")),
            None => (rate.parse().expect("N"), 1u32),
        };
        let r = match what {
            "corn" => None,
            "wood" => Some(GiftKind::Res(Resource::Wood)),
            "stone" => Some(GiftKind::Res(Resource::Stone)),
            "gold" => Some(GiftKind::Res(Resource::Gold)),
            "temple" => Some(GiftKind::Temple),
            other => panic!("--gift: unknown resource {other:?}"),
        };
        (r, n, d)
    });
    // `--grant A3` or `--grant A1,R1` — the levels appear at day 0 for free.
    // Paired with the matching `--force` arm this splits the causal number in
    // two: `force = benefit - price`, `grant = benefit`, so `price = grant -
    // force`. Without it a negative `--force` result cannot distinguish "the
    // levels are worthless" from "the levels are fine and the board charges
    // more than they are worth".
    let grant: Option<([u8; 4], u8)> = arg("--grant").map(|g| {
        let (body, day) = match g.split_once('@') {
            Some((b, d)) => (b.to_string(), d.parse().expect("--grant: bad day")),
            None => (g.clone(), 0u8),
        };
        let mut lv = [0u8; 4];
        for item in body.split(',') {
            let mut ch = item.chars();
            let t = ch.next().and_then(science_of)
                .unwrap_or_else(|| panic!("--grant: bad track in {item:?}; use A R C T"));
            let l: u8 = ch.as_str().parse()
                .unwrap_or_else(|_| panic!("--grant: bad level in {item:?}"));
            assert!((1..=3).contains(&l), "--grant: level in {item:?} must be 1..3");
            lv[t.idx()] = l;
        }
        (lv, day)
    });
    let blocks: usize = arg("--blocks").and_then(|s| s.parse().ok()).unwrap_or(50);
    let jobs: usize = arg("--jobs").and_then(|s| s.parse().ok()).unwrap_or(3);
    let out = arg("--out").map(PathBuf::from);

    let mut done: Vec<u64> = Vec::new();
    if flag("--resume") {
        if let Some(o) = &out {
            if let Ok(txt) = std::fs::read_to_string(o) {
                for line in txt.lines() {
                    if let Some(s) = json_u64(line, "seed") {
                        done.push(s);
                    }
                }
            }
        }
    }
    let all_seeds: Vec<u64> = match arg("--seeds") {
        Some(f) => std::fs::read_to_string(&f)
            .unwrap_or_else(|e| panic!("--seeds {f}: {e}"))
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .take(blocks)
            .collect(),
        None => (0..blocks as u64).map(|i| seed0 + i).collect(),
    };
    let todo: Vec<u64> = all_seeds
        .into_iter()
        .filter(|s| !done.contains(s))
        .collect();

    eprintln!(
        "rlab: agent={} force={} blocks={} (todo {}) jobs={}",
        spec.name(),
        plan.label,
        blocks,
        todo.len(),
        jobs
    );

    let cfg = GameConfig::evaluation();
    let sink = out.as_ref().map(|o| {
        Mutex::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(o)
                .expect("open --out"),
        )
    });
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("thread pool");

    let rows: Mutex<Vec<String>> = Mutex::new(Vec::new());
    pool.install(|| {
        todo.par_iter().for_each(|&seed| {
            let b = play_block(seed, &plan, &spec, &cfg, gift, grant);
            let line = b.to_json();
            if let Some(f) = &sink {
                let mut f = f.lock().unwrap();
                let _ = writeln!(f, "{line}");
                let _ = f.flush();
            }
            rows.lock().unwrap().push(line);
        });
    });

    // Summarise everything on disk if there is a file, else this run's rows.
    let all: Vec<String> = match &out {
        Some(o) => std::fs::read_to_string(o)
            .unwrap_or_default()
            .lines()
            .map(|s| s.to_string())
            .collect(),
        None => rows.into_inner().unwrap(),
    };
    report(&all, &plan, &spec.name(), started.elapsed());
}

fn json_f(line: &str, key: &str) -> Option<f64> {
    let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = &line[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}
fn json_u64(line: &str, key: &str) -> Option<u64> {
    json_f(line, key).map(|x| x as u64)
}
fn json_b(line: &str, key: &str) -> bool {
    line.contains(&format!("\"{key}\":true"))
}

fn stat(name: &str, xs: &[f64]) {
    let s = Summary::of(xs);
    let star = if s.ci.is_finite() && s.mean.abs() > s.ci {
        " *"
    } else {
        ""
    };
    println!(
        "  {name:<16} {:+8.3}  [{:+.3}, {:+.3}]  n={}{star}",
        s.mean,
        s.mean - s.ci,
        s.mean + s.ci,
        s.n
    );
}

fn report(lines: &[String], plan: &Plan, agent: &str, took: Duration) {
    let mut seen: Vec<u64> = Vec::new();
    let mut keep: Vec<&String> = Vec::new();
    for l in lines {
        if let Some(s) = json_u64(l, "seed") {
            if !seen.contains(&s) {
                seen.push(s);
                keep.push(l);
            }
        }
    }
    let col = |k: &str| -> Vec<f64> { keep.iter().filter_map(|l| json_f(l, k)).collect() };
    println!(
        "\n=== rlab  agent={agent}  force={}  blocks={}  {:.1}s",
        plan.label,
        keep.len(),
        took.as_secs_f64()
    );
    stat("centred", &col("centred"));
    stat("vs_base", &col("vs_base"));
    stat("win (null .25)", &col("win"));
    stat("levels", &col("lv"));
    stat("levels(base)", &col("lv_base"));
    stat("levels d14", &col("lv_mid"));
    stat("tracks maxed", &col("maxed"));
    stat("monuments", &col("mons"));
    stat("monuments(b)", &col("mons_base"));
    stat("buildings", &col("builds"));
    stat("buildings(b)", &col("builds_base"));
    stat("corn to place", &col("corn_place"));
    stat("corn place(b)", &col("corn_place_base"));
    stat("workers placed", &col("placed"));
    stat("placed(base)", &col("placed_base"));
    stat("begs", &col("begs"));
    stat("begs(base)", &col("begs_base"));
    stat("temple sum", &col("temples"));
    stat("temple(base)", &col("temples_base"));
    stat("corn held", &col("corn_hold"));
    stat("corn held(b)", &col("corn_hold_base"));
    stat("forced turns", &col("hits"));

    // The monument split: #11 pays 9/20/33 for maxed tracks, #12 pays 3 a
    // level. Both are dealt about one row in five together.
    for (name, pred) in [
        ("both 11&12", 3usize),
        ("11 or 12", 1),
        ("neither", 0),
    ] {
        let xs: Vec<f64> = keep
            .iter()
            .filter(|l| {
                let m = (json_b(l, "mon11") as usize) + 2 * (json_b(l, "mon12") as usize);
                match pred {
                    3 => m == 3,
                    1 => m == 1 || m == 2,
                    _ => m == 0,
                }
            })
            .filter_map(|l| json_f(l, "centred"))
            .collect();
        if !xs.is_empty() {
            stat(name, &xs);
        }
    }
}

// =======================================================================
// --evalcurve
//
// The causal `--grant A3@D` sweep asks what the levels are worth if they
// appear on day D. This asks what `eval::heuristic` *thinks* they are worth on
// day D, on the same states, by evaluating each position twice — once as
// played, once with the levels written in — and differencing. Put beside the
// causal curve it reads the `uses` shape off the evaluator directly, in points,
// with no reverse-engineering of constants.
// =======================================================================

fn eval_curve(spec: &AgentSpec, seed0: u64, n: u64, grant: String) {
    let mut lv = [0u8; 4];
    for item in grant.split(',') {
        let mut ch = item.chars();
        let t = ch.next().and_then(science_of).expect("--grant track");
        lv[t.idx()] = ch.as_str().parse().expect("--grant level");
    }
    let cfg = GameConfig::evaluation();
    // day -> (sum of eval delta, count)
    let acc: Mutex<Vec<(f64, u32)>> = Mutex::new(vec![(0.0, 0); LAST_DAY as usize + 2]);
    let next = AtomicU32::new(0);
    std::thread::scope(|sc| {
        for _ in 0..4 {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed) as u64;
                if i >= n {
                    return;
                }
                let seed = seed0 + i;
                let base = spec.instance();
                let agents: [&dyn Agent; N_PLAYERS] =
                    std::array::from_fn(|_| base.as_ref() as &dyn Agent);
                let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
                let mut state = record::new_drafted_game(seed, &agents, &mut rng);
                let mut local: Vec<(f64, u32)> = vec![(0.0, 0); LAST_DAY as usize + 2];
                let mut guard = 0u32;
                while !state.over && guard < 400 {
                    guard += 1;
                    let d = state.day.min(LAST_DAY) as usize;
                    for p in PlayerId::ALL {
                        // Only seats with none of the granted levels, so the
                        // delta is always "gain the whole track".
                        if lv.iter().enumerate().any(|(s, &l)| l > 0 && state.research[p.idx()][s] > 0) {
                            continue;
                        }
                        let before = eval::heuristic(&state, p);
                        let saved = state.research[p.idx()];
                        for (s, &l) in lv.iter().enumerate() {
                            state.research[p.idx()][s] = state.research[p.idx()][s].max(l);
                        }
                        let after = eval::heuristic(&state, p);
                        state.research[p.idx()] = saved;
                        local[d].0 += (after - before) as f64;
                        local[d].1 += 1;
                    }
                    state.current = state.first_player;
                    for _ in 0..N_PLAYERS {
                        let p = state.current;
                        let temp = (cfg.temperature)(state.day);
                        if let Some(out) = agents[p.idx()].play_turn(&state, p, temp, &mut rng) {
                            apply_move(&mut state, p, &out.mv);
                            state.refill_buildings();
                        }
                        state.current = state.current.next(1);
                    }
                    let claimer = state.resolve_first_player();
                    let mut days = 1u8;
                    if let Some(p) = claimer {
                        if state.may_take_extra_day(p) {
                            let (take, _) = agents[p.idx()].extra_day(&state, p, &mut rng);
                            if take {
                                state.spend_extra_day(p);
                                days = 2;
                            }
                        }
                    }
                    state.advance_days(days);
                }
                let mut g = acc.lock().unwrap();
                for (i, (s, c)) in local.iter().enumerate() {
                    g[i].0 += s;
                    g[i].1 += c;
                }
            });
        }
    });
    let g = acc.lock().unwrap();
    println!("evalcurve: grant={grant} over {n} games");
    println!("  day   n      eval delta   rel to day 0");
    let d0 = g[0].0 / g[0].1.max(1) as f64;
    for (d, (s, c)) in g.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let m = s / *c as f64;
        println!("  {d:>3} {c:>6}      {m:+8.4}       {:.3}", m / d0);
    }
}

// =======================================================================
// --pricecurve
//
// The same double-evaluation trick as `--evalcurve`, applied to every currency
// at once: corn, a block, a temple step, a building, and each maxed research
// track. Every row is `eval::heuristic` with the thing and without it, on the
// same real positions. Beside the causal price of the same thing (the `--gift`
// arms, the cross-arm regression, the `--grant` arms) this is the evaluator's
// full calibration table, and every entry is measured the same way, so the
// ratios between rows are meaningful.
// =======================================================================

fn price_curve(spec: &AgentSpec, seed0: u64, n: u64) {
    const NROW: usize = 12;
    let names = [
        "corn +1", "corn +3", "wood +1", "stone +1", "gold +1", "skull +1",
        "temple step (best)", "building +1",
        "Agriculture max", "Extraction max", "Architecture max", "Theology max",
    ];
    let cfg = GameConfig::evaluation();
    let acc: Mutex<[(f64, u32); NROW]> = Mutex::new([(0.0, 0); NROW]);
    let next = AtomicU32::new(0);
    std::thread::scope(|sc| {
        for _ in 0..4 {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed) as u64;
                if i >= n {
                    return;
                }
                let seed = seed0 + i;
                let base = spec.instance();
                let agents: [&dyn Agent; N_PLAYERS] =
                    std::array::from_fn(|_| base.as_ref() as &dyn Agent);
                let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
                let mut state = record::new_drafted_game(seed, &agents, &mut rng);
                let mut local = [(0.0f64, 0u32); NROW];
                let mut guard = 0u32;
                while !state.over && guard < 400 {
                    guard += 1;
                    // Only the first six days: the question is what an early
                    // action is worth, and that is where the forcing bites.
                    if state.day <= 6 {
                        for p in PlayerId::ALL {
                            let before = eval::heuristic(&state, p);
                            macro_rules! bump {
                                ($row:expr, $st:expr) => {{
                                    let after = eval::heuristic($st, p);
                                    local[$row].0 += (after - before) as f64;
                                    local[$row].1 += 1;
                                }};
                            }
                            let saved_corn = state.players[p.idx()].corn;
                            state.players[p.idx()].corn = saved_corn.saturating_add(1);
                            bump!(0, &state);
                            state.players[p.idx()].corn = saved_corn.saturating_add(3);
                            bump!(1, &state);
                            state.players[p.idx()].corn = saved_corn;
                            for (row, r) in [Resource::Wood, Resource::Stone,
                                             Resource::Gold, Resource::Skull]
                                .iter().enumerate()
                            {
                                let sv = state.players[p.idx()].res[*r as usize];
                                state.players[p.idx()].res[*r as usize] = sv.saturating_add(1);
                                bump!(2 + row, &state);
                                state.players[p.idx()].res[*r as usize] = sv;
                            }
                            // Best legal single temple step, which is what a
                            // Tikal 5 action actually buys half of.
                            let mut best = f32::NEG_INFINITY;
                            for t in Temple::ALL {
                                if state.can_temple_step(p, t, 1) {
                                    state.temple_step(p, t, 1);
                                    best = best.max(eval::heuristic(&state, p));
                                    state.temple_step(p, t, -1);
                                }
                            }
                            if best > f32::NEG_INFINITY {
                                local[6].0 += (best - before) as f64;
                                local[6].1 += 1;
                            }
                            let sb = state.players[p.idx()].buildings;
                            state.players[p.idx()].buildings = sb + 1;
                            bump!(7, &state);
                            state.players[p.idx()].buildings = sb;
                            for (row, sc) in Science::ALL.iter().enumerate() {
                                let sv = state.research[p.idx()][sc.idx()];
                                if sv > 0 { continue; }
                                state.research[p.idx()][sc.idx()] = 3;
                                bump!(8 + row, &state);
                                state.research[p.idx()][sc.idx()] = sv;
                            }
                        }
                    }
                    state.current = state.first_player;
                    for _ in 0..N_PLAYERS {
                        let p = state.current;
                        let temp = (cfg.temperature)(state.day);
                        if let Some(out) = agents[p.idx()].play_turn(&state, p, temp, &mut rng) {
                            apply_move(&mut state, p, &out.mv);
                            state.refill_buildings();
                        }
                        state.current = state.current.next(1);
                    }
                    let claimer = state.resolve_first_player();
                    let mut days = 1u8;
                    if let Some(p) = claimer {
                        if state.may_take_extra_day(p) {
                            let (take, _) = agents[p.idx()].extra_day(&state, p, &mut rng);
                            if take {
                                state.spend_extra_day(p);
                                days = 2;
                            }
                        }
                    }
                    state.advance_days(days);
                }
                let mut g = acc.lock().unwrap();
                for (i, (s, c)) in local.iter().enumerate() {
                    g[i].0 += s;
                    g[i].1 += c;
                }
            });
        }
    });
    let g = acc.lock().unwrap();
    println!("pricecurve: days 0-6 over {n} games");
    println!("  {:<22} {:>8}  {:>7}", "thing", "eval", "n");
    for (i, nm) in names.iter().enumerate() {
        if g[i].1 == 0 { continue; }
        println!("  {:<22} {:>+8.4}  {:>7}", nm, g[i].0 / g[i].1 as f64, g[i].1);
    }
}

// =======================================================================
// --census
// =======================================================================

fn census(spec: &AgentSpec, seed0: u64, n: u64, out: Option<PathBuf>) {
    let cfg = GameConfig::evaluation();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(arg("--jobs").and_then(|s| s.parse().ok()).unwrap_or(3))
        .build()
        .unwrap();
    let rows: Mutex<Vec<CensusRow>> = Mutex::new(Vec::new());
    pool.install(|| {
        (0..n).into_par_iter().for_each(|i| {
            let seed = seed0 + i;
            let a = spec.instance();
            let agents: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|_| a.as_ref());
            let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
            let t = play_traced(seed, &agents, &cfg, &mut rng, true, None, None);
            rows.lock().unwrap().extend(t.census);
        });
    });
    let rows = rows.into_inner().unwrap();
    if let Some(o) = out {
        let mut f = std::fs::File::create(o).unwrap();
        writeln!(f, "day,seat,hand,corn,depth,d1,d2,d3,d4,d5,d6,placed").unwrap();
        for r in &rows {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                r.day,
                r.seat,
                r.hand,
                r.corn,
                r.depth,
                r.depth_plus[0],
                r.depth_plus[1],
                r.depth_plus[2],
                r.depth_plus[3],
                r.depth_plus[4],
                r.depth_plus[5],
                r.placed
            )
            .unwrap();
        }
    }
    // A summary a human can read without a spreadsheet.
    println!("census: {} turns over {} games", rows.len(), n);
    for h in 0..=6u32 {
        let sub: Vec<&CensusRow> = rows.iter().filter(|r| r.hand == h).collect();
        if sub.is_empty() {
            continue;
        }
        let mean = |f: &dyn Fn(&CensusRow) -> f64| -> f64 {
            sub.iter().map(|r| f(r)).sum::<f64>() / sub.len() as f64
        };
        let d0 = mean(&|r| r.depth as f64);
        println!(
            "  hand={h}  n={:<7} corn={:5.2}  depth={:4.2}  +1={:+.3} +2={:+.3} +3={:+.3} +6={:+.3}  placed={:4.2}",
            sub.len(),
            mean(&|r| r.corn as f64),
            d0,
            mean(&|r| r.depth_plus[0] as f64) - d0,
            mean(&|r| r.depth_plus[1] as f64) - d0,
            mean(&|r| r.depth_plus[2] as f64) - d0,
            mean(&|r| r.depth_plus[5] as f64) - d0,
            mean(&|r| r.placed as f64),
        );
    }
    // Same, by day, so "corn buys actions" can be read against the calendar.
    for lo in [0u8, 7, 14, 21] {
        let sub: Vec<&CensusRow> = rows
            .iter()
            .filter(|r| r.day >= lo && r.day < lo + 7)
            .collect();
        if sub.is_empty() {
            continue;
        }
        let mean = |f: &dyn Fn(&CensusRow) -> f64| -> f64 {
            sub.iter().map(|r| f(r)).sum::<f64>() / sub.len() as f64
        };
        let d0 = mean(&|r| r.depth as f64);
        println!(
            "  day {lo:2}-{:<2} n={:<7} hand={:4.2} corn={:5.2} depth={:4.2}  +1={:+.3} +3={:+.3}",
            lo + 6,
            sub.len(),
            mean(&|r| r.hand as f64),
            mean(&|r| r.corn as f64),
            d0,
            mean(&|r| r.depth_plus[0] as f64) - d0,
            mean(&|r| r.depth_plus[2] as f64) - d0,
        );
    }
}

// =======================================================================
// --verify
// =======================================================================

/// The driver above is a copy of `record::play_game`. This is what stops the
/// copy drifting: same seeds, same agents, same RNG stream, and the scores must
/// agree exactly.
fn verify(spec: &AgentSpec, seed0: u64, n: u64) {
    let cfg = GameConfig::evaluation();
    let mut bad = 0;
    for i in 0..n {
        let seed = seed0 + i;
        let a = spec.instance();
        let agents: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|_| a.as_ref());

        let mut r1 = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
        let t = play_traced(seed, &agents, &cfg, &mut r1, false, None, None);

        let mut r2 = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
        let g = record::play_game(seed, &agents, &cfg, &mut r2);

        if t.scores != g.scores || t.days != g.days {
            bad += 1;
            println!(
                "seed {seed}: traced {:?} d{} vs play_game {:?} d{}",
                t.scores, t.days, g.scores, g.days
            );
        }
    }
    println!("--verify: {} seeds, {bad} disagreements", n);
    if bad > 0 {
        std::process::exit(1);
    }
}

/// Silences the unused-import warning when the file is compiled without the
/// census path being referenced; also documents the horizon the plans use.
#[allow(dead_code)]
const _LAST_DAY: u8 = LAST_DAY;

#[allow(dead_code)]
fn _unused(m: &Move) -> usize {
    m.n_workers()
}
