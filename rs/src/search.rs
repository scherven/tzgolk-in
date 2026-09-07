//! Paranoid minimax with alpha-beta pruning over whole turns.
//!
//! # Why paranoid
//!
//! Alpha-beta is a two-player, zero-sum algorithm and Tzolk'in seats four. The
//! standard reduction is *paranoid* search: collapse the four scores into the
//! single quantity [`eval::margin`] — how far the root player stands ahead of
//! whichever opponent is currently doing best — and let every opponent minimise
//! it. That is a genuine two-valued game, so alpha-beta prunes soundly rather
//! than "mostly", which is the failure mode of pruning a `max^n` tree.
//!
//! It is pessimistic by construction: three opponents will not in fact
//! coordinate against one player. The price is that the search underrates moves
//! whose refutation requires an opponent to hurt themselves, and that is the
//! right way to be wrong for a game where the alternative — assuming everyone
//! plays their own best move — cannot prune at all.
//!
//! # The shape of the tree
//!
//! One ply is one player's whole turn. The turn is the unit because that is
//! what `moves::Move` is and what `moves::apply_move` consumes; the factored
//! sub-decision chain in `tree.rs` exists for MCTS and the policy head, and
//! nothing here needs it.
//!
//! Round boundaries resolve inside [`advance_seat`]: when the seat wraps back
//! to the first player, the first-player space is handed over, the extra
//! calendar day is decided, and the calendar advances with its food day and
//! scoring. A search that skipped that would never see a food day coming, which
//! is exactly the thing the evaluator most needs help with.
//!
//! # Where the cost goes, and what is culled
//!
//! **The root is exhaustive.** Every legal move is generated and scored — no
//! cap, no sampling — because that is the one place where missing a move means
//! never playing it. Scoring is streamed through [`eval::TopK`]-style bounded
//! retention, so a 1.9M-move node costs time but not memory.
//!
//! Everything below the root is culled, and the static evaluation is what does
//! the culling: at each interior node the candidate moves are scored one ply
//! deep, sorted, and only the best [`Config::widths`] of them are searched.
//! Good ordering is also what makes alpha-beta pay — a beam that is already
//! sorted best-first produces a cutoff on the first or second child most of the
//! time.

use crate::eval::{self, Ranking};
use crate::ids::*;
use crate::moves::{self, Move};
use crate::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustc_hash::FxHashMap;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

const INF: f32 = 1e9;

/// How wide the search is at each ply, and when it must stop.
#[derive(Clone, Debug)]
pub struct Config {
    /// Plies to look ahead, counting the root's own move as ply 1. Four is one
    /// full round: my move and one reply from each opponent.
    pub max_depth: u8,
    /// Moves searched at each ply, index 0 being the root. Past the end of the
    /// slice the last entry repeats.
    pub widths: Vec<usize>,
    /// Candidate moves enumerated at an *interior* node before the walk is cut
    /// short. The root ignores this.
    pub interior_cap: usize,
    /// Rollout-policy draws mixed into an interior node's candidates when the
    /// walk above was cut short. See [`Search::candidates`].
    pub top_up: usize,
    /// Wall-clock budget for one `search` call. An iteration that runs out is
    /// discarded whole, so a truncated search returns the last depth that
    /// finished rather than a half-updated ranking.
    pub budget: Duration,
    /// Ranked root moves to report.
    pub keep: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_depth: 4,
            widths: vec![12, 6, 4, 3, 2],
            interior_cap: 3_000,
            top_up: 24,
            budget: Duration::from_millis(1_500),
            keep: 10,
        }
    }
}

impl Config {
    /// A budget in milliseconds, at the default shape.
    pub fn with_budget_ms(mut self, ms: u64) -> Self {
        self.budget = Duration::from_millis(ms);
        self
    }

    pub fn with_depth(mut self, d: u8) -> Self {
        self.max_depth = d.max(1);
        self
    }

    /// Moves searched at `ply`, the last configured width repeating past the
    /// end of the slice.
    pub fn width_at(&self, ply: usize) -> usize {
        *self
            .widths
            .get(ply)
            .or_else(|| self.widths.last())
            .unwrap_or(&3)
    }
}

/// What one search did, for the status line and for tuning.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Interior nodes whose children were searched.
    pub nodes: u64,
    /// Leaves scored by the static evaluator.
    pub leaves: u64,
    /// Beta cutoffs — branches alpha-beta never had to look at.
    pub cutoffs: u64,
    /// Nodes answered from the transposition table.
    pub tt_hits: u64,
    /// Deepest ply reached. When `partial`, only `deepened` root moves got it.
    pub depth: u8,
    /// Root moves searched at `depth`. The rest carry `depth - 1` scores.
    pub deepened: usize,
    /// Whether the budget cut the last iteration short.
    pub partial: bool,
    /// Legal moves at the root. Always the whole set.
    pub root_moves: usize,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    /// The true value is at least this.
    Lower,
    /// The true value is at most this.
    Upper,
}

/// The depth an entry was proved to is part of the key, not the payload, so
/// a probe can never read a shallower search's answer as a deeper one's.
#[derive(Clone, Copy)]
struct Entry {
    value: f32,
    bound: Bound,
}

/// One search. Owns the transposition table, so reusing it across turns of the
/// same game keeps the table warm.
pub struct Search {
    pub cfg: Config,
    root: PlayerId,
    /// Keyed on the whole state rather than a hash of it. `GameState` is 312
    /// bytes and `Copy`, so this is a memcpy per insert and no collision can
    /// ever return another position's score — worth it for a table that also
    /// backs an analysis display.
    tt: FxHashMap<(GameState, u8), Entry>,
    rng: StdRng,
    stats: Stats,
    deadline: Instant,
    /// Set when the budget ran out mid-iteration; the iteration is then thrown
    /// away rather than reported.
    out_of_time: bool,
}

impl Search {
    pub fn new(cfg: Config) -> Self {
        Search {
            cfg,
            root: PlayerId(0),
            tt: FxHashMap::default(),
            rng: StdRng::seed_from_u64(0x7201_C1),
            stats: Stats::default(),
            deadline: Instant::now(),
            out_of_time: false,
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Drop the transposition table. Call between games, not between turns.
    pub fn clear(&mut self) {
        self.tt.clear();
    }

    /// Search `g` for `p` and return its moves, best first.
    ///
    /// Ply 1 is exhaustive: every legal move is generated and scored. The best
    /// `widths[0]` of those are then deepened by iterative deepening until the
    /// budget runs out.
    pub fn search(&mut self, g: &GameState, p: PlayerId) -> Ranking {
        let started = Instant::now();
        self.deadline = started + self.cfg.budget;
        self.root = p;
        self.stats = Stats::default();
        self.tt.clear();

        // ---- ply 1: every move, scored ----------------------------------
        let keep = self.cfg.keep.max(self.cfg.width_at(0));
        let mut roots = eval::rank_all(g, p, keep);
        self.stats.root_moves = roots.total;
        if roots.moves.is_empty() {
            roots.note = "no legal move".into();
            return roots;
        }

        // ---- deepening --------------------------------------------------
        let beam = self.cfg.width_at(0).min(roots.moves.len());
        let mut best_depth = 1u8;
        let mut deepened = beam;

        for depth in 2..=self.cfg.max_depth {
            self.out_of_time = false;
            let mut scored: Vec<(Move, f32)> = Vec::with_capacity(roots.moves.len());
            let mut done = 0usize;

            for (i, (m, prev)) in roots.moves.iter().enumerate() {
                // Outside the beam, or out of time: carry the previous score
                // forward. It is an honest number, just a shallower one.
                if i >= beam || self.out_of_time {
                    scored.push((m.clone(), *prev));
                    continue;
                }
                if self.expired() {
                    self.out_of_time = true;
                    scored.push((m.clone(), *prev));
                    continue;
                }
                let child = after_turn(g, p, m);
                // A full window per root move. Narrowing to `(alpha, INF)` would
                // prune more, but every move outside the principal variation
                // would come back as a bound rather than a value — and these
                // numbers are shown to a human, so they have to mean something
                // on their own.
                let v = self.ab(&child, depth - 1, 1, -INF, INF);
                if self.out_of_time {
                    // The budget ran out *inside* this subtree, so its value is
                    // a truncated search's guess. Keep the shallower number.
                    scored.push((m.clone(), *prev));
                    continue;
                }
                scored.push((m.clone(), v));
                done += 1;
            }

            // A partial iteration is still worth keeping: the root list was
            // already ordered best-first, so the moves that got the deeper look
            // are the ones that mattered. What is *not* acceptable is dropping
            // to a shallower answer than the one already in hand, so an
            // iteration that deepened nothing is discarded.
            if done == 0 {
                break;
            }
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            roots.moves = scored;
            best_depth = depth;
            deepened = done;
            if self.out_of_time {
                break;
            }
        }

        self.stats.depth = best_depth;
        self.stats.deepened = deepened;
        self.stats.partial = deepened < beam;
        self.stats.elapsed = started.elapsed();
        roots.moves.truncate(self.cfg.keep);

        let dupes = roots.total - roots.distinct;
        let restated = if dupes > 0 {
            format!(" ({} restatements)", dupes)
        } else {
            String::new()
        };
        let depth_note = if self.stats.partial {
            format!("depth {best_depth} on {deepened}/{beam}, else {}", best_depth - 1)
        } else {
            format!("depth {best_depth} on top {beam}")
        };
        roots.note = format!(
            "{depth_note} of all {} root moves{restated} · {} nodes, {} cutoffs · {:.0} ms",
            self.stats.root_moves,
            self.stats.nodes,
            self.stats.cutoffs,
            self.stats.elapsed.as_secs_f64() * 1e3
        );
        roots
    }

    /// The best move alone.
    pub fn best_move(&mut self, g: &GameState, p: PlayerId) -> Option<Move> {
        self.search(g, p).moves.into_iter().next().map(|(m, _)| m)
    }

    #[inline]
    fn expired(&self) -> bool {
        Instant::now() >= self.deadline
    }

    /// Alpha-beta over `g`, whose mover is `g.current`.
    ///
    /// Maximising when the mover is the root player, minimising otherwise —
    /// three minimising plies in a row is the paranoid assumption made concrete.
    fn ab(&mut self, g: &GameState, depth: u8, ply: usize, mut alpha: f32, mut beta: f32) -> f32 {
        if g.over || depth == 0 {
            self.stats.leaves += 1;
            return eval::margin(g, self.root);
        }
        if self.expired() {
            self.out_of_time = true;
            self.stats.leaves += 1;
            return eval::margin(g, self.root);
        }

        if let Some(e) = self.tt.get(&(*g, depth)) {
            let e = *e;
            match e.bound {
                Bound::Exact => {
                    self.stats.tt_hits += 1;
                    return e.value;
                }
                Bound::Lower if e.value >= beta => {
                    self.stats.tt_hits += 1;
                    return e.value;
                }
                Bound::Upper if e.value <= alpha => {
                    self.stats.tt_hits += 1;
                    return e.value;
                }
                _ => {}
            }
        }

        let mover = g.current;
        let maximizing = mover == self.root;
        let (alpha0, beta0) = (alpha, beta);

        let cands = self.candidates(g, mover, self.cfg.width_at(ply));
        if cands.is_empty() {
            // No legal move at all should be impossible — the pity rule always
            // leaves something — but a search must not invent a score for a
            // position it could not expand.
            self.stats.leaves += 1;
            return eval::margin(g, self.root);
        }

        self.stats.nodes += 1;
        let mut best = if maximizing { -INF } else { INF };

        for m in &cands {
            let child = after_turn(g, mover, m);
            let v = self.ab(&child, depth - 1, ply + 1, alpha, beta);
            if maximizing {
                if v > best {
                    best = v;
                }
                if best > alpha {
                    alpha = best;
                }
            } else {
                if v < best {
                    best = v;
                }
                if best < beta {
                    beta = best;
                }
            }
            if alpha >= beta {
                self.stats.cutoffs += 1;
                break;
            }
        }

        // A value found after the budget expired is not a value; do not poison
        // the table with it.
        if !self.out_of_time {
            let bound = if best <= alpha0 {
                Bound::Upper
            } else if best >= beta0 {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.tt.insert((*g, depth), Entry { value: best, bound });
        }
        best
    }

    /// The moves an interior node will actually search, best first.
    ///
    /// This is where culling happens. The walk is cut off at
    /// [`Config::interior_cap`], which covers the great majority of positions
    /// whole — the median node has 42 moves — but a wide one gets a *prefix* in
    /// traversal order, and traversal order is not a sample: placements come
    /// before retrievals, and retrievals come worker by worker. So when the walk
    /// is cut short, [`Config::top_up`] draws from `sample_legal_move` are mixed
    /// in, which is the project's own rollout policy and is shaped like real
    /// play. The beam is then the best `width` by one-ply static score.
    ///
    /// Ordering uses the *mover's* own estimate rather than the root's margin:
    /// it is a quarter of the cost (one `heuristic` call, not four) and ordering
    /// only has to be roughly right. `ab` computes the real value.
    fn candidates(&mut self, g: &GameState, mover: PlayerId, width: usize) -> Vec<Move> {
        let cap = self.cfg.interior_cap;
        let mut seen: Vec<(Move, f32)> = Vec::new();
        let mut n = 0usize;

        let flow = moves::visit_legal_moves(g, mover, |m| {
            n += 1;
            let succ = eval::successor(g, mover, m);
            seen.push((m.clone(), eval::heuristic(&succ, mover)));
            if n >= cap {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });

        if flow.is_break() {
            for _ in 0..self.cfg.top_up {
                let Some(m) = moves::sample_legal_move(g, mover, &mut self.rng) else {
                    break;
                };
                if seen.iter().any(|(x, _)| *x == m) {
                    continue;
                }
                let succ = eval::successor(g, mover, &m);
                let s = eval::heuristic(&succ, mover);
                seen.push((m, s));
            }
        }

        // Sorted by the mover's own estimate in both directions: an opponent
        // minimises the root's margin by playing *well*, not by playing badly,
        // so the ordering is the same whoever is to move.
        seen.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Fill the beam with moves that actually differ. Equivalent placements
        // score identically and so land adjacent after the sort; without this
        // a width of six is routinely two positions searched three times each.
        let width = width.max(1);
        let mut out: Vec<Move> = Vec::with_capacity(width);
        for (m, _) in seen {
            if out.iter().any(|k| k.same_effect(&m)) {
                continue;
            }
            out.push(m);
            if out.len() >= width {
                break;
            }
        }
        out
    }
}

// ---- turn and round flow ------------------------------------------------

/// The position the next mover faces after `p` plays `m`.
///
/// Mirrors `record::play_game` exactly: apply, refill the building row, hand the
/// seat on, and resolve the round when the seat wraps.
pub fn after_turn(g: &GameState, p: PlayerId, m: &Move) -> GameState {
    debug_assert_eq!(g.current, p, "search advanced a seat out of turn");
    let mut next = eval::successor(g, p, m);
    advance_seat(&mut next);
    next
}

/// Hand the seat on, resolving the end of the round when it wraps.
pub fn advance_seat(g: &mut GameState) {
    g.current = g.current.next(1);
    if g.current != g.first_player {
        return;
    }

    let claimer = g.resolve_first_player();
    let mut days = 1u8;
    if let Some(q) = claimer {
        if g.may_take_extra_day(q) && prefers_extra_day(g, q) {
            g.spend_extra_day(q);
            days = 2;
        }
    }
    g.advance_days(days);
    // `resolve_first_player` may have moved the marker.
    g.current = g.first_player;
}

/// Whether the claimer would take the extra calendar day.
///
/// Modelled greedily on the claimer's own estimate, the same rule
/// `GreedyAgent::extra_day` uses, so the two agree about what a round boundary
/// does. Making it a search node instead would double the tree for a binary
/// decision that is rarely close.
pub fn prefers_extra_day(g: &GameState, q: PlayerId) -> bool {
    let mut one = *g;
    one.advance_days(1);

    let mut two = *g;
    two.spend_extra_day(q);
    two.advance_days(2);

    eval::heuristic(&two, q) > eval::heuristic(&one, q)
}
