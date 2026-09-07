//! Paranoid minimax with alpha-beta pruning over whole turns.
//!
//! # What the other three seats are assumed to do
//!
//! Alpha-beta is a two-player, zero-sum algorithm and Tzolk'in seats four. The
//! textbook reduction is *paranoid* search: collapse the four scores into the
//! single quantity [`eval::margin`] — how far the root player stands ahead of
//! whichever opponent is currently doing best — and let every opponent minimise
//! it. That is a genuine two-valued game, so alpha-beta prunes soundly rather
//! than "mostly", which is the failure mode of pruning a `max^n` tree.
//!
//! It is also false, and measurably so. Three opponents do not coordinate, and
//! against agents that simply play their own best move the paranoid search is
//! **not distinguishable from no search at all**: +0.98 centred score against
//! `heuristic:full` over 120 games, 95% CI -3.74..+5.70. Swapping in
//! [`Opponents::Greedy`] — each opponent plays the move its own evaluator likes
//! best, and nothing else is tried at that ply — gives +5.80 (+1.86..+9.75) at
//! the same depth and +12.95 (+8.89..+17.01) at depth 8. The greedy model
//! cannot prune, because a one-child node has nothing to cut; it wins anyway,
//! because being cheap buys depth and because it is *true*.
//!
//! Both are kept and both are the default of nothing: [`Config::opponents`]
//! chooses, and `Greedy` is what `Config::default` picks. Paranoid is the one
//! to reach for against an opponent that is actually trying to stop you.
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
//!
//! **Enumeration is the cost, not evaluation.** Expanding one interior node
//! walks ~200 moves to return one or two, and generating those moves is nearly
//! the whole bill: 2.16 us/move to walk the generator alone against 2.29 us to
//! walk it *and* apply *and* score each move, so `successor` plus `heuristic`
//! are about 6% between them. A depth-8 turn enumerates 23k moves to reach 150
//! leaves — 150 generated moves per leaf, where a textbook alpha-beta spends
//! about one.
//! Anything that makes this search faster has to enumerate less
//! ([`Config::cap_per_width`]) or enumerate once and remember
//! ([`Search::tt`]); nothing else is big enough to matter.
//!
//! # What was measured and did not work
//!
//! Recorded here because each of these is the obvious next thing to reach for,
//! and the measurement says not to. Arena figures are head-to-head against
//! `minimax:8:200::greedy`, solo mode, null 0.
//!
//! * **Depth.** The search is not short of it. A depth-8 turn finishes in
//!   95 ms of its 200 ms budget because it ran out of *plies*, not time — and
//!   spending the rest buys nothing: depth 16 scores **+0.26** centred
//!   (-2.14..+2.65, p = 0.83, 60 blocks / 240 games) and a depth-40 single
//!   line **-0.83** (-3.58..+1.92, p = 0.55, 60 blocks). Under `Greedy` the
//!   extra plies are extra turns of a *fictional* opponent, and the fiction
//!   stops paying long before the plies run out.
//! * **Anything built on a beta cutoff.** This search does not prune. Under
//!   `Greedy` every opponent node is one child wide, so beta never leaves
//!   `INF`, and the measured cutoff count is **0.0 per turn** at depths 4, 8
//!   and 12 alike. Killer moves, a history heuristic, PVS and aspiration
//!   windows all sharpen a cutoff that never happens. [`Config::opp_width`]
//!   above 1 is the only thing that would give them something to bite on.
//! * **The horizon story.** The worry was that a placement scores nothing until
//!   it is retrieved rounds later, so a one-ply root beam would rank every
//!   placement below every retrieval and never deepen one. It is the other way
//!   round: placements are 8.8% of the legal move list (3420/38724) but 40.8%
//!   of the root beam of 12 and 51.9% of the one-ply best pick. The beam
//!   *over*-selects placements more than fourfold, and deepening overturns the
//!   one-ply choice on 61% of positions, so it is not rubber-stamping either.
//!   A retrieval-selective extension would be solving a problem this search
//!   does not have. (210 positions from `heuristic:16` games.)
//! * **A wider root beam.** Doubling it to 24 on top of the cheap enumeration
//!   below is **+1.38** (-7.79..+10.55, p = 0.74, 11 blocks / 44 games) — a
//!   run stopped early because the interval was going nowhere. The exhaustive
//!   first ply already ranks every move; the 13th-best is not where the points
//!   are.
//!
//! The one thing that did work is [`Config::cap_per_width`], and it worked by
//! making nodes cheaper rather than by searching more of them.
//!
//! And the larger result, which is not about this file: at the same ~20 ms a
//! turn, `mcts:2048` beats the tuned alpha-beta by **+9.32** centred (95% CI
//! +6.09..+12.55, 39 blocks / 156 games). See the header of `crate::mcts`.
//! Whole-turn alpha-beta over a four-player game with a fabricated opponent
//! model is not obviously the right instrument here, and the arena says so.

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

/// What the search assumes the other three seats will do.
///
/// This is the load-bearing choice in a four-player search, and neither answer
/// is right — that is why it is a knob and not a constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opponents {
    /// Every opponent plays the move that hurts the root player most.
    ///
    /// The textbook reduction: it collapses four scores to one and makes the
    /// game two-valued, so alpha-beta prunes soundly. It is also false —
    /// three opponents do not coordinate — and the error is not random. It
    /// systematically overrates lines whose refutation costs the refuter
    /// nothing and underrates anything an opponent would have to hurt itself
    /// to punish.
    Paranoid,
    /// Every opponent plays the move its own evaluation likes best, and nothing
    /// else is considered at that ply.
    ///
    /// A `max^n` model with a width-1 opponent beam. It cannot prune — there is
    /// only one child, so there is nothing to cut — but it is far cheaper for
    /// the same reason, which buys depth. It matches what the agents in the
    /// arena actually do, at the cost of never seeing a move an opponent
    /// happens to have available and the model did not pick.
    Greedy,
}

/// How wide the search is at each ply, and when it must stop.
#[derive(Clone, Debug)]
pub struct Config {
    /// What the other three seats are assumed to do. See [`Opponents`].
    pub opponents: Opponents,
    /// Plies to look ahead, counting the root's own move as ply 1. Four is one
    /// full round: my move and one reply from each opponent.
    pub max_depth: u8,
    /// Moves searched at each ply, index 0 being the root. Past the end of the
    /// slice the last entry repeats.
    pub widths: Vec<usize>,
    /// Candidate moves enumerated at an *interior* node before the walk is cut
    /// short, **per move kind**. Placements and retrievals get this budget
    /// each; the root ignores it entirely.
    pub interior_cap: usize,
    /// Rollout-policy draws mixed into an interior node's candidates when the
    /// walk above was cut short. See [`Search::candidates`].
    pub top_up: usize,
    /// Wall-clock budget for *deepening*. The exhaustive first ply is not
    /// covered by it — see [`Config::root_budget`] — so a turn costs up to the
    /// two added together.
    pub budget: Duration,
    /// Wall-clock budget for the exhaustive first ply.
    ///
    /// `None` means walk the whole move list however long that takes, which is
    /// right when a human is waiting on one position and wrong for anything
    /// driving a game: the widest turns a strong player reaches run to millions
    /// of moves. A search that hit this reports `exhaustive: false` rather than
    /// claiming a shortlist it did not earn.
    pub root_budget: Option<Duration>,
    /// Ranked root moves to report.
    pub keep: usize,
    /// AB-HARNESS (temporary): turn the shortlist cache off, to race the two
    /// against each other in one arena process. Delete with the `nocache` spec
    /// flag once the measurement is banked.
    pub cache: bool,
    /// AB-HARNESS (temporary): index `widths` by the *mover's own* turn number
    /// rather than by ply. Under `Greedy` the three opponent plies are width 1
    /// whatever `widths` says, so ply indexing spends entries 1..3 on nobody
    /// and hands the root player's second turn `widths[4]`.
    pub own_width: bool,
    /// AB-HARNESS (temporary): keep the shortlist cache across turns of a game.
    pub keep_tt: bool,
    /// Rank moves the search deepened above moves it did not.
    ///
    /// Without it the deepening loop sorts the whole retained list on score
    /// alone, and a move holding only its one-ply margin can finish first and
    /// be played — see [`Stats::best_deepened`]. Sorting the deepened set ahead
    /// of the rest restores what the module header describes: the beam is the
    /// best `widths[0]` at one ply, and the answer comes from inside it.
    pub beam_first: bool,
    /// How many moves an opponent is allowed under [`Opponents::Greedy`].
    ///
    /// One is the model as originally written: the opponent plays its own best
    /// move and nothing else is tried. Raising it hedges — the opponent is
    /// assumed to play one of the `k` moves *its own* evaluator likes, and the
    /// search takes the worst of those for the root player. That is weaker than
    /// paranoia, which lets an opponent play any move at all however much it
    /// costs them, and stronger than assuming they find exactly the move the
    /// model predicted.
    ///
    /// It is also the only way this search prunes. Under width-1 opponents no
    /// min node has a sibling, so beta never moves off `INF` and the measured
    /// cutoff count is **0.0 per turn** at every depth — killer moves, history
    /// ordering and PVS all have nothing to bite on. At `k >= 2` min nodes
    /// branch, beta tightens, and alpha-beta starts doing its job.
    ///
    /// Cheap in the place that matters: the extra moves come out of the same
    /// `candidates` call, so it costs subtree and not enumeration, and
    /// enumeration is 94% of a node.
    pub opp_width: usize,
    /// Per-kind enumeration cap scaled by the node's own beam width,
    /// `k * width`, clamped above by [`Config::interior_cap`]. `None` restores
    /// the flat [`Config::interior_cap`] everywhere.
    ///
    /// This is the one change that made the search both faster and stronger,
    /// and it works because of what a node actually costs. Expanding one
    /// interior node enumerates ~200 moves to return one or two, and
    /// enumeration is 94% of that; under `Greedy` three quarters of those
    /// expansions are opponent plies of width 1, spending the full 400-per-kind
    /// budget to report a single argmax. Scaling the cap by width spends the
    /// budget only where the beam is wide enough to use a ranked list.
    ///
    /// At `Some(25)` a turn costs **19.8 ms against 95.3 ms** flat — 4.8x — and
    /// reaches depth 8.0 on 11.2 of 12 root moves where the flat cap manages
    /// 7.7 on 10.2. It is worth **+3.24** centred score head to head against
    /// the flat-cap search (95% CI +0.35..+6.13, p = 0.022, 32 blocks / 128
    /// games), so the narrower opponent model costs nothing measurable and
    /// finishing the beam is worth real points. Cheaper *and* stronger, which
    /// is only surprising until you notice the flat cap was spending 400 moves
    /// per kind to pick an opponent's single reply.
    ///
    /// Note this is *not* the same as lowering `interior_cap` to 25, which
    /// would starve the root player's own turns too; the point is the taper.
    pub cap_per_width: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            // Measured against `heuristic:full` over 120 games each: paranoid
            // at depth 4 scores +0.98 centred (95% CI -3.74..+5.70, p = 0.66 --
            // not distinguishable from one ply), greedy at the same depth
            // +5.80 (+1.86..+9.75, p = 0.003), and greedy at depth 8 +12.95
            // (+8.89..+17.01). Paranoid is the model alpha-beta was built for
            // and it prunes far harder, but it is modelling opponents that do
            // not exist; the cheaper, truer assumption wins and keeps winning
            // as it goes deeper, which is why the depth default follows it up.
            //
            // It stops paying at 8. Depth 16 is +0.26 centred against depth 8
            // head to head (-2.14..+2.65, p = 0.83, 60 blocks / 240 games) and
            // a depth-40 single line is -0.83 (-3.58..+1.92) -- so 8 is not a
            // compromise with the clock, it is where the greedy opponent model
            // stops being worth believing. A depth-8 turn finishes in 95 ms of
            // its 200 ms budget; the rest of the budget has nowhere to go.
            opponents: Opponents::Greedy,
            max_depth: 8,
            widths: vec![12, 6, 4, 3, 2],
            interior_cap: 400,
            top_up: 16,
            budget: Duration::from_millis(600),
            root_budget: Some(crate::eval::FULL_BUDGET),
            keep: 10,
            cache: true,
            own_width: false,
            keep_tt: false,
            // Both measured; see each field. `beam_first` is a correctness
            // fix worth no points, `cap_per_width` is worth +3.24 at 4.8x less
            // wall clock, and the pair of them is what this default is for.
            beam_first: true,
            opp_width: 1,
            cap_per_width: Some(25),
        }
    }
}

impl Config {
    /// A deepening budget in milliseconds, and a root budget scaled to match.
    ///
    /// The two move together because a caller asking for a 120 ms search is
    /// asking for a fast turn, and paying the default 2.5 s root would make
    /// that number meaningless. The root gets the larger share: it is the one
    /// ply that has to see everything, and a beam over a shortlist it never
    /// finished building is worth less than a shallower search over a complete
    /// one.
    pub fn with_budget_ms(mut self, ms: u64) -> Self {
        self.budget = Duration::from_millis(ms);
        self.root_budget = Some(Duration::from_millis(ms.saturating_mul(3).max(50)));
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
    /// Calls to [`Search::candidates`] -- one per interior node expanded.
    pub cand_calls: u64,
    /// Moves enumerated *and statically scored* inside those calls. This is the
    /// search's real unit of work: one `successor` + one `heuristic` each.
    pub cand_moves: u64,
    /// Deepest ply reached. When `partial`, only `deepened` root moves got it.
    pub depth: u8,
    /// Root moves searched at `depth`. The rest carry `depth - 1` scores.
    pub deepened: usize,
    /// Whether the budget cut the last iteration short.
    pub partial: bool,
    /// Whether the move finally returned is one the search actually deepened.
    ///
    /// It can fail to be. `MinimaxAgent` raises `Config::keep` to `MAX_VISITS`
    /// (24) so a recorded turn has a ranked list to store, but the beam is
    /// `widths[0]` (12) — and the deepening loop sorts all 24 together, so a
    /// move carrying nothing but its one-ply margin competes against moves that
    /// were searched eight plies, and can win.
    ///
    /// In practice it almost never does: **1 turn in 210** at depth 8, because
    /// the deep score of a move the one-ply pass already liked stays near the
    /// top. So [`Config::beam_first`] is a correctness fix rather than a
    /// strength one — an effect this rare is far below what the arena can
    /// resolve, and this counter, not a match result, is the instrument that
    /// settles it.
    pub best_deepened: bool,
    /// Legal moves at the root. Always the whole set.
    pub root_moves: usize,
    /// How long the exhaustive first ply took. The rest of `elapsed` is
    /// deepening, so the two together say which half a slow turn was spent in.
    pub root_elapsed: Duration,
    pub elapsed: Duration,
}

/// One node's shortlist, kept so the next deepening pass does not rebuild it.
///
/// The table this lives in used to hold *values* keyed on `(GameState, depth)`,
/// which measured 0.3 hits per turn against ~100 interior nodes — because the
/// depth was in the key, so iteration `d + 1` could never read iteration `d`'s
/// answer, and because a turn-granular tree of this shape barely transposes at
/// all. Caching the shortlist instead hits on nearly every node above the
/// frontier, for the reason iterative deepening is supposed to be cheap: pass
/// `d + 1` walks the same nodes pass `d` did. It matters far more here than in
/// a normal alpha-beta because a node in this game costs ~200 enumerated moves
/// to *expand* — `visit_moves_of` plus a `successor` and a `heuristic` per
/// candidate — and almost nothing to search.
///
/// **It earns its place on cost, not on strength.** At depth 8 and a 200 ms
/// budget the cache hits 70% of nodes and gets the turn done in 73 ms at depth
/// 7.9 on 11.2 of 12 root moves; without it the same turn takes 159 ms and
/// reaches depth 7.2 on 8.5 of 12. That is a 2.2x saving and a deeper answer —
/// and it is worth **+1.73** centred score head to head (95% CI -0.84..+4.29,
/// p = 0.18, 60 blocks / 240 games), which is to say not measurably anything.
/// Keep it because it is free, not because it wins games: the extra depth it
/// buys is depth the arena has already shown does not pay (see the module
/// header). The `nocache` spec flag is what raced the two.
struct Entry {
    /// Best first, already deduplicated by `Move::same_effect`.
    moves: Vec<Move>,
    /// How many were asked for when it was built. A probe wanting more has to
    /// rebuild; in practice a state always recurs at the same ply, so it never
    /// does.
    width: usize,
}

/// Shortlists retained before the table is dropped wholesale.
///
/// Nodes per search run to the low thousands, so this is a guard against a
/// pathological position rather than a working limit. Clearing outright beats
/// evicting: entries are worth something only within one search, and the cost
/// of getting it wrong is one recomputation.
const MAX_ENTRIES: usize = 60_000;

/// One search. Owns the shortlist cache, which is scoped to a single
/// `search` call.
pub struct Search {
    pub cfg: Config,
    root: PlayerId,
    /// Keyed on the whole state rather than a hash of it. `GameState` is 312
    /// bytes and `Copy`, so this is a memcpy per insert and no collision can
    /// ever hand one position another's shortlist — which would be an *illegal
    /// move*, not merely a wrong score, so the key stays exact.
    ///
    /// `g.current` is the mover at every interior node, so the state alone
    /// pins the player too.
    tt: FxHashMap<GameState, Entry>,
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
    ///
    /// `p` must be the player to move. The seat is written into the working
    /// copy rather than asserted, so that a caller analysing a position out of
    /// turn gets the answer it asked for instead of a search that hands the
    /// turn to the wrong player at every round boundary.
    pub fn search(&mut self, g: &GameState, p: PlayerId) -> Ranking {
        let started = Instant::now();
        self.deadline = started + self.cfg.budget;
        self.root = p;
        self.stats = Stats::default();
        if !self.cfg.keep_tt {
            self.tt.clear();
        }

        let g = &{
            let mut probe = *g;
            probe.current = p;
            probe
        };

        // ---- ply 1: every move, scored ----------------------------------
        let keep = self.cfg.keep.max(self.cfg.width_at(0));
        let mut roots = eval::rank_all_within(g, p, keep, self.cfg.root_budget, |s| {
            eval::margin(s, p)
        });
        self.stats.root_moves = roots.total;
        self.stats.root_elapsed = started.elapsed();
        if roots.moves.is_empty() {
            roots.note = "no legal move".into();
            return roots;
        }

        // ---- deepening --------------------------------------------------
        let beam = self.cfg.width_at(0).min(roots.moves.len());
        let mut best_depth = 1u8;
        let mut deepened = beam;

        // The third field is whether this move's score came from a real
        // deepening rather than from the one-ply root pass. It has to ride
        // along rather than be recomputed, because a move can be deepened in
        // one iteration and fall outside the beam in the next while keeping the
        // deep score it earned.
        let mut work: Vec<(Move, f32, bool)> = roots
            .moves
            .iter()
            .map(|(m, s)| (m.clone(), *s, false))
            .collect();

        for depth in 2..=self.cfg.max_depth {
            self.out_of_time = false;
            let mut scored: Vec<(Move, f32, bool)> = Vec::with_capacity(work.len());
            let mut done = 0usize;

            for (i, (m, prev, was_deep)) in work.iter().enumerate() {
                // Outside the beam, or out of time: carry the previous score
                // forward. It is an honest number, just a shallower one.
                if i >= beam || self.out_of_time {
                    scored.push((m.clone(), *prev, *was_deep));
                    continue;
                }
                if self.expired() {
                    self.out_of_time = true;
                    scored.push((m.clone(), *prev, *was_deep));
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
                    scored.push((m.clone(), *prev, *was_deep));
                    continue;
                }
                scored.push((m.clone(), v, true));
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
            if self.cfg.beam_first {
                // See [`Config::beam_first`]: searched beats unsearched before
                // score is even consulted.
                scored.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.total_cmp(&a.1)));
            } else {
                scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            }
            work = scored;
            best_depth = depth;
            deepened = done;
            if self.out_of_time {
                break;
            }
        }

        self.stats.best_deepened = work.first().map(|t| t.2).unwrap_or(false);
        roots.moves = work.into_iter().map(|(m, s, _)| (m, s)).collect();

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
        let root_note = if roots.exhaustive {
            format!("all {} root moves{restated}", self.stats.root_moves)
        } else {
            format!(
                "{} root moves{restated} — the list is wider than the root budget",
                self.stats.root_moves
            )
        };
        roots.note = format!(
            "{depth_note} of {root_note} · {} nodes, {} cutoffs · {:.0} ms",
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

        let mover = g.current;
        let maximizing = mover == self.root;

        // Under `Greedy`, an opponent's ply is one move wide: whichever its own
        // evaluation prefers. The min over a single child is that child, so the
        // rest of this function needs no special case.
        let width = if maximizing || self.cfg.opponents == Opponents::Paranoid {
            // Under `Greedy` a ply index counts three opponent plies that are
            // width 1 no matter what `widths` holds, so `widths[1..3]` is spent
            // on nobody and the root player's second turn gets `widths[4]`.
            // Counting the mover's own turns instead makes the taper mean what
            // it reads like.
            if self.cfg.own_width && self.cfg.opponents == Opponents::Greedy {
                self.cfg.width_at(ply / N_PLAYERS)
            } else {
                self.cfg.width_at(ply)
            }
        } else {
            // See [`Config::opp_width`]: 1 is the pure greedy model, and the
            // reason this search records 0.0 cutoffs a turn.
            self.cfg.opp_width.max(1)
        };
        let cands = self.candidates(g, mover, width);
        if cands.is_empty() {
            // No legal move at all should be impossible — the pity rule always
            // leaves something — but a search must not invent a score for a
            // position it could not expand.
            self.stats.leaves += 1;
            return eval::margin(g, self.root);
        }

        self.stats.nodes += 1;
        let mut best = if maximizing { -INF } else { INF };

        for (i, m) in cands.iter().enumerate() {
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
                // Only a break that skips a sibling is a cutoff. Under
                // `Greedy` every opponent node is one child wide, so the
                // window test fires on the last child constantly and the old
                // unconditional count read 89 per turn at depth 8 where the
                // true figure is a fifth of that.
                if i + 1 < cands.len() {
                    self.stats.cutoffs += 1;
                }
                break;
            }
        }

        best
    }

    /// The moves an interior node will actually search, best first.
    ///
    /// This is where culling happens, and it is the one place in the search
    /// where a legal move can go unconsidered. Three things keep that honest:
    ///
    /// * **The budget is per kind.** Placements and retrievals are walked
    ///   separately, each capped at [`Config::interior_cap`]. A single capped
    ///   walk would spend the whole budget on placements — they come first in
    ///   traversal order and three workers in hand already make a couple of
    ///   hundred — and never reach a retrieval, which is where the points are.
    /// * **A cut-short walk is topped up by sampling.** Within a kind the
    ///   prefix is still traversal order, not a sample, so
    ///   [`Config::top_up`] draws from `sample_legal_move` are mixed in — the
    ///   project's own rollout policy, shaped like real play.
    /// * **The root does none of this.** Ply 1 is exhaustive.
    ///
    /// Ordering uses the *mover's* own estimate rather than the root's margin:
    /// it is a quarter of the cost (one `heuristic` call, not four) and ordering
    /// only has to be roughly right. `ab` computes the real value.
    fn candidates(&mut self, g: &GameState, mover: PlayerId, width: usize) -> Vec<Move> {
        let width = width.max(1);
        if self.cfg.cache {
        if let Some(e) = self.tt.get(g) {
            if e.width >= width || e.moves.len() < e.width {
                // `moves.len() < width` means the node really is that narrow,
                // so a shorter list is the whole list and not a truncation.
                self.stats.tt_hits += 1;
                let n = width.min(e.moves.len());
                return e.moves[..n].to_vec();
            }
        }
        }
        // A width-1 node needs an argmax, not a ranked shortlist, so it has no
        // use for the full budget -- see [`Config::cap_per_width`].
        let cap = match self.cfg.cap_per_width {
            Some(k) => k.saturating_mul(width).clamp(k.max(1), self.cfg.interior_cap),
            None => self.cfg.interior_cap,
        };
        let mut seen: Vec<(Move, f32)> = Vec::new();
        let mut capped = false;

        self.stats.cand_calls += 1;
        for kind in [moves::Kinds::Placements, moves::Kinds::Retrievals] {
            let mut n = 0usize;
            let flow = moves::visit_moves_of(g, mover, kind, |m| {
                n += 1;
                let succ = eval::successor(g, mover, m);
                seen.push((m.clone(), eval::heuristic(&succ, mover)));
                if n >= cap {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            capped |= flow.is_break();
            self.stats.cand_moves += n as u64;
        }

        if seen.is_empty() {
            // No ordinary move exists, so the gods take pity. That set is tiny
            // and `visit_legal_moves` is the only thing that generates it.
            let _ = moves::visit_legal_moves(g, mover, |m| {
                let succ = eval::successor(g, mover, m);
                seen.push((m.clone(), eval::heuristic(&succ, mover)));
                ControlFlow::Continue(())
            });
        } else if capped {
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

        if !self.cfg.cache {
            return out;
        }
        if self.tt.len() >= MAX_ENTRIES {
            self.tt.clear();
        }
        self.tt.insert(
            *g,
            Entry {
                moves: out.clone(),
                width,
            },
        );
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
