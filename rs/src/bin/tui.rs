//! Watch a game. Read-only for now; human play drops in as extra key handlers.
//!
//!     cargo run --release --bin tui -- [seed]

use std::io;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use rand::SeedableRng;
use ratatui::prelude::*;

use tzolkin::eval::{self, margin, Ranking};
use tzolkin::game::Game;
use tzolkin::ids::PlayerId;
use tzolkin::moves::sample_legal_move;
use tzolkin::record::{Agent, TurnOutcome};
use tzolkin::ui::{self, App, MoveSource, Thinking};

const SHOWN: usize = 10;

/// How long the `full` view will walk a move list before giving up and saying
/// the position is wider than it could finish.
const FULL_VIEW_BUDGET: Duration = Duration::from_secs(5);

const HELP: &str = "\
tui -- watch a game

USAGE
  cargo run --release --bin tui -- [SEED] [--agent SPEC] [--view VIEW]

OPTIONS
  --agent SPEC   who plays when you press `n` or turn on autoplay, and whose
                 ranking the `agent` view shows. Without it the rollout policy
                 plays and the move list is scored by the built-in evaluator.
  --view VIEW    which move list to open on: sampled | full | agent.
                 Defaults to `agent` with --agent, `full` without.

AGENT SPECS  (the same ones arena and selfplay take)
  random                              the rollout policy
  heuristic[:K]                       one ply, over K sampled turns
  heuristic:full                      one ply, over EVERY legal move
  minimax[:DEPTH[:MS[:WIDTH]]]        paranoid alpha-beta; exhaustive first ply
  greedy:K:EVAL / greedy:full:EVAL    one ply over any evaluator
  mcts:SIMS[:EVAL][:FLAGS]            tree search; FLAGS is a comma-separated
                                      key=value list over MctsConfig, e.g.
                                      mcts:2048:pri=1ply,ptemp=4
  PATH.safetensors                    shorthand for mcts on a trained net

VIEWS  (cycle with `t`)
  sampled   ten draws from the rollout policy -- instant, and biased
  full      the ten best of EVERY legal move, by eval::margin
  agent     the agent's own ranking. For minimax and the one-ply agents that
            is every legal move, scored -- backed-up search values in the first
            case, the evaluator's in the second. For mcts it is the whole turns
            the tree actually walked, scored by visit share in percent, and the
            status line says so: a tree holds only the paths its simulations
            took, so that shortlist is not drawn from the whole move space and
            the panel does not claim it is.

KEYS
  j/k    move the selection        n    let the current player act
  enter  play the highlighted move r    recompute the list
  t      next view                 a    autoplay          q  quit

WHILE IT IS THINKING
  Every search runs on a worker thread, so the screen keeps redrawing and the
  Thinking pane says which search of the two a turn costs is running, what the
  first one returned, and how long it is taking against the last one. The keys
  that would change the position are refused until it lands; j/k and q are not.

Scores are `margin` -- this player's estimated final score less the best
opponent's, so positive means ahead -- except in the `agent` view under an mcts
agent, where the column is a visit share in percent. The status line always
names the scale in use.
";

fn main() -> io::Result<()> {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        return Ok(());
    }

    let seed: u64 = std::env::args()
        .skip(1)
        .find_map(|v| v.parse().ok())
        .unwrap_or(7);

    // `--agent SPEC` uses the same specs as the arena and self-play:
    //   heuristic:32                          the one-ply baseline
    //   mcts:800                              search on the heuristic evaluator
    //   mcts:800:ckpt/gen0007.safetensors     search on a trained net
    //   ckpt/gen0007.safetensors              shorthand for the above
    let spec = std::env::args()
        .position(|a| a == "--agent")
        .and_then(|i| std::env::args().nth(i + 1));
    // Records its decision nodes — that is what fills the search panel, and
    // with plain `parse_agent(s, false)` the panel came up empty the first
    // time. It does **not** explore: `record` used to imply self-play, so this
    // viewer was running MCTS with 0.25 Dirichlet noise deliberately scrambling
    // its root priors and playout-cap randomisation dropping seven turns in
    // eight to an eighth of the budget. Noise is a replay-buffer device; a
    // human watching wants the policy the search actually believes.
    // `Arc`, not `Box`: the viewer hands a clone to a worker thread so a
    // multi-second turn does not block the draw loop.
    let agent: Option<std::sync::Arc<dyn tzolkin::record::Agent>> = match &spec {
        Some(s) => match tzolkin::record::parse_analysis_agent(s) {
            Ok(a) => Some(std::sync::Arc::from(a)),
            Err(e) => {
                eprintln!("tui: --agent {s}: {e}");
                std::process::exit(2);
            }
        },
        None => None,
    };
    let agent_name = agent.as_ref().map(|a| a.name()).unwrap_or_default();

    // Opening on the agent's own ranking is the point of passing `--agent`;
    // without one there is nothing to ask, so the built-in exhaustive scoring
    // is the default instead.
    let view = std::env::args()
        .position(|a| a == "--view")
        .and_then(|i| std::env::args().nth(i + 1));
    let source = match view.as_deref() {
        Some("sampled") => MoveSource::Sampled,
        Some("full") => MoveSource::Full,
        Some("agent") => MoveSource::Agent,
        Some(other) => {
            eprintln!("tui: --view {other}: try sampled, full or agent");
            std::process::exit(2);
        }
        None if agent.is_some() => MoveSource::Agent,
        None => MoveSource::Full,
    };

    // With an agent named, let it draft its own starting tiles — otherwise the
    // board you are watching it play was set up by a coin flip.
    let mut game = Game::new(seed);
    if let Some(a) = &agent {
        let (fresh, deal) = Game::new_undrafted(seed);
        game = fresh;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xD8AF7);
        for p in PlayerId::ALL {
            let kept = a.draft(&game.state, p, deal[p.idx()], &mut rng);
            game.keep_tiles(p, kept);
        }
    }

    let mut app = App {
        game,
        agent,
        agent_name,
        thinking: None,
        last_decisions: Vec::new(),
        last_played: None,
        ranking: Ranking::default(),
        selected: 0,
        source,
        focus: None,
        autoplay: false,
        status: format!("seed {seed}"),
    };
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let res = run(&mut term, &mut app);

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    term.show_cursor()?;
    res
}

/// A search running off the draw loop.
///
/// One detached thread per job, not a persistent worker: jobs are at most a few
/// a second, `Agent` is `Send + Sync`, and a `spawn` costs nothing measurable
/// against 8,192 simulations. Quitting mid-search drops the receiver, the
/// worker's `send` fails, and the thread ends on its own.
struct Bg {
    rx: mpsc::Receiver<Job>,
    started: Instant,
}

/// Which kind of search a `Bg` is running. Only [`Cost`] needs to tell them
/// apart — the two cost very differently and the pane compares each against its
/// own kind — and the finished `Job` says which arrived.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The agent choosing what to play.
    Turn,
    /// Rebuilding the shortlist for the position now on screen. Under an mcts
    /// agent this is `ranked_turns` — a second full-budget search, and the one
    /// that used to freeze the terminal after *every* move rather than only on
    /// `n`.
    Rank,
}

enum Job {
    Turn(PlayerId, Box<Option<TurnOutcome>>),
    Rank(Box<Ranking>),
}

/// What each kind of search cost last time. The Thinking pane has no simulation
/// counter to draw a fraction from, so this is the only honest scale it can put
/// the running clock against.
#[derive(Default)]
struct Cost {
    turn: Option<Duration>,
    rank: Option<Duration>,
}

impl Cost {
    fn of(&self, k: Kind) -> Option<Duration> {
        match k {
            Kind::Turn => self.turn,
            Kind::Rank => self.rank,
        }
    }
}

fn run<B: Backend>(term: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    let mut last_step = Instant::now();
    let mut bg: Option<Bg> = None;
    let mut cost = Cost::default();
    // The opening list is a full-budget search of its own. Starting it here
    // rather than before the alternate screen is why the board is visible, with
    // a running clock on it, instead of a blank terminal for the first second.
    begin_rank(app, &mut bg, &cost, Vec::new(), None);

    loop {
        // The clock the pane draws comes from the job's own start, so a frame
        // the loop was late for shows the real elapsed time rather than a count
        // of how many frames were drawn.
        if let (Some(t), Some(b)) = (app.thinking.as_mut(), bg.as_ref()) {
            t.elapsed = b.started.elapsed();
        }
        term.draw(|f| ui::draw(f, app))?;

        if let Some(b) = &bg {
            match b.rx.try_recv() {
                Ok(job) => {
                    let took = b.started.elapsed();
                    bg = None;
                    land(app, job, took, &mut cost, &mut bg);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    bg = None;
                    app.thinking = None;
                    app.status = "the search thread stopped without an answer".into();
                }
            }
        }

        // Short enough that the spinner reads as animation rather than as a
        // stutter, and short enough that a finished search is picked up within
        // one frame of finishing.
        let wait = if bg.is_some() { 40 } else { 90 };
        if event::poll(Duration::from_millis(wait))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                // Refusing the mutating keys mid-search is not politeness: a
                // second `n` would queue a turn against a position the first one
                // is about to replace. `help` stops offering them at the same
                // moment, so the footer never advertises a key that is refused.
                let busy = bg.is_some();
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('j') | KeyCode::Down => app.step_selection(1),
                    KeyCode::Char('k') | KeyCode::Up => app.step_selection(-1),
                    KeyCode::Char('a') => app.autoplay = !app.autoplay,
                    // Pinning the Board panel's gear reads nothing the search
                    // owns, so it is allowed mid-search like `j`/`k`.
                    KeyCode::Char('g') => app.step_focus(),
                    _ if busy => {
                        app.status = "still searching — that key is refused until it lands".into()
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => play_selected(app, &mut bg, &cost),
                    KeyCode::Char('n') => step(app, &mut bg, &cost),
                    KeyCode::Char('r') => begin_rank(app, &mut bg, &cost, Vec::new(), None),
                    KeyCode::Char('t') => {
                        app.source = app.source.next();
                        begin_rank(app, &mut bg, &cost, Vec::new(), None);
                    }
                    _ => {}
                }
            }
        }

        // Autoplay waits for the search rather than racing it: the gap is a
        // pause for the human between turns, not a timer the engine has to
        // beat.
        if app.autoplay && bg.is_none() && last_step.elapsed() > Duration::from_millis(250) {
            step(app, &mut bg, &cost);
            last_step = Instant::now();
            if app.game.state.over {
                app.autoplay = false;
            }
        }
    }
}

/// Take a finished search and, if it was only the first half of a turn, start
/// the second.
fn land(app: &mut App, job: Job, took: Duration, cost: &mut Cost, bg: &mut Option<Bg>) {
    match job {
        Job::Turn(p, outcome) => {
            cost.turn = Some(took);
            // `decisions_from` keeps only the nodes whose visit indices really
            // are edge indices, so a one-ply or minimax agent leaves this empty
            // and the search pane says so instead of inventing rows.
            app.last_decisions.clear();
            let colour = app.game.state.players[p.idx()].color;
            let mut known = vec![format!("{:.1}s to choose the move", took.as_secs_f64())];
            if let Some(o) = *outcome {
                app.last_decisions = ui::decisions_from(&o.nodes);
                let text = ui::row_text(&o.mv);
                known.push(format!("chose {text}"));
                app.last_played = Some(format!("{colour} played {text}"));
                app.game.play(p, &o.mv);
            } else {
                known.push("no legal move — passed".into());
                app.last_played = Some(format!("{colour} had no legal move"));
            }
            advance(app);
            // Stage 2. The move is settled and named in `known`, so the pane is
            // reporting a result while the second search runs rather than
            // showing a bare clock for the whole turn.
            begin_rank(app, bg, cost, known, Some(2));
        }
        Job::Rank(r) => {
            cost.rank = Some(took);
            app.ranking = *r;
            app.selected = 0;
            app.thinking = None;
            app.status = format!("{} · {:.0} ms", app.ranking.note, took.as_secs_f64() * 1e3);
        }
    }
}

/// Play whichever move is highlighted.
fn play_selected(app: &mut App, bg: &mut Option<Bg>, cost: &Cost) {
    if app.game.state.over {
        app.status = "game over".into();
        return;
    }
    let Some(m) = app.selected_move().cloned() else {
        return;
    };
    let p = app.game.state.current;
    app.game.play(p, &m);
    // A hand-played move is not the search's, so the pane must stop attributing
    // the last agent turn's reasoning to the position on screen.
    app.last_decisions.clear();
    app.last_played = None;
    advance(app);
    begin_rank(app, bg, cost, vec![format!("you played {}", ui::row_text(&m))], None);
}

/// Let the current player take its own turn, then advance.
fn step(app: &mut App, bg: &mut Option<Bg>, cost: &Cost) {
    if app.game.state.over {
        app.status = "game over".into();
        return;
    }
    if app.agent.is_some() {
        begin_turn(app, bg, cost);
        return;
    }

    // With no agent configured, `n` plays whatever the current view recommends:
    // the top of a scored list, or a fresh draw from the rollout policy.
    let p = app.game.state.current;
    let played = match app.source {
        MoveSource::Full | MoveSource::Agent => app.ranking.moves.first().map(|(m, _)| m.clone()),
        MoveSource::Sampled => sample_legal_move(&app.game.state, p, &mut app.game.rng),
    };
    if let Some(m) = &played {
        app.game.play(p, m);
    }
    advance(app);
    begin_rank(app, bg, cost, Vec::new(), None);
}

/// Hand the agent's turn to a worker thread.
///
/// The state is `Copy` and the agent is behind an `Arc`, so the whole job is a
/// move into the closure and nothing is borrowed across the search — which is
/// what lets the draw loop keep running while it happens.
fn begin_turn(app: &mut App, bg: &mut Option<Bg>, cost: &Cost) {
    use rand::SeedableRng;
    let Some(agent) = app.agent.clone() else { return };
    let p = app.game.state.current;
    let state = app.game.state;
    let seed = state.day as u64 * 977 + p.0 as u64;
    let colour = state.players[p.idx()].color;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let out = agent.play_turn(&state, p, 0.0, &mut rng);
        let _ = tx.send(Job::Turn(p, Box::new(out)));
    });

    app.thinking = Some(Thinking {
        what: format!("searching {colour}'s turn"),
        detail: format!(
            "{} · one search per sub-decision, then one more to rank the alternatives",
            ui::short_agent(&app.agent_name)
        ),
        elapsed: Duration::ZERO,
        stage: Some((1, 2)),
        known: Vec::new(),
        prior: cost.of(Kind::Turn),
    });
    app.status = "searching".into();
    *bg = Some(Bg { rx, started: Instant::now() });
}

/// Rebuild the candidate list for whoever is to move.
///
/// `stage` is `Some(2)` when this is the second half of a turn the agent just
/// played, so the pane can say which of the two searches is running.
fn begin_rank(
    app: &mut App,
    bg: &mut Option<Bg>,
    cost: &Cost,
    known: Vec<String>,
    stage: Option<u32>,
) {
    app.selected = 0;
    app.ranking = Ranking::default();
    app.thinking = None;

    if app.game.state.over {
        let scores = app.game.scores();
        let w: Vec<String> = app
            .game
            .winners()
            .iter()
            .map(|p| app.game.state.players[p.idx()].color.to_string())
            .collect();
        app.status = format!("game over — {scores:?} — winner {}", w.join("/"));
        return;
    }

    let p = app.game.state.current;
    // The one view that cannot go to a thread — it draws from `game.rng`, and
    // it is instant by construction, which is the reason it exists.
    if app.source == MoveSource::Sampled {
        app.ranking = sampled(app, p);
        app.status = app.ranking.note.clone();
        return;
    }

    let state = app.game.state;
    let agent = app.agent.clone();
    let source = app.source;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(Job::Rank(Box::new(rank(&state, p, source, agent))));
    });

    let colour = app.game.state.players[p.idx()].color;
    app.thinking = Some(Thinking {
        what: match stage {
            Some(_) => format!("ranking what {colour} passed over"),
            None => format!("scoring {colour}'s options"),
        },
        detail: match source {
            MoveSource::Agent if app.agent.is_some() => format!(
                "{} · one search over whole turns, for the shortlist only",
                ui::short_agent(&app.agent_name)
            ),
            _ => format!("eval::margin over every legal move, capped at {FULL_VIEW_BUDGET:?}"),
        },
        elapsed: Duration::ZERO,
        stage: stage.map(|k| (k, 2)),
        known,
        prior: cost.of(Kind::Rank),
    });
    app.status = "scoring".into();
    *bg = Some(Bg { rx, started: Instant::now() });
}

/// The shortlist, computed off the draw loop.
///
/// Split out of `begin_rank` because it must not touch `App`: everything it
/// reads is a `Copy` of the state or an `Arc`, which is what makes it safe to
/// run on the worker while the main thread keeps drawing that same position.
fn rank(
    state: &tzolkin::state::GameState,
    p: PlayerId,
    source: MoveSource,
    agent: Option<Arc<dyn Agent>>,
) -> Ranking {
    match source {
        // A longer leash than an agent gets: nothing is waiting on this but a
        // person looking at one position, and seeing the whole list is the
        // point of the view. It still has a leash, because the widest turns run
        // to millions of moves.
        MoveSource::Full | MoveSource::Sampled => {
            eval::rank_all_capped(state, p, SHOWN, FULL_VIEW_BUDGET)
        }
        MoveSource::Agent => match agent {
            // The agent scores the whole move list itself, so `minimax` reports
            // backed-up search values here rather than a one-ply guess.
            Some(a) => a
                .ranked_moves(state, p, SHOWN)
                .unwrap_or_else(|| eval::rank_all(state, p, SHOWN)),
            None => {
                let mut r = eval::rank_all_capped(state, p, SHOWN, FULL_VIEW_BUDGET);
                r.note = format!("{} — no --agent, so this is the built-in scoring", r.note);
                r
            }
        },
    }
}

/// Move to the next player, closing out the round when it wraps.
fn advance(app: &mut App) {
    let g = &mut app.game;
    g.state.current = g.state.current.next(1);

    if g.state.current == g.state.first_player {
        g.end_round_public();
        g.state.current = g.state.first_player;
    }
}

/// Distinct draws from the rollout policy, so the list is varied.
///
/// The one view that stays instant no matter how wide the position is, which is
/// why it is kept: `Full` and `Agent` both walk the whole move list.
fn sampled(app: &mut App, p: PlayerId) -> Ranking {
    let mut moves: Vec<(tzolkin::moves::Move, f32)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..SHOWN * 20 {
        if moves.len() >= SHOWN {
            break;
        }
        if let Some(m) = sample_legal_move(&app.game.state, p, &mut app.game.rng) {
            if seen.insert(m.clone()) {
                let succ = eval::successor(&app.game.state, p, &m);
                moves.push((m, margin(&succ, p)));
            }
        }
    }
    moves.sort_by(|a, b| b.1.total_cmp(&a.1));
    let n = moves.len();
    Ranking {
        moves,
        total: n,
        distinct: n,
        // These are draws, not the move list: nothing here says how many moves
        // the position actually has, and the panel must not imply that it does.
        exhaustive: false,
        note: format!("{n} draws from the rollout policy"),
    }
}
