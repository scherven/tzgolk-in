//! Watch a game. Read-only for now; human play drops in as extra key handlers.
//!
//!     cargo run --release --bin tui -- [seed]

use std::io;
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
use tzolkin::ui::{self, App, MoveSource};

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
        ranking: Ranking::default(),
        selected: 0,
        source,
        autoplay: false,
        status: format!("seed {seed}"),
    };
    refresh(&mut app);

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let res = run(&mut term, &mut app);

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    term.show_cursor()?;
    res
}

fn run<B: Backend>(term: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    let mut last_step = Instant::now();

    loop {
        term.draw(|f| ui::draw(f, app))?;

        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('j') | KeyCode::Down => {
                        let n = app.candidates().len();
                        if n > 0 {
                            app.selected = (app.selected + 1) % n;
                        }
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        let n = app.candidates().len();
                        if n > 0 {
                            app.selected = (app.selected + n - 1) % n;
                        }
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => play_selected(app),
                    KeyCode::Char('n') => step(app),
                    KeyCode::Char('r') => refresh(app),
                    KeyCode::Char('t') => {
                        app.source = app.source.next();
                        refresh(app);
                    }
                    KeyCode::Char('a') => app.autoplay = !app.autoplay,
                    _ => {}
                }
            }
        }

        if app.autoplay && last_step.elapsed() > Duration::from_millis(250) {
            step(app);
            last_step = Instant::now();
            if app.game.state.over {
                app.autoplay = false;
            }
        }
    }
}

/// Play whichever move is highlighted.
fn play_selected(app: &mut App) {
    if app.game.state.over {
        app.status = "game over".into();
        return;
    }
    let Some(m) = app.selected_move().cloned() else {
        return;
    };
    let p = app.game.state.current;
    app.game.play(p, &m);
    advance(app);
}

/// Let the current player take its own turn, then advance.
fn step(app: &mut App) {
    if app.game.state.over {
        app.status = "game over".into();
        return;
    }
    if app.agent.is_some() {
        agent_step(app);
        return;
    }

    // With no agent configured, `n` plays whatever the current view recommends:
    // the top of a scored list, or a fresh draw from the rollout policy.
    match app.source {
        MoveSource::Full | MoveSource::Agent => play_best(app),
        MoveSource::Sampled => {
            let p = app.game.state.current;
            if let Some(m) = sample_legal_move(&app.game.state, p, &mut app.game.rng) {
                app.game.play(p, &m);
            }
            advance(app);
        }
    }
}

/// One turn played by the configured agent, recording what its search did.
fn agent_step(app: &mut App) {
    use rand::SeedableRng;
    let p = app.game.state.current;
    let agent = app.agent.take().expect("agent_step without an agent");
    let mut rng = rand::rngs::StdRng::seed_from_u64(app.game.state.day as u64 * 977 + p.0 as u64);

    let outcome = agent.play_turn(&app.game.state, p, 0.0, &mut rng);
    app.agent = Some(agent);

    // `decisions_from` keeps only the nodes whose visit indices really are edge
    // indices, so a one-ply or minimax agent leaves this empty and the preview
    // pane shows the move diff instead — which is the honest thing to show for
    // an agent that never built a tree.
    app.last_decisions.clear();
    if let Some(o) = &outcome {
        app.last_decisions = ui::decisions_from(&o.nodes);
        app.game.play(p, &o.mv);
    }
    advance(app);
}

fn play_best(app: &mut App) {
    let p = app.game.state.current;
    if let Some((m, _)) = app.ranking.moves.first().cloned() {
        app.game.play(p, &m);
    }
    advance(app);
}

/// Move to the next player, closing out the round when it wraps.
fn advance(app: &mut App) {
    let g = &mut app.game;
    g.state.current = g.state.current.next(1);

    if g.state.current == g.state.first_player {
        g.end_round_public();
        g.state.current = g.state.first_player;
    }
    refresh(app);
}

/// Rebuild the candidate list for whoever is to move.
fn refresh(app: &mut App) {
    app.selected = 0;
    app.ranking = Ranking::default();

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
    let started = Instant::now();

    app.ranking = match app.source {
        MoveSource::Sampled => sampled(app, p),
        // A longer leash than an agent gets: nothing is waiting on this but a
        // person looking at one position, and seeing the whole list is the
        // point of the view. It still has a leash, because the widest turns run
        // to millions of moves and a frozen terminal explains nothing.
        MoveSource::Full => eval::rank_all_capped(&app.game.state, p, SHOWN, FULL_VIEW_BUDGET),
        MoveSource::Agent => match app.agent.as_ref() {
            // The agent scores the whole move list itself, so `minimax` reports
            // backed-up search values here rather than a one-ply guess.
            Some(a) => a
                .ranked_moves(&app.game.state, p, SHOWN)
                .unwrap_or_else(|| eval::rank_all(&app.game.state, p, SHOWN)),
            None => {
                let mut r =
                    eval::rank_all_capped(&app.game.state, p, SHOWN, FULL_VIEW_BUDGET);
                r.note = format!("{} — no --agent, so this is the built-in scoring", r.note);
                r
            }
        },
    };

    app.status = format!(
        "{} · {:.0} ms",
        app.ranking.note,
        started.elapsed().as_secs_f64() * 1e3
    );
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
