//! Watch a game. Read-only for now; human play drops in as extra key handlers.
//!
//!     cargo run --release --bin tui -- [seed]

use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;

use tzolkin::eval::{heuristic, rank};
use tzolkin::game::Game;
use tzolkin::moves::{sample_legal_move, Move};
use tzolkin::ui::{self, App, MoveSource};

const SHOWN: usize = 10;
/// Above this, ranking every move is too slow to do on each redraw.
const RANK_BUDGET: usize = 20_000;

const HELP: &str = "\
tui -- watch a game

USAGE
  cargo run --release --bin tui -- [SEED] [--agent SPEC]

OPTIONS
  --agent SPEC   who plays when you press `n` or turn on autoplay.
                 Without it the rollout policy plays and the move list is
                 ranked by the placeholder heuristic.

AGENT SPECS  (the same ones arena and selfplay take)
  random                              the rollout policy
  heuristic[:K]                       one-ply greedy over K sampled turns
  mcts:SIMS                           search on the heuristic evaluator
  mcts:SIMS:PATH.safetensors          search on a trained net
  PATH.safetensors                    shorthand for mcts at the default budget

KEYS
  j/k    move the selection        n    let the current player act
  enter  play the highlighted move r    redraw the candidate list
  t      sampled <-> ranked        a    autoplay          q  quit

With --agent, the preview pane shows the search's visit share at each
sub-decision of the last turn -- what the agent actually considered, not a
heuristic ranking of the alternatives.
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
    // `record: true` is what makes the agent hand back its decision nodes;
    // with `false` it plays the same move and reports nothing, which is how the
    // search panel came up empty the first time.
    let agent = match &spec {
        Some(s) => match tzolkin::record::parse_agent(s, true) {
            Ok(a) => Some(a),
            Err(e) => {
                eprintln!("tui: --agent {s}: {e}");
                std::process::exit(2);
            }
        },
        None => None,
    };
    let agent_name = agent.as_ref().map(|a| a.name()).unwrap_or_default();

    let mut app = App {
        game: Game::new(seed),
        agent,
        agent_name,
        last_decisions: Vec::new(),
        candidates: Vec::new(),
        selected: 0,
        source: MoveSource::Sampled,
        autoplay: false,
        total_moves: None,
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
                        if !app.candidates.is_empty() {
                            app.selected = (app.selected + 1) % app.candidates.len();
                        }
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        if !app.candidates.is_empty() {
                            app.selected = (app.selected + app.candidates.len() - 1)
                                % app.candidates.len();
                        }
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => play_selected(app),
                    KeyCode::Char('n') => step(app),
                    KeyCode::Char('r') => refresh(app),
                    KeyCode::Char('t') => {
                        app.source = match app.source {
                            MoveSource::Sampled => MoveSource::Ranked,
                            MoveSource::Ranked => MoveSource::Sampled,
                        };
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
    let Some((m, _)) = app.candidates.get(app.selected).cloned() else {
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

    // Play the best candidate when ranking, otherwise a sampled one.
    match app.source {
        MoveSource::Ranked => play_selected_best(app),
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

    app.last_decisions.clear();
    if let Some(o) = &outcome {
        app.last_decisions = ui::decisions_from(&o.nodes);
        for node in &o.nodes {
            let total: u32 = node.visits.iter().map(|(_, n)| *n as u32).sum();
            if total == 0 {
                continue;
            }
            let (best, n) = node
                .visits
                .iter()
                .max_by_key(|(_, n)| *n)
                .map(|(s, n)| (format!("{s:?}"), *n as u32))
                .unwrap_or_default();
            // `Step`'s Debug is long; the variant name carries the meaning.
            let short = best.split(['(', ' ']).next().unwrap_or(&best).to_string();
            app.last_decisions.push((
                format!("{:?}", node.phase),
                short,
                n as f32 / total as f32,
            ));
        }
        app.game.play(p, &o.mv);
    }
    advance(app);
}

fn play_selected_best(app: &mut App) {
    let p = app.game.state.current;
    if let Some((m, _)) = app.candidates.first().cloned() {
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
    app.candidates.clear();
    app.total_moves = None;

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
    match app.source {
        MoveSource::Sampled => {
            // Distinct draws from the rollout policy, so the list is varied.
            let mut seen = std::collections::HashSet::new();
            for _ in 0..SHOWN * 20 {
                if app.candidates.len() >= SHOWN {
                    break;
                }
                if let Some(m) = sample_legal_move(&app.game.state, p, &mut app.game.rng) {
                    if seen.insert(m.clone()) {
                        let mut probe = app.game.state;
                        tzolkin::moves::apply_move(&mut probe, p, &m);
                        app.candidates.push((m, heuristic(&probe, p)));
                    }
                }
            }
            app.status = format!("seed view · {} sampled", app.candidates.len());
        }
        MoveSource::Ranked => {
            // Ranking needs the real list, which is unbounded in the late game.
            let all: Vec<Move> =
                tzolkin::moves::legal_moves_capped(&app.game.state, p, RANK_BUDGET);
            let capped = all.len() >= RANK_BUDGET;
            let total = if capped {
                None
            } else {
                Some(all.len())
            };
            app.total_moves = total;
            for (i, s) in rank(&app.game.state, p, &all).into_iter().take(SHOWN) {
                app.candidates.push((all[i].clone(), s));
            }
            app.status = if capped {
                format!("ranked first {RANK_BUDGET} of a larger set")
            } else {
                format!("ranked all {}", all.len())
            };
        }
    }

    // Keep the exact count available when it is cheap to know.
    if app.total_moves.is_none() && app.source == MoveSource::Sampled {
        let mut n = 0usize;
        let _ = tzolkin::moves::visit_legal_moves(&app.game.state, p, |_| {
            n += 1;
            if n > RANK_BUDGET {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        });
        app.total_moves = (n <= RANK_BUDGET).then_some(n);
    }
}
