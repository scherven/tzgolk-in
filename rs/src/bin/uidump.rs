//! Render the TUI to an offscreen buffer and print it.
//!
//! Lets the layout be checked (and diffed in review) without a terminal.
//!
//!     cargo run --release --bin uidump -- [seed] [rounds] [cols] [rows]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use tzolkin::eval::{self, margin, Ranking};
use tzolkin::game::Game;
use tzolkin::moves::sample_legal_move;
use tzolkin::ui::{self, App, MoveSource};

fn main() {
    let arg = |n: usize, d: u64| -> u64 {
        std::env::args().nth(n).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let (seed, rounds, cols, rows) = (arg(1, 7), arg(2, 9), arg(3, 132) as u16, arg(4, 44) as u16);

    let mut game = Game::new(seed);
    for _ in 0..rounds {
        game.play_round();
    }

    let full = std::env::args().any(|a| a == "--full" || a == "--ranked");
    let p = game.state.current;

    let ranking = if full {
        eval::rank_all(&game.state, p, 10)
    } else {
        let mut moves = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            if moves.len() >= 10 {
                break;
            }
            if let Some(m) = sample_legal_move(&game.state, p, &mut game.rng) {
                if seen.insert(m.clone()) {
                    let succ = eval::successor(&game.state, p, &m);
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
            exhaustive: false,
            note: format!("{n} draws from the rollout policy"),
        }
    };

    // `--agent SPEC` plays one turn with that agent first, so the render shows
    // the search panel rather than the heuristic move list.
    let spec = std::env::args()
        .position(|a| a == "--agent")
        .and_then(|i| std::env::args().nth(i + 1));
    let mut agent_name = String::new();
    let mut last_decisions = Vec::new();
    let mut agent_ranking = None;
    let mut last_move = None;
    let mut played_by = game.state.current;
    if let Some(sp) = &spec {
        use rand::SeedableRng;
        let agent = tzolkin::record::parse_analysis_agent(sp).expect("bad --agent");
        agent_name = agent.name();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // Every turn is at the full budget now that recording no longer drags
        // playout-cap randomisation along with it, so this loop is a guard
        // against a forced turn rather than against a 1-in-4 write rate.
        //
        // The ranking is asked for whatever came back. Requiring a populated
        // decision list first meant an agent that builds no tree — `minimax`,
        // `heuristic:full` — played twelve turns and then rendered the
        // *heuristic* fallback, so the one view that would have shown its score
        // scale was the one view unreachable for it. Those scales differ (see
        // `docs/FINDINGS-tui.md` T1), which is exactly what wanted checking.
        for _ in 0..12 {
            let p = game.state.current;
            match agent.play_turn(&game.state, p, 0.0, &mut rng) {
                Some(o) => {
                    last_decisions = ui::decisions_from(&o.nodes);
                    last_move = Some(ui::row_text(&o.mv));
                    played_by = p;
                    game.play(p, &o.mv);
                    game.state.current = game.state.current.next(1);
                    // The `agent` view: the agent's own ranking of whole
                    // turns, which is the thing the panel claims to show
                    // and used to substitute a one-ply heuristic for.
                    agent_ranking = agent.ranked_moves(&game.state, game.state.current, 10);
                    if agent_ranking.is_some() {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    // `--thinking SECS` renders the mid-search screen. There is no other way to
    // review it: by the time a search returns, the thing being checked is gone.
    // `--stage2` renders it as the *second* of a turn's two searches, which is
    // the interesting one — it is the state that carries a settled result and a
    // prior to measure the clock against, and the two halves lay out
    // differently.
    let stage2 = std::env::args().any(|a| a == "--stage2");
    let thinking = std::env::args()
        .position(|a| a == "--thinking")
        .map(|i| {
            let secs: f64 = std::env::args().nth(i + 1).and_then(|v| v.parse().ok()).unwrap_or(2.4);
            let colour = game.state.players[game.state.current.idx()].color;
            let short = ui::short_agent(&agent_name);
            ui::Thinking {
                what: if stage2 {
                    format!("ranking what {colour} passed over")
                } else {
                    format!("searching {colour}'s turn")
                },
                detail: if stage2 {
                    format!("{short} · one search over whole turns, for the shortlist only")
                } else {
                    format!("{short} · one search per sub-decision, then one more to rank the alternatives")
                },
                elapsed: std::time::Duration::from_secs_f64(secs),
                stage: Some((if stage2 { 2 } else { 1 }, 2)),
                known: if stage2 {
                    vec![
                        "3.1s to choose the move".into(),
                        format!(
                            "chose {}",
                            last_move.as_deref().unwrap_or("(nothing yet)")
                        ),
                    ]
                } else {
                    Vec::new()
                },
                // Only the second stage has a comparable search behind it, so
                // only the second stage gets the bar. That is the whole point of
                // being able to dump both.
                prior: stage2.then(|| std::time::Duration::from_secs_f64(3.1)),
            }
        });

    let source = match (&agent_ranking, full) {
        (Some(_), _) => MoveSource::Agent,
        (None, true) => MoveSource::Full,
        (None, false) => MoveSource::Sampled,
    };
    let ranking = agent_ranking.unwrap_or(ranking);
    // The note is where a ranking says what its numbers mean -- for MCTS, that
    // the column is a visit share and the shortlist is not the move list -- and
    // the TUI puts it in the status line, so a dump that dropped it would be
    // reviewing a different screen.
    let status = format!("seed {seed}, {rounds} rounds in · {}", ranking.note);
    let last_played = last_move
        .as_ref()
        .map(|m| format!("{} played {m}", game.state.players[played_by.idx()].color));
    let app = App {
        game,
        agent: None,
        agent_name,
        thinking,
        last_decisions,
        last_played,
        ranking,
        selected: std::env::args()
            .position(|a| a == "--select")
            .and_then(|i| std::env::args().nth(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        source,
        autoplay: false,
        status,
    };

    let mut term = Terminal::new(TestBackend::new(cols, rows)).unwrap();
    term.draw(|f| ui::draw(f, &app)).unwrap();

    let buf = term.backend().buffer();
    for y in 0..rows {
        let mut line = String::new();
        for x in 0..cols {
            line.push_str(buf[(x, y)].symbol());
        }
        println!("{}", line.trim_end());
    }
}
