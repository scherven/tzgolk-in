//! Render the TUI to an offscreen buffer and print it.
//!
//! Lets the layout be checked (and diffed in review) without a terminal.
//!
//!     cargo run --release --bin uidump -- [seed] [rounds] [cols] [rows]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use tzolkin::eval::heuristic;
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

    let ranked = std::env::args().any(|a| a == "--ranked");
    let p = game.state.current;
    let mut candidates = Vec::new();
    let mut total = None;

    if ranked {
        let all = tzolkin::moves::legal_moves_capped(&game.state, p, 20_000);
        total = Some(all.len());
        for (i, s) in tzolkin::eval::rank(&game.state, p, &all).into_iter().take(10) {
            candidates.push((all[i].clone(), s));
        }
    } else {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            if candidates.len() >= 10 {
                break;
            }
            if let Some(m) = sample_legal_move(&game.state, p, &mut game.rng) {
                if seen.insert(m.clone()) {
                    let mut probe = game.state;
                    tzolkin::moves::apply_move(&mut probe, p, &m);
                    candidates.push((m, heuristic(&probe, p)));
                }
            }
        }
    }

    // `--agent SPEC` plays one turn with that agent first, so the render shows
    // the search panel rather than the heuristic move list.
    let spec = std::env::args()
        .position(|a| a == "--agent")
        .and_then(|i| std::env::args().nth(i + 1));
    let mut agent_name = String::new();
    let mut last_decisions = Vec::new();
    if let Some(sp) = &spec {
        use rand::SeedableRng;
        let agent = tzolkin::record::parse_agent(sp, true).expect("bad --agent");
        agent_name = agent.name();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // Only about a quarter of turns run the full budget and record nodes,
        // so keep playing until one does.
        for _ in 0..12 {
            let p = game.state.current;
            match agent.play_turn(&game.state, p, 0.0, &mut rng) {
                Some(o) => {
                    last_decisions = ui::decisions_from(&o.nodes);
                    game.play(p, &o.mv);
                    game.state.current = game.state.current.next(1);
                    if !last_decisions.is_empty() {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    let app = App {
        game,
        agent: None,
        agent_name,
        last_decisions,
        candidates,
        selected: 0,
        source: if ranked { MoveSource::Ranked } else { MoveSource::Sampled },
        autoplay: false,
        total_moves: total,
        status: format!("seed {seed}, {rounds} rounds in"),
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
