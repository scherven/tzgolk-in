//! Scratch smoke test for the new evaluator and search.
use std::time::Instant;
use tzolkin::game::Game;
use tzolkin::search::{Config, Search};

fn main() {
    for spec in ["heuristic", "heuristic:full", "heuristic:64", "minimax", "minimax:3",
                 "minimax:4:400", "minimax:4:400:8", "greedy:full:heuristic", "mcts:16"] {
        match tzolkin::record::AgentSpec::parse(spec, false) {
            Ok(a) => println!("{spec:<24} -> {}", a.name()),
            Err(e) => println!("{spec:<24} -> ERR {e}"),
        }
    }
    for bad in ["heuristic:zero", "minimax:0", "minimax:4:400:8:9", "nonsense"] {
        println!("{bad:<24} -> {:?}", tzolkin::record::AgentSpec::parse(bad, false).err());
    }

    for rounds in [0usize, 6, 13, 20, 26] {
        let mut g = Game::new(7);
        for _ in 0..rounds {
            if g.state.over { break; }
            g.play_round();
        }
        if g.state.over { println!("rounds {rounds}: game over"); continue; }
        let p = g.state.current;

        let t = Instant::now();
        let full = tzolkin::eval::rank_all(&g.state, p, 10);
        let full_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut s = Search::new(Config::default().with_depth(4).with_budget_ms(1500));
        let t = Instant::now();
        let r = s.search(&g.state, p);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let st = s.stats();

        println!("\n=== day {} seat {:?} ===", g.state.day, p);
        println!("  full rank : {} moves ({} distinct), {full_ms:.0} ms, best {:+.2}", full.total, full.distinct,
                 full.moves.first().map(|(_, s)| *s).unwrap_or(0.0));
        println!("  minimax   : {} ({ms:.0} ms, depth {}, {} nodes, {} cutoffs, {} tt)",
                 r.note, st.depth, st.nodes, st.cutoffs, st.tt_hits);
        for (i, (m, sc)) in r.moves.iter().take(5).enumerate() {
            println!("    {i}. {sc:+8.2}  {m}");
        }
    }
}
