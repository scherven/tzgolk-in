//! Scratch: where does the alpha-beta actually spend its time? (delete me)

use std::time::{Duration, Instant};
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::search::{Config, Opponents, Search};
use tzolkin::state::GameState;

fn positions(seeds: u64, at: &[usize]) -> Vec<GameState> {
    let mut out = Vec::new();
    for seed in 0..seeds {
        for &rounds in at {
            let mut g = Game::new(seed + 900);
            for _ in 0..rounds {
                if g.state.over { break; }
                g.play_round();
            }
            if !g.state.over { out.push(g.state); }
        }
    }
    out
}

fn run(label: &str, cfg: Config, ps: &[GameState]) {
    let mut s = Search::new(cfg);
    let (mut nodes, mut leaves, mut cuts, mut tt, mut cc, mut cm) = (0u64,0u64,0u64,0u64,0u64,0u64);
    let (mut depth, mut deep, mut rootn) = (0u64, 0u64, 0u64);
    let t = Instant::now();
    for g in ps {
        let _ = s.search(g, g.current);
        let st = s.stats();
        nodes += st.nodes; leaves += st.leaves; cuts += st.cutoffs; tt += st.tt_hits;
        cc += st.cand_calls; cm += st.cand_moves;
        depth += st.depth as u64; deep += st.deepened as u64; rootn += st.root_moves as u64;
    }
    let n = ps.len() as f64;
    println!("{label:<34} {:>7.0}ms/turn  nodes {:>7.1}  leaves {:>6.1}  cut {:>5.1}  tt {:>5.1}  candcalls {:>6.1}  candmoves {:>9.0}  depth {:>4.1}  deep {:>4.1}  rootmoves {:>8.0}",
        t.elapsed().as_secs_f64()*1e3/n, nodes as f64/n, leaves as f64/n, cuts as f64/n, tt as f64/n,
        cc as f64/n, cm as f64/n, depth as f64/n, deep as f64/n, rootn as f64/n);
}

fn main() {
    let ps = positions(4, &[0, 4, 8, 12, 16, 20, 24]);
    println!("{} positions\n", ps.len());
    let base = || Config::default();
    for d in [4u8, 8, 12, 16, 24, 40] {
        run(&format!("greedy d{d} 200ms"), Config { max_depth: d, ..base() }.with_budget_ms(200), &ps);
    }
    println!();
    for w in [vec![12usize,6,4,3,2], vec![12,6,4,3,3], vec![12,1,1,1,4,1,1,1,3], vec![24,6,4,3,2], vec![8,6,4,3,2]] {
        run(&format!("greedy d16 200ms widths {w:?}"), Config { max_depth: 16, widths: w, ..base() }.with_budget_ms(200), &ps);
    }
    println!();
    for c in [25usize, 50, 100, 200, 400, 1600] {
        run(&format!("greedy d16 200ms cap{c}"), Config { max_depth: 16, interior_cap: c, ..base() }.with_budget_ms(200), &ps);
    }
    println!();
    for d in [4u8, 8, 12] {
        run(&format!("paranoid d{d} 200ms"), Config { max_depth: d, opponents: Opponents::Paranoid, ..base() }.with_budget_ms(200), &ps);
    }
    let _ = Duration::from_millis(1);
    let _ = PlayerId(0);
}
