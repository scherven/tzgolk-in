//! Where does move generation actually cost us? Samples branching and timing
//! across whole games so the AI architecture can be chosen from data.

use std::time::Instant;
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::{legal_moves, MoveKind};
use tzolkin::state::GameState;

fn main() {
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    let mut counts: Vec<usize> = Vec::new();
    let mut gen_nanos: Vec<u128> = Vec::new();
    // branching bucketed by how many workers the player has on the board
    let mut by_workers: Vec<Vec<usize>> = vec![Vec::new(); 7];
    let mut place_v_retrieve = (0u64, 0u64);
    let mut total_gen = 0u128;
    let mut total_play = 0u128;

    for seed in 0..games {
        let mut g = Game::new(seed);
        while !g.state.over {
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                let on_board = g.state.on_board(p).count();

                let t = Instant::now();
                let moves = legal_moves(&g.state, p);
                let elapsed = t.elapsed().as_nanos();

                counts.push(moves.len());
                gen_nanos.push(elapsed);
                total_gen += elapsed;
                by_workers[on_board.min(6)].push(moves.len());
                for m in &moves {
                    match m.kind {
                        MoveKind::Place(_) => place_v_retrieve.0 += 1,
                        MoveKind::Retrieve(_) => place_v_retrieve.1 += 1,
                        MoveKind::Pity { .. } => {}
                    }
                }

                if !moves.is_empty() {
                    let i = seed as usize % moves.len();
                    let t = Instant::now();
                    let m = moves[i].clone();
                    g.play(p, &m);
                    total_play += t.elapsed().as_nanos();
                }
                g.state.current = g.state.current.next(1);
            }
            g.resolve_first_player();
            g.rotate_for_test();
        }
    }

    counts.sort_unstable();
    gen_nanos.sort_unstable();
    let pct = |v: &[usize], q: f64| v[((v.len() as f64 - 1.0) * q) as usize];
    let pctn = |v: &[u128], q: f64| v[((v.len() as f64 - 1.0) * q) as usize];

    println!("turns sampled: {}", counts.len());
    println!("\nbranching factor");
    for q in [0.5, 0.75, 0.9, 0.99, 0.999, 1.0] {
        println!("  p{:<6} {:>10}", format!("{:.1}", q * 100.0), pct(&counts, q));
    }
    println!(
        "  mean   {:>10.0}",
        counts.iter().sum::<usize>() as f64 / counts.len() as f64
    );

    println!("\nlegal_moves() time");
    for q in [0.5, 0.9, 0.99, 1.0] {
        println!(
            "  p{:<6} {:>10.3} ms",
            format!("{:.0}", q * 100.0),
            pctn(&gen_nanos, q) as f64 / 1e6
        );
    }

    println!("\nbranching by workers on board");
    for (n, v) in by_workers.iter().enumerate() {
        if v.is_empty() {
            continue;
        }
        let mut v = v.clone();
        v.sort_unstable();
        println!(
            "  {n} workers: n={:<7} median {:>7}  p99 {:>9}  max {:>9}",
            v.len(),
            pct(&v, 0.5),
            pct(&v, 0.99),
            v[v.len() - 1]
        );
    }

    let (pl, re) = place_v_retrieve;
    println!("\nmoves generated: {pl} placement, {re} retrieval ({:.1}% retrieval)",
        re as f64 * 100.0 / (pl + re) as f64);
    println!(
        "\ntime: {:.1}s generating, {:.2}s applying  ({:.0}x)",
        total_gen as f64 / 1e9,
        total_play as f64 / 1e9,
        total_gen as f64 / total_play.max(1) as f64
    );
    println!("  mean per turn: {:.3} ms generate, {:.4} ms apply",
        total_gen as f64 / counts.len() as f64 / 1e6,
        total_play as f64 / counts.len() as f64 / 1e6);
    println!("\nGameState {} bytes", std::mem::size_of::<GameState>());
}
