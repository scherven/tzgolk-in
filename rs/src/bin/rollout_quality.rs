//! How much signal does a random rollout actually carry?
//!
//! For a fixed position, take K candidate moves, run R random playouts after
//! each, and compare the spread of their means against the noise of a single
//! rollout. That ratio is what decides whether rollouts can rank moves at all.

use rand::{Rng, SeedableRng};
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::legal_moves;

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1).max(1) as f64).sqrt()
}

fn main() {
    let rollouts: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(400);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);

    // A mid-game position.
    let mut g = Game::new(11);
    for _ in 0..9 {
        g.play_round();
    }
    let root = g.state;
    let p = root.first_player;

    let moves = legal_moves(&root, p);
    println!("position: day {}, {} legal moves for {}", root.day, moves.len(), root.players[p.idx()].color);

    // Sample up to 12 candidate moves.
    let k = 12.min(moves.len());
    let step = moves.len() / k;
    let candidates: Vec<_> = (0..k).map(|i| moves[i * step].clone()).collect();

    let mut per_move_means = Vec::new();
    let mut all_sds = Vec::new();

    for (i, m) in candidates.iter().enumerate() {
        let mut scores = Vec::with_capacity(rollouts);
        for _ in 0..rollouts {
            let mut sim = Game::new(rng.gen());
            sim.state = root;
            sim.play(p, m);
            sim.state.current = p.next(1);
            for _ in 1..N_PLAYERS {
                sim.take_turn_sampled();
                sim.state.current = sim.state.current.next(1);
            }
            sim.resolve_first_player();
            sim.rotate_for_test();
            sim.run_sampled();
            scores.push(sim.state.players[p.idx()].points as f64);
        }
        let (m_, s_) = (mean(&scores), sd(&scores));
        per_move_means.push(m_);
        all_sds.push(s_);
        if i < 6 {
            println!("  move {i:<2} mean {m_:>7.2}  sd {s_:>6.2}  stderr {:>5.2}", s_ / (rollouts as f64).sqrt());
        }
    }

    let spread = per_move_means.iter().cloned().fold(f64::MIN, f64::max)
        - per_move_means.iter().cloned().fold(f64::MAX, f64::min);
    let noise = mean(&all_sds);

    println!("\n{rollouts} rollouts per move, {k} candidate moves");
    println!("  spread of move means : {spread:.2}");
    println!("  sd of a single rollout: {noise:.2}");
    println!("  signal-to-noise      : {:.3}", spread / noise);

    // Rollouts needed for the standard error to be a quarter of the spread,
    // i.e. roughly enough to rank two adjacent moves apart.
    let target = spread / 4.0;
    let needed = (noise / target).powi(2);
    println!("\n  rollouts to resolve the full spread to +/- 1/4: {needed:.0} per move");
    println!("  ...for all {} moves at this node: {:.0}", moves.len(), needed * moves.len() as f64);
}
