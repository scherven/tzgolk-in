//! Playout throughput: enumerate-then-pick versus sample-directly.

use std::time::Instant;
use tzolkin::game::Game;

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(300);

    let t = Instant::now();
    for seed in 0..n {
        Game::new(seed).run_random();
    }
    let enumerated = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for seed in 0..n {
        Game::new(seed).run_sampled();
    }
    let sampled = t.elapsed().as_secs_f64();

    println!("{n} full playouts");
    println!(
        "  enumerate then pick : {enumerated:>7.2}s   {:>8.0} playouts/s   {:>7.2} ms each",
        n as f64 / enumerated,
        enumerated * 1000.0 / n as f64
    );
    println!(
        "  sample directly     : {sampled:>7.2}s   {:>8.0} playouts/s   {:>7.2} ms each",
        n as f64 / sampled,
        sampled * 1000.0 / n as f64
    );
    println!("  speedup: {:.0}x", enumerated / sampled);
    println!(
        "\n10,000 rollouts would take {:.1}s enumerating, {:.2}s sampling",
        10_000.0 * enumerated / n as f64,
        10_000.0 * sampled / n as f64
    );

    // The rollout policy is not uniform over legal moves, so check it still
    // produces game-shaped games rather than degenerate ones.
    let stats = |sampled: bool| {
        let (mut sum, mut lo, mut hi, mut starved) = (0i64, i16::MAX, i16::MIN, 0u32);
        for seed in 0..n {
            let mut g = Game::new(seed);
            if sampled { g.run_sampled() } else { g.run_random() }
            for s in g.scores() {
                sum += s as i64;
                lo = lo.min(s);
                hi = hi.max(s);
                if s < 0 { starved += 1; }
            }
        }
        (sum as f64 / (n * 4) as f64, lo, hi, starved as f64 * 100.0 / (n * 4) as f64)
    };
    let (em, elo, ehi, eneg) = stats(false);
    let (sm, slo, shi, sneg) = stats(true);
    println!("\nfinal scores");
    println!("  enumerate: mean {em:>6.1}  range {elo}..{ehi}  {eneg:.0}% negative");
    println!("  sample:    mean {sm:>6.1}  range {slo}..{shi}  {sneg:.0}% negative");
}
