//! SCRATCH: where does the late-game move space actually come from?
//!
//! Plays whole games with a real agent (random play never reaches the positions
//! that hurt) and, at every turn, measures both the size of the move space and
//! the structure that produced it: workers on board, options per worker, how
//! much of each worker's option list is the pay-to-step-down multiplier, and how
//! fat the two combinatorial spaces (Uxmal's mirror, Tikal's double build) are.
//!
//! Delete before finishing.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use tzolkin::effect::Effect;
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::{self, Kinds, MoveKind};
use tzolkin::record::parse_agent;
use tzolkin::spaces::choices_at;
use tzolkin::state::GameState;

fn set_prune(on: bool) {
    tzolkin::options::PRUNE.store(on, std::sync::atomic::Ordering::Relaxed);
}

struct Turn {
    day: u8,
    on_board: usize,
    corn: u8,
    place: usize,
    retrieve: usize,
    nanos: u128,
    capped: bool,
    /// Product of per-worker option counts: the retrieval tree before the
    /// state memo collapses commuting orders.
    product: f64,
    worker_opts: Vec<usize>,
    /// Same, if a worker could only take the action of its own space.
    own_opts: Vec<usize>,
}

fn measure(g: &GameState, p: PlayerId, cap: Duration) -> Turn {
    let mut place = 0usize;
    let mut retrieve = 0usize;
    let mut capped = false;

    let t = Instant::now();
    let mut n = 0u64;
    let _ = moves::visit_moves_of(g, p, Kinds::Placements, |_| {
        place += 1;
        ControlFlow::Continue(())
    });
    let _ = moves::visit_moves_of(g, p, Kinds::Retrievals, |_| {
        retrieve += 1;
        n += 1;
        if n % 4096 == 0 && t.elapsed() > cap {
            capped = true;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    let nanos = t.elapsed().as_nanos();

    let mut worker_opts = Vec::new();
    let mut own_opts = Vec::new();
    let mut product = 1.0f64;
    for w in g.on_board(p) {
        let Some((gear, pos)) = g.loc(w).on_board() else {
            continue;
        };
        let total = moves::choices_for_worker(g, p, gear, pos).len();
        // The "no step-down" counterfactual: the space's own action plus skip.
        let mut own = choices_at(g, p, gear, pos);
        own.retain(|c| !c.is_skip());
        worker_opts.push(total);
        own_opts.push(own.len() + 1);
        product *= total as f64;
    }

    Turn {
        day: g.day,
        on_board: g.on_board(p).count(),
        corn: g.players[p.idx()].corn,
        place,
        retrieve,
        nanos,
        capped,
        product,
        worker_opts,
        own_opts,
    }
}

/// Per-space option counts, so the fat spaces name themselves.
fn space_census(g: &GameState, p: PlayerId, census: &mut Vec<(Gear, u8, usize, usize, usize)>) {
    for gear in Gear::ALL {
        for i in 0..gear.size() {
            let v = choices_at(g, p, gear, Pos(i));
            let two_builds = v
                .iter()
                .filter(|c| c.0.iter().filter(|e| matches!(e, Effect::Build(_))).count() >= 2)
                .count();
            let mirrored = if gear == Gear::Uxmal {
                tzolkin::spaces::uxmal::mirror_choices(g, p, tzolkin::spaces::MAX_DEPTH).len()
            } else {
                0
            };
            census.push((gear, i, v.len(), two_builds, mirrored));
        }
    }
}

fn pct<T: Copy>(v: &[T], q: f64) -> T {
    v[((v.len() as f64 - 1.0) * q) as usize]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    let spec = args.next().unwrap_or_else(|| "heuristic:32".into());
    let cap_ms: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1500);
    let cap = Duration::from_millis(cap_ms);

    let agent = parse_agent(&spec, false).expect("agent");
    let mut turns: Vec<Turn> = Vec::new();
    let mut pruned: Vec<Turn> = Vec::new();
    let mut census: Vec<(Gear, u8, usize, usize, usize)> = Vec::new();

    let wall = Instant::now();
    for seed in 0..games {
        let mut g = Game::new(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_1234);
        let mut guard = 0;
        while !g.state.over && guard < 60 {
            guard += 1;
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                set_prune(false);
                turns.push(measure(&g.state, p, cap));
                if g.state.day >= 18 {
                    space_census(&g.state, p, &mut census);
                }
                set_prune(true);
                pruned.push(measure(&g.state, p, cap));
                // Positions come from the *unpruned* engine, so the trajectory
                // is the one the pre-change code would have played.
                set_prune(false);
                if let Some(o) = agent.play_turn(&g.state, p, 0.0, &mut rng) {
                    g.play(p, &o.mv);
                }
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
        eprintln!(
            "seed {seed} done ({} turns, {:.1}s elapsed)",
            turns.len(),
            wall.elapsed().as_secs_f64()
        );
    }

    let mut counts: Vec<usize> = turns.iter().map(|t| t.place + t.retrieve).collect();
    counts.sort_unstable();
    let mut nanos: Vec<u128> = turns.iter().map(|t| t.nanos).collect();
    nanos.sort_unstable();

    println!("agent {spec}, {games} games, {} turns", turns.len());
    println!(
        "capped turns (>{cap_ms}ms): {}",
        turns.iter().filter(|t| t.capped).count()
    );

    println!("\nbranching factor (>= when capped)");
    for q in [0.5, 0.75, 0.9, 0.99, 0.999, 1.0] {
        println!("  p{:<6} {:>12}", format!("{:.1}", q * 100.0), pct(&counts, q));
    }
    println!(
        "  mean   {:>12.0}",
        counts.iter().sum::<usize>() as f64 / counts.len() as f64
    );

    println!("\nvisit time");
    for q in [0.5, 0.9, 0.99, 1.0] {
        println!(
            "  p{:<6} {:>10.3} ms",
            format!("{:.0}", q * 100.0),
            pct(&nanos, q) as f64 / 1e6
        );
    }
    println!(
        "  total  {:>10.2} s",
        nanos.iter().sum::<u128>() as f64 / 1e9
    );

    let pl: usize = turns.iter().map(|t| t.place).sum();
    let re: usize = turns.iter().map(|t| t.retrieve).sum();
    println!(
        "\nmoves generated: {pl} placement, {re} retrieval ({:.1}% retrieval)",
        re as f64 * 100.0 / (pl + re).max(1) as f64
    );

    println!("\nbranching by workers on board");
    for n in 0..=6 {
        let mut v: Vec<usize> = turns
            .iter()
            .filter(|t| t.on_board == n)
            .map(|t| t.place + t.retrieve)
            .collect();
        if v.is_empty() {
            continue;
        }
        v.sort_unstable();
        println!(
            "  {n} workers: n={:<6} median {:>8}  p99 {:>10}  max {:>10}",
            v.len(),
            pct(&v, 0.5),
            pct(&v, 0.99),
            v[v.len() - 1]
        );
    }

    println!("\nby day (late game)");
    for d in [0u8, 4, 8, 12, 16, 20, 24] {
        let mut v: Vec<usize> = turns
            .iter()
            .filter(|t| t.day >= d && t.day < d + 4)
            .map(|t| t.place + t.retrieve)
            .collect();
        if v.is_empty() {
            continue;
        }
        v.sort_unstable();
        let w: f64 = turns
            .iter()
            .filter(|t| t.day >= d && t.day < d + 4)
            .map(|t| t.on_board as f64)
            .sum::<f64>()
            / v.len() as f64;
        println!(
            "  days {d:>2}-{:<2}: n={:<5} median {:>8}  p99 {:>10}  mean workers {w:.2}",
            d + 3,
            v.len(),
            pct(&v, 0.5),
            pct(&v, 0.99)
        );
    }

    // The step-down multiplier: the ratio of the whole option list to the
    // subset a worker could take without paying to walk down its gear.
    let mut ratios: Vec<f64> = Vec::new();
    let mut opt_hist: Vec<usize> = Vec::new();
    for t in &turns {
        for (i, &o) in t.worker_opts.iter().enumerate() {
            opt_hist.push(o);
            let own = t.own_opts[i].max(1);
            ratios.push(o as f64 / own as f64);
        }
    }
    opt_hist.sort_unstable();
    ratios.sort_by(|a, b| a.total_cmp(b));
    if !opt_hist.is_empty() {
        println!("\noptions per worker on board (choices_for_worker)");
        for q in [0.5, 0.9, 0.99, 1.0] {
            println!(
                "  p{:<6} {:>8}   step-down ratio p{:.0} {:.2}x",
                format!("{:.0}", q * 100.0),
                pct(&opt_hist, q),
                q * 100.0,
                pct(&ratios, q)
            );
        }
        println!(
            "  mean {:.1} options, mean step-down ratio {:.2}x",
            opt_hist.iter().sum::<usize>() as f64 / opt_hist.len() as f64,
            ratios.iter().sum::<f64>() / ratios.len() as f64
        );
    }

    let mut prods: Vec<f64> = turns
        .iter()
        .filter(|t| t.on_board > 0)
        .map(|t| t.product)
        .collect();
    prods.sort_by(|a, b| a.total_cmp(b));
    if !prods.is_empty() {
        println!("\nproduct of per-worker option counts (unmemoised retrieval tree width)");
        for q in [0.5, 0.9, 0.99, 1.0] {
            println!(
                "  p{:<6} {:>14.0}",
                format!("{:.0}", q * 100.0),
                pct(&prods, q)
            );
        }
    }

    if !census.is_empty() {
        println!("\nspace census (day >= 18), mean options per space");
        let mut by: Vec<(Gear, u8, f64, f64, f64, usize)> = Vec::new();
        for gear in Gear::ALL {
            for i in 0..gear.size() {
                let rows: Vec<_> = census
                    .iter()
                    .filter(|r| r.0 == gear && r.1 == i)
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                let n = rows.len();
                by.push((
                    gear,
                    i,
                    rows.iter().map(|r| r.2 as f64).sum::<f64>() / n as f64,
                    rows.iter().map(|r| r.3 as f64).sum::<f64>() / n as f64,
                    rows.iter().map(|r| r.4 as f64).sum::<f64>() / n as f64,
                    rows.iter().map(|r| r.2).max().unwrap(),
                ));
            }
        }
        by.sort_by(|a, b| b.2.total_cmp(&a.2));
        println!("  {:<14} {:>3} {:>10} {:>10} {:>10} {:>8}", "gear", "pos", "mean opts", "2-build", "mirror", "max");
        for (gear, i, mean, tb, mi, mx) in by.iter().take(20) {
            println!(
                "  {:<14} {:>3} {:>10.1} {:>10.1} {:>10.1} {:>8}",
                gear.name(),
                i,
                mean,
                tb,
                mi,
                mx
            );
        }
    }

    // The worst turn, spelled out.
    if let Some(worst) = turns.iter().max_by_key(|t| t.place + t.retrieve) {
        println!(
            "\nworst turn: day {} corn {} workers {} -> {} moves ({} place, {} retrieve){}",
            worst.day,
            worst.corn,
            worst.on_board,
            worst.place + worst.retrieve,
            worst.place,
            worst.retrieve,
            if worst.capped { " [CAPPED]" } else { "" }
        );
        println!("  per-worker options: {:?}", worst.worker_opts);
        println!("  own-space only:     {:?}", worst.own_opts);
    }

    // ---- A/B --------------------------------------------------------
    println!("\n================ pruned vs unpruned, same positions ================");
    let mut a: Vec<usize> = turns.iter().map(|t| t.place + t.retrieve).collect();
    let mut b: Vec<usize> = pruned.iter().map(|t| t.place + t.retrieve).collect();
    a.sort_unstable();
    b.sort_unstable();
    println!("{:<10} {:>14} {:>14} {:>9}", "quantile", "before", "after", "ratio");
    for q in [0.5, 0.75, 0.9, 0.99, 0.999, 1.0] {
        let (x, y) = (pct(&a, q), pct(&b, q));
        println!(
            "  p{:<7} {:>14} {:>14} {:>8.2}x",
            format!("{:.1}", q * 100.0),
            x,
            y,
            x as f64 / y.max(1) as f64
        );
    }
    let (sa, sb) = (a.iter().sum::<usize>(), b.iter().sum::<usize>());
    println!("  {:<8} {:>14} {:>14} {:>8.2}x", "total", sa, sb, sa as f64 / sb.max(1) as f64);

    let mut ta: Vec<u128> = turns.iter().map(|t| t.nanos).collect();
    let mut tb: Vec<u128> = pruned.iter().map(|t| t.nanos).collect();
    ta.sort_unstable();
    tb.sort_unstable();
    println!("\nvisit time (ms)");
    for q in [0.5, 0.9, 0.99, 1.0] {
        let (x, y) = (pct(&ta, q) as f64 / 1e6, pct(&tb, q) as f64 / 1e6);
        println!(
            "  p{:<7} {:>14.3} {:>14.3} {:>8.2}x",
            format!("{:.0}", q * 100.0),
            x,
            y,
            x / y.max(1e-9)
        );
    }
    let (na, nb) = (
        ta.iter().sum::<u128>() as f64 / 1e9,
        tb.iter().sum::<u128>() as f64 / 1e9,
    );
    println!("  {:<8} {:>14.2} {:>14.2} {:>8.2}x", "total s", na, nb, na / nb.max(1e-9));
    println!(
        "  capped turns: before {}, after {}",
        turns.iter().filter(|t| t.capped).count(),
        pruned.iter().filter(|t| t.capped).count()
    );

    let mut oa: Vec<usize> = turns.iter().flat_map(|t| t.worker_opts.clone()).collect();
    let mut ob: Vec<usize> = pruned.iter().flat_map(|t| t.worker_opts.clone()).collect();
    oa.sort_unstable();
    ob.sort_unstable();
    if !oa.is_empty() {
        println!("\noptions per worker (choices_for_worker)");
        for q in [0.5, 0.9, 0.99, 1.0] {
            let (x, y) = (pct(&oa, q), pct(&ob, q));
            println!(
                "  p{:<7} {:>14} {:>14} {:>8.2}x",
                format!("{:.0}", q * 100.0),
                x,
                y,
                x as f64 / y.max(1) as f64
            );
        }
        println!(
            "  {:<8} {:>14.1} {:>14.1} {:>8.2}x",
            "mean",
            oa.iter().sum::<usize>() as f64 / oa.len() as f64,
            ob.iter().sum::<usize>() as f64 / ob.len() as f64,
            (oa.iter().sum::<usize>() as f64) / (ob.iter().sum::<usize>() as f64).max(1.0)
        );
    }

    println!("\nwall {:.1}s", wall.elapsed().as_secs_f64());
    let _ = MoveKind::Place(Default::default());
}
