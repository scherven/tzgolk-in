//! SCRATCH: where does the late-game move space actually come from?
//!
//! Phase 1 plays whole games with a real agent (random play never reaches the
//! positions that hurt) and keeps every turn's position. Phase 2 replays each
//! position twice -- pruning off, then on -- so before/after is measured on
//! *identical* positions rather than on two different games. Phase 3 attributes
//! the option count to workers and spaces.
//!
//! Delete before finishing.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use tzolkin::effect::Effect;
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::{self, Kinds};
use tzolkin::record::parse_agent;
use tzolkin::spaces::choices_at;
use tzolkin::state::GameState;

fn set_prune(on: bool) {
    tzolkin::options::PRUNE.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct Sized_ {
    place: usize,
    retrieve: usize,
    nanos: u128,
    capped: bool,
}

fn size_of_move_space(g: &GameState, p: PlayerId, cap: Duration) -> Sized_ {
    let mut place = 0usize;
    let mut retrieve = 0usize;
    let mut capped = false;
    let t = Instant::now();
    let _ = moves::visit_moves_of(g, p, Kinds::Placements, |_| {
        place += 1;
        ControlFlow::Continue(())
    });
    let mut n = 0u64;
    let _ = moves::visit_moves_of(g, p, Kinds::Retrievals, |_| {
        retrieve += 1;
        n += 1;
        if n % 2048 == 0 && t.elapsed() > cap {
            capped = true;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    Sized_ {
        place,
        retrieve,
        nanos: t.elapsed().as_nanos(),
        capped,
    }
}

fn pct<T: Copy>(v: &[T], q: f64) -> T {
    v[((v.len() as f64 - 1.0) * q) as usize]
}

fn quantiles(label: &str, a: &[usize], b: &[usize]) {
    println!("\n{label}");
    println!("  {:<8} {:>13} {:>13} {:>9}", "", "before", "after", "ratio");
    for q in [0.5, 0.75, 0.9, 0.99, 0.999, 1.0] {
        let (x, y) = (pct(a, q), pct(b, q));
        println!(
            "  p{:<7} {:>13} {:>13} {:>8.2}x",
            format!("{:.1}", q * 100.0),
            x,
            y,
            x as f64 / y.max(1) as f64
        );
    }
    let (sa, sb) = (a.iter().sum::<usize>(), b.iter().sum::<usize>());
    println!(
        "  {:<8} {:>13.0} {:>13.0} {:>8.2}x",
        "mean",
        sa as f64 / a.len() as f64,
        sb as f64 / b.len() as f64,
        sa as f64 / sb.max(1) as f64
    );
    println!("  {:<8} {:>13} {:>13} {:>8.2}x", "total", sa, sb, sa as f64 / sb.max(1) as f64);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    let spec = args.next().unwrap_or_else(|| "heuristic:32".into());
    let cap_ms: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(20000);
    let cap = Duration::from_millis(cap_ms);

    let agent = parse_agent(&spec, false).expect("agent");
    let wall = Instant::now();

    // ---- phase 1: reach the positions -----------------------------------
    set_prune(false);
    let mut positions: Vec<(GameState, PlayerId)> = Vec::new();
    for seed in 0..games {
        let mut g = Game::new(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_1234);
        let mut guard = 0;
        while !g.state.over && guard < 60 {
            guard += 1;
            g.state.current = g.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = g.state.current;
                positions.push((g.state, p));
                if let Some(o) = agent.play_turn(&g.state, p, 0.0, &mut rng) {
                    g.play(p, &o.mv);
                }
                g.state.current = g.state.current.next(1);
            }
            g.end_round_public();
        }
    }
    eprintln!(
        "phase 1: {} positions from {games} games ({:.1}s)",
        positions.len(),
        wall.elapsed().as_secs_f64()
    );

    // ---- phase 3 (attribution) ------------------------------------------
    for on in [false, true] {
    set_prune(on);
    println!("\n########## PRUNING {} ##########", if on { "ON" } else { "OFF" });
    let mut worker_opts: Vec<usize> = Vec::new();
    let mut worker_states: Vec<usize> = Vec::new();
    let mut worker_own: Vec<usize> = Vec::new();
    let mut by_space: Vec<(Gear, u8, usize, usize, usize)> = Vec::new(); // gear,pos,n,sum len,sum distinct
    let mut hot: Vec<(Gear, u8, usize, usize)> = Vec::new();
    for &(ref g, p) in &positions {
        for w in g.on_board(p) {
            let Some((gear, pos)) = g.loc(w).on_board() else {
                continue;
            };
            let cs = moves::choices_for_worker(g, p, gear, pos);
            let mut seen = HashSet::new();
            for c in &cs {
                let mut probe = *g;
                c.apply(&mut probe, p);
                probe.retrieve_worker(w);
                seen.insert(probe);
            }
            worker_opts.push(cs.len());
            worker_states.push(seen.len());
            worker_own.push(choices_at(g, p, gear, pos).len());
            hot.push((gear, pos.0, cs.len(), seen.len()));
        }
        if g.day >= 14 {
            for gear in Gear::ALL {
                for i in 0..gear.size() {
                    let v = choices_at(g, p, gear, Pos(i));
                    let mut seen = HashSet::new();
                    for c in &v {
                        let mut probe = *g;
                        c.apply(&mut probe, p);
                        seen.insert(probe);
                    }
                    let row = by_space
                        .iter_mut()
                        .find(|r| r.0 == gear && r.1 == i);
                    match row {
                        Some(r) => {
                            r.2 += 1;
                            r.3 += v.len();
                            r.4 += seen.len();
                        }
                        None => by_space.push((gear, i, 1, v.len(), seen.len())),
                    }
                }
            }
        }
    }
    eprintln!("phase 3 done ({:.1}s)", wall.elapsed().as_secs_f64());

    println!("=== where the options are (UNPRUNED, {} positions) ===", positions.len());
    let mut wo = worker_opts.clone();
    let mut ws = worker_states.clone();
    let mut ww = worker_own.clone();
    wo.sort_unstable();
    ws.sort_unstable();
    ww.sort_unstable();
    println!("\noptions for one worker on the board");
    println!(
        "  {:<8} {:>12} {:>12} {:>12}",
        "", "own space", "with step-down", "distinct states"
    );
    for q in [0.5, 0.9, 0.99, 1.0] {
        println!(
            "  p{:<7} {:>12} {:>12} {:>12}",
            format!("{:.0}", q * 100.0),
            pct(&ww, q),
            pct(&wo, q),
            pct(&ws, q)
        );
    }
    println!(
        "  {:<8} {:>12.1} {:>12.1} {:>12.1}",
        "mean",
        ww.iter().sum::<usize>() as f64 / ww.len() as f64,
        wo.iter().sum::<usize>() as f64 / wo.len() as f64,
        ws.iter().sum::<usize>() as f64 / ws.len() as f64
    );

    // Which occupied spaces carry the mass?
    hot.sort_by_key(|r| std::cmp::Reverse(r.2));
    let mut agg: Vec<(Gear, u8, usize, usize, usize)> = Vec::new();
    for (gear, pos, len, st) in &hot {
        match agg.iter_mut().find(|r| r.0 == *gear && r.1 == *pos) {
            Some(r) => {
                r.2 += 1;
                r.3 += len;
                r.4 += st;
            }
            None => agg.push((*gear, *pos, 1, *len, *st)),
        }
    }
    agg.sort_by_key(|r| std::cmp::Reverse(r.3));
    println!("\nwhere workers actually sit: total options contributed");
    println!(
        "  {:<14} {:>3} {:>8} {:>12} {:>10} {:>12}",
        "gear", "pos", "workers", "sum options", "mean", "sum distinct"
    );
    for (gear, pos, n, len, st) in agg.iter().take(14) {
        println!(
            "  {:<14} {:>3} {:>8} {:>12} {:>10.1} {:>12}",
            gear.name(),
            pos,
            n,
            len,
            *len as f64 / *n as f64,
            st
        );
    }

    by_space.sort_by(|a, b| (b.3 as f64 / b.2 as f64).total_cmp(&(a.3 as f64 / a.2 as f64)));
    println!("\nspace census (day >= 14): mean options and mean distinct states");
    println!("  {:<14} {:>3} {:>12} {:>14} {:>8}", "gear", "pos", "mean opts", "mean distinct", "dup x");
    for (gear, pos, n, len, st) in by_space.iter().take(16) {
        println!(
            "  {:<14} {:>3} {:>12.1} {:>14.1} {:>8.2}",
            gear.name(),
            pos,
            *len as f64 / *n as f64,
            *st as f64 / *n as f64,
            *len as f64 / (*st).max(1) as f64
        );
    }
    }
    set_prune(false);

    // ---- phase 2: A/B on identical positions ----------------------------
    let mut a: Vec<usize> = Vec::new();
    let mut b: Vec<usize> = Vec::new();
    let mut ta: Vec<u128> = Vec::new();
    let mut tb: Vec<u128> = Vec::new();
    let mut capped_a = 0;
    let mut capped_b = 0;
    let mut worst = (0usize, 0usize, 0u8, 0usize);
    let mut place_a = 0usize;
    let mut retr_a = 0usize;
    for (i, &(ref g, p)) in positions.iter().enumerate() {
        set_prune(false);
        let x = size_of_move_space(g, p, cap);
        set_prune(true);
        let y = size_of_move_space(g, p, cap);
        if x.capped {
            capped_a += 1;
        }
        if y.capped {
            capped_b += 1;
        }
        place_a += x.place;
        retr_a += x.retrieve;
        // Only pairs where neither side hit the cap are comparable.
        if !x.capped && !y.capped {
            a.push(x.place + x.retrieve);
            b.push(y.place + y.retrieve);
            ta.push(x.nanos);
            tb.push(y.nanos);
            if x.place + x.retrieve > worst.0 {
                worst = (x.place + x.retrieve, y.place + y.retrieve, g.day, g.on_board(p).count());
            }
        }
        if i % 100 == 0 {
            eprintln!("phase 2: {i}/{} ({:.1}s)", positions.len(), wall.elapsed().as_secs_f64());
        }
    }
    set_prune(true);

    println!(
        "\n=== pruned vs unpruned on identical positions ({} of {} comparable, cap {cap_ms}ms) ===",
        a.len(),
        positions.len()
    );
    println!(
        "unpruned mix: {place_a} placement, {retr_a} retrieval ({:.1}% retrieval)",
        retr_a as f64 * 100.0 / (place_a + retr_a).max(1) as f64
    );
    println!("capped turns: before {capped_a}, after {capped_b}");
    quantiles("branching factor", &a, &b);

    let tam: Vec<usize> = ta.iter().map(|&v| v as usize).collect();
    let tbm: Vec<usize> = tb.iter().map(|&v| v as usize).collect();
    let mut tas = tam.clone();
    let mut tbs = tbm.clone();
    tas.sort_unstable();
    tbs.sort_unstable();
    println!("\nlegal_moves() time (ms)");
    println!("  {:<8} {:>13} {:>13} {:>9}", "", "before", "after", "ratio");
    for q in [0.5, 0.9, 0.99, 1.0] {
        let (x, y) = (pct(&tas, q) as f64 / 1e6, pct(&tbs, q) as f64 / 1e6);
        println!(
            "  p{:<7} {:>13.3} {:>13.3} {:>8.2}x",
            format!("{:.0}", q * 100.0),
            x,
            y,
            x / y.max(1e-9)
        );
    }
    let (na, nb) = (
        tas.iter().sum::<usize>() as f64 / 1e9,
        tbs.iter().sum::<usize>() as f64 / 1e9,
    );
    println!("  {:<8} {:>13.2} {:>13.2} {:>8.2}x", "total s", na, nb, na / nb.max(1e-9));
    println!(
        "\nworst comparable position: day {} with {} workers on board: {} -> {} ({:.2}x)",
        worst.2,
        worst.3,
        worst.0,
        worst.1,
        worst.0 as f64 / worst.1.max(1) as f64
    );

    // per-worker options, pruned
    set_prune(true);
    let mut wo2: Vec<usize> = Vec::new();
    for &(ref g, p) in &positions {
        for w in g.on_board(p) {
            if let Some((gear, pos)) = g.loc(w).on_board() {
                wo2.push(moves::choices_for_worker(g, p, gear, pos).len());
            }
        }
    }
    wo2.sort_unstable();
    set_prune(false);
    let mut wo1: Vec<usize> = Vec::new();
    for &(ref g, p) in &positions {
        for w in g.on_board(p) {
            if let Some((gear, pos)) = g.loc(w).on_board() {
                wo1.push(moves::choices_for_worker(g, p, gear, pos).len());
            }
        }
    }
    wo1.sort_unstable();
    set_prune(true);
    quantiles("options per worker (choices_for_worker)", &wo1, &wo2);

    let _ = Effect::UnlockWorker;
    println!("\nwall {:.1}s", wall.elapsed().as_secs_f64());
}
