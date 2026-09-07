//! Scratch: where does the alpha-beta actually spend its time? (delete me)

use std::time::Instant;
use tzolkin::ids::*;
use tzolkin::record::AgentSpec;
use tzolkin::search::{Config, Opponents, Search};
use tzolkin::state::GameState;

/// Positions from games driven by a real agent, not the uniform-over-enumeration
/// driver `Game::play_round` uses: branching is a function of play quality
/// (`docs/SEARCH.md` 1.1), so positions from bad play understate every cost.
fn positions(games: u64, every: usize) -> Vec<GameState> {
    use rand::SeedableRng;
    let a = AgentSpec::parse("heuristic:16", false).unwrap().instance();
    let mut out = Vec::new();
    for seed in 0..games {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xABCD);
        let mut st = tzolkin::game::Game::new(seed + 5000).state;
        let mut k = 0usize;
        let mut guard = 0;
        while !st.over && guard < 60 {
            st.current = st.first_player;
            for _ in 0..N_PLAYERS {
                let p = st.current;
                if k % every == 0 {
                    out.push(st);
                }
                k += 1;
                if let Some(o) = a.play_turn(&st, p, 0.0, &mut rng) {
                    tzolkin::moves::apply_move(&mut st, p, &o.mv);
                    st.refill_buildings();
                }
                st.current = st.current.next(1);
            }
            tzolkin::search::advance_seat(&mut st);
            guard += 1;
        }
    }
    out
}

fn run(label: &str, cfg: Config, ps: &[GameState]) {
    let mut s = Search::new(cfg);
    let (mut nodes, mut leaves, mut tt, mut cm) = (0u64, 0u64, 0u64, 0u64);
    let (mut depth, mut deep, mut rootn, mut rootms) = (0f64, 0f64, 0f64, 0f64);
    let t = Instant::now();
    for g in ps {
        let _ = s.search(g, g.current);
        let st = s.stats();
        nodes += st.nodes;
        leaves += st.leaves;
        tt += st.tt_hits;
        cm += st.cand_moves;
        depth += st.depth as f64;
        deep += st.deepened as f64;
        rootn += st.root_moves as f64;
        rootms += st.root_elapsed.as_secs_f64() * 1e3;
    }
    let n = ps.len() as f64;
    println!(
        "{label:<40} {:>7.0}ms/turn (root {:>6.0}ms)  nodes {:>7.1}  leaves {:>7.1}  tt {:>7.1}  candmoves {:>8.0}  depth {:>4.1}  deep {:>4.1}  rootmoves {:>9.0}",
        t.elapsed().as_secs_f64() * 1e3 / n,
        rootms / n,
        nodes as f64 / n,
        leaves as f64 / n,
        tt as f64 / n,
        cm as f64 / n,
        depth / n,
        deep / n,
        rootn / n
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let which = argv.get(1).cloned().unwrap_or_else(|| "depth".into());
    let ps = positions(6, 7);
    println!("{} positions from heuristic:16 games\n", ps.len());
    let base = || Config::default();
    match which.as_str() {
        "depth" => {
            for d in [4u8, 8, 12, 16, 24] {
                run(&format!("greedy d{d} 200ms"), Config { max_depth: d, ..base() }.with_budget_ms(200), &ps);
            }
        }
        "nocache" => {
            for d in [4u8, 8, 12] {
                run(&format!("greedy d{d} 200ms nocache"), Config { max_depth: d, cache: false, ..base() }.with_budget_ms(200), &ps);
                run(&format!("greedy d{d} 200ms cache"), Config { max_depth: d, ..base() }.with_budget_ms(200), &ps);
            }
        }
        "width" => {
            for w in [vec![12usize, 6, 4, 3, 2], vec![24, 12, 8, 6, 4], vec![48, 24, 16, 12, 8]] {
                run(&format!("greedy d8 200ms w{w:?}"), Config { max_depth: 8, widths: w, ..base() }.with_budget_ms(200), &ps);
            }
        }
        "cap" => {
            for c in [25usize, 50, 100, 200, 400, 800] {
                run(&format!("greedy d8 200ms cap{c}"), Config { max_depth: 8, interior_cap: c, ..base() }.with_budget_ms(200), &ps);
            }
        }
        "paranoid" => {
            for d in [4u8, 8, 12] {
                run(&format!("paranoid d{d} 200ms"), Config { max_depth: d, opponents: Opponents::Paranoid, ..base() }.with_budget_ms(200), &ps);
            }
        }
        "kinds" => {
            // Does a placement ever survive the one-ply root beam? The README
            // says the heuristic "barely discriminates between placements", so
            // the beam the search deepens may be all retrievals -- in which
            // case the search can never repair the thing it is worst at.
            let mut s = Search::new(Config { max_depth: 8, ..Config::default() }.with_budget_ms(200));
            let (mut plc, mut ret, mut n) = (0usize, 0usize, 0usize);
            let (mut top_plc, mut top_n) = (0usize, 0usize);
            let (mut all_plc, mut all_n) = (0usize, 0usize);
            let mut changed = 0usize;
            for g in &ps {
                // What the whole move list looks like, by kind.
                let mut ap = 0usize; let mut an = 0usize;
                let _ = tzolkin::moves::visit_legal_moves(g, g.current, |m| {
                    an += 1;
                    if matches!(m.kind, tzolkin::moves::MoveKind::Place(_)) { ap += 1; }
                    std::ops::ControlFlow::Continue(())
                });
                all_plc += ap; all_n += an;

                // The one-ply shortlist, before deepening.
                let one = tzolkin::eval::rank_all_capped(g, g.current, 12, std::time::Duration::from_millis(2500));
                let one_best = one.moves.first().map(|(m, _)| m.clone());
                for (m, _) in &one.moves {
                    n += 1;
                    if matches!(m.kind, tzolkin::moves::MoveKind::Place(_)) { plc += 1; } else { ret += 1; }
                }
                if let Some((m, _)) = one.moves.first() {
                    top_n += 1;
                    if matches!(m.kind, tzolkin::moves::MoveKind::Place(_)) { top_plc += 1; }
                }
                // Does the deep search overturn the one-ply pick?
                let r = s.search(g, g.current);
                if let (Some(a), Some((b, _))) = (one_best, r.moves.first()) {
                    if !a.same_effect(b) { changed += 1; }
                }
            }
            println!("whole move list:  {:.1}% placements ({all_plc}/{all_n})", 100.0 * all_plc as f64 / all_n as f64);
            println!("root beam of 12:  {:.1}% placements ({plc} placements, {ret} retrievals, n={n})", 100.0 * plc as f64 / n as f64);
            println!("one-ply best:     {:.1}% placements ({top_plc}/{top_n})", 100.0 * top_plc as f64 / top_n as f64);
            println!("deep search overturned the one-ply pick on {changed}/{} positions ({:.0}%)", ps.len(), 100.0 * changed as f64 / ps.len() as f64);
        }
        "mcts" => {
            use rand::SeedableRng;
            // Wall-clock per turn for each agent spec, on the same positions,
            // so an mcts budget can be matched to a minimax one.
            for spec in argv.iter().skip(2) {
                let a = AgentSpec::parse(spec, false).unwrap_or_else(|e| panic!("{spec}: {e}"));
                let inst = a.instance();
                let mut rng = rand::rngs::StdRng::seed_from_u64(7);
                let t = Instant::now();
                for g in &ps {
                    let _ = inst.play_turn(g, g.current, 0.0, &mut rng);
                }
                println!("{:<28} {:>8.1} ms/turn", a.name(), t.elapsed().as_secs_f64() * 1e3 / ps.len() as f64);
            }
        }
        "shape" => {
            let v = |w: Vec<usize>, d: u8, ow: bool| Config { max_depth: d, widths: w, own_width: ow, ..Config::default() }.with_budget_ms(200);
            run("d8  w[12,6,4,3,2]        (bar)", v(vec![12,6,4,3,2], 8, false), &ps);
            run("d16 w[12,6,4,3,2]", v(vec![12,6,4,3,2], 16, false), &ps);
            run("d24 w[12,1]  greedy line", v(vec![12,1], 24, false), &ps);
            run("d40 w[12,1]  greedy line", v(vec![12,1], 40, false), &ps);
            run("d24 w[12,2,1]", v(vec![12,2,1], 24, false), &ps);
            run("d12 ownwidth w[12,6,4,3,2]", v(vec![12,6,4,3,2], 12, true), &ps);
            run("d12 ownwidth w[12,3,2]", v(vec![12,3,2], 12, true), &ps);
            run("d16 ownwidth w[12,3,2,1]", v(vec![12,3,2,1], 16, true), &ps);
        }
        "micro" => {
            use std::ops::ControlFlow;
            // Split an interior node's cost three ways: walking the generator,
            // applying each move, and scoring the result.
            let cap = 400usize;
            let mut n = 0u64;
            let t = Instant::now();
            for g in &ps {
                for kind in [tzolkin::moves::Kinds::Placements, tzolkin::moves::Kinds::Retrievals] {
                    let mut k = 0usize;
                    let _ = tzolkin::moves::visit_moves_of(g, g.current, kind, |_m| {
                        k += 1; n += 1;
                        if k >= cap { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
                    });
                }
            }
            let walk = t.elapsed().as_secs_f64();
            println!("walk only            {n} moves in {:.3}s = {:.2} us/move", walk, walk * 1e6 / n as f64);

            let mut n2 = 0u64; let mut acc = 0f32;
            let t = Instant::now();
            for g in &ps {
                for kind in [tzolkin::moves::Kinds::Placements, tzolkin::moves::Kinds::Retrievals] {
                    let mut k = 0usize;
                    let _ = tzolkin::moves::visit_moves_of(g, g.current, kind, |m| {
                        k += 1; n2 += 1;
                        let s = tzolkin::eval::successor(g, g.current, m);
                        acc += s.day as f32;
                        if k >= cap { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
                    });
                }
            }
            let succ = t.elapsed().as_secs_f64();
            println!("walk + successor     {n2} moves in {:.3}s = {:.2} us/move  (+{:.2})", succ, succ * 1e6 / n2 as f64, (succ - walk) * 1e6 / n2 as f64);

            let mut n3 = 0u64;
            let t = Instant::now();
            for g in &ps {
                for kind in [tzolkin::moves::Kinds::Placements, tzolkin::moves::Kinds::Retrievals] {
                    let mut k = 0usize;
                    let _ = tzolkin::moves::visit_moves_of(g, g.current, kind, |m| {
                        k += 1; n3 += 1;
                        let s = tzolkin::eval::successor(g, g.current, m);
                        acc += tzolkin::eval::heuristic(&s, g.current);
                        if k >= cap { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
                    });
                }
            }
            let full = t.elapsed().as_secs_f64();
            println!("walk + succ + heur   {n3} moves in {:.3}s = {:.2} us/move  (+{:.2} for heuristic)  [{acc}]", full, full * 1e6 / n3 as f64, (full - succ) * 1e6 / n3 as f64);
        }
        other => println!("unknown mode {other}"),
    }
}
