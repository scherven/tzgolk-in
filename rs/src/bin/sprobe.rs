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
    let (mut nodes, mut leaves, mut tt, mut cm, mut cc, mut cut) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut depth, mut deep, mut rootn, mut rootms) = (0f64, 0f64, 0f64, 0f64);
    let t = Instant::now();
    for g in ps {
        let _ = s.search(g, g.current);
        let st = s.stats();
        nodes += st.nodes;
        leaves += st.leaves;
        tt += st.tt_hits;
        cm += st.cand_moves;
        cc += st.cand_calls;
        cut += st.cutoffs;
        depth += st.depth as f64;
        deep += st.deepened as f64;
        rootn += st.root_moves as f64;
        rootms += st.root_elapsed.as_secs_f64() * 1e3;
    }
    let n = ps.len() as f64;
    // `per-expand` is the number that matters: enumerating and statically
    // scoring the candidates of one *fresh* node. `cut` proves whether
    // alpha-beta is doing anything -- under Greedy no min node has a sibling,
    // so beta never moves and the answer is zero.
    println!(
        "{label:<34} {:>6.0}ms (root {:>5.0}ms) nodes {:>7.1} leaves {:>6.1} tt {:>7.1} ({:>3.0}%) expand {:>6.1} candmv {:>8.0} per-expand {:>5.0} cut {:>6.1} depth {:>4.1} deep {:>4.1} rootmv {:>7.0}",
        t.elapsed().as_secs_f64() * 1e3 / n,
        rootms / n,
        nodes as f64 / n,
        leaves as f64 / n,
        tt as f64 / n,
        100.0 * tt as f64 / (tt + cc).max(1) as f64,
        cc as f64 / n,
        cm as f64 / n,
        cm as f64 / cc.max(1) as f64,
        cut as f64 / n,
        depth / n,
        deep / n,
        rootn / n
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let which = argv.get(1).cloned().unwrap_or_else(|| "depth".into());
    // The wide `Take` nodes are late-game, high on a gear, with resources to
    // spend: a 6-game sample every 7th turn tops out around 84 edges and never
    // meets the cap at all. `SPROBE_GAMES`/`SPROBE_EVERY` are how a run gets to
    // the thousand-edge nodes the cap was written for.
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let ps = positions(env("SPROBE_GAMES", 6) as u64, env("SPROBE_EVERY", 7));
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
        "beam" => {
            // How often does the move the agent actually plays come from
            // outside the deepened beam? `MinimaxAgent` sets `keep` to
            // MAX_VISITS = 24 while the beam is widths[0] = 12, and the
            // deepening loop sorts all 24 on score alone -- so a move holding
            // nothing but its one-ply margin can finish first.
            for (label, keep, bf) in [
                ("as MinimaxAgent plays it (keep=24, beam=12)", 24usize, false),
                ("beamfirst                                  ", 24, true),
                ("keep=12 == beam (no tail to lose to)       ", 12, false),
            ] {
                let mut s = Search::new(Config { keep, beam_first: bf, max_depth: 8, ..Config::default() }.with_budget_ms(200));
                let (mut undeep, mut n) = (0usize, 0usize);
                for g in &ps {
                    let _ = s.search(g, g.current);
                    n += 1;
                    if !s.stats().best_deepened { undeep += 1; }
                }
                println!("{label}  played an un-deepened move on {undeep}/{n} turns ({:.0}%)", 100.0 * undeep as f64 / n as f64);
            }
        }
        "mcts" => {
            use rand::SeedableRng;
            // Wall-clock per turn for each agent spec, on the same positions,
            // so an mcts simulation budget can be matched to a minimax depth.
            //
            // **Interleaved, position by position, not spec by spec.** The
            // whole point of this number is to say two budgets are the same
            // size, and running one spec to completion and then the next
            // charges whichever went second for whatever else started on the
            // machine in between. Interleaving makes contention a common-mode
            // error instead of a bias, which is the same reason the arena
            // rotates seats within a block.
            let specs: Vec<_> = argv
                .iter()
                .skip(2)
                .map(|s| {
                    let a = AgentSpec::parse(s, false).unwrap_or_else(|e| panic!("{s}: {e}"));
                    let inst = a.instance();
                    (a.name(), inst, 0f64, 0usize)
                })
                .collect();
            let mut specs = specs;
            let mut rng = rand::rngs::StdRng::seed_from_u64(7);
            for _ in 0..3 {
                for g in &ps {
                    for (_, inst, secs, n) in specs.iter_mut() {
                        let t = Instant::now();
                        let _ = inst.play_turn(g, g.current, 0.0, &mut rng);
                        *secs += t.elapsed().as_secs_f64();
                        *n += 1;
                    }
                }
            }
            for (name, _, secs, n) in &specs {
                println!("{name:<46} {:>8.2} ms/turn  ({n} turns)", secs * 1e3 / *n as f64);
            }
        }
        "nodes" => {
            // Is §2.6's cap set anywhere near where the widths actually are?
            // `cap_per_width` was the alpha-beta's biggest single win and the
            // principle behind it -- do not spend enumeration budget out of
            // proportion to how much of the result you will use -- has exactly
            // one analogue here, which is `max_edges` plus widening.
            use tzolkin::mcts::{Mcts, MctsConfig, Priors};
            use tzolkin::phase::{HeuristicEvaluator, Phase};
            let sims: u32 = argv.get(2).and_then(|v| v.parse().ok()).unwrap_or(2048);
            let priors = match argv.get(3).map(|s| s.as_str()) {
                Some("1ply") => Priors::OnePly,
                _ => Priors::Evaluator,
            };
            let cfg = MctsConfig { priors, dirichlet_eps: 0.0, temperature: 0.0, ..MctsConfig::default() };
            let mut m = Mcts::new(HeuristicEvaluator, cfg);
            let mut widths: Vec<usize> = Vec::new();
            let (mut capped, mut widened, mut opened, mut n) = (0usize, 0usize, 0usize, 0usize);
            let (mut nodes, mut decisions, mut reused, mut discarded) = (0usize, 0usize, 0usize, 0usize);
            let mut all: Vec<(u32, u32)> = Vec::new();
            let (mut conc, mut uniform, mut nconc) = (0f64, 0f64, 0usize);
            let t = Instant::now();
            for g in ps.iter().take(60) {
                let mut st = *g;
                let mut at = (Phase::Beg, st.current, 0u8);
                for _ in 0..32 {
                    let (phase, turn, done) = at;
                    let r = m.search_at(&st, phase, turn, done, sims);
                    decisions += 1;
                    nodes += r.nodes;
                    let (ru, di) = m.reuse_stats();
                    reused += ru;
                    discarded += di;
                    if r.sims > 0 {
                        n += 1;
                        widths.push(r.legal_edges);
                        opened += r.visits.len();
                        // How concentrated is the visit distribution? With
                        // PUCT's exploration term tuned for a value scale it
                        // does not have, the root comes out near-uniform and
                        // the search is spending its budget proving that every
                        // edge is about the same.
                        let tot: u32 = r.visits.iter().map(|(_, n)| n).sum();
                        if tot > 0 {
                            let top = r.visits.iter().map(|(_, n)| *n).max().unwrap_or(0);
                            conc += top as f64 / tot as f64;
                            uniform += 1.0 / r.visits.len() as f64;
                            nconc += 1;
                        }
                        if r.legal_edges > cfg.max_edges {
                            capped += 1;
                            if r.visits.len() > cfg.max_edges {
                                widened += 1;
                            }
                        }
                    }
                    all.extend(m.node_widths());
                    let tr = tzolkin::tree::apply_step(&mut st, phase, turn, done, &r.step);
                    let committed = tr.committed();
                    match tr.next() { None => break, Some(nx) => at = nx }
                    if committed { break; }
                }
            }
            // Interior nodes as well: the roots of a chain are narrow by
            // construction and say nothing about where the cap sits.
            let mut inner: Vec<u32> = all.iter().map(|&(l, _)| l).collect();
            inner.sort_unstable();
            if !inner.is_empty() {
                let iq = |f: f64| inner[((inner.len() - 1) as f64 * f) as usize];
                let over = all.iter().filter(|&&(l, _)| l as usize > cfg.max_edges).count();
                let part = all.iter().filter(|&&(l, a)| a < l && (l as usize) <= cfg.max_edges).count();
                println!(
                    "arena nodes (interior included) n={}  legal p50 {} p90 {} p99 {} max {}  \
                     over max_edges={} on {} ({:.2}%)  still-widening below the cap {} ({:.1}%)",
                    inner.len(), iq(0.5), iq(0.9), iq(0.99), inner[inner.len() - 1],
                    cfg.max_edges, over, 100.0 * over as f64 / inner.len() as f64,
                    part, 100.0 * part as f64 / inner.len() as f64
                );
            }
            widths.sort_unstable();
            let q = |f: f64| widths[((widths.len() - 1) as f64 * f) as usize];
            println!("{n} searched nodes over {decisions} sub-decisions, {:.0} ms/decision", t.elapsed().as_secs_f64() * 1e3 / decisions as f64);
            println!("legal edges  p50 {}  p90 {}  p99 {}  max {}  mean {:.1}", q(0.5), q(0.9), q(0.99), widths[widths.len()-1], widths.iter().sum::<usize>() as f64 / n as f64);
            println!("mean edges opened {:.1} of {:.1} legal ({:.0}%)", opened as f64 / n as f64, widths.iter().sum::<usize>() as f64 / n as f64, 100.0 * opened as f64 / widths.iter().sum::<usize>() as f64);
            println!("wider than max_edges={} on {capped}/{n} ({:.1}%); widening opened past it on {widened}", cfg.max_edges, 100.0 * capped as f64 / n as f64);
            println!("top-edge visit share {:.1}% at searched roots; a uniform split would be {:.1}%", 100.0 * conc / nconc as f64, 100.0 * uniform / nconc as f64);
            println!("arena {:.0} nodes/decision; reuse kept {:.0} and threw {:.0} per decision", nodes as f64 / decisions as f64, reused as f64 / decisions as f64, discarded as f64 / decisions as f64);
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
        "trunc" => {
            // What does the §2.6 cap actually delete? The cap sorts on the
            // prior and truncates -- and under `HeuristicEvaluator` the prior
            // is uniform, so the sort is a stable no-op and the survivors are
            // whichever edges `Choice`'s derived `Ord` happened to emit first.
            // This scores every edge of every wide node one ply deep, which is
            // the best ground truth available without a net, and asks how often
            // the winner is in the part that gets kept.
            use tzolkin::mcts::Gradient;
            use tzolkin::phase::Phase;
            let max_edges: usize = argv.get(2).and_then(|v| v.parse().ok()).unwrap_or(32);
            let caps: Vec<usize> = argv
                .get(3)
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_else(|| vec![128, 512, usize::MAX]);

            // One row per (ordering, cap): how often the one-ply best edge is
            // deleted, and how much worse the best survivor is.
            struct Row { excl_cap: usize, excl_act: usize, regret_cap: f64, regret_act: f64 }
            let mut rows: Vec<(String, usize, Row)> = Vec::new();
            for ord in ["generation", "gradient"] {
                for &c in &caps {
                    rows.push((ord.into(), c, Row { excl_cap: 0, excl_act: 0, regret_cap: 0.0, regret_act: 0.0 }));
                }
            }

            let (mut wide, mut nodes, mut wide_mass, mut mass) = (0usize, 0usize, 0u64, 0u64);
            let mut widths: Vec<usize> = Vec::new();
            let (mut t_grad, mut t_oneply, mut n_edges) = (0f64, 0f64, 0u64);
            let mut worst = 0usize;

            for g in ps.iter() {
                for &p in PlayerId::ALL.iter() {
                    let workers: Vec<_> = g.on_board(p).collect();
                    for w in workers {
                        let phase = Phase::Take { worker: w };
                        let steps = tzolkin::tree::legal_steps(g, phase, p, 0);
                        if steps.len() < 2 { continue; }
                        nodes += 1;
                        mass += steps.len() as u64;
                        if steps.len() <= max_edges { continue; }
                        wide += 1;
                        wide_mass += steps.len() as u64;
                        widths.push(steps.len());
                        worst = worst.max(steps.len());

                        // Ground truth: the one-ply score of every edge.
                        let t = Instant::now();
                        let one: Vec<f32> = steps.iter().map(|st| {
                            let mut next = *g;
                            let _ = tzolkin::tree::apply_step(&mut next, phase, p, 0, st);
                            tzolkin::eval::heuristic(&next, p)
                        }).collect();
                        t_oneply += t.elapsed().as_secs_f64();

                        let t = Instant::now();
                        let grad = Gradient::new(g, p);
                        let key: Vec<f32> = steps.iter().map(|st| grad.step(st)).collect();
                        t_grad += t.elapsed().as_secs_f64();
                        n_edges += steps.len() as u64;

                        let best = one.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                        let by_gradient = {
                            let mut o: Vec<u32> = (0..steps.len() as u32).collect();
                            o.sort_unstable_by(|&a, &b| key[b as usize].total_cmp(&key[a as usize]).then(a.cmp(&b)));
                            o
                        };
                        let by_generation: Vec<u32> = (0..steps.len() as u32).collect();

                        for (ord, cap, row) in rows.iter_mut() {
                            let order = if ord == "gradient" { &by_gradient } else { &by_generation };
                            let kept = order.len().min((*cap).max(max_edges));
                            // Two windows matter and they are different
                            // questions. `cap` is what survives at all --
                            // deleted edges can never be reopened. `max_edges`
                            // is what `active` opens immediately; the rest wait
                            // on widening and are only delayed.
                            let top = |n: usize| order[..n.min(order.len())]
                                .iter().map(|&i| one[i as usize])
                                .fold(f32::NEG_INFINITY, f32::max);
                            let bc = top(kept);
                            let ba = top(max_edges);
                            if bc < best { row.excl_cap += 1; }
                            if ba < best { row.excl_act += 1; }
                            row.regret_cap += (best - bc) as f64;
                            row.regret_act += (best - ba) as f64;
                        }
                    }
                }
            }
            widths.sort_unstable();
            let q = |f: f64| if widths.is_empty() { 0 } else { widths[((widths.len()-1) as f64 * f) as usize] };
            println!("{} Take nodes, {wide} wider than max_edges={max_edges} ({:.1}%)", nodes, 100.0 * wide as f64 / nodes.max(1) as f64);
            println!("wide nodes hold {}/{} edges = {:.1}% of all edge mass; width p50 {} p90 {} max {}",
                     wide_mass, mass, 100.0 * wide_mass as f64 / mass.max(1) as f64, q(0.5), q(0.9), worst);
            println!("per edge: gradient {:.1} ns   one-ply {:.1} ns   ({:.1}x)",
                     t_grad * 1e9 / n_edges as f64, t_oneply * 1e9 / n_edges as f64, t_oneply / t_grad.max(1e-12));
            println!();
            println!("{:<11} {:>8} | {:>16} {:>10} | {:>16} {:>10}", "order", "cap", "best deleted", "regret", "best not in top32", "regret");
            for (ord, cap, r) in &rows {
                let capname = if *cap == usize::MAX { "none".to_string() } else { cap.to_string() };
                println!("{:<11} {:>8} | {:>7} ({:>5.1}%) {:>10.3} | {:>7} ({:>5.1}%) {:>10.3}",
                    ord, capname,
                    r.excl_cap, 100.0 * r.excl_cap as f64 / wide.max(1) as f64, r.regret_cap / wide.max(1) as f64,
                    r.excl_act, 100.0 * r.excl_act as f64 / wide.max(1) as f64, r.regret_act / wide.max(1) as f64);
            }
        }
        other => println!("unknown mode {other}"),
    }
}
