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
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::time::Instant;
use tzolkin::effect::{Choice, Effect};
use tzolkin::game::Game;
use tzolkin::ids::*;
use tzolkin::moves::{self, Kinds};
use tzolkin::phase::{Phase, Step};
use tzolkin::record::parse_agent;
use tzolkin::spaces::choices_at;
use tzolkin::state::GameState;
use tzolkin::tree;

/// 0 generates everything, 1 is the corn-axis dominance that predates this
/// pass, 2 adds the wealth-vector rules.
fn set_prune(level: u8) {
    tzolkin::options::PRUNE.store(level, std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct Sized_ {
    place: usize,
    retrieve: usize,
    nanos: u128,
    capped: bool,
}

fn size_of_move_space(g: &GameState, p: PlayerId, cap: usize) -> Sized_ {
    let mut place = 0usize;
    let mut retrieve = 0usize;
    let mut capped = false;
    let t = Instant::now();
    let _ = moves::visit_moves_of(g, p, Kinds::Placements, |_| {
        place += 1;
        ControlFlow::Continue(())
    });
    // Capped by move count, not by wall clock: the cut has to fall in the same
    // place in both runs however loaded the machine is.
    let _ = moves::visit_moves_of(g, p, Kinds::Retrievals, |_| {
        retrieve += 1;
        if retrieve >= cap {
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
    // Sort here rather than trusting the caller: the first version of this read
    // quantiles straight off the position-ordered vectors and printed p75 above
    // p90, which made every branching number in the A/B table meaningless.
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.sort_unstable();
    b.sort_unstable();
    let (a, b) = (&a[..], &b[..]);
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

/// Everything about a position that is *not* the acting player's liquid wealth.
///
/// Two choices whose reached states agree here differ only in what `p` is
/// holding, so componentwise-richer is the whole of the dominance test.
fn context(g: &GameState, p: PlayerId) -> impl std::hash::Hash + Eq {
    let mut h = *g;
    let pl = &mut h.players[p.idx()];
    pl.corn = 0;
    pl.res = [0; 4];
    pl.points = 0;
    h
}

/// The axes on which more is never worse: corn (only begging reads a ceiling,
/// and it hands back a temple step *down*), blocks and skulls (no upkeep, no
/// hand limit), and points.
fn wealth(g: &GameState, p: PlayerId) -> [i32; 6] {
    let pl = &g.players[p.idx()];
    [
        pl.corn as i32,
        pl.res[0] as i32,
        pl.res[1] as i32,
        pl.res[2] as i32,
        pl.res[3] as i32,
        pl.points as i32,
    ]
}

/// Where is the redundancy that survives the current pruning?
///
/// Groups each worker's option list by the *state* it reaches, then looks for
/// options another option strictly beats. Both are ground truth -- they compare
/// positions, not effect spellings -- so anything they report is a real
/// redundancy the effect-level rules are missing.
fn audit(positions: &[(GameState, PlayerId)]) {
    use std::collections::HashMap;
    let mut total = 0usize;
    let mut distinct = 0usize;
    let mut undominated = 0usize;
    // gear,pos -> (workers, options, distinct states, undominated)
    let mut by_space: HashMap<(Gear, u8), [usize; 4]> = HashMap::new();
    let mut samples: Vec<(Gear, u8, String, String)> = Vec::new();
    let mut dups: Vec<(Gear, u8, String, String)> = Vec::new();

    for (g, p) in positions {
        let p = *p;
        for w in g.on_board(p) {
            let Some((gear, pos)) = g.loc(w).on_board() else {
                continue;
            };
            let cs = moves::choices_for_worker(g, p, gear, pos);
            let mut reached: Vec<(GameState, &tzolkin::effect::Choice)> = Vec::new();
            let mut seen: HashMap<GameState, usize> = HashMap::new();
            for c in &cs {
                let mut probe = *g;
                c.apply(&mut probe, p);
                match seen.get(&probe) {
                    Some(&j) => {
                        if dups.len() < 40 {
                            dups.push((gear, pos.0, format!("{c}"), format!("{}", reached[j].1)));
                        }
                    }
                    None => {
                        seen.insert(probe, reached.len());
                        reached.push((probe, c));
                    }
                }
            }
            // Group by context, then find the Pareto front of each group.
            let mut groups: HashMap<_, Vec<usize>> = HashMap::new();
            for (i, (s, _)) in reached.iter().enumerate() {
                groups.entry(context(s, p)).or_default().push(i);
            }
            let mut live = 0usize;
            for idxs in groups.values() {
                for &i in idxs {
                    let wi = wealth(&reached[i].0, p);
                    let beaten = idxs.iter().any(|&j| {
                        j != i && {
                            let wj = wealth(&reached[j].0, p);
                            // Strictly better on some axis, no worse on any.
                            // Ties break by index so one of a pair survives.
                            (0..6).all(|k| wj[k] >= wi[k])
                                && ((0..6).any(|k| wj[k] > wi[k]) || j < i)
                        }
                    });
                    if beaten {
                        if samples.len() < 40 {
                            let j = *idxs
                                .iter()
                                .find(|&&j| {
                                    j != i && (0..6).all(|k| {
                                        wealth(&reached[j].0, p)[k] >= wi[k]
                                    })
                                })
                                .unwrap();
                            samples.push((
                                gear,
                                pos.0,
                                format!("{}", reached[i].1),
                                format!("{}", reached[j].1),
                            ));
                        }
                    } else {
                        live += 1;
                    }
                }
            }
            total += cs.len();
            distinct += reached.len();
            undominated += live;
            let e = by_space.entry((gear, pos.0)).or_insert([0; 4]);
            e[0] += 1;
            e[1] += cs.len();
            e[2] += reached.len();
            e[3] += live;
        }
    }

    println!("\n=== audit: redundancy surviving the current pruning ===");
    println!(
        "options {total}, distinct states {distinct} ({:.1}% dup), undominated {undominated} ({:.1}% of options)",
        100.0 - distinct as f64 * 100.0 / total.max(1) as f64,
        undominated as f64 * 100.0 / total.max(1) as f64
    );
    let mut rows: Vec<_> = by_space.into_iter().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v[1]));
    println!(
        "  {:<14} {:>3} {:>8} {:>10} {:>10} {:>12} {:>10}",
        "gear", "pos", "workers", "options", "distinct", "undominated", "shrink"
    );
    for ((gear, pos), v) in rows.iter().take(16) {
        println!(
            "  {:<14} {:>3} {:>8} {:>10} {:>10} {:>12} {:>9.2}x",
            gear.name(),
            pos,
            v[0],
            v[1],
            v[2],
            v[3],
            v[1] as f64 / v[3].max(1) as f64
        );
    }
    println!("\nsample exact state duplicates (two spellings, one position):");
    dups.sort();
    dups.dedup();
    for (gear, pos, a, b) in dups.iter().take(20) {
        println!("  {}:{pos}  [{a}]  ==  [{b}]", gear.name());
    }
    println!("\nsample dominated pairs (dominated <= dominator):");
    samples.sort();
    samples.dedup();
    for (gear, pos, bad, good) in samples.iter().take(30) {
        println!("  {}:{pos}  [{bad}]  <=  [{good}]", gear.name());
    }
}

/// How much redundancy is left *between whole moves* rather than within one
/// worker's option list?
///
/// Per-worker Pareto fronts do not compose: two workers each offering an
/// antichain can still combine into a dominated sum, so this is the honest
/// upper bound on what a generator that could see the whole move set would
/// remove. Nothing here is implementable inside `visit_legal_moves`, which is a
/// streaming visitor precisely so a six-figure move set never has to exist at
/// once -- this measures the ceiling, it does not propose reaching it.
fn move_audit(positions: &[(GameState, PlayerId)], cap: usize) {
    use std::collections::HashMap;
    let mut moves_total = 0usize;
    let mut front_total = 0usize;
    let mut seen = 0usize;
    for (g, p) in positions {
        let p = *p;
        let mut states: Vec<GameState> = Vec::new();
        let mut over = false;
        let _ = moves::visit_moves_of(g, p, Kinds::Retrievals, |m| {
            let mut probe = *g;
            moves::apply_move(&mut probe, p, m);
            states.push(probe);
            if states.len() > cap {
                over = true;
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        });
        if over || states.len() < 500 {
            continue;
        }
        seen += 1;
        let mut groups: HashMap<_, Vec<usize>> = HashMap::new();
        for (i, s) in states.iter().enumerate() {
            groups.entry(context(s, p)).or_default().push(i);
        }
        let mut front = 0usize;
        for idxs in groups.values() {
            for &i in idxs {
                let wi = wealth(&states[i], p);
                let beaten = idxs.iter().any(|&j| {
                    j != i && {
                        let wj = wealth(&states[j], p);
                        (0..6).all(|k| wj[k] >= wi[k]) && ((0..6).any(|k| wj[k] > wi[k]) || j < i)
                    }
                });
                if !beaten {
                    front += 1;
                }
            }
        }
        moves_total += states.len();
        front_total += front;
    }
    println!(
        "\n=== move-level ceiling over {seen} positions of 500..{cap} retrievals ===\n\
         retrievals {moves_total}, Pareto front {front_total} ({:.1}% removable if the \
         whole set could be compared at once)",
        100.0 - front_total as f64 * 100.0 / moves_total.max(1) as f64
    );
}

/// Would culling cross-worker dominated moves change what the consumer picks?
///
/// `move_audit` prices the *ceiling* on count. A count is only worth paying for
/// if the consumer notices, and the consumer is a beam that keeps 3-6 moves
/// ordered by `eval::heuristic` on the mover's own successor -- exactly what
/// `search::candidates` does. So: score every retrieval that way, then ask
/// where the dominated ones land in that order, and where the *cap* lands.
///
/// Dominance is checked only against the moves whose rank is being questioned,
/// not pairwise over the whole set, because `move_audit` already established
/// the population figure and this is O(k * group) instead of O(n^2).
fn beam_audit(positions: &[(GameState, PlayerId)], cap: usize) {
    use std::collections::HashMap;
    // `search::Config::cap_per_width = Some(25)` makes the real per-kind budget
    // `25 * width` clamped to [25, interior_cap], so a width-2 opponent node
    // sees 50 and the width-12 root node 300. 400 is the flat cap.
    const PREFIXES: [usize; 5] = [25, 50, 150, 300, 400];
    const KS: [usize; 4] = [1, 3, 6, 12];

    let mut seen = 0usize;
    let mut n_moves = 0usize;
    let mut dom_in_top = [0usize; KS.len()];
    let mut top_total = [0usize; KS.len()];
    let mut argmax_dominated = 0usize;
    let mut prefix_dom = [0usize; PREFIXES.len()];
    let mut prefix_n = [0usize; PREFIXES.len()];
    // Rank, in the full heuristic ordering, of the best move the cap can see.
    let mut prefix_best_rank: [Vec<usize>; PREFIXES.len()] = Default::default();
    let mut prefix_regret: [Vec<f64>; PREFIXES.len()] = Default::default();
    // Where the genuinely best move sits in traversal order.
    let mut best_rank: Vec<usize> = Vec::new();
    let mut spread: Vec<f64> = Vec::new();

    for (g, p) in positions {
        let p = *p;
        // (post-apply state for dominance, heuristic of the refilled successor)
        let mut rows: Vec<(GameState, f32)> = Vec::new();
        let mut over = false;
        let _ = moves::visit_moves_of(g, p, Kinds::Retrievals, |m| {
            let mut probe = *g;
            moves::apply_move(&mut probe, p, m);
            let dom = probe;
            probe.refill_buildings();
            rows.push((dom, tzolkin::eval::heuristic(&probe, p)));
            if rows.len() > cap {
                over = true;
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        });
        if over || rows.len() < 500 {
            continue;
        }
        seen += 1;
        n_moves += rows.len();

        // Context groups, so a dominance question is answered against the only
        // moves that can answer it.
        let mut groups: HashMap<_, Vec<usize>> = HashMap::new();
        for (i, (s, _)) in rows.iter().enumerate() {
            groups.entry(context(s, p)).or_default().push(i);
        }
        let members: Vec<Vec<usize>> = groups.into_values().collect();
        let mut of_group = vec![0usize; rows.len()];
        for (gi, idxs) in members.iter().enumerate() {
            for &i in idxs {
                of_group[i] = gi;
            }
        }
        let dominated = |i: usize| -> bool {
            let idxs = &members[of_group[i]];
            let wi = wealth(&rows[i].0, p);
            idxs.iter().any(|&j| {
                j != i && {
                    let wj = wealth(&rows[j].0, p);
                    (0..6).all(|k| wj[k] >= wi[k]) && ((0..6).any(|k| wj[k] > wi[k]) || j < i)
                }
            })
        };

        // Full ranking by the mover's own estimate, ties broken by traversal
        // index so the order is a function of the position and not of the sort.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by(|&a, &b| rows[b].1.total_cmp(&rows[a].1).then(a.cmp(&b)));
        let rank_of: Vec<usize> = {
            let mut v = vec![0usize; rows.len()];
            for (r, &i) in order.iter().enumerate() {
                v[i] = r;
            }
            v
        };
        let best = rows[order[0]].1 as f64;
        spread.push(best - rows[order[rows.len() / 2]].1 as f64);
        best_rank.push(order[0]);
        if dominated(order[0]) {
            argmax_dominated += 1;
        }
        for (ki, &k) in KS.iter().enumerate() {
            for &i in order.iter().take(k) {
                top_total[ki] += 1;
                if dominated(i) {
                    dom_in_top[ki] += 1;
                }
            }
        }
        for (pi, &pre) in PREFIXES.iter().enumerate() {
            let pre = pre.min(rows.len());
            let mut b = 0usize;
            for i in 0..pre {
                if rows[i].1 > rows[b].1 {
                    b = i;
                }
                prefix_n[pi] += 1;
                if dominated(i) {
                    prefix_dom[pi] += 1;
                }
            }
            prefix_best_rank[pi].push(rank_of[b]);
            prefix_regret[pi].push(best - rows[b].1 as f64);
        }
    }

    let q = |v: &[usize], t: f64| -> usize {
        let mut v = v.to_vec();
        v.sort_unstable();
        v[((v.len() as f64 - 1.0) * t) as usize]
    };
    let qf = |v: &[f64], t: f64| -> f64 {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() as f64 - 1.0) * t) as usize]
    };

    println!("\n=== beam audit over {seen} positions of 500..{cap} retrievals ({n_moves} moves) ===");
    println!("argmax by eval::heuristic is a dominated move in {argmax_dominated} of {seen} positions");
    println!("\ndominated moves inside the top K of the heuristic ordering");
    println!("  {:<6} {:>10} {:>10} {:>8}", "K", "slots", "dominated", "share");
    for (ki, &k) in KS.iter().enumerate() {
        println!(
            "  {:<6} {:>10} {:>10} {:>7.1}%",
            k,
            top_total[ki],
            dom_in_top[ki],
            dom_in_top[ki] as f64 * 100.0 / top_total[ki].max(1) as f64
        );
    }
    println!("\nwhat a capped prefix of traversal order actually costs");
    println!(
        "  {:<7} {:>9} {:>12} {:>12} {:>12} {:>10}",
        "cap", "dom %", "rank p50", "rank p90", "rank p99", "regret p90"
    );
    for (pi, &pre) in PREFIXES.iter().enumerate() {
        println!(
            "  {:<7} {:>8.1}% {:>12} {:>12} {:>12} {:>10.2}",
            pre,
            prefix_dom[pi] as f64 * 100.0 / prefix_n[pi].max(1) as f64,
            q(&prefix_best_rank[pi], 0.5),
            q(&prefix_best_rank[pi], 0.9),
            q(&prefix_best_rank[pi], 0.99),
            qf(&prefix_regret[pi], 0.9)
        );
    }
    println!(
        "\ntraversal-order index of the best move: p50 {}, p90 {}, max {}",
        q(&best_rank, 0.5),
        q(&best_rank, 0.9),
        q(&best_rank, 1.0)
    );
    println!(
        "heuristic spread best-minus-median: p50 {:.2}, p90 {:.2}",
        qf(&spread, 0.5),
        qf(&spread, 0.9)
    );
}

/// A stand-in for the `plan::Appetite` this pass wants: a linear price on the
/// resolved `Effect` vocabulary, and nothing else.
///
/// The constraint that shapes it is where it has to run -- once per `Choice`,
/// on option lists whose p99 is 157, at every node of the retrieval walk -- so
/// it may not probe the state. `Effect` is a closed 14-variant enum of
/// value-resolved data (`effect.rs`), which is exactly enough: "I am racing the
/// green temple" is a price on `TempleStep(Green, +n)`, not a state query.
#[derive(Clone, Copy)]
struct Appetite {
    line: Option<tzolkin::plan::Line>,
}

impl Appetite {
    /// The prices `eval` already implies for liquid wealth -- corn 1, wood 2,
    /// stone 3, gold 4, which is `ids.rs`'s scoring table -- plus a guess at the
    /// structural effects. `line` multiplies whatever the leading line eats.
    fn price(&self, e: &Effect) -> i32 {
        use tzolkin::plan::Line;
        let l = self.line;
        let building = matches!(l, Some(Line::Construction));
        match *e {
            Effect::Corn(n) => n as i32,
            Effect::SetCorn(n) => n as i32,
            Effect::Res(Resource::Wood, n) => n as i32 * if building { 6 } else { 2 },
            Effect::Res(Resource::Stone, n) => n as i32 * if building { 8 } else { 3 },
            Effect::Res(Resource::Gold, n) => n as i32 * if building { 10 } else { 4 },
            Effect::Res(Resource::Skull, n) => {
                n as i32 * if matches!(l, Some(Line::Skulls)) { 12 } else { 5 }
            }
            Effect::Points(n) => n as i32 * 4,
            // `plan::Line::Temples` does not say *which* temple, so this can
            // only price every climb alike -- see the interface request in the
            // report. A line that named the track would price the other two at
            // 4 and this one at 14.
            Effect::TempleStep(_, n) => n as i32 * if l == Some(Line::Temples) { 14 } else { 4 },
            Effect::AdvanceResearch(s) => {
                if matches!(l, Some(Line::Construction)) && s == Science::Architecture {
                    12
                } else {
                    5
                }
            }
            Effect::UnlockWorker => 8,
            Effect::FreeWorker(n) => n as i32 * 6,
            Effect::WorkerDiscount(n) => n as i32 * 6,
            Effect::TakePalenqueTile(..) => 2,
            Effect::BurnPalenqueWood(..) => -2,
            Effect::FillChichen(..) => 0,
            Effect::Build(..) => 8,
            Effect::TakeMonument(..) => 20,
        }
    }
}

impl Appetite {
    /// The whole choice, priced. Public so the prior audit can call it without
    /// going through the `Priority` seam, which the factored search does not use.
    fn raw(&self, c: &Choice) -> i32 {
        c.0.iter().map(|e| self.price(e)).sum()
    }
}

impl moves::Priority for Appetite {
    fn choice(&self, _gear: Gear, _pos: Pos, c: &tzolkin::effect::Choice) -> i32 {
        self.raw(c)
    }
    /// A worker high on a gear has the whole gear below it, so it is where the
    /// interesting choices are; the plan does not need to say so.
    fn worker(&self, _gear: Gear, pos: Pos) -> i32 {
        pos.0 as i32
    }
}

/// Worker order alone, with no opinion about choices.
///
/// Separates the two halves of the hint. A worker high on a gear has every
/// space below it available for a corn a step, so it is where the interesting
/// choices are and where `retrieve_rec` should spend the shallow part of the
/// walk -- and that needs no plan, only `pos`. Everything the priced orders
/// below win *beyond* this row is what pricing the choices is worth.
struct ByPos;

impl moves::Priority for ByPos {
    fn worker(&self, _gear: Gear, pos: Pos) -> i32 {
        pos.0 as i32
    }
}

/// The price this pass actually proposes: `eval`'s own local gradient.
///
/// A hand-written table of effect prices is a second opinion about the value
/// function, and the `flat`/`plan` rows below show what that costs at the tail.
/// This asks `eval` instead: probe `+1` of each axis once per turn -- fifteen
/// scalar probes plus one per distinct card id -- and price a `Choice` as the
/// dot product of its resolved effects with that gradient. Linear, so it misses
/// every interaction, which is fine for a hint and is why it is a hint.
///
/// Cost is ~30 `heuristic` calls for a whole turn, against the ~200 the node
/// already spends scoring candidates, and it is paid once rather than per node
/// of the retrieval walk.
struct Gradient {
    g: GameState,
    p: PlayerId,
    base: f32,
    /// The proposed interface, filled from probes: see `options::EffectPrice`.
    price: tzolkin::options::EffectPrice,
    /// Effects that name a card, tile or space: too many to probe eagerly, few
    /// enough per position to cache.
    cached: std::cell::RefCell<std::collections::HashMap<Effect, f32>>,
    /// SCRATCH ablation: when false the named effects take `EffectPrice`'s flat
    /// constants instead of a probe, which is what an implementation that
    /// refused to touch the state at all would have to do.
    probe_named: bool,
}

impl Gradient {
    fn probe(g: &GameState, p: PlayerId, base: f32, e: Effect) -> f32 {
        let mut probe = *g;
        Choice::one(e).apply(&mut probe, p);
        tzolkin::eval::heuristic(&probe, p) - base
    }

    fn new(g: &GameState, p: PlayerId) -> Gradient {
        let base = tzolkin::eval::heuristic(g, p);
        let pr = |e| Gradient::probe(g, p, base, e);
        let points = pr(Effect::Points(1));
        // The card-naming constants come from the flat table, scaled into
        // `heuristic` points by the point price so the two halves of the sum
        // are commensurate. Probing them per card is worth 0.005 points of
        // regret in the first 32 edges, which is why they are constants.
        let flat = Appetite { line: None };
        let k = |e| flat.price(&e) as f32 * points / 4.0;
        Gradient {
            g: *g,
            p,
            base,
            price: tzolkin::options::EffectPrice {
                corn: pr(Effect::Corn(1)),
                res: std::array::from_fn(|i| pr(Effect::Res(Resource::ALL[i], 1))),
                points,
                temple: std::array::from_fn(|i| pr(Effect::TempleStep(Temple::ALL[i], 1))),
                science: std::array::from_fn(|i| pr(Effect::AdvanceResearch(Science::ALL[i]))),
                unlock_worker: pr(Effect::UnlockWorker),
                free_worker: pr(Effect::FreeWorker(1)),
                worker_discount: pr(Effect::WorkerDiscount(1)),
                palenque_tile: k(Effect::TakePalenqueTile(Pos(0), TileKind::Corn)),
                burn_wood: k(Effect::BurnPalenqueWood(Pos(0))),
                fill_chichen: k(Effect::FillChichen(Pos(0))),
                build: k(Effect::Build(BuildingId(0))),
                monument: k(Effect::TakeMonument(MonumentId(0))),
            },
            cached: Default::default(),
            probe_named: true,
        }
    }

    fn without_card_probes(mut self) -> Gradient {
        self.probe_named = false;
        self
    }

    fn cached(&self, e: Effect) -> f32 {
        if let Some(&v) = self.cached.borrow().get(&e) {
            return v;
        }
        let v = Gradient::probe(&self.g, self.p, self.base, e);
        self.cached.borrow_mut().insert(e, v);
        v
    }

    /// A choice priced as the dot product of its resolved effects with the
    /// gradient. No state copy, no `heuristic` call: `Choice` is already the
    /// summary, which is the whole point of `effect.rs` resolving values at
    /// generation time.
    fn score(&self, c: &Choice) -> f32 {
        if !self.probe_named {
            return self.price.choice(c);
        }
        c.0.iter()
            .map(|&e| match e {
                Effect::TakePalenqueTile(..)
                | Effect::BurnPalenqueWood(_)
                | Effect::FillChichen(_)
                | Effect::Build(_)
                | Effect::TakeMonument(_) => self.cached(e),
                other => self.price.of(other),
            })
            .sum()
    }

    /// Only the axes `options::dominated_dedup` already sums: corn, the three
    /// blocks, and points. Everything structural -- a building, a temple step,
    /// a research advance -- prices at zero.
    fn wealth_only(&self, c: &Choice) -> f32 {
        let mut v = 0.0;
        for e in &c.0 {
            v += match *e {
                Effect::Corn(n) => n as f32 * self.price.corn,
                Effect::Res(r, n) if r != Resource::Skull => {
                    n as f32 * self.price.res[r.idx()]
                }
                Effect::Points(n) => n as f32 * self.price.points,
                _ => 0.0,
            };
        }
        v
    }
}

impl moves::Priority for Gradient {
    fn choice(&self, _gear: Gear, _pos: Pos, c: &tzolkin::effect::Choice) -> i32 {
        (self.score(c) * 256.0) as i32
    }
    fn worker(&self, _gear: Gear, pos: Pos) -> i32 {
        pos.0 as i32
    }
}

/// The ceiling for a per-choice key: price each choice by what one call to
/// `eval::heuristic` says it is worth on its own.
///
/// This is not implementable -- it probes the state once per distinct choice --
/// but it answers the design question the hand-written price cannot. If this
/// orders well, the interface shape (a scalar key per `Choice`, evaluated
/// without reference to the prefix) is right and only the numbers are wrong. If
/// it does not, no `Priority` implementation can be made to work and the
/// factoring is what needs changing.
struct Oracle {
    g: GameState,
    p: PlayerId,
    base: f32,
    seen: std::cell::RefCell<std::collections::HashMap<(u8, u8, tzolkin::effect::Choice), i32>>,
}

impl moves::Priority for Oracle {
    fn choice(&self, gear: Gear, pos: Pos, c: &tzolkin::effect::Choice) -> i32 {
        let key = (gear as u8, pos.0, c.clone());
        if let Some(&v) = self.seen.borrow().get(&key) {
            return v;
        }
        let mut probe = self.g;
        c.apply(&mut probe, self.p);
        // Scaled so the key stays an integer without collapsing the ordering.
        let v = ((tzolkin::eval::heuristic(&probe, self.p) - self.base) * 256.0) as i32;
        self.seen.borrow_mut().insert(key, v);
        v
    }
    fn worker(&self, _gear: Gear, pos: Pos) -> i32 {
        pos.0 as i32
    }
}

/// The leading line, the way `plan::focus` takes it.
fn leading_line(g: &GameState, p: PlayerId) -> Option<tzolkin::plan::Line> {
    tzolkin::plan::Line::ALL
        .iter()
        .map(|&l| (l, tzolkin::plan::line_progress(g, p, l)))
        .fold(None, |best: Option<(tzolkin::plan::Line, f32)>, x| match best {
            Some(b) if b.1 >= x.1 => Some(b),
            _ => Some(x),
        })
        .filter(|&(_, v)| v > 0.0)
        .map(|(l, _)| l)
}

/// What ordering costs the caller that does *not* stop early, and whether it
/// changes what that caller gets.
///
/// `NoOrder::ORDERS = false` puts every sort behind `if false`, so the first
/// column is the generator exactly as it was. The counts must agree: a walk
/// emits one move per state it reaches for the first time, and a depth-first
/// walk with a visited set reaches the same states whatever order the edges
/// come in, so a mismatch would mean the hint had changed the move space rather
/// than its order.
fn order_cost(positions: &[(GameState, PlayerId)], cap: usize) {
    let mut t = [0u128; 2];
    let mut n = 0usize;
    let mut total = 0usize;
    let mut mismatched = 0usize;
    for (g, p) in positions {
        let p = *p;
        let grad = Gradient::new(g, p);
        let mut c = [0usize; 2];
        let mut ns = [0u128; 2];
        let mut over = false;
        for which in 0..2 {
            let mut k = 0usize;
            let s = Instant::now();
            let mut count = |_: &moves::Move| {
                k += 1;
                if k > cap {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            let flow = if which == 0 {
                moves::visit_moves_of(g, p, Kinds::Retrievals, &mut count)
            } else {
                moves::visit_moves_of_by(g, p, Kinds::Retrievals, &grad, &mut count)
            };
            ns[which] = s.elapsed().as_nanos();
            over |= flow.is_break();
            c[which] = k;
        }
        if over || c[0] < 500 {
            continue;
        }
        if c[0] != c[1] {
            mismatched += 1;
        }
        n += 1;
        total += c[0];
        t[0] += ns[0];
        t[1] += ns[1];
    }
    println!(
        "\n=== cost of ordering an uncapped walk, {n} positions / {total} retrievals ===\n\
         NoOrder {:.2} s, gradient-ordered {:.2} s ({:.2}x); \
         move counts differed in {mismatched} positions",
        t[0] as f64 / 1e9,
        t[1] as f64 / 1e9,
        t[1] as f64 / t[0].max(1) as f64
    );
}

/// The best move a capped walk finds, under each traversal order.
///
/// This runs the real walk through `visit_moves_of_by`, so what it measures is
/// what a capped consumer would actually get -- not top-N under a score, which
/// is a ceiling a depth-first walk does not reach. Regret is in `eval::heuristic`
/// units against the best move in the whole retrieval set, which is what
/// `search::candidates` would rank first if it could afford to see everything.
fn order_audit(positions: &[(GameState, PlayerId)], cap: usize) {
    use moves::Priority;
    const PREFIXES: [usize; 4] = [25, 50, 150, 400];
    const NAMES: [&str; 6] =
        ["traversal", "worker only", "flat price", "plan price", "gradient", "oracle"];
    let mut seen = 0usize;
    let mut lined = 0usize;
    let mut regret: [[Vec<f64>; PREFIXES.len()]; NAMES.len()] = Default::default();
    let mut nanos = [0u128; NAMES.len()];
    let mut grad_nanos = 0u128;

    for (gs, p) in positions {
        let p = *p;
        // The whole retrieval set, to know what the cap is missing.
        let mut best = f32::MIN;
        let mut total = 0usize;
        let mut over = false;
        let _ = moves::visit_moves_of(gs, p, Kinds::Retrievals, |m| {
            let mut probe = *gs;
            moves::apply_move(&mut probe, p, m);
            probe.refill_buildings();
            best = best.max(tzolkin::eval::heuristic(&probe, p));
            total += 1;
            if total > cap {
                over = true;
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        });
        if over || total < 500 {
            continue;
        }
        seen += 1;
        let line = leading_line(gs, p);
        if line.is_some() {
            lined += 1;
        }

        let flat = Appetite { line: None };
        let plan = Appetite { line };
        let t = Instant::now();
        let gradient = Gradient::new(gs, p);
        grad_nanos += t.elapsed().as_nanos();
        let oracle = Oracle {
            g: *gs,
            p,
            base: tzolkin::eval::heuristic(gs, p),
            seen: Default::default(),
        };

        // One capped walk per order; the shorter prefixes are prefixes of it.
        let mut run = |which: usize, run_walk: &dyn Fn(&mut dyn FnMut(&moves::Move) -> ControlFlow<()>)| {
            let mut marks = [f32::MIN; PREFIXES.len()];
            let mut n = 0usize;
            let mut run_best = f32::MIN;
            let t = Instant::now();
            run_walk(&mut |m: &moves::Move| {
                let mut probe = *gs;
                moves::apply_move(&mut probe, p, m);
                probe.refill_buildings();
                run_best = run_best.max(tzolkin::eval::heuristic(&probe, p));
                n += 1;
                for (pi, &pre) in PREFIXES.iter().enumerate() {
                    if n == pre {
                        marks[pi] = run_best;
                    }
                }
                if n >= PREFIXES[PREFIXES.len() - 1] {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            nanos[which] += t.elapsed().as_nanos();
            for (pi, m) in marks.iter().enumerate() {
                let m = if *m == f32::MIN { run_best } else { *m };
                regret[which][pi].push((best - m) as f64);
            }
        };

        run(0, &|f| {
            let _ = moves::visit_moves_of(gs, p, Kinds::Retrievals, f);
        });
        run(1, &|f| {
            let _ = moves::visit_moves_of_by(gs, p, Kinds::Retrievals, &ByPos, f);
        });
        run(2, &|f| {
            let _ = moves::visit_moves_of_by(gs, p, Kinds::Retrievals, &flat, f);
        });
        run(3, &|f| {
            let _ = moves::visit_moves_of_by(gs, p, Kinds::Retrievals, &plan, f);
        });
        run(4, &|f| {
            let _ = moves::visit_moves_of_by(gs, p, Kinds::Retrievals, &gradient, f);
        });
        run(5, &|f| {
            let _ = moves::visit_moves_of_by(gs, p, Kinds::Retrievals, &oracle, f);
        });
        let _ = oracle.choice(Gear::Tikal, Pos(0), &tzolkin::effect::Choice::skip());
    }

    let qf = |v: &[f64], t: f64| -> f64 {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() as f64 - 1.0) * t) as usize]
    };
    println!(
        "\n=== capped-walk regret over {seen} positions ({lined} with a leading plan line) ===\n\
         regret = eval::heuristic of the best move the cap saw, below the best in the whole set"
    );
    println!(
        "  {:<11} {:>6} {:>9} {:>9} {:>9} {:>9}",
        "order", "cap", "mean", "p50", "p90", "p99"
    );
    for (wi, name) in NAMES.iter().enumerate() {
        for (pi, &pre) in PREFIXES.iter().enumerate() {
            let v = &regret[wi][pi];
            println!(
                "  {:<11} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
                if pi == 0 { *name } else { "" },
                pre,
                v.iter().sum::<f64>() / v.len() as f64,
                qf(v, 0.5),
                qf(v, 0.9),
                qf(v, 0.99),
            );
        }
    }
    println!(
        "\nbuilding the gradient cost {:.3} s over {seen} positions ({:.0} us each)",
        grad_nanos as f64 / 1e9,
        grad_nanos as f64 / 1e3 / seen.max(1) as f64
    );
    println!("wall clock for the capped walks (400 moves each), total over {seen} positions:");
    for (wi, name) in NAMES.iter().enumerate() {
        println!("  {:<11} {:>8.3} s", name, nanos[wi] as f64 / 1e9);
    }
}

// ---- what MCTS actually pays: per-node edge counts ----------------------

/// One `Take` node as the factored search meets it.
///
/// `done > 0` nodes are the ones the flat move list never showed: the second
/// and third worker of a retrieval sit behind a `PickWorker` that a whole-turn
/// enumeration folds away. MCTS pays for them one at a time.
struct TakeNode {
    state: GameState,
    turn: PlayerId,
    worker: WorkerId,
    done: u8,
    gear: Gear,
    pos: u8,
}

/// Walk the turn chain the way a simulation does, collecting every `Take` node
/// the descent stands on.
///
/// `Mode` is forced to `Retrieve` when retrieval is legal. A uniform descent
/// picks placement half the time and placement has no `Take` node at all, so an
/// unbiased walk would spend most of its samples measuring nodes of width <= 7
/// -- and the whole question here is the width of the wide ones.
fn take_nodes(g: &GameState, p: PlayerId, rng: &mut StdRng, descents: usize, out: &mut Vec<TakeNode>) {
    for _ in 0..descents {
        let mut s = *g;
        let mut phase = Phase::Beg;
        let mut done = 0u8;
        for _ in 0..64 {
            let steps = tree::legal_steps(&s, phase, p, done);
            if steps.is_empty() {
                break;
            }
            if let Phase::Take { worker } = phase {
                if let Some((gear, pos)) = s.loc(worker).on_board() {
                    out.push(TakeNode {
                        state: s,
                        turn: p,
                        worker,
                        done,
                        gear,
                        pos: pos.0,
                    });
                }
            }
            let pick = match phase {
                Phase::Mode => steps
                    .iter()
                    .position(|st| matches!(st, Step::Mode(tzolkin::phase::ModeChoice::Retrieve)))
                    .unwrap_or_else(|| rng.gen_range(0..steps.len())),
                _ => rng.gen_range(0..steps.len()),
            };
            match tree::step_within_turn(&mut s, phase, p, done, &steps[pick]) {
                Some((ph, d)) => {
                    phase = ph;
                    done = d;
                }
                None => break,
            }
        }
    }
}

fn qs(v: &[usize], t: f64) -> usize {
    let mut v = v.to_vec();
    v.sort_unstable();
    v[((v.len() as f64 - 1.0) * t) as usize]
}

fn qsf(v: &[f64], t: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 - 1.0) * t) as usize]
}

/// Per-node edge counts across the whole turn chain, by phase and by space.
///
/// This is the number MCTS pays, and it is not the number `legal_moves`
/// reports: a turn is ~8 nodes and the search never multiplies them out. Run at
/// two pruning levels on *identical* positions, so the rules are priced against
/// each other and not against a drifting position set.
fn node_audit(positions: &[(GameState, PlayerId)], descents: usize) {
    use std::collections::HashMap;
    // [level][phase tag] -> widths
    let mut by_phase: [Vec<Vec<usize>>; 2] = Default::default();
    for v in by_phase.iter_mut() {
        v.resize(Phase::COUNT, Vec::new());
    }
    // [level] -> (gear, pos) -> (nodes, sum width, max width)
    let mut by_space: [HashMap<(Gear, u8), (usize, usize, usize)>; 2] = Default::default();
    let mut takes: [Vec<usize>; 2] = Default::default();
    let mut over_cap: [usize; 2] = [0; 2];
    let mut over_k: [usize; 2] = [0; 2];

    for (i, &(ref g, p)) in positions.iter().enumerate() {
        for (li, lvl) in [0u8, 2].into_iter().enumerate() {
            set_prune(lvl);
            // The same seed at both levels, so the two descents follow the same
            // decisions wherever the rules leave them available.
            let mut rng = StdRng::seed_from_u64(i as u64 ^ 0xf00d);
            let mut nodes = Vec::new();
            take_nodes(g, p, &mut rng, descents, &mut nodes);
            // Every node of the chain, not just `Take`: the report needs to say
            // what fraction of the search's nodes are the wide ones.
            let mut rng2 = StdRng::seed_from_u64(i as u64 ^ 0xf00d);
            let mut s = *g;
            let mut phase = Phase::Beg;
            let mut done = 0u8;
            for _ in 0..64 {
                let steps = tree::legal_steps(&s, phase, p, done);
                if steps.is_empty() {
                    break;
                }
                by_phase[li][phase.tag() as usize].push(steps.len());
                let pick = match phase {
                    Phase::Mode => steps
                        .iter()
                        .position(|st| {
                            matches!(st, Step::Mode(tzolkin::phase::ModeChoice::Retrieve))
                        })
                        .unwrap_or_else(|| rng2.gen_range(0..steps.len())),
                    _ => rng2.gen_range(0..steps.len()),
                };
                match tree::step_within_turn(&mut s, phase, p, done, &steps[pick]) {
                    Some((ph, d)) => {
                        phase = ph;
                        done = d;
                    }
                    None => break,
                }
            }

            for n in &nodes {
                let w = tree::take_candidates(&n.state, n.turn, n.worker).len();
                takes[li].push(w);
                if w > 128 {
                    over_cap[li] += 1;
                }
                if w > 32 {
                    over_k[li] += 1;
                }
                let e = by_space[li].entry((n.gear, n.pos)).or_insert((0, 0, 0));
                e.0 += 1;
                e.1 += w;
                e.2 = e.2.max(w);
            }
        }
    }
    set_prune(2);

    println!(
        "\n=== per-node edge counts over {} positions x {descents} descents ===",
        positions.len()
    );
    println!("(what PUCT pays; the whole-turn move count is never materialised)");
    const PHASES: [&str; 8] = [
        "Beg", "Mode", "Placing", "PickWorker", "Take", "ExtraDay", "PityPlace", "DraftTile",
    ];
    println!(
        "\n  {:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "phase", "nodes", "mean", "p50", "p90", "p99", "max"
    );
    for t in 0..Phase::COUNT {
        let v = &by_phase[1][t];
        if v.is_empty() {
            continue;
        }
        println!(
            "  {:<12} {:>8} {:>8.1} {:>8} {:>8} {:>8} {:>8}",
            PHASES[t],
            v.len(),
            v.iter().sum::<usize>() as f64 / v.len() as f64,
            qs(v, 0.5),
            qs(v, 0.9),
            qs(v, 0.99),
            qs(v, 1.0)
        );
    }

    println!("\nTake-node width, pruning level 0 vs 2, identical descents");
    println!(
        "  {:<10} {:>12} {:>12} {:>9}",
        "", "level 0", "level 2", "ratio"
    );
    for q in [0.5, 0.75, 0.9, 0.99, 1.0] {
        let (a, b) = (qs(&takes[0], q), qs(&takes[1], q));
        println!(
            "  p{:<9} {:>12} {:>12} {:>8.2}x",
            format!("{:.1}", q * 100.0),
            a,
            b,
            a as f64 / b.max(1) as f64
        );
    }
    let m: [f64; 2] = std::array::from_fn(|k| {
        takes[k].iter().sum::<usize>() as f64 / takes[k].len().max(1) as f64
    });
    println!(
        "  {:<10} {:>12.1} {:>12.1} {:>8.2}x",
        "mean", m[0], m[1], m[0] / m[1].max(1e-9)
    );
    println!(
        "  {:<10} {:>11.1}% {:>11.1}%   nodes above max_edges=32",
        "> K",
        over_k[0] as f64 * 100.0 / takes[0].len().max(1) as f64,
        over_k[1] as f64 * 100.0 / takes[1].len().max(1) as f64
    );
    println!(
        "  {:<10} {:>11.2}% {:>11.2}%   nodes above widen_cap=128 (edges dropped outright)",
        "> cap",
        over_cap[0] as f64 * 100.0 / takes[0].len().max(1) as f64,
        over_cap[1] as f64 * 100.0 / takes[1].len().max(1) as f64
    );

    let mut rows: Vec<_> = by_space[1].iter().map(|(k, v)| (*k, *v)).collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.1));
    let grand: usize = rows.iter().map(|(_, v)| v.1).sum();
    println!("\nwhere the Take-node width lives (level 2)");
    println!(
        "  {:<12} {:>3} {:>8} {:>10} {:>8} {:>8} {:>9} {:>9}",
        "gear", "pos", "nodes", "sum edges", "share", "mean", "max", "lvl0 mean"
    );
    for ((gear, pos), v) in rows.iter().take(14) {
        let z = by_space[0].get(&(*gear, *pos)).copied().unwrap_or((0, 0, 0));
        println!(
            "  {:<12} {:>3} {:>8} {:>10} {:>7.1}% {:>8.1} {:>9} {:>9.1}",
            gear.name(),
            pos,
            v.0,
            v.1,
            v.1 as f64 * 100.0 / grand.max(1) as f64,
            v.1 as f64 / v.0.max(1) as f64,
            v.2,
            z.1 as f64 / z.0.max(1) as f64
        );
    }
    let tikal: usize = rows
        .iter()
        .filter(|((g, _), _)| *g == Gear::Tikal)
        .map(|(_, v)| v.1)
        .sum();
    println!(
        "  Tikal is {:.1}% of all Take-node edges ({tikal} of {grand})",
        tikal as f64 * 100.0 / grand.max(1) as f64
    );

    // What a per-sub-space factoring would buy, without building it.
    //
    // A `Take` node's edge set is a union the generator builds in two nested
    // layers: `choices_for_worker` unions the spaces the worker may pay to step
    // down to, and a free-choice space's own arm unions every space below it for
    // nothing. Either layer is a decision a player makes out loud -- "I'll take
    // the Tikal-4 action" comes before "with which two cards" -- so promoting it
    // to a node of its own is a refactor of the chain, not a change to the legal
    // set. This is that split's arithmetic: outer = one edge per space with
    // anything to offer, inner = the choices at one space.
    let mut before: Vec<usize> = Vec::new();
    let mut after_outer: Vec<usize> = Vec::new();
    let mut after_inner: Vec<usize> = Vec::new();
    // Every worker on the board once, rather than once per descent: this is the
    // population the earlier per-worker figures were quoted over, and it is not
    // the same population as the descent sample, which revisits the wide nodes.
    let mut flat: [Vec<usize>; 2] = Default::default();
    for (li, lvl) in [0u8, 2].into_iter().enumerate() {
        set_prune(lvl);
        for &(ref g, p) in positions {
            for w in g.on_board(p) {
                if let Some((gear, pos)) = g.loc(w).on_board() {
                    flat[li].push(moves::choices_for_worker(g, p, gear, pos).len());
                }
            }
        }
    }
    println!(
        "\noptions per worker on board ({} workers, one sample each)",
        flat[0].len()
    );
    println!("  {:<10} {:>12} {:>12} {:>9}", "", "level 0", "level 2", "ratio");
    for q in [0.5, 0.9, 0.99, 1.0] {
        let (a, b) = (qs(&flat[0], q), qs(&flat[1], q));
        println!(
            "  p{:<9} {:>12} {:>12} {:>8.2}x",
            format!("{:.0}", q * 100.0),
            a,
            b,
            a as f64 / b.max(1) as f64
        );
    }
    let fm: [f64; 2] = std::array::from_fn(|k| {
        flat[k].iter().sum::<usize>() as f64 / flat[k].len().max(1) as f64
    });
    println!(
        "  {:<10} {:>12.1} {:>12.1} {:>8.2}x",
        "mean", fm[0], fm[1], fm[0] / fm[1].max(1e-9)
    );

    set_prune(2);
    for &(ref g, p) in positions {
        for w in g.on_board(p) {
            let Some((gear, pos)) = g.loc(w).on_board() else {
                continue;
            };
            let whole = moves::choices_for_worker(g, p, gear, pos).len();
            if whole < 8 {
                continue;
            }
            before.push(whole);
            // A free-choice space repeats spaces `0..k` for nothing, so its
            // partition is over those and not over a step-down fee it never
            // charges; `is_free_choice` and the catch-all arms agree on `k`.
            let k = if gear == Gear::Chichen { 10 } else { 6 };
            let mut parts = 0usize;
            if tzolkin::spaces::is_free_choice(gear, pos) {
                for i in 0..k {
                    let n = tzolkin::options::dominated_dedup(choices_at(g, p, gear, Pos(i))).len();
                    if n > 0 {
                        parts += 1;
                        after_inner.push(n);
                    }
                }
            } else {
                for j in 0..=pos.0 {
                    let mut probe = *g;
                    let fee = pos.0 - j;
                    if fee > probe.players[p.idx()].corn {
                        continue;
                    }
                    probe.players[p.idx()].corn -= fee;
                    let n =
                        tzolkin::options::dominated_dedup(choices_at(&probe, p, gear, Pos(j))).len();
                    if n > 0 {
                        parts += 1;
                        after_inner.push(n);
                    }
                }
            }
            after_outer.push(parts.max(1));
        }
    }
    // One level down again: Tikal 4's double build is a single space whose own
    // option list is quadratic, so splitting the node by space leaves it whole.
    // These are the two halves it would split into.
    let mut db_before: Vec<usize> = Vec::new();
    let mut db_first: Vec<usize> = Vec::new();
    let mut db_second: Vec<usize> = Vec::new();
    for &(ref g, p) in positions {
        let firsts = tzolkin::options::building_choices(g, p, None, true, 1);
        if firsts.len() < 4 {
            continue;
        }
        let mut whole = firsts.len();
        db_first.push(firsts.len());
        for first in &firsts {
            let built = first.0.iter().find_map(|e| match e {
                Effect::Build(id) => Some(*id),
                _ => None,
            });
            let mut probe = *g;
            first.apply(&mut probe, p);
            let n = tzolkin::options::building_choices(&probe, p, built, false, 0).len();
            db_second.push(n);
            whole += n;
        }
        db_before.push(whole);
    }
    if !db_before.is_empty() {
        println!(
            "\nand one level below that: Tikal 4's double build, over {} positions\n  \
             {:<16} {:>8} {:>8} {:>8} {:>8}",
            db_before.len(),
            "",
            "mean",
            "p50",
            "p90",
            "max"
        );
        for (name, v) in [
            ("one node now", &db_before),
            ("first card", &db_first),
            ("second card", &db_second),
        ] {
            println!(
                "  {:<16} {:>8.1} {:>8} {:>8} {:>8}",
                name,
                v.iter().sum::<usize>() as f64 / v.len() as f64,
                qs(v, 0.5),
                qs(v, 0.9),
                qs(v, 1.0)
            );
        }
    }

    println!(
        "\nif a `Take` node were split by sub-space (needs a Phase variant; see report)\n  \
         nodes >= 8 edges: {}\n  \
         {:<14} {:>8} {:>8} {:>8} {:>8} {:>8}",
        before.len(),
        "",
        "mean",
        "p50",
        "p90",
        "p99",
        "max"
    );
    for (name, v) in [
        ("one node now", &before),
        ("outer (space)", &after_outer),
        ("inner (choice)", &after_inner),
    ] {
        if v.is_empty() {
            continue;
        }
        println!(
            "  {:<14} {:>8.1} {:>8} {:>8} {:>8} {:>8}",
            name,
            v.iter().sum::<usize>() as f64 / v.len() as f64,
            qs(v, 0.5),
            qs(v, 0.9),
            qs(v, 0.99),
            qs(v, 1.0)
        );
    }
}

/// Can a prior be built without applying the choice?
///
/// `Priors::OnePly` costs one `apply_step` and one `eval::heuristic` per edge.
/// At a 124-edge Tikal node that is the expansion. The alternative on offer is
/// that a `Choice` is *already* a resolved effect vector, so a linear price on
/// its effects is a prior that costs no state copy at all -- the question is
/// only whether it orders like the thing it replaces, because at a node wider
/// than `max_edges` the prior order decides which edges the search ever sees.
fn prior_audit(positions: &[(GameState, PlayerId)], descents: usize) {
    const NAMES: [&str; 6] = [
        "generation",
        "wealth only",
        "flat price",
        "grad@turn",
        "grad-nocards",
        "grad@node",
    ];
    const KS: [usize; 3] = [8, 32, 128];
    set_prune(2);

    let mut n_nodes = 0usize;
    let mut widths: Vec<usize> = Vec::new();
    // [order][K] -> regret in heuristic points; and top-K recall against oneply
    let mut regret: [[Vec<f64>; KS.len()]; NAMES.len()] = Default::default();
    let mut recall: [[Vec<f64>; KS.len()]; NAMES.len()] = Default::default();
    let mut argmax_rank: [Vec<usize>; NAMES.len()] = Default::default();
    const REPEATS: usize = 3;
    let mut nanos = [0u128; NAMES.len() + 2];
    // Split at `widen_cap`: a narrow node's two-stage cost is one-ply's plus the
    // price, because there is nothing to cut. The saving lives entirely in the
    // wide band, which is also where the expansion cost lives.
    let mut wide_nanos: [[u128; NAMES.len() + 2]; 2] = Default::default();
    let mut wide_edges = [0usize; 2];
    let mut wide_nodes = [0usize; 2];
    // Two-stage: the cheap price picks `widen_cap` survivors, one `apply` and
    // one `heuristic` reorder those. Timed for real rather than extrapolated.
    let mut two_nanos = 0u128;
    let mut edges_scored = 0usize;
    let mut grad_build_nanos = 0u128;
    let mut grad_builds = 0usize;
    // Edges lost outright to the 128 truncation, and whether the best went.
    let mut trunc_nodes = 0usize;
    let mut trunc_lost_best: [usize; NAMES.len()] = [0; NAMES.len()];

    for (i, &(ref g, p)) in positions.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(i as u64 ^ 0xbeef);
        let mut nodes = Vec::new();
        take_nodes(g, p, &mut rng, descents, &mut nodes);
        // The affordable gradient: probed once at the turn's root, then reused
        // at every `Take` node of that turn. `heuristic` is not linear, so this
        // is the approximation `grad@node` bounds.
        // Min of the same repeats the per-node costs use: this is one
        // single-shot measurement per position and the machine is shared.
        let mut best_build = u128::MAX;
        for _ in 0..REPEATS {
            let t = Instant::now();
            std::hint::black_box(Gradient::new(g, p));
            best_build = best_build.min(t.elapsed().as_nanos());
        }
        grad_build_nanos += best_build;
        grad_builds += 1;
        let turn_grad = Gradient::new(g, p);
        let turn_nc = Gradient::new(g, p).without_card_probes();

        for node in &nodes {
            let cs = tree::take_candidates(&node.state, node.turn, node.worker);
            if cs.len() < 8 {
                continue;
            }
            n_nodes += 1;
            widths.push(cs.len());
            edges_scored += cs.len();

            // The reference: exactly what `mcts::one_ply` computes.
            let one_ply_all = |out: &mut Vec<f32>| {
                out.clear();
                for c in &cs {
                    let mut next = node.state;
                    let _ = tree::apply_step(
                        &mut next,
                        Phase::Take {
                            worker: node.worker,
                        },
                        node.turn,
                        node.done,
                        &Step::Take(c.clone()),
                    );
                    out.push(tzolkin::eval::heuristic(&next, node.turn));
                }
            };
            let mut truth: Vec<f32> = Vec::with_capacity(cs.len());
            let flat = Appetite { line: None };
            let mut s_wealth: Vec<f32> = Vec::new();
            let mut s_flat: Vec<f32> = Vec::new();
            let mut s_turn: Vec<f32> = Vec::new();
            let mut s_nc: Vec<f32> = Vec::new();
            let mut s_node: Vec<f32> = Vec::new();

            // Interleaved repeats, minimum taken. Four other processes are on
            // this machine and a single pass drifted 40% between runs; the
            // minimum of interleaved repeats is the one number contention can
            // only inflate, never deflate.
            let mut rep = [u128::MAX; NAMES.len() + 2];
            for _ in 0..REPEATS {
                let t = Instant::now();
                one_ply_all(&mut truth);
                rep[NAMES.len()] = rep[NAMES.len()].min(t.elapsed().as_nanos());

                // The five-axis net `dominated_dedup` already computes for
                // every choice, priced by the same gradient. If this ordered as
                // well as the full effect walk, generation could hand the prior
                // a vector it has in hand and the walk would be pure waste.
                let t = Instant::now();
                s_wealth = cs
                    .iter()
                    .map(|c| turn_grad.wealth_only(c))
                    .collect();
                rep[1] = rep[1].min(t.elapsed().as_nanos());

                let t = Instant::now();
                s_flat = cs.iter().map(|c| flat.raw(c) as f32).collect();
                rep[2] = rep[2].min(t.elapsed().as_nanos());

                let t = Instant::now();
                s_turn = cs.iter().map(|c| turn_grad.score(c)).collect();
                rep[3] = rep[3].min(t.elapsed().as_nanos());

                let t = Instant::now();
                s_nc = cs.iter().map(|c| turn_nc.score(c)).collect();
                rep[4] = rep[4].min(t.elapsed().as_nanos());

                let t = Instant::now();
                let node_grad = Gradient::new(&node.state, node.turn);
                s_node = cs.iter().map(|c| node_grad.score(c)).collect();
                rep[5] = rep[5].min(t.elapsed().as_nanos());

                // The two-stage prior, end to end: price every edge, keep
                // `widen_cap`, then pay the one-ply only on those.
                let t = Instant::now();
                let s_pre: Vec<f32> = cs.iter().map(|c| turn_grad.score(c)).collect();
                std::hint::black_box(&s_pre);
                let mut keep: Vec<usize> = (0..cs.len()).collect();
                keep.sort_by(|&a, &b| s_pre[b].total_cmp(&s_pre[a]).then(a.cmp(&b)));
                keep.truncate(128);
                for &j in &keep {
                    let mut next = node.state;
                    let _ = tree::apply_step(
                        &mut next,
                        Phase::Take {
                            worker: node.worker,
                        },
                        node.turn,
                        node.done,
                        &Step::Take(cs[j].clone()),
                    );
                    std::hint::black_box(tzolkin::eval::heuristic(&next, node.turn));
                }
                rep[NAMES.len() + 1] = rep[NAMES.len() + 1].min(t.elapsed().as_nanos());
            }
            let band = if cs.len() > 128 { 1 } else { 0 };
            for k in 0..rep.len() {
                nanos[k] += rep[k];
                wide_nanos[band][k] += rep[k];
            }
            wide_edges[band] += cs.len();
            wide_nodes[band] += 1;
            two_nanos += rep[NAMES.len() + 1];
            // Generation order is what a uniform prior leaves behind: `sort_by`
            // is stable, so an all-equal prior is a no-op and the edges stay in
            // the order `choices_for_worker` produced them.
            let s_gen: Vec<f32> = (0..cs.len()).map(|k| -(k as f32)).collect();

            let order_of = |s: &[f32]| -> Vec<usize> {
                let mut o: Vec<usize> = (0..s.len()).collect();
                o.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
                o
            };
            let truth_order = order_of(&truth);
            let best = truth[truth_order[0]] as f64;
            let truth_rank: Vec<usize> = {
                let mut v = vec![0usize; cs.len()];
                for (r, &j) in truth_order.iter().enumerate() {
                    v[j] = r;
                }
                v
            };
            let wide = cs.len() > 128;
            if wide {
                trunc_nodes += 1;
            }

            for (oi, s) in [&s_gen, &s_wealth, &s_flat, &s_turn, &s_nc, &s_node]
                .into_iter()
                .enumerate()
            {
                let o = order_of(s);
                argmax_rank[oi].push(o.iter().position(|&j| j == truth_order[0]).unwrap_or(0));
                if wide && !o[..128].contains(&truth_order[0]) {
                    trunc_lost_best[oi] += 1;
                }
                for (ki, &k) in KS.iter().enumerate() {
                    let k = k.min(cs.len());
                    let got = o[..k]
                        .iter()
                        .map(|&j| truth[j] as f64)
                        .fold(f64::MIN, f64::max);
                    regret[oi][ki].push(best - got);
                    let hit = o[..k].iter().filter(|&&j| truth_rank[j] < k).count();
                    recall[oi][ki].push(hit as f64 / k as f64);
                }
            }
        }
    }

    println!(
        "\n=== prior orderings at {n_nodes} Take nodes of >= 8 edges ({edges_scored} edges) ===\n\
         width mean {:.1}, p50 {}, p90 {}, p99 {}, max {}",
        widths.iter().sum::<usize>() as f64 / widths.len().max(1) as f64,
        qs(&widths, 0.5),
        qs(&widths, 0.9),
        qs(&widths, 0.99),
        qs(&widths, 1.0)
    );
    println!(
        "\nregret: eval::heuristic points below the best edge, among the first K in prior order\n\
         (K=32 is `max_edges`, the window a node opens with; K=128 is `widen_cap`)"
    );
    println!(
        "  {:<12} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "order", "K", "mean", "p50", "p90", "p99", "recall"
    );
    for (oi, name) in NAMES.iter().enumerate() {
        for (ki, &k) in KS.iter().enumerate() {
            let v = &regret[oi][ki];
            println!(
                "  {:<12} {:>4} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>8.1}%",
                if ki == 0 { *name } else { "" },
                k,
                v.iter().sum::<f64>() / v.len().max(1) as f64,
                qsf(v, 0.5),
                qsf(v, 0.9),
                qsf(v, 0.99),
                recall[oi][ki].iter().sum::<f64>() * 100.0
                    / recall[oi][ki].len().max(1) as f64
            );
        }
    }
    println!("\nrank of the one-ply best edge under each cheap order");
    println!("  {:<12} {:>8} {:>8} {:>8} {:>8}", "order", "p50", "p90", "p99", "max");
    for (oi, name) in NAMES.iter().enumerate() {
        let v = &argmax_rank[oi];
        println!(
            "  {:<12} {:>8} {:>8} {:>8} {:>8}",
            name,
            qs(v, 0.5),
            qs(v, 0.9),
            qs(v, 0.99),
            qs(v, 1.0)
        );
    }
    println!(
        "\ntruncation: {trunc_nodes} of {n_nodes} nodes exceed widen_cap=128, so edges are dropped"
    );
    for (oi, name) in NAMES.iter().enumerate() {
        println!(
            "  {:<12} drops the one-ply best edge in {} of them",
            name, trunc_lost_best[oi]
        );
    }
    println!(
        "\ncost of building the prior: min of {REPEATS} interleaved repeats per node"
    );
    let rows: [(&str, usize); NAMES.len() + 1] = [
        ("generation", 0),
        ("wealth only", 1),
        ("flat price", 2),
        ("grad@turn", 3),
        ("grad-nocards", 4),
        ("grad@node", 5),
        ("one-ply", NAMES.len()),
    ];
    for (bi, (label, edges, nodes)) in [
        ("all nodes", edges_scored, n_nodes),
        ("nodes > 128 edges", wide_edges[1], wide_nodes[1]),
    ]
    .into_iter()
    .enumerate()
    {
        let t = if bi == 0 { &nanos } else { &wide_nanos[1] };
        let one = t[NAMES.len()] as f64 / edges.max(1) as f64;
        println!(
            "\n  {label}: {nodes} nodes, {edges} edges\n  {:<14} {:>10} {:>12} {:>11}",
            "order", "ns/edge", "vs one-ply", "us/node"
        );
        for (name, k) in rows {
            if k == 0 {
                continue;
            }
            let x = t[k] as f64 / edges.max(1) as f64;
            println!(
                "  {:<14} {:>10.1} {:>11.1}x {:>11.2}",
                name,
                x,
                one / x.max(1e-9),
                t[k] as f64 / 1e3 / nodes.max(1) as f64
            );
        }
        let x = t[NAMES.len() + 1] as f64 / edges.max(1) as f64;
        println!(
            "  {:<14} {:>10.1} {:>11.1}x {:>11.2}   grad@turn on every edge, one-ply on \
             the 128 it keeps",
            "two-stage",
            x,
            one / x.max(1e-9),
            t[NAMES.len() + 1] as f64 / 1e3 / nodes.max(1) as f64
        );
    }
    println!(
        "\n  gradient build: {:.1} us each, {grad_builds} builds for {n_nodes} nodes \
         ({:.2} us amortised per node)",
        grad_build_nanos as f64 / 1e3 / grad_builds.max(1) as f64,
        grad_build_nanos as f64 / 1e3 / n_nodes.max(1) as f64
    );
    println!(
        "  two-stage's top 32 is the one-ply best 32 inside grad@turn's top 128, so its \
         regret is the grad@turn K=128 row above, not a row of its own"
    );
    let _ = two_nanos;
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    let spec = args.next().unwrap_or_else(|| "heuristic:32".into());
    let cap: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let cap_ms = cap;

    let agent = parse_agent(&spec, false).expect("agent");
    let wall = Instant::now();

    // ---- phase 1: reach the positions -----------------------------------
    set_prune(0);
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

    // The MCTS-facing pass. `Take`-node width, not moves per turn: the
    // factored search never enumerates a whole turn, so the six-figure number
    // is one nobody pays.
    if std::env::var("NODES").is_ok() {
        let d: usize = std::env::var("DESCENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        node_audit(&positions, d);
        if std::env::var("PRIORS").is_ok() {
            prior_audit(&positions, d);
        }
        println!("\nwall {:.1}s", wall.elapsed().as_secs_f64());
        return;
    }

    if std::env::var("AUDIT").is_ok() {
        set_prune(2);
        audit(&positions);
        if std::env::var("MOVEAUDIT").is_ok() {
            move_audit(&positions, 40_000);
        }
        if std::env::var("BEAM").is_ok() {
            beam_audit(&positions, 40_000);
        }
        if std::env::var("ORDER").is_ok() {
            order_audit(&positions, 40_000);
            order_cost(&positions, 40_000);
        }
        println!("\nwall {:.1}s", wall.elapsed().as_secs_f64());
        return;
    }

    // ---- phase 3 (attribution) ------------------------------------------
    for on in [0u8, 1, 2] {
    set_prune(on);
    println!("\n########## PRUNING LEVEL {on} ##########");
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
    set_prune(0);

    // ---- phase 2: A/B/C on identical positions --------------------------
    //
    // Three levels in one pass, so the two generations of rules are priced
    // against the same positions and against each other.
    let mut n: [Vec<usize>; 3] = Default::default();
    let mut t: [Vec<usize>; 3] = Default::default();
    let mut capped = [0usize; 3];
    let mut worst = ([0usize; 3], 0u8, 0usize);
    let mut place_a = 0usize;
    let mut retr_a = 0usize;
    for (i, &(ref g, p)) in positions.iter().enumerate() {
        let mut row = [Sized_ { place: 0, retrieve: 0, nanos: 0, capped: false }; 3];
        for lvl in 0..3u8 {
            set_prune(lvl);
            row[lvl as usize] = size_of_move_space(g, p, cap);
            if row[lvl as usize].capped {
                capped[lvl as usize] += 1;
            }
        }
        place_a += row[0].place;
        retr_a += row[0].retrieve;
        // Only rows where no level hit the cap are comparable.
        if row.iter().all(|r| !r.capped) {
            let sizes: [usize; 3] = std::array::from_fn(|k| row[k].place + row[k].retrieve);
            for k in 0..3 {
                n[k].push(sizes[k]);
                t[k].push(row[k].nanos as usize);
            }
            if sizes[0] > worst.0[0] {
                worst = (sizes, g.day, g.on_board(p).count());
            }
        }
        if i % 100 == 0 {
            eprintln!("phase 2: {i}/{} ({:.1}s)", positions.len(), wall.elapsed().as_secs_f64());
        }
    }
    set_prune(2);

    println!(
        "\n=== branching by pruning level on identical positions ({} of {} comparable, cap {cap_ms} moves) ===",
        n[0].len(),
        positions.len()
    );
    println!(
        "level 0 mix: {place_a} placement, {retr_a} retrieval ({:.1}% retrieval)",
        retr_a as f64 * 100.0 / (place_a + retr_a).max(1) as f64
    );
    println!("capped turns: {capped:?}");
    quantiles("branching, level 0 -> level 1 (rules already in place)", &n[0], &n[1]);
    quantiles("branching, level 1 -> level 2 (this pass)", &n[1], &n[2]);
    quantiles("branching, level 0 -> level 2 (everything)", &n[0], &n[2]);

    println!("\nlegal_moves() time (ms)");
    println!(
        "  {:<8} {:>11} {:>11} {:>11} {:>9}",
        "", "level 0", "level 1", "level 2", "0/2"
    );
    let mut ts: [Vec<usize>; 3] = t.clone();
    for v in ts.iter_mut() {
        v.sort_unstable();
    }
    for q in [0.5, 0.9, 0.99, 1.0] {
        let x: [f64; 3] = std::array::from_fn(|k| pct(&ts[k], q) as f64 / 1e6);
        println!(
            "  p{:<7} {:>11.3} {:>11.3} {:>11.3} {:>8.2}x",
            format!("{:.0}", q * 100.0),
            x[0],
            x[1],
            x[2],
            x[0] / x[2].max(1e-9)
        );
    }
    let tot: [f64; 3] = std::array::from_fn(|k| ts[k].iter().sum::<usize>() as f64 / 1e9);
    println!(
        "  {:<8} {:>11.2} {:>11.2} {:>11.2} {:>8.2}x",
        "total s", tot[0], tot[1], tot[2], tot[0] / tot[2].max(1e-9)
    );
    println!(
        "\nworst comparable position: day {} with {} workers on board: {} -> {} -> {}",
        worst.1, worst.2, worst.0[0], worst.0[1], worst.0[2]
    );

    // per-worker options, pruned
    set_prune(2);
    let mut wo2: Vec<usize> = Vec::new();
    for &(ref g, p) in &positions {
        for w in g.on_board(p) {
            if let Some((gear, pos)) = g.loc(w).on_board() {
                wo2.push(moves::choices_for_worker(g, p, gear, pos).len());
            }
        }
    }
    wo2.sort_unstable();
    set_prune(0);
    let mut wo1: Vec<usize> = Vec::new();
    for &(ref g, p) in &positions {
        for w in g.on_board(p) {
            if let Some((gear, pos)) = g.loc(w).on_board() {
                wo1.push(moves::choices_for_worker(g, p, gear, pos).len());
            }
        }
    }
    wo1.sort_unstable();
    set_prune(2);
    quantiles("options per worker (choices_for_worker)", &wo1, &wo2);

    let _ = Effect::UnlockWorker;
    println!("\nwall {:.1}s", wall.elapsed().as_secs_f64());
}
