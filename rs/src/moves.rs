//! Legal move enumeration.
//!
//! A `Move` owns its placements and choices outright. The Go equivalent built
//! moves with `append(m.Workers, worker)` on a shared backing array, so sibling
//! branches off the same parent overwrote each other's last entry once capacity
//! allowed -- from the fourth worker onward. That class of bug cannot be written
//! here.

use crate::effect::{Choice, Effect};
use crate::ids::*;
use crate::spaces::choices_at;
use crate::state::GameState;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::ops::ControlFlow;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Placement {
    Gear(Gear, Pos),
    FirstPlayer,
}

impl Placement {
    /// Corn surcharge for the space itself.
    /// Corn surcharge for the space itself. Public because the factored search
    /// tree needs the budget check and must not restate it.
    pub fn space_cost(self) -> u8 {
        match self {
            Placement::Gear(_, pos) => pos.0,
            Placement::FirstPlayer => 0,
        }
    }
}

pub type Placements = SmallVec<[(WorkerId, Placement); 6]>;
pub type Retrievals = SmallVec<[(WorkerId, Choice); 6]>;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MoveKind {
    Place(Placements),
    Retrieve(Retrievals),
    /// "The gods take pity": with no workers on the gears, nothing affordable to
    /// place, and no temple left to beg from, a player may place exactly one
    /// worker on the cheapest space and give all their corn to the bank.
    /// Available only when no other move is.
    Pity { worker: WorkerId, spot: Placement },
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Move {
    pub kind: MoveKind,
    /// Beg for corn first, at the cost of one step down a temple.
    pub beg: Option<Temple>,
    /// Corn paid to place. Costs carried inside choices are separate.
    pub corn_cost: u8,
}

impl Move {
    pub fn n_workers(&self) -> usize {
        match &self.kind {
            MoveKind::Place(v) => v.len(),
            MoveKind::Retrieve(v) => v.len(),
            MoveKind::Pity { .. } => 1,
        }
    }

    /// Whether two moves do the same thing, under different names.
    ///
    /// `place_rec` enumerates *assignments*, and a player's available workers
    /// are interchangeable, so `w0->Tikal:0 w1->Uxmal:0` and `w0->Uxmal:0
    /// w1->Tikal:0` are two spellings of one move. Two workers going on the
    /// same gear are forced into ascending order by `lowest_free`, so the
    /// multiset of spots also pins the corn cost — which is what makes this an
    /// equivalence rather than a resemblance.
    ///
    /// Retrieval order is *not* interchangeable: a contested building, a
    /// Palenque tile, corn a later action needs. Retrievals therefore compare
    /// exactly, and the state memo in `retrieve_rec` has already collapsed the
    /// orderings that genuinely commute.
    ///
    /// This is a display and search-ordering concern, not a rules one:
    /// `legal_moves` still returns every spelling, because the equivalence is
    /// only true of the *state* reached and a caller checking generation should
    /// see what generation produced.
    pub fn same_effect(&self, other: &Move) -> bool {
        if self.beg != other.beg || self.corn_cost != other.corn_cost {
            return false;
        }
        match (&self.kind, &other.kind) {
            (MoveKind::Place(a), MoveKind::Place(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                let key = |v: &Placements| {
                    let mut k: SmallVec<[Placement; 6]> = v.iter().map(|&(_, s)| s).collect();
                    k.sort_unstable();
                    k
                };
                key(a) == key(b)
            }
            _ => self.kind == other.kind,
        }
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(t) = self.beg {
            write!(f, "[beg {}] ", t.letter())?;
        }
        match &self.kind {
            MoveKind::Place(v) => {
                write!(f, "place({} corn)", self.corn_cost)?;
                for (w, p) in v {
                    match p {
                        Placement::Gear(g, pos) => write!(f, " w{}->{}:{}", w.0, g.name(), pos.0)?,
                        Placement::FirstPlayer => write!(f, " w{}->first", w.0)?,
                    }
                }
            }
            MoveKind::Retrieve(v) => {
                f.write_str("retrieve")?;
                for (w, c) in v {
                    write!(f, " w{}[{c}]", w.0)?;
                }
            }
            MoveKind::Pity { worker, spot } => {
                write!(f, "pity w{}->", worker.0)?;
                match spot {
                    Placement::Gear(g, pos) => write!(f, "{}:{}", g.name(), pos.0)?,
                    Placement::FirstPlayer => f.write_str("first")?,
                }
            }
        }
        Ok(())
    }
}

/// Visit every legal move for `p` without materialising the list.
///
/// This is the primitive; `legal_moves` collects from it. A search node with a
/// six-figure action space must not build a `Vec` of it -- at 264 bytes per
/// `Move`, the worst position measured would be ~66 MB for one node -- and
/// progressive widening only ever wants the first handful anyway.
///
/// The callback returns `ControlFlow::Break` to stop early. Traversal order is
/// deterministic: the memo decides *whether* a move is emitted, never in what
/// order, so a capped walk returns the same moves every run.
pub fn visit_legal_moves<F>(g: &GameState, p: PlayerId, mut f: F) -> ControlFlow<()>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    // Shared across beg variants: begging one temple down and then stepping it
    // back up lands in the same position as begging a different temple, so the
    // variants converge and would otherwise emit duplicate moves.
    let mut any = false;
    {
        let mut wrapped = |m: &Move| {
            any = true;
            f(m)
        };
        visit_normal_moves(g, p, &mut wrapped)?;
    }
    if !any {
        for m in pity_moves(g, p) {
            f(&m)?;
        }
    }
    ControlFlow::Continue(())
}

/// Everything except the pity fallback. Separate so that the pity rule can ask
/// "is there any other move?" without recursing into itself.
fn visit_normal_moves<F>(g: &GameState, p: PlayerId, f: &mut F) -> ControlFlow<()>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    let mut v = Walk {
        f,
        reached: HashSet::new(),
    };

    for beg in beg_options(g, p) {
        let mut probe = *g;
        let budget = match beg {
            Some(t) => {
                probe.players[p.idx()].corn = 3;
                probe.temple_step(p, t, -1);
                3
            }
            None => g.players[p.idx()].corn,
        };
        visit_placements(&probe, p, budget, beg, &mut v)?;
        visit_retrievals(&probe, p, beg, &mut v)?;
    }
    ControlFlow::Continue(())
}

/// The cheapest spaces a pitied player may take, giving up all their corn.
pub fn pity_moves(g: &GameState, p: PlayerId) -> Vec<Move> {
    let Some(worker) = g.available(p).next() else {
        return Vec::new();
    };
    let mut spots: SmallVec<[Placement; 6]> = SmallVec::new();
    for gear in Gear::ALL {
        if let Some(pos) = g.lowest_free(gear) {
            spots.push(Placement::Gear(gear, pos));
        }
    }
    if g.first_player_space.is_none() {
        spots.push(Placement::FirstPlayer);
    }
    let Some(cheapest) = spots.iter().map(|s| s.space_cost()).min() else {
        return Vec::new();
    };
    spots
        .into_iter()
        .filter(|s| s.space_cost() == cheapest)
        .map(|spot| Move {
            kind: MoveKind::Pity { worker, spot },
            beg: None,
            corn_cost: g.players[p.idx()].corn,
        })
        .collect()
}

struct Walk<'a, F> {
    f: &'a mut F,
    reached: HashSet<GameState>,
}

impl<F> Walk<'_, F>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    #[inline]
    fn emit(&mut self, m: &Move) -> ControlFlow<()> {
        (self.f)(m)
    }
}

/// Every legal move, in canonical order.
pub fn legal_moves(g: &GameState, p: PlayerId) -> Vec<Move> {
    let mut out = Vec::new();
    let _ = visit_legal_moves(g, p, |m| {
        out.push(m.clone());
        ControlFlow::Continue(())
    });
    out.sort();
    out
}

/// The first `max` legal moves in traversal order.
///
/// Deterministic, but *not* a canonical subset: which moves you get depends on
/// traversal order, so this is for bounded expansion, not for reasoning about
/// the move set.
pub fn legal_moves_capped(g: &GameState, p: PlayerId, max: usize) -> Vec<Move> {
    let mut out = Vec::with_capacity(max.min(64));
    if max == 0 {
        return out;
    }
    let _ = visit_legal_moves(g, p, |m| {
        out.push(m.clone());
        if out.len() >= max {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    out
}

/// Whether `p` has any legal move, without building any of them.
pub fn has_legal_move(g: &GameState, p: PlayerId) -> bool {
    visit_legal_moves(g, p, |_| ControlFlow::Break(())).is_break()
}

/// How many legal moves exist, without holding them.
pub fn count_legal_moves(g: &GameState, p: PlayerId) -> usize {
    let mut n = 0usize;
    let _ = visit_legal_moves(g, p, |_| {
        n += 1;
        ControlFlow::Continue(())
    });
    n
}

/// Begging is available to a player with fewer than three corn who still has
/// somewhere to fall on a temple.
pub fn beg_options(g: &GameState, p: PlayerId) -> Vec<Option<Temple>> {
    let mut out = vec![None];
    if g.players[p.idx()].corn < 3 {
        for t in Temple::ALL {
            if g.can_temple_step(p, t, -1) {
                out.push(Some(t));
            }
        }
    }
    out
}

// ---- placement ---------------------------------------------------------

fn visit_placements<F>(
    g: &GameState,
    p: PlayerId,
    budget: u8,
    beg: Option<Temple>,
    v: &mut Walk<F>,
) -> ControlFlow<()>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    // A player's available workers are interchangeable, so they are consumed in
    // id order. Enumerating which *identity* goes where would only produce
    // permutations of the same move.
    let avail: SmallVec<[WorkerId; 6]> = g.available(p).collect();
    if avail.is_empty() {
        return ControlFlow::Continue(());
    }
    let mut probe = *g;
    let mut cur = Placements::new();
    place_rec(&mut probe, p, &avail, 0, 0, budget, beg, &mut cur, v)
}

#[allow(clippy::too_many_arguments)]
fn place_rec<F>(
    g: &mut GameState,
    p: PlayerId,
    avail: &[WorkerId],
    depth: usize,
    cost: u8,
    budget: u8,
    beg: Option<Temple>,
    cur: &mut Placements,
    v: &mut Walk<F>,
) -> ControlFlow<()>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    if depth > 0 {
        v.emit(&Move {
            kind: MoveKind::Place(cur.clone()),
            beg,
            corn_cost: cost,
        })?;
    }
    if depth == avail.len() {
        return ControlFlow::Continue(());
    }

    let w = avail[depth];
    let mut spots: SmallVec<[Placement; 6]> = SmallVec::new();
    for gear in Gear::ALL {
        if let Some(pos) = g.lowest_free(gear) {
            spots.push(Placement::Gear(gear, pos));
        }
    }
    if g.first_player_space.is_none() {
        spots.push(Placement::FirstPlayer);
    }

    for spot in spots {
        // The nth worker placed this turn costs n extra corn, on top of the
        // space's own index.
        let step = depth as u8 + spot.space_cost();
        if cost + step > budget {
            continue;
        }

        let saved = *g;
        match spot {
            Placement::Gear(gear, pos) => g.place_worker(w, gear, pos),
            Placement::FirstPlayer => g.place_on_first_player(w),
        }
        cur.push((w, spot));

        let flow = place_rec(g, p, avail, depth + 1, cost + step, budget, beg, cur, v);

        cur.pop();
        *g = saved;
        flow?;
    }
    ControlFlow::Continue(())
}

// ---- retrieval ---------------------------------------------------------

/// Everything a worker on `gear`:`pos` may do when it is picked up.
///
/// A worker may take the action of its own space or of any *lower* space on the
/// same gear, paying one corn per space it steps down. The Go version had this
/// commented out with the note "first attempt broke".
pub fn choices_for_worker(g: &GameState, p: PlayerId, gear: Gear, pos: Pos) -> Vec<Choice> {
    let corn = g.players[p.idx()].corn;
    // "Do nothing (except pick up the worker)" is always one of the three
    // options the rules give for a retrieved worker. Go only offered it when a
    // space had nothing else to give, which forced players to buy an unwanted
    // worker on Uxmal 3, spend a block on Tikal 5, or build on Tikal 2 and 4.
    let mut out = vec![Choice::skip()];

    for j in 0..=pos.0 {
        let fee = pos.0 - j;
        if fee > corn {
            continue;
        }
        // Generate against a state with the fee already paid, so a discounted
        // action cannot spend corn the fee consumed.
        let mut probe = *g;
        probe.players[p.idx()].corn -= fee;

        for c in choices_at(&probe, p, gear, Pos(j)) {
            if fee == 0 {
                out.push(c);
            } else if !c.is_skip() {
                // Paying corn to reach a space that does nothing is strictly
                // dominated by doing nothing for free.
                out.push(Choice::one(Effect::Corn(-(fee as i16))).chain(&c));
            }
        }
    }
    crate::options::dedup(out)
}

fn visit_retrievals<F>(
    g: &GameState,
    p: PlayerId,
    beg: Option<Temple>,
    v: &mut Walk<F>,
) -> ControlFlow<()>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    let on_board: SmallVec<[WorkerId; 6]> = g.on_board(p).collect();
    if on_board.is_empty() {
        return ControlFlow::Continue(());
    }
    let mut probe = *g;
    let mut cur = Retrievals::new();
    retrieve_rec(&mut probe, p, &on_board, &mut cur, beg, v)
}

/// Enumerate retrievals in every order, deduplicated by the position they reach.
///
/// Order genuinely matters -- a contested building, a Palenque tile, corn needed
/// to pay for a later action -- so every ordering has to be explored. But most
/// orderings commute, and two that land in the same state are the same move for
/// every purpose. The set of reached states doubles as a memo: a state already
/// visited has had its whole subtree explored, so it is never expanded twice.
///
/// Cost is therefore proportional to the number of distinct reachable positions
/// rather than to the number of orderings, which is only possible because the
/// state is `Copy + Hash`.
fn retrieve_rec<F>(
    g: &mut GameState,
    p: PlayerId,
    board: &[WorkerId],
    cur: &mut Retrievals,
    beg: Option<Temple>,
    v: &mut Walk<F>,
) -> ControlFlow<()>
where
    F: FnMut(&Move) -> ControlFlow<()>,
{
    if !cur.is_empty() {
        if !v.reached.insert(*g) {
            return ControlFlow::Continue(());
        }
        v.emit(&Move {
            kind: MoveKind::Retrieve(cur.clone()),
            beg,
            corn_cost: 0,
        })?;
    }

    for &w in board {
        let Some((gear, pos)) = g.loc(w).on_board() else {
            continue;
        };
        for choice in choices_for_worker(g, p, gear, pos) {
            let saved = *g;
            choice.apply(g, p);
            g.retrieve_worker(w);

            cur.push((w, choice));
            let flow = retrieve_rec(g, p, board, cur, beg, v);
            cur.pop();

            *g = saved;
            flow?;
        }
    }
    ControlFlow::Continue(())
}

/// Apply a move to the state. The move must have come from `legal_moves` for
/// this state.
pub fn apply_move(g: &mut GameState, p: PlayerId, m: &Move) {
    if let Some(t) = m.beg {
        g.players[p.idx()].corn = 3;
        g.temple_step(p, t, -1);
    }

    match &m.kind {
        MoveKind::Place(v) => {
            let corn = &mut g.players[p.idx()].corn;
            debug_assert!(*corn >= m.corn_cost, "placement not affordable");
            *corn = corn.saturating_sub(m.corn_cost);
            for &(w, spot) in v {
                match spot {
                    Placement::Gear(gear, pos) => g.place_worker(w, gear, pos),
                    Placement::FirstPlayer => g.place_on_first_player(w),
                }
            }
        }
        MoveKind::Retrieve(v) => {
            for (w, choice) in v {
                choice.apply(g, p);
                g.retrieve_worker(*w);
            }
        }
        MoveKind::Pity { worker, spot } => {
            g.players[p.idx()].corn = 0;
            match *spot {
                Placement::Gear(gear, pos) => g.place_worker(*worker, gear, pos),
                Placement::FirstPlayer => g.place_on_first_player(*worker),
            }
        }
    }
}

/// Whether any move other than the pity fallback exists.
pub fn has_normal_move(g: &GameState, p: PlayerId) -> bool {
    let mut found = false;
    let mut f = |_: &Move| {
        found = true;
        ControlFlow::Break(())
    };
    let _ = visit_normal_moves(g, p, &mut f);
    found
}

/// Verify a move is actually payable, step by step, without clamping.
///
/// `apply` clamps at zero so a bad move can never corrupt the state; this is
/// what turns "the generator produced something unaffordable" into a loud
/// failure instead of a silent one.
pub fn check_move(g: &GameState, p: PlayerId, m: &Move) -> Result<(), String> {
    let mut probe = *g;
    if let Some(t) = m.beg {
        if probe.players[p.idx()].corn >= 3 {
            return Err(format!("begged on {t:?} while holding 3+ corn"));
        }
        if !probe.can_temple_step(p, t, -1) {
            return Err(format!("begged on {t:?} from the bottom of the track"));
        }
        probe.players[p.idx()].corn = 3;
        probe.temple_step(p, t, -1);
    }

    match &m.kind {
        MoveKind::Place(v) => {
            if v.is_empty() {
                return Err("placement move with no workers".into());
            }
            if probe.players[p.idx()].corn < m.corn_cost {
                return Err(format!(
                    "placement costs {} corn, player has {}",
                    m.corn_cost,
                    probe.players[p.idx()].corn
                ));
            }
            let mut expected = 0u8;
            for (n, &(w, spot)) in v.iter().enumerate() {
                if probe.loc(w) != crate::state::WorkerLoc::Available {
                    return Err(format!("placed worker {} which is {:?}", w.0, probe.loc(w)));
                }
                expected += n as u8 + spot.space_cost();
                match spot {
                    Placement::Gear(gear, pos) => {
                        if probe.lowest_free(gear) != Some(pos) {
                            return Err(format!(
                                "placed on {}:{} which is not the lowest free space",
                                gear.name(),
                                pos.0
                            ));
                        }
                        probe.place_worker(w, gear, pos);
                    }
                    Placement::FirstPlayer => {
                        if probe.first_player_space.is_some() {
                            return Err("placed on an occupied first player space".into());
                        }
                        probe.place_on_first_player(w);
                    }
                }
            }
            if expected != m.corn_cost {
                return Err(format!(
                    "move claims {} corn but the placements cost {expected}",
                    m.corn_cost
                ));
            }
        }
        MoveKind::Pity { worker, spot } => {
            if has_normal_move(g, p) {
                return Err("pity move offered while an ordinary move exists".into());
            }
            if probe.loc(*worker) != crate::state::WorkerLoc::Available {
                return Err(format!("pity move used worker {}, which is not free", worker.0));
            }
            if let Placement::Gear(gear, pos) = *spot {
                if probe.lowest_free(gear) != Some(pos) {
                    return Err("pity move did not take the lowest free space".into());
                }
            }
        }
        MoveKind::Retrieve(v) => {
            if v.is_empty() {
                return Err("retrieval move with no workers".into());
            }
            for (w, choice) in v {
                if probe.loc(*w).on_board().is_none() {
                    return Err(format!(
                        "retrieved worker {} which is {:?}",
                        w.0,
                        probe.loc(*w)
                    ));
                }
                if !choice.affordable(&probe, p) {
                    return Err(format!("choice [{choice}] is not affordable"));
                }
                choice.apply(&mut probe, p);
                probe.retrieve_worker(*w);
            }
        }
    }
    Ok(())
}

// ---- sampling ----------------------------------------------------------

/// Draw one legal move without enumerating the rest.
///
/// This is a **rollout policy**, not a uniform sample of `legal_moves`. It walks
/// the same recursion and calls the same space generators, so it tracks any
/// rules change automatically, but it decides at each step rather than branching
/// — which is the whole point, since a rollout throws away every move but one.
///
/// Where it deliberately departs from uniform:
///   * Begging is tried as a *fallback* (plus a small voluntary chance) rather
///     than as an equal quarter of the options. Uniform-over-structure would beg
///     three times out of four whenever a player dipped under three corn, which
///     spends a temple step for nothing.
///   * The number of workers used is geometric rather than uniform, favouring
///     the one-to-three-worker moves that dominate real play.
pub fn sample_legal_move<R: rand::Rng>(
    g: &GameState,
    p: PlayerId,
    rng: &mut R,
) -> Option<Move> {
    for beg in beg_candidates(g, p, rng) {
        let mut probe = *g;
        let budget = match beg {
            Some(t) => {
                probe.players[p.idx()].corn = 3;
                probe.temple_step(p, t, -1);
                3
            }
            None => g.players[p.idx()].corn,
        };

        let retrieve_first = rng.gen_bool(0.5);
        for retrieve in [retrieve_first, !retrieve_first] {
            let m = if retrieve {
                sample_retrieval(&probe, p, beg, rng)
            } else {
                sample_placement(&probe, p, budget, beg, rng)
            };
            if m.is_some() {
                return m;
            }
        }
    }

    // Nothing ordinary is available, so the gods take pity. Without this the
    // sampler silently skipped the turn in exactly the position the rule exists
    // for -- and passing is not legal.
    let pity = pity_moves(g, p);
    if pity.is_empty() {
        return None;
    }
    Some(pity[rng.gen_range(0..pity.len())].clone())
}

/// Beg options in the order they should be *tried*: rarely up front, otherwise
/// only once nothing else works.
fn beg_candidates<R: rand::Rng>(
    g: &GameState,
    p: PlayerId,
    rng: &mut R,
) -> SmallVec<[Option<Temple>; 3]> {
    let legal: SmallVec<[Temple; 3]> = Temple::ALL
        .iter()
        .copied()
        .filter(|&t| g.can_temple_step(p, t, -1))
        .collect();
    let can_beg = g.players[p.idx()].corn < 3 && !legal.is_empty();

    let mut out = SmallVec::new();
    let pick = |rng: &mut R| legal[rng.gen_range(0..legal.len())];
    if can_beg && rng.gen_bool(0.15) {
        out.push(Some(pick(rng)));
    }
    out.push(None);
    if can_beg {
        out.push(Some(pick(rng)));
    }
    out
}

fn sample_placement<R: rand::Rng>(
    g: &GameState,
    p: PlayerId,
    budget: u8,
    beg: Option<Temple>,
    rng: &mut R,
) -> Option<Move> {
    let avail: SmallVec<[WorkerId; 6]> = g.available(p).collect();
    let mut probe = *g;
    let mut cur = Placements::new();
    let mut cost = 0u8;

    for (depth, &w) in avail.iter().enumerate() {
        if depth > 0 && !rng.gen_bool(0.55) {
            break;
        }

        let mut spots: SmallVec<[Placement; 6]> = SmallVec::new();
        for gear in Gear::ALL {
            if let Some(pos) = probe.lowest_free(gear) {
                if cost + depth as u8 + pos.0 <= budget {
                    spots.push(Placement::Gear(gear, pos));
                }
            }
        }
        if probe.first_player_space.is_none() && cost + depth as u8 <= budget {
            spots.push(Placement::FirstPlayer);
        }
        if spots.is_empty() {
            break;
        }

        let spot = spots[rng.gen_range(0..spots.len())];
        cost += depth as u8 + spot.space_cost();
        match spot {
            Placement::Gear(gear, pos) => probe.place_worker(w, gear, pos),
            Placement::FirstPlayer => probe.place_on_first_player(w),
        }
        cur.push((w, spot));
    }

    (!cur.is_empty()).then(|| Move {
        kind: MoveKind::Place(cur),
        beg,
        corn_cost: cost,
    })
}

fn sample_retrieval<R: rand::Rng>(
    g: &GameState,
    p: PlayerId,
    beg: Option<Temple>,
    rng: &mut R,
) -> Option<Move> {
    let mut board: SmallVec<[WorkerId; 6]> = g.on_board(p).collect();
    if board.is_empty() {
        return None;
    }
    // Retrieval order matters, so the order itself is part of the sample.
    for i in (1..board.len()).rev() {
        board.swap(i, rng.gen_range(0..=i));
    }

    let mut probe = *g;
    let mut cur = Retrievals::new();

    for (n, &w) in board.iter().enumerate() {
        if n > 0 && !rng.gen_bool(0.45) {
            break;
        }
        let Some((gear, pos)) = probe.loc(w).on_board() else {
            continue;
        };
        let choices = choices_for_worker(&probe, p, gear, pos);
        if choices.is_empty() {
            continue;
        }
        let choice = choices[rng.gen_range(0..choices.len())].clone();
        choice.apply(&mut probe, p);
        probe.retrieve_worker(w);
        cur.push((w, choice));
    }

    (!cur.is_empty()).then(|| Move {
        kind: MoveKind::Retrieve(cur),
        beg,
        corn_cost: 0,
    })
}
