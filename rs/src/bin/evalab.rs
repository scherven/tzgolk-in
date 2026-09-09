//! SCRATCH — the evaluator variant lab. Delete before finishing.
//!
//! `bin/arena` cannot answer "is this evaluator better than that one", because
//! both of its agent specs resolve to whatever `eval::heuristic` is compiled
//! into the binary. This holds a **parameterised copy** of `src/eval.rs` as it
//! stood at 6a99f2f — `V::HEAD` reproduces it exactly, checked by `--eqcheck`
//! against the live `eval::heuristic` — and every structural idea is a field on
//! `V`, so two arms of a comparison differ in one line and nothing else.
//!
//! Both agent kinds are wired to the same `V`:
//!
//!     cargo run --release --bin evalab -- --ab derived --agent mcts:256 --games 400
//!     cargo run --release --bin evalab -- --ab derived --agent greedy:full --games 400
//!
//! and the arena design is `bin/arena`'s: one candidate rotated through all
//! four seats against three baselines, the **rotation block** (not the game) as
//! the independent unit, null centred score 0, null win rate 25%.
//!
//! Diagnostics that need no games:
//!
//!     --eqcheck     V::HEAD == eval::heuristic on 800 real positions
//!     --cost        ns/call for heuristic and for each variant
//!     --midturn     the evaluation trajectory across a turn (tree::apply_step)
//!     --spacetab    the hand table against the derived one, space by space
//!
//! Progress is appended to JSONL as each block completes, so a Ctrl-C loses
//! nothing and `--resume` picks up where it stopped.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use tzolkin::ids::*;
use tzolkin::phase::{Evaluation, Evaluator, Phase};
use tzolkin::record::{
    play_game, Agent, Candidates, GameConfig, GreedyAgent, Summary,
};
use tzolkin::state::GameState;

// =======================================================================
// The parameterised evaluator: `src/eval.rs` at 6a99f2f with every idea
// under test on a switch. `V::HEAD` is the committed file.
// =======================================================================
#[allow(dead_code)]
pub mod v {
    use tzolkin::data::buildings::def as bdef;
    use tzolkin::data::monuments::def as mdef;
    use tzolkin::data::temples::TEMPLES;
    use tzolkin::ids::*;
    use tzolkin::options::EffectPrice;
    use tzolkin::state::{GameState, LAST_DAY, POINT_DAYS, RESOURCE_DAYS};

    // ---- committed constants ------------------------------------------
    const CORN_PER_POINT: f32 = 4.0;
    const CORN_PREMIUM: f32 = 0.10;
    const LIQUID_CORN: f32 = 16.0;
    const BLOCK_PREMIUM: f32 = 0.55;
    const SKULL_PREMIUM: f32 = 2.2;
    const STARVE_POINTS: f32 = 3.0;
    const CORN_INCOME_PER_ROUND: f32 = 1.9;
    const ROUNDS_PER_ACTION: f32 = 2.6;
    const RESEARCH_SCALE: f32 = 0.5;
    const MONUMENT_SHARE: f32 = 0.45;
    const TEMPLE_NEAR: f32 = 0.85;
    const TEMPLE_FAR: f32 = 0.55;
    const TEMPLE_CLIMB: f32 = 0.10;
    const BLOCK_BREADTH: f32 = 0.50;
    const TILE_PREMIUM: f32 = 0.25;

    /// Where `board_position` gets a space's point-equivalent.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Space {
        /// The committed hand table.
        Table,
        /// The hand table times a per-space factor measured by `--promise`:
        /// what taking the action there actually moves `heuristic` by, divided
        /// by what the table says it is worth. `space_scale` renormalises the
        /// level so this is a change of *ranking* only.
        Calib,
        /// Price the `Effect`s `spaces::choices_at` emits, and take the best
        /// choice. The whole point of the experiment.
        Derived,
        /// Derived, but a space whose generator returns nothing but `skip`
        /// falls back to the hand table — isolating "exhausted options price at
        /// zero" from the pricing itself.
        DerivedFloor,
        /// Derived from `mcts::Gradient` instead of the hand price list: the
        /// evaluator's own marginal for `+1` of each axis, probed against this
        /// very position. Removes "your price list is bad" from the argument.
        Probed,
        /// The hand table with every state-dependent gate removed — the
        /// ungated number at every space. Isolates what the gates are worth
        /// from what the numbers are worth.
        TableFlat,
        /// The hand table with **only the deferred-payoff spaces** repriced to
        /// what the derived table says they are worth: Tikal 1 and 3
        /// (research), Tikal 5 (two temple steps) and Uxmal 1 (corn to a temple
        /// step). Every other space keeps its hand price. If the derived
        /// table's regression is about *shape* rather than gating or scale,
        /// this alone reproduces most of it.
        TableDefer,
        /// The corpus mean of `Space::Derived`, frozen into a constant table.
        /// The derived table's *shape and scale* at the hand table's cost, so
        /// the comparison can be run at MCTS scale at all — `Derived` itself
        /// is ~700x `heuristic`, which no search can pay.
        DerivedStatic,
    }

    /// `Space::DerivedStatic`: the mean of `Space::Derived` over 1500 turn
    /// roots (`--spacetab`), by gear and position.
    /// `delivery / table` per space, from `evalab --promise --corpus 300`
    /// (53,704 standing workers): the factor by which taking the action at that
    /// space moves `heuristic` more than the hand table says it is worth.
    /// Position 0 and the unused Chichen slots are 1.0 — the table is 0 there.
    static RCAL: [[f32; 11]; 5] = [
        [1.000, 2.923, 2.702, 3.577, 3.492, 3.483, 3.534, 3.289, 1.000, 1.000, 1.000],
        [1.000, 3.658, 3.128, 3.033, 2.506, 3.569, 3.133, 3.085, 1.000, 1.000, 1.000],
        [1.000, 0.302, 1.927, 0.944, 1.330, 5.189, 3.139, 2.867, 1.000, 1.000, 1.000],
        [1.000, 3.725, 2.430, 0.584, 3.493, 4.623, 2.736, 2.544, 1.000, 1.000, 1.000],
        [1.000, 1.956, 1.636, 1.502, 1.467, 1.148, 1.258, 1.052, 1.301, 1.233, 1.082],
    ];

    const DERIVED_STATIC: [[f32; 11]; 5] = [
        // Palenque
        [0.00, 1.06, 2.75, 3.71, 4.76, 5.81, 5.81, 5.81, 0.0, 0.0, 0.0],
        // Yaxchilan
        [0.00, 1.41, 1.73, 2.26, 5.21, 3.64, 5.21, 5.21, 0.0, 0.0, 0.0],
        // Tikal
        [0.00, 0.04, 4.37, 0.12, 5.69, 2.14, 6.11, 6.11, 0.0, 0.0, 0.0],
        // Uxmal
        [0.00, 0.20, 0.82, 7.20, 1.47, 4.59, 7.96, 7.96, 0.0, 0.0, 0.0],
        // Chichen
        [0.06, 0.18, 0.41, 0.64, 1.02, 1.27, 1.66, 1.97, 2.54, 2.84, 2.99],
    ];

    /// Every knob a variant can turn. `HEAD` is `src/eval.rs` at 6a99f2f.
    #[derive(Clone, Copy, Debug)]
    pub struct V {
        pub action_value: f32,
        pub tempo: f32,
        pub building_value: f32,
        pub space: Space,
        /// Multiply the derived space value by this before it meets
        /// `tempo` — the units of a derived table are not the units the
        /// hand table was swept in.
        pub space_scale: f32,
        /// Credit a worker still in hand with the best space it could be
        /// placed on this turn, less the corn. Makes `heuristic` roughly
        /// invariant to *where in a placement* a turn has got to.
        pub hand_worker: bool,
        /// Value corn by the placement depth it buys rather than as a flat
        /// premium under a cap.
        pub corn_depth: bool,
        /// Price a building by the monuments that count its colour rather
        /// than at a flat rate.
        pub building_colour: bool,
        /// Contention: an exclusive temple top, a monument an opponent is
        /// closer to, a draining skull bank.
        pub contention: bool,
        /// Split `contention` in two, so the report can say which half works.
        pub contend_temple: bool,
        pub contend_monument: bool,
        /// Charge a worker on a gear one of its generic actions: the action it
        /// is about to take is priced by `board_position`, so `engine_value`
        /// paying it again counts the same action twice and makes a placement
        /// a free lunch of the whole space value.
        pub charge_placed: bool,
        /// Rounds of tempo an in-hand worker is charged for not being on the
        /// board yet, when `hand_worker` is on.
        pub hand_lag: f32,
        /// Post-multipliers on the `Components` terms, applied where
        /// `plan::Schedule` applies its weights, so "this term is over-priced
        /// by k" is a single number rather than a rewrite of the term.
        pub board_scale: f32,
        pub engine_scale: f32,
        pub temple_scale: f32,
        /// The two-level confidence on a projected temple payout. `t_half > 0`
        /// replaces both with a hyperbolic discount in the *distance* to the
        /// day, `t_half / (t_half + (day - g.day))`, which is the thing the
        /// committed pair cannot say: a payout one day out and one twelve days
        /// out are both "the near one" today.
        pub t_near: f32,
        pub t_far: f32,
        pub t_half: f32,
        /// `TEMPLE_CLIMB`, and whether the climb credit respects
        /// `GameState::temple_ceiling` — the top step of a track an opponent
        /// already stands on cannot be climbed to, and `eval.rs` credits it.
        pub climb: f32,
        pub ceiling: bool,
        /// Rounds of riding a placed worker may be credited for. `board_position`
        /// maxes over every space the calendar still allows, which at
        /// `rounds_left = 20` is the whole gear; this caps the look-ahead.
        pub reach: f32,
        /// A flat credit per worker still in hand, added *after* `board_scale`
        /// so the number is in points. F17a: a placing turn under-states the
        /// turn it is about to play by 1.81 points because `board_position`
        /// pays a worker only once it is standing on a gear, and `hand_worker`
        /// closes that gap by crediting the best space on the board — which is
        /// +3.32 greedy and −3.28 MCTS (S4). This is the same correction
        /// without the max: one number, gated on being able to pay for the
        /// placement and on there being a round left to take the action in.
        pub hand_flat: f32,
        /// Exponent on the `--promise` ratio when `space` is `Calib`. 0 is the
        /// hand table, 1 is the fully delivery-consistent one.
        pub calib_alpha: f32,
        /// `RESEARCH_SCALE`. `--promise` says the first research space delivers
        /// 0.60 where the table prices it at 2.00 — the widest disagreement on
        /// the board — so either the table is wrong there or `engine_value` is.
        pub research_scale: f32,
        /// The `held_premium` price list, and `starvation_risk`'s income
        /// assumption. Every one of these is a hand number that has only ever
        /// been moved as part of the whole term (`held=` scales all of them at
        /// once and reads +0.35 / −1.64 at 0.5 / 2.0), never on its own.
        pub corn_premium: f32,
        pub block_premium: f32,
        pub block_breadth: f32,
        pub skull_premium: f32,
        pub tile_premium: f32,
        pub corn_income: f32,
        /// How many blocks short a face-up monument may be and still be
        /// credited. `monument_outlook` is inert (F4, F18) and this is the gate
        /// most likely to be why.
        pub monument_gate: i32,
        /// A flat bonus for the player to move. `heuristic` cannot see the
        /// `Phase`, but it *can* see `g.current`, so this is the one correction
        /// for F17a's root-side under-statement that does not touch any
        /// within-turn ranking: every sibling of a node shares it, and only a
        /// commit edge — which hands the move on — sees it change.
        pub initiative: f32,
        /// The cap on how much held corn earns `corn_premium`. Only ever moved
        /// with the premium itself, and the two trade off directly.
        pub liquid_corn: f32,
        /// `starvation_risk`'s discount on a shortfall that is still several
        /// rounds out: `(1 / (1 + rounds * urgency)).max(urgency_floor)`.
        pub urgency: f32,
        pub urgency_floor: f32,
        /// Scale on `engine_value`'s permanent-food-discount line, the corn a
        /// free worker or a worker discount saves over every food day left.
        /// Never swept; it becomes worth more once `starvation_risk` stops
        /// assuming income (F23).
        pub food_saving: f32,
        /// Per-space multipliers on the *gated* hand table, for the five spaces
        /// `--promise` says disagree most with the rest of the evaluator.
        /// Multiplying rather than replacing keeps every affordability gate,
        /// which F2 showed is most of what the table is worth.
        pub tik1: f32,
        pub tik3: f32,
        pub tik5: f32,
        pub uxm3: f32,
        pub chi: f32,
        /// Whole-gear multipliers on the *gated* hand table. `--promise` reads
        /// the table's error as one common factor plus outliers, but the common
        /// factor is not common: n-weighted `want/table` is 4.18 on Palenque,
        /// 3.27 Yaxchilan, 3.08 Uxmal, 2.03 Tikal, 1.52 Chichen against an
        /// all-board 2.33 (F26). `space_scale` cannot express that tilt and
        /// `Space::Calib` moves every space at once; these move one gear.
        pub g_pal: f32,
        pub g_yax: f32,
        pub g_tik: f32,
        pub g_uxm: f32,
        /// Weight on the food day *after* the next one in `starvation_risk`.
        ///
        /// The committed term looks at one food day and stops, so a player who
        /// can feed six workers this time and has nothing coming scores zero
        /// risk. That was defensible while the search saw one turn; a search
        /// that descends six turns already sees the next food day itself, and
        /// what it cannot see is the one after. Zero reproduces the committed
        /// term exactly.
        pub starve_next2: f32,
        /// Extra multiplier on the Palenque table for a player who cannot pay
        /// the *next* food bill out of the corn in hand.
        ///
        /// `pal` is a flat re-price of the corn gear and it peaks at 1.5 (F26c),
        /// but a corn space is not worth the same to everyone: after
        /// `CORN_INCOME_PER_ROUND = 0.0`, corn's value to a player who is short
        /// is the 3 points a head that `starvation_risk` is charging them, and
        /// to a player who is not it is 1/4 a point. A flat multiplier cannot
        /// say that; this is the gate-shaped version of the same correction.
        pub pal_need: f32,
        /// Three constants inside `board_position` that have never been swept:
        /// the flat part of the first-player space, the unused first-player
        /// tile's free calendar day, and the discount on a worker sitting on
        /// the top space of a gear (it is picked up next rotation whether its
        /// owner wants it or not, so half its space is what it is worth).
        /// Points subtracted from every entry of a gear whose action *charges*
        /// the player for its payout, floored at the ungated 0.1.
        ///
        /// F26d: `space_value` prices what a space hands over and never what it
        /// takes, so the gears that charge — Chichen a skull, Tikal blocks for
        /// research and building, Uxmal corn — are over-paid relative to
        /// Palenque and Yaxchilan, which charge nothing. A *shift* is the right
        /// shape for that and a scale is not: the cost is the same at the
        /// bottom of a gear as at the top, so it is the low spaces that are
        /// mispriced most, and those are exactly where `dead` is highest.
        pub chi_sub: f32,
        pub tik_sub: f32,
        pub uxm_sub: f32,
        pub fp_bonus: f32,
        pub skip_day: f32,
        pub top_worker: f32,
        /// The cap on `actions_each` in `engine_value`. At 2.6 rounds an action
        /// and a 27-day calendar the cap binds for the first eleven days, so
        /// this — not `ROUNDS_PER_ACTION`, which is collinear with
        /// `ACTION_VALUE` — is the knob that prices a worker's early-game
        /// throughput.
        pub action_cap: f32,
        /// The `lvl == 2` "one short of the top" bonus in `engine_value`.
        /// Never swept; it survived `RESEARCH_SCALE` untouched because it is
        /// outside the scaled sum.
        pub near_top: f32,
        /// The rounds over which `held_premium`'s spendability fades in:
        /// `(rounds_left / spend_fade).min(1.0)`. Never swept, and it gates
        /// every holding premium at once.
        pub spend_fade: f32,
        pub held_scale: f32,
        pub monument_scale: f32,
        pub starve_scale: f32,
        /// Post-multiplier on `liquidation`, the one `Components` term that had
        /// no scale knob. F34 predicts a *concrete* term — corn and skulls the
        /// player is already holding, priced at the rate the end-game will pay
        /// for them — should hold or want raising at depth, and that cannot be
        /// tested through `lc`/`sp`/`cp`, which also move `held_premium` and
        /// `temple_outlook`. This moves nothing but the term.
        pub liq_scale: f32,
        /// The three gates the hand table is missing: a Palenque space whose
        /// tiles are gone, a research space with no blocks to pay the advance,
        /// and the corn exchange with nothing to trade.
        pub gates: bool,
    }

    /// Must track `eval.rs` exactly — `--eqcheck` is the assertion, and every
    /// A/B here is meaningless the moment it drifts. Because this is a *copy*
    /// and not an import, landing a change in `eval.rs` does not move the
    /// baseline arm on its own; update it here in the same edit, then re-run
    /// `--eqcheck`. That is also the property that lets a run keep measuring
    /// against a pinned baseline while `eval.rs` is being changed underneath.
    ///
    /// `action_value`, `board_scale` and `temple_scale` were moved here from
    /// (1.2, 1.0, 1.0) when `docs/FINDINGS-eval.md` F11 landed the re-pricing,
    /// `research_scale` (0.5), `corn_income` (1.9) and `ceiling` (false) when
    /// F24 landed the second one, `g_pal` (1.0) and `chi` (1.0) when F28 landed
    /// `GEAR_SCALE`, and `pal_need` (0.0) and `board_scale` (0.5) when F29
    /// landed `HUNGRY_CORN` and re-fitted `BOARD_SCALE` on the new table. So
    /// the null is now `--ab 'board=0.9,av=0.2,temple=1.2,rs=0.05,ci=0.0,
    /// ceiling,pal=1.5,chi=0.7,pneed=0.25'` and must measure 0 against this.
    /// `board_scale` moved again (0.65 -> 0.8) with F35 and (0.8 -> 0.9) with
    /// F47, which also took `temple_scale` 1.4 -> 1.2. The pre-F47 evaluator —
    /// the one every number before this session was measured against — is
    /// `--base 'board=0.8,temple=1.4'`.
    /// The pre-F24 evaluator is `--base 'rs=0.5,ci=1.9,noceiling'`, the pre-F28
    /// one is `--base 'pal=1,chi=1,pneed=0,board=0.5'`, and the pre-F29 one is
    /// `--base 'pneed=0,board=0.5'`.
    pub const HEAD: V = V {
        g_pal: 1.5,
        chi: 0.7,
        pal_need: 0.25,
        action_value: 0.2,
        tempo: 0.52,
        building_value: 0.45,
        space: Space::Table,
        space_scale: 1.0,
        hand_worker: false,
        corn_depth: false,
        building_colour: false,
        contention: false,
        contend_temple: false,
        contend_monument: false,
        charge_placed: false,
        hand_lag: 1.0,
        board_scale: 0.9,
        engine_scale: 1.0,
        temple_scale: 1.2,
        t_near: 0.85,
        t_far: 0.55,
        t_half: 0.0,
        climb: 0.10,
        ceiling: true,
        reach: 99.0,
        hand_flat: 0.0,
        calib_alpha: 0.0,
        research_scale: 0.05,
        corn_premium: 0.10,
        block_premium: 0.55,
        block_breadth: 0.50,
        skull_premium: 2.2,
        tile_premium: 0.25,
        corn_income: 0.0,
        monument_gate: 4,
        initiative: 0.0,
        liquid_corn: 16.0,
        urgency: 0.25,
        urgency_floor: 0.3,
        food_saving: 1.0,
        tik1: 1.0,
        tik3: 1.0,
        tik5: 1.0,
        uxm3: 1.0,
        g_yax: 1.0,
        g_tik: 1.0,
        g_uxm: 1.0,
        starve_next2: 0.0,
        chi_sub: 0.0,
        tik_sub: 0.0,
        uxm_sub: 0.0,
        fp_bonus: 1.2,
        skip_day: 0.6,
        top_worker: 0.5,
        action_cap: 6.0,
        near_top: 0.4,
        spend_fade: 6.0,
        held_scale: 1.0,
        monument_scale: 1.0,
        starve_scale: 1.0,
        liq_scale: 1.0,
        gates: false,
    };

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Components {
        pub banked: f32,
        pub liquidation: f32,
        pub held: f32,
        pub temple: f32,
        pub engine: f32,
        pub board: f32,
        pub monument: f32,
        pub starvation: f32,
    }

    impl Components {
        pub const NAMES: [&'static str; 8] = [
            "banked", "liquidation", "held", "temple", "engine", "board", "monument", "starve",
        ];
        pub fn terms(&self) -> [f32; 8] {
            [
                self.banked,
                self.liquidation,
                self.held,
                self.temple,
                self.engine,
                self.board,
                self.monument,
                -self.starvation,
            ]
        }
        pub fn total(&self) -> f32 {
            self.terms().iter().sum()
        }
    }

    #[inline]
    pub fn heuristic(vr: &V, g: &GameState, p: PlayerId) -> f32 {
        components(vr, g, p).total()
    }

    pub fn components(vr: &V, g: &GameState, p: PlayerId) -> Components {
        let pl = &g.players[p.idx()];
        let mut c = Components {
            banked: pl.points as f32,
            ..Components::default()
        };
        if g.over {
            return c;
        }
        c.liquidation = liquidation(g, p) * vr.liq_scale;
        let rounds_left = (LAST_DAY.saturating_sub(g.day)) as f32;
        let horizon = rounds_left / LAST_DAY as f32;
        c.held = held_premium(vr, g, p, rounds_left) * vr.held_scale;
        c.temple = temple_outlook(vr, g, p) * vr.temple_scale;
        c.engine = engine_value(vr, g, p, rounds_left, horizon) * vr.engine_scale;
        c.board = board_position(vr, g, p, rounds_left) * vr.board_scale
            + hand_flat(vr, g, p, rounds_left);
        c.monument = monument_outlook(vr, g, p, horizon) * vr.monument_scale;
        if vr.initiative != 0.0 && g.current == p {
            c.banked += vr.initiative;
        }
        c.starvation = starvation_risk(vr, g, p) * vr.starve_scale;
        c
    }

    pub fn margin(vr: &V, g: &GameState, p: PlayerId) -> f32 {
        let mine = heuristic(vr, g, p);
        let best_other = PlayerId::ALL
            .iter()
            .filter(|&&q| q != p)
            .map(|&q| heuristic(vr, g, q))
            .fold(f32::NEG_INFINITY, f32::max);
        mine - best_other
    }

    fn liquidation(g: &GameState, p: PlayerId) -> f32 {
        let pl = &g.players[p.idx()];
        let mut v = (pl.total_corn() / 4) as f32;
        v += pl.get(Resource::Skull) as f32 * 3.0;
        for id in pl.monument_ids() {
            v += (mdef(id).score)(g, p) as f32;
        }
        v
    }

    fn held_premium(vr: &V, g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
        let pl = &g.players[p.idx()];
        let spendable = (rounds_left / vr.spend_fade).min(1.0);
        if spendable <= 0.0 {
            return 0.0;
        }

        let mut v = if vr.corn_depth {
            corn_value(g, p) * spendable
        } else {
            (pl.corn as f32).min(vr.liquid_corn) * vr.corn_premium * spendable
        };

        let counts = [
            pl.get(Resource::Wood) as f32,
            pl.get(Resource::Stone) as f32,
            pl.get(Resource::Gold) as f32,
        ];
        let total: f32 = counts.iter().sum();
        v += total.min(9.0) * vr.block_premium * spendable;
        let breadth = counts.iter().copied().fold(f32::INFINITY, f32::min);
        v += breadth.min(3.0) * vr.block_breadth * spendable;

        let chichen_left = 9u32.saturating_sub(g.chichen_filled.count_ones()) as f32;
        if chichen_left > 0.0 {
            let usable = (pl.get(Resource::Skull) as f32).min(chichen_left);
            v += usable * vr.skull_premium * spendable;
        }

        if g.face_up_monuments().next().is_some() {
            v += (pl.corn_tiles + pl.wood_tiles) as f32 * vr.tile_premium * spendable;
        }
        v
    }

    /// Corn priced by the *placement depth* it buys.
    ///
    /// A turn placing `k` workers on the lowest free spaces pays
    /// `sum(i + pos_i)`, so the marginal corn is worth a whole action while a
    /// player is short of one and nothing once they can already afford every
    /// worker they hold. Flat `CORN_PREMIUM` under a cap cannot say that.
    fn corn_value(g: &GameState, p: PlayerId) -> f32 {
        let pl = &g.players[p.idx()];
        let hand = g.available(p).count() as f32;
        let corn = pl.corn as f32;
        // Cheapest full placement of everything in hand: 0+1+..+(k-1) plus the
        // lowest free space on each of the k cheapest gears.
        let mut floors: Vec<f32> = Gear::ALL
            .iter()
            .filter_map(|&gr| g.lowest_free(gr).map(|q| q.0 as f32))
            .collect();
        floors.sort_by(f32::total_cmp);
        let k = (hand as usize).min(floors.len());
        let need: f32 = (0..k).map(|i| i as f32 + floors[i]).sum();
        // Below `need` a corn is worth a fraction of the action it unlocks;
        // above it, only what it liquidates for plus a small spending premium.
        let short = (need - corn).max(0.0).min(corn);
        short * 0.28 + (corn - short).min(LIQUID_CORN) * CORN_PREMIUM
    }

    fn temple_outlook(vr: &V, g: &GameState, p: PlayerId) -> f32 {
        let mut v = 0.0;
        // Confidence in a payout `d` days out. The committed rule is two
        // levels keyed on *which* remaining day it is; `t_half` keys it on how
        // far away the day actually is.
        let conf = |n: usize, d: u8| -> f32 {
            if vr.t_half > 0.0 {
                let gap = (d.saturating_sub(g.day)) as f32;
                vr.t_half / (vr.t_half + gap)
            } else if n == 0 {
                vr.t_near
            } else {
                vr.t_far
            }
        };
        for (n, &d) in POINT_DAYS.iter().filter(|&&d| d > g.day).enumerate() {
            let age = 1 + POINT_DAYS.iter().filter(|&&x| x < d).count() as u8;
            let pts = g.temple_points(p, age) as f32;
            v += pts * conf(n, d);
        }
        for (n, &dd) in RESOURCE_DAYS.iter().filter(|&&d| d > g.day).enumerate() {
            let mut haul = 0.0;
            for t in Temple::ALL {
                let step = g.temple_pos(p, t);
                for &(at, r) in TEMPLES[t.idx()].resources {
                    if step >= at {
                        haul += match r {
                            Resource::Skull => 3.0 + vr.skull_premium,
                            other => other.corn_value() as f32 / CORN_PER_POINT + vr.block_premium,
                        };
                    }
                }
            }
            v += haul * conf(n, dd);
        }

        let mut climb = 0.0;
        for t in Temple::ALL {
            let d = &TEMPLES[t.idx()];
            let step = g.temple_pos(p, t) as usize;
            // `temple_ceiling` is private to `state.rs`; this is the same rule.
            let top = (d.steps - 1) as usize;
            let ceiling = if vr.ceiling
                && PlayerId::ALL
                    .iter()
                    .any(|&q| q != p && g.temple_pos(q, t) as usize == top)
            {
                top - 1
            } else {
                top
            };
            if step + 1 <= ceiling {
                climb += (d.points[step + 1] - d.points[step]) as f32;
            }
        }
        v += climb * vr.climb;

        if vr.contention || vr.contend_temple {
            v += temple_contention(g, p);
        }
        v
    }

    /// What the *other three* players' standings do to a projected payout.
    ///
    /// `temple_points` already splits the majority prize, so the level is
    /// modelled; what is not is that the level is about to move. Sole
    /// occupancy of a step nobody else can reach before the next scoring day
    /// is worth more than a shared one, and being one step behind a rival on a
    /// track that pays a majority is worth less than the raw number says.
    fn temple_contention(g: &GameState, p: PlayerId) -> f32 {
        let next = POINT_DAYS.iter().copied().find(|&d| d > g.day);
        let Some(next) = next else { return 0.0 };
        let rounds = (next - g.day) as f32;
        // Two rounds is about one temple step for a player who is trying.
        let swing = (rounds / 2.0).min(3.0);
        let mut v = 0.0;
        for t in Temple::ALL {
            let mine = g.temple_pos(p, t) as f32;
            let best_other = PlayerId::ALL
                .iter()
                .filter(|&&q| q != p)
                .map(|&q| g.temple_pos(q, t) as f32)
                .fold(0.0, f32::max);
            let lead = mine - best_other;
            let d = &TEMPLES[t.idx()];
            let top = d.points[(d.steps - 1) as usize] as f32;
            // Uncontested at the top of a track pays; a lead inside the noise
            // of what an opponent can climb before the day is worth nothing.
            if lead > swing {
                v += top * 0.03;
            } else if lead < -swing {
                v -= top * 0.02;
            }
        }
        v
    }

    fn engine_value(vr: &V, g: &GameState, p: PlayerId, rounds_left: f32, horizon: f32) -> f32 {
        let pl = &g.players[p.idx()];
        let mut v = 0.0;

        let actions_each = (rounds_left / ROUNDS_PER_ACTION).min(vr.action_cap);
        let workers = g.n_unlocked(p) as f32;
        let mut acts = workers * actions_each;
        if vr.charge_placed {
            // The action a worker on a gear is about to take is priced by
            // `board_position`; paying it the generic rate as well counts it
            // twice, and that double count is exactly the amount by which a
            // placement improves the estimate for free.
            acts = (acts - g.on_board(p).count() as f32).max(0.0);
        }
        v += acts * vr.action_value;

        let food_days_left = RESOURCE_DAYS
            .iter()
            .chain(POINT_DAYS.iter())
            .filter(|&&d| d > g.day)
            .count() as f32;
        let saved =
            pl.free_workers as f32 * 2.0 + (pl.worker_discount as f32).min(2.0) * workers;
        v += saved * food_days_left / CORN_PER_POINT * vr.food_saving;

        let uses = (rounds_left / 3.0).min(7.0);
        for s in Science::ALL {
            let lvl = g.level(p, s);
            for l in 1..=lvl {
                v += research_step_value(s, l) * uses * vr.research_scale;
            }
            if lvl == 2 && horizon > 0.15 {
                v += vr.near_top;
            }
        }

        if vr.building_colour {
            v += building_residual(g, p);
        } else {
            v += pl.n_buildings() as f32 * vr.building_value;
        }
        v
    }

    /// A built card's residual, priced by the monuments that count it.
    ///
    /// `liquidation` already pays the exact score of every monument the player
    /// *owns*, so this is only about the ones still on the table: #2/#4/#5 pay
    /// 4 per building of their colour, #9 pays 2 per card of any colour, and a
    /// colour nothing on the table counts is worth nothing at all.
    fn building_residual(g: &GameState, p: PlayerId) -> f32 {
        let pl = &g.players[p.idx()];
        let mut per_colour = [0.0f32; 4];
        let mut flat = 0.0f32;
        for id in g.face_up_monuments() {
            let d = mdef(id);
            match d.id.0 {
                2 => per_colour[Color::Green as usize] += 4.0,
                4 => per_colour[Color::Red as usize] += 4.0,
                5 => per_colour[Color::Blue as usize] += 4.0,
                9 => flat += 2.0,
                _ => {}
            }
        }
        // A quarter of face value: the card has to be bought, and the row is
        // shared. `MONUMENT_SHARE` is 0.45 for a monument already within reach;
        // this is a card that may never be affordable.
        let mut v = 0.0;
        for id in pl.building_ids() {
            v += (per_colour[bdef(id).color as usize] + flat) * 0.11;
        }
        v
    }

    fn research_step_value(s: Science, level: u8) -> f32 {
        match (s, level) {
            (Science::Agriculture, 1) => 0.35,
            (Science::Agriculture, 2) => 0.45,
            (Science::Agriculture, 3) => 0.75,
            (Science::Extraction, 1) => 0.55,
            (Science::Extraction, 2) => 0.65,
            (Science::Extraction, 3) => 0.75,
            (Science::Architecture, 1) => 0.30,
            (Science::Architecture, 2) => 0.85,
            (Science::Architecture, 3) => 0.80,
            (Science::Theology, 1) => 0.45,
            (Science::Theology, 2) => 0.65,
            (Science::Theology, 3) => 0.90,
            _ => 0.0,
        }
    }

    /// What the workers still in hand are worth over the generic action
    /// `engine_value` already pays them.
    ///
    /// Counted only for the workers the player could actually put down: the
    /// n-th placement of a turn costs `n + space_cost` corn, so the cheapest
    /// possible k placements cost `k(k-1)/2`.
    fn hand_flat(vr: &V, g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
        if vr.hand_flat == 0.0 || rounds_left <= 0.0 {
            return 0.0;
        }
        let corn = g.players[p.idx()].corn as u32;
        let hand = g.available(p).count() as u32;
        let mut k = 0u32;
        while k < hand && k * (k + 1) / 2 <= corn {
            k += 1;
        }
        // The action still has to be taken, so the credit fades with the
        // calendar the same way `held_premium`'s does.
        k as f32 * vr.hand_flat * (rounds_left / 6.0).min(1.0)
    }

    fn board_position(vr: &V, g: &GameState, p: PlayerId, rounds_left: f32) -> f32 {
        let mut v = 0.0;
        if rounds_left <= 0.0 {
            return 0.0;
        }
        let price = match vr.space {
            // The tabulated variants never consult a `Pricer`, and building one
            // eagerly would put `derived_price`'s cost on every call for
            // nothing.
            Space::Table | Space::TableFlat | Space::TableDefer | Space::DerivedStatic => {
                Pricer::None
            }
            Space::Probed => Pricer::Probe(tzolkin::mcts::Gradient::new(g, p)),
            _ => Pricer::List(derived_price(g, p)),
        };

        for w in g.on_board(p) {
            let Some((gear, pos)) = g.loc(w).on_board() else {
                continue;
            };
            v += rider_worth(vr, g, p, gear, pos.0, rounds_left, &price);
        }

        // A worker still in hand will be placed, and the space it can be
        // placed on is as real a feature of the position as the space one is
        // standing on. Without this the estimate jumps by the whole space
        // value the moment a placement lands, which is a step change *inside*
        // a turn that the search reads as a reason to keep placing.
        if vr.hand_worker {
            let mut floors: Vec<(Gear, u8)> = Gear::ALL
                .iter()
                .filter_map(|&gr| g.lowest_free(gr).map(|q| (gr, q.0)))
                .collect();
            let corn = g.players[p.idx()].corn as i32;
            let mut spent = 0i32;
            let hand = g.available(p).count();
            for n in 0..hand {
                // The placement a player would actually make next: the best
                // space they can still pay for, at the price the n-th worker
                // of a turn costs.
                let mut best: Option<(f32, usize)> = None;
                for (i, &(gr, pos)) in floors.iter().enumerate() {
                    let cost = n as i32 + pos as i32;
                    if spent + cost > corn {
                        continue;
                    }
                    let worth = rider_worth(vr, g, p, gr, pos, rounds_left, &price)
                        - cost as f32 / CORN_PER_POINT
                        - vr.hand_lag * vr.tempo;
                    if best.map_or(true, |(b, _)| worth > b) {
                        best = Some((worth, i));
                    }
                }
                let Some((worth, i)) = best else { break };
                spent += n as i32 + floors[i].1 as i32;
                floors[i].1 += 1;
                v += worth;
            }
        }

        if let Some(w) = g.first_player_space {
            if w.owner() == p {
                v += g.accumulated_corn as f32 / CORN_PER_POINT + vr.fp_bonus;
            }
        }
        if g.players[p.idx()].may_skip_day && rounds_left > 2.0 {
            v += vr.skip_day;
        }
        v
    }

    /// What one worker sitting at `(gear, pos)` is worth: the best space it can
    /// still ride to, less a round of its own throughput per space ridden.
    /// `rider_worth` / `space_value` for the tabulated variants, so
    /// `--promise` can ask what the committed evaluator pays a worker without
    /// building a `Pricer` it will not consult.
    pub fn rider_worth_head(vr: &V, g: &GameState, p: PlayerId, gear: Gear, pos: u8, rl: f32) -> f32 {
        rider_worth(vr, g, p, gear, pos, rl, &Pricer::None)
    }
    pub fn space_value_head(vr: &V, g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
        space_value(vr, g, p, gear, pos, &Pricer::None)
    }

    fn rider_worth(
        vr: &V,
        g: &GameState,
        p: PlayerId,
        gear: Gear,
        pos: u8,
        rounds_left: f32,
        price: &Pricer,
    ) -> f32 {
        let last = gear.size() - 1;
        let ride = rounds_left.min(vr.reach) as u8;
        let reach = last.min(pos.saturating_add(ride));
        let sv = |j: u8| space_value(vr, g, p, gear, j, price);
        let mut worth = sv(pos);
        for j in (pos + 1)..=reach {
            let waited = (j - pos) as f32 * vr.tempo;
            worth = worth.max(sv(j) - waited);
        }
        if pos == last {
            worth *= vr.top_worker;
        }
        worth
    }

    // ---- the space table, hand-written and derived ----------------------

    /// A price per unit of everything an `Effect` hands out, in the points
    /// `heuristic` returns — the same idea as `mcts::Gradient`, but written
    /// from the evaluator's own constants rather than probed, because probing
    /// inside `board_position` would recurse.
    pub fn derived_price(g: &GameState, p: PlayerId) -> EffectPrice {
        let block = |r: Resource| r.corn_value() as f32 / CORN_PER_POINT + BLOCK_PREMIUM;
        EffectPrice {
            corn: 1.0 / CORN_PER_POINT + CORN_PREMIUM,
            res: [
                block(Resource::Wood),
                block(Resource::Stone),
                block(Resource::Gold),
                3.0 + SKULL_PREMIUM,
            ],
            points: 1.0,
            temple: std::array::from_fn(|i| {
                let d = &TEMPLES[i];
                let step = g.temple_pos(p, Temple::ALL[i]) as usize;
                if step + 1 < d.steps as usize {
                    // The step's own jump, plus the projected payout it moves.
                    (d.points[step + 1] - d.points[step]) as f32 * TEMPLE_CLIMB
                        + (d.points[step + 1] - d.points[step]) as f32 * TEMPLE_NEAR
                } else {
                    0.0
                }
            }),
            science: std::array::from_fn(|i| {
                let s = Science::ALL[i];
                let lvl = g.level(p, s);
                if lvl >= 3 {
                    0.0
                } else {
                    research_step_value(s, lvl + 1) * 4.0 * RESEARCH_SCALE
                }
            }),
            unlock_worker: 6.0 * 1.2 / 2.6 * 2.6,
            free_worker: 2.0 * 5.0 / CORN_PER_POINT,
            worker_discount: 4.0 * 5.0 / CORN_PER_POINT,
            palenque_tile: TILE_PREMIUM + 1.0,
            burn_wood: -1.0,
            fill_chichen: 0.0,
            build: 0.45 + 2.0,
            monument: 8.0,
        }
    }

    fn space_value(
        vr: &V,
        g: &GameState,
        p: PlayerId,
        gear: Gear,
        pos: u8,
        price: &Pricer,
    ) -> f32 {
        match (vr.space, price) {
            (Space::TableFlat, _) => flat_space_value(gear, pos),
            (Space::DerivedStatic, _) => {
                DERIVED_STATIC[gear as usize][(pos as usize).min(10)] * vr.space_scale
            }
            // The four spaces whose payoff is entirely deferred, at the
            // derived table's conditional mean for them (`--spacetab`).
            (Space::TableDefer, _) => match (gear, pos) {
                (Gear::Tikal, 1) => 0.08,
                (Gear::Tikal, 3) => 0.23,
                (Gear::Tikal, 5) => 2.27,
                (Gear::Uxmal, 1) => 0.91,
                _ => table_space_value(vr, g, p, gear, pos),
            },
            // The table's *ranking* replaced by the one `--promise` measured,
            // gates and all, because this multiplies the gated value rather
            // than replacing it (F2: the gates are most of the table's value).
            (Space::Calib, _) => {
                table_space_value(vr, g, p, gear, pos)
                    * RCAL[gear as usize][(pos as usize).min(10)].powf(vr.calib_alpha)
                    * vr.space_scale
            }
            (Space::Table, _) | (_, Pricer::None) => {
                let mut base =
                    table_space_value(vr, g, p, gear, pos) * space_mul_g(vr, g, p, gear, pos);
                // The action's price to the player, for the gears that have one.
                let sub = match gear {
                    Gear::Chichen => vr.chi_sub,
                    // Tikal 6/7 are the "any of the above" spaces and 0 is the
                    // entry space; only the five that really charge are moved.
                    Gear::Tikal if (1..=5).contains(&pos) => vr.tik_sub,
                    Gear::Uxmal if pos == 1 || pos == 4 => vr.uxm_sub,
                    _ => 0.0,
                };
                if sub > 0.0 && base > 0.1 {
                    base = (base - sub).max(0.1);
                }
                if vr.gates {
                    base.min(gate_ceiling(g, p, gear, pos))
                } else {
                    base
                }
            }
            (Space::DerivedFloor, pr) => {
                let cs = tzolkin::spaces::choices_at(g, p, gear, Pos(pos));
                if cs.iter().all(|c| c.is_skip()) {
                    table_space_value(vr, g, p, gear, pos)
                } else {
                    cs.iter().map(|c| pr.of(c)).fold(0.0, f32::max) * vr.space_scale
                }
            }
            (_, pr) => tzolkin::spaces::choices_at(g, p, gear, Pos(pos))
                .iter()
                .map(|c| pr.of(c))
                .fold(0.0, f32::max)
                * vr.space_scale,
        }
    }

    /// Where a derived space value's per-effect prices come from.
    pub enum Pricer {
        None,
        /// A price list written from the evaluator's own constants.
        List(EffectPrice),
        /// `eval::heuristic`'s local gradient at this position, which is what
        /// `mcts::EdgeOrder::Gradient` already orders edges on.
        Probe(tzolkin::mcts::Gradient),
    }

    impl Pricer {
        #[inline]
        pub fn of(&self, c: &tzolkin::effect::Choice) -> f32 {
            match self {
                Pricer::None => 0.0,
                Pricer::List(pr) => pr.choice(c),
                // `Gradient` prices a `Step`, and only `Take` carries a choice.
                Pricer::Probe(gr) => gr.step(&tzolkin::phase::Step::Take(c.clone())),
            }
        }
    }

    /// Every choice the space generator offers, priced, best first.
    pub fn derived_space_value(g: &GameState, p: PlayerId, gear: Gear, pos: u8, pr: &Pricer) -> f32 {
        tzolkin::spaces::choices_at(g, p, gear, Pos(pos))
            .iter()
            .map(|c| pr.of(c))
            .fold(0.0, f32::max)
    }

    /// The three gates the hand table does not have, as a ceiling on its
    /// number.
    ///
    /// Taking the gates *off* the table costs 40.14 centred points at
    /// `mcts:256` (150 blocks) — far more than replacing every number in it —
    /// so the ones it is missing are worth looking for. These are the three
    /// places where `spaces::choices_at` returns nothing but `skip` and the
    /// table still pays: an exhausted Palenque stack, a research space the
    /// player cannot pay the blocks for (levels 1/2/3 cost 1/2/3 blocks,
    /// `options::recurse`), and the corn exchange with nothing to trade.
    fn gate_ceiling(g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
        let pl = &g.players[p.idx()];
        match (gear, pos) {
            (Gear::Palenque, 2) => {
                if g.palenque[2].corn > 0 || g.irrigation(p) {
                    f32::INFINITY
                } else {
                    0.2
                }
            }
            (Gear::Palenque, 3..=5) => {
                let st = g.palenque[pos as usize];
                if st.wood > 0 || st.corn_showing() || g.irrigation(p) {
                    f32::INFINITY
                } else {
                    0.2
                }
            }
            (Gear::Tikal, 1) | (Gear::Tikal, 3) => {
                let cheapest = Science::ALL
                    .iter()
                    .map(|&s| g.level(p, s) + 1)
                    .min()
                    .unwrap_or(4) as u32;
                if pl.n_blocks() >= cheapest {
                    f32::INFINITY
                } else {
                    0.3
                }
            }
            (Gear::Uxmal, 2) => {
                if pl.corn >= 2 || pl.n_blocks() >= 1 {
                    f32::INFINITY
                } else {
                    0.2
                }
            }
            _ => f32::INFINITY,
        }
    }

    /// `Space::DerivedStatic`'s table, exposed for the diagnostics.
    pub fn static_space_value(gear: Gear, pos: u8) -> f32 {
        DERIVED_STATIC[gear as usize][(pos as usize).min(10)]
    }

    /// The hand table with every gate taken off: the number the space pays
    /// when the player *can* use it, unconditionally.
    pub fn flat_space_value(gear: Gear, pos: u8) -> f32 {
        match gear {
            Gear::Palenque => [0.0, 0.9, 1.6, 1.6, 2.1, 2.6, 2.6, 2.6][(pos as usize).min(7)],
            Gear::Yaxchilan => {
                [0.0, 0.8, 1.3, 1.8, 3.0 + SKULL_PREMIUM * 0.5, 2.7, 3.4, 3.4][(pos as usize).min(7)]
            }
            Gear::Tikal => [0.0, 2.0, 3.2, 3.4, 5.2, 4.0, 5.2, 5.2][(pos as usize).min(7)],
            Gear::Uxmal => [0.0, 2.2, 1.1, 3.4, 3.0, 4.0, 4.0, 4.0][(pos as usize).min(7)],
            Gear::Chichen => {
                [0.0, 3.0, 3.8, 4.6, 5.4, 6.2, 6.8, 8.0, 9.0, 10.5, 10.5][(pos as usize).min(10)]
            }
        }
    }

    /// The targeted per-space multipliers, 1.0 unless a variant sets one.
    fn space_mul_g(vr: &V, g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
        let mut m = space_mul(vr, gear, pos);
        if vr.pal_need > 0.0 && gear == Gear::Palenque {
            // Short of the next food bill on the corn in hand. The same test
            // `starvation_risk` fires on, without its arithmetic.
            let pl = &g.players[p.idx()];
            if owed_at(g, p).is_some_and(|owed| owed > pl.corn as f32) {
                m *= 1.0 + vr.pal_need;
            }
        }
        m
    }

    fn space_mul(vr: &V, gear: Gear, pos: u8) -> f32 {
        let per_gear = match gear {
            Gear::Palenque => vr.g_pal,
            Gear::Yaxchilan => vr.g_yax,
            Gear::Tikal => vr.g_tik,
            Gear::Uxmal => vr.g_uxm,
            Gear::Chichen => vr.chi,
        };
        per_gear
            * match (gear, pos) {
                (Gear::Tikal, 1) => vr.tik1,
                (Gear::Tikal, 3) => vr.tik3,
                (Gear::Tikal, 5) => vr.tik5,
                (Gear::Uxmal, 3) => vr.uxm3,
                _ => 1.0,
            }
    }

    pub fn table_space_value(vr: &V, g: &GameState, p: PlayerId, gear: Gear, pos: u8) -> f32 {
        let pl = &g.players[p.idx()];
        match gear {
            Gear::Palenque => match pos {
                0 => 0.0,
                1 => 0.9,
                2 => 1.6,
                3 => 1.6,
                4 => 2.1,
                5 => 2.6,
                _ => 2.6,
            },
            Gear::Yaxchilan => match pos {
                0 => 0.0,
                1 => 0.8,
                2 => 1.3,
                3 => 1.8,
                4 => {
                    if g.skulls_remaining > 0 {
                        3.0 + vr.skull_premium * 0.5
                    } else {
                        0.2
                    }
                }
                5 => 2.7,
                _ => 3.4,
            },
            Gear::Tikal => {
                let can_build = pl.n_blocks() >= 2;
                match pos {
                    0 => 0.0,
                    1 => 2.0,
                    2 => {
                        if can_build {
                            3.2
                        } else {
                            0.4
                        }
                    }
                    3 => 3.4,
                    4 => {
                        if can_build {
                            5.2
                        } else {
                            0.6
                        }
                    }
                    5 => {
                        if pl.n_blocks() >= 1 {
                            4.0
                        } else {
                            0.3
                        }
                    }
                    _ => 5.2,
                }
            }
            Gear::Uxmal => match pos {
                0 => 0.0,
                1 => {
                    if pl.corn >= 3 {
                        2.2
                    } else {
                        0.3
                    }
                }
                2 => 1.1,
                3 => {
                    let locked = WORKERS_PER_PLAYER - g.n_unlocked(p);
                    if locked > 0 {
                        3.4
                    } else {
                        0.2
                    }
                }
                4 => {
                    if pl.corn >= 4 {
                        3.0
                    } else {
                        0.3
                    }
                }
                5 => {
                    if pl.corn >= 1 {
                        4.0
                    } else {
                        0.2
                    }
                }
                _ => 4.0,
            },
            Gear::Chichen => {
                if pl.get(Resource::Skull) == 0 {
                    return 0.1;
                }
                if pos >= 1 && pos <= 9 && g.chichen_is_full(Pos(pos)) {
                    return if g.foresight(p) { 3.0 } else { 0.1 };
                }
                match pos {
                    0 => 0.0,
                    1 => 3.0,
                    2 => 3.8,
                    3 => 4.6,
                    4 => 5.4,
                    5 => 6.2,
                    6 => 6.8,
                    7 => 8.0,
                    8 => 9.0,
                    9 => 10.5,
                    _ => 10.5,
                }
            }
        }
    }

    fn monument_outlook(vr: &V, g: &GameState, p: PlayerId, horizon: f32) -> f32 {
        if horizon <= 0.10 {
            return 0.0;
        }
        let pl = &g.players[p.idx()];
        let mut best: f32 = 0.0;

        for id in g.face_up_monuments() {
            let d = mdef(id);
            let short: i32 = Resource::ALL
                .iter()
                .map(|&r| (d.cost[r.idx()] as i32 - pl.get(r) as i32).max(0))
                .sum();
            if short > vr.monument_gate {
                continue;
            }
            let pays = (d.score)(g, p) as f32;
            if pays <= 0.0 {
                continue;
            }
            let mut w = pays * MONUMENT_SHARE / (1.0 + short as f32);
            if vr.contention || vr.contend_monument {
                // The row is shared and a monument is taken, not scored: an
                // opponent who can already afford it takes it first.
                let beaten = PlayerId::ALL.iter().any(|&q| {
                    q != p && {
                        let o = &g.players[q.idx()];
                        let os: i32 = Resource::ALL
                            .iter()
                            .map(|&r| (d.cost[r.idx()] as i32 - o.get(r) as i32).max(0))
                            .sum();
                        os < short
                    }
                });
                if beaten {
                    w *= 0.5;
                }
            }
            best = best.max(w);
        }
        best * horizon.min(1.0)
    }

    fn starvation_risk(vr: &V, g: &GameState, p: PlayerId) -> f32 {
        let mut days = RESOURCE_DAYS
            .iter()
            .chain(POINT_DAYS.iter())
            .copied()
            .filter(|&d| d > g.day)
            .collect::<Vec<_>>();
        days.sort_unstable();
        let Some(&next) = days.first() else {
            return 0.0;
        };
        let mut v = one_food_day(vr, g, p, next, 0.0);
        // The day after the next one, at `starve_next2`. A player who can pay
        // this bill and has nothing coming scores zero under the committed
        // term; the corn it will have spent on the first day is charged before
        // the second is priced.
        if vr.starve_next2 > 0.0 {
            if let Some(&after) = days.get(1) {
                let owed_now = owed_at(g, p).unwrap_or(0.0);
                v += vr.starve_next2 * one_food_day(vr, g, p, after, owed_now);
            }
        }
        v
    }

    /// The feeding bill, or `None` when the player owes nothing on any day.
    fn owed_at(g: &GameState, p: PlayerId) -> Option<f32> {
        let pl = &g.players[p.idx()];
        let mouths = g.n_unlocked(p);
        let free = (pl.free_workers as usize).min(mouths);
        let each = 2u8.saturating_sub(pl.worker_discount);
        if each == 0 {
            return None;
        }
        Some((mouths - free) as f32 * each as f32)
    }

    /// Points expected to be lost on the food day at `day`, with `spent` corn
    /// already committed to earlier days.
    fn one_food_day(vr: &V, g: &GameState, p: PlayerId, day: u8, spent: f32) -> f32 {
        let pl = &g.players[p.idx()];
        let each = 2u8.saturating_sub(pl.worker_discount);
        let Some(owed) = owed_at(g, p) else {
            return 0.0;
        };
        let rounds = (day - g.day) as f32;
        let expected = (pl.corn as f32 - spent).max(0.0) + rounds * vr.corn_income;
        let short = owed - expected;
        if short <= 0.0 {
            return 0.0;
        }
        let unfed = short / each as f32;
        let urgency = (1.0 / (1.0 + rounds * vr.urgency)).max(vr.urgency_floor);
        unfed * STARVE_POINTS * urgency
    }
}

// =======================================================================
// Wiring a variant into the two agent kinds
// =======================================================================

/// `phase::Evaluator` over a variant. Byte-for-byte `HeuristicEvaluator` apart
/// from which `heuristic` it calls, so the two arms of a comparison differ in
/// the evaluator and in nothing else.
#[derive(Clone, Copy)]
struct VEval(v::V, &'static str);

impl Evaluator for VEval {
    fn evaluate(&self, s: &GameState, _ph: Phase, _t: PlayerId, n_edges: usize) -> Evaluation {
        let raw: Vec<f32> = PlayerId::ALL
            .iter()
            .map(|&q| v::heuristic(&self.0, s, q))
            .collect();
        let mean = raw.iter().sum::<f32>() / N_PLAYERS as f32;
        let value = std::array::from_fn(|i| ((raw[i] - mean) / 25.0).tanh());
        let p = if n_edges == 0 { 0.0 } else { 1.0 / n_edges as f32 };
        Evaluation {
            priors: vec![p; n_edges],
            value,
        }
    }
    fn name(&self) -> String {
        self.1.into()
    }
}

/// The named variants. `head` is the committed evaluator and is always the
/// baseline arm.
fn variant(name: &str) -> Option<v::V> {
    use v::Space;
    let h = v::HEAD;
    Some(match name {
        "head" => h,
        // ---- the derived-table decomposition ----
        "derived" => v::V { space: Space::Derived, ..h },
        "derived-probed" => v::V { space: Space::Probed, ..h },
        "flat" => v::V { space: Space::TableFlat, ..h },
        "defer" => v::V { space: Space::TableDefer, ..h },
        "derived-static" => v::V { space: Space::DerivedStatic, ..h },
        "derived-static-scaled" => v::V { space: Space::DerivedStatic, space_scale: 0.62, ..h },
        "charge" => v::V { charge_placed: true, ..h },
        "gates" => v::V { gates: true, ..h },
        "gates+charge" => v::V { gates: true, charge_placed: true, ..h },
        "hand2" => v::V { hand_worker: true, hand_lag: 2.0, ..h },
        "hand0" => v::V { hand_worker: true, hand_lag: 0.0, ..h },
        "derived-probed-scaled" => v::V { space: Space::Probed, space_scale: 0.55, ..h },
        "derived-floor" => v::V { space: Space::DerivedFloor, ..h },
        "derived-scaled" => v::V { space: Space::Derived, space_scale: 0.60, ..h },
        "derived-scaled-tempo" => v::V {
            space: Space::Derived,
            space_scale: 0.60,
            tempo: 0.52,
            ..h
        },
        "derived-floor-scaled" => v::V {
            space: Space::DerivedFloor,
            space_scale: 0.60,
            ..h
        },
        // ---- structural ideas ----
        "hand" => v::V { hand_worker: true, ..h },
        "corn" => v::V { corn_depth: true, ..h },
        "colour" => v::V { building_colour: true, ..h },
        "contend" => v::V { contention: true, ..h },
        "contend-temple" => v::V { contend_temple: true, ..h },
        "contend-monument" => v::V { contend_monument: true, ..h },
        "hand+corn" => v::V { hand_worker: true, corn_depth: true, ..h },
        _ => {
            // `av=1.2,tempo=0.52` style, for the joint sweep.
            // `charge,av=0.8,tempo=0.40`: bare words switch a structural
            // idea on, `k=v` sets a constant. Anything else is a typo, and a
            // typo must not silently measure HEAD against itself.
            let mut out = h;
            for part in name.split(',') {
                match part.split_once('=') {
                    Some((k, val)) => {
                        let f: f32 = val.parse().ok()?;
                        match k {
                            "av" => out.action_value = f,
                            "tempo" => out.tempo = f,
                            "bv" => out.building_value = f,
                            "scale" => out.space_scale = f,
                            "lag" => out.hand_lag = f,
                            "board" => out.board_scale = f,
                            "engine" => out.engine_scale = f,
                            "temple" => out.temple_scale = f,
                            "held" => out.held_scale = f,
                            "monu" => out.monument_scale = f,
                            "starve" => out.starve_scale = f,
                            "liq" => out.liq_scale = f,
                            "tnear" => out.t_near = f,
                            "tfar" => out.t_far = f,
                            "thalf" => out.t_half = f,
                            "climb" => out.climb = f,
                            "reach" => out.reach = f,
                            "handv" => out.hand_flat = f,
                            "rs" => out.research_scale = f,
                            "cp" => out.corn_premium = f,
                            "bp" => out.block_premium = f,
                            "bb" => out.block_breadth = f,
                            "sp" => out.skull_premium = f,
                            "tp" => out.tile_premium = f,
                            "ci" => out.corn_income = f,
                            "mgate" => out.monument_gate = f as i32,
                            "init" => out.initiative = f,
                            "lc" => out.liquid_corn = f,
                            "urg" => out.urgency = f,
                            "ufl" => out.urgency_floor = f,
                            "fs" => out.food_saving = f,
                            "tik1" => out.tik1 = f,
                            "tik3" => out.tik3 = f,
                            "tik5" => out.tik5 = f,
                            "uxm3" => out.uxm3 = f,
                            "chi" => out.chi = f,
                            "pal" => out.g_pal = f,
                            "yax" => out.g_yax = f,
                            "tik" => out.g_tik = f,
                            "uxm" => out.g_uxm = f,
                            "sn2" => out.starve_next2 = f,
                            "pneed" => out.pal_need = f,
                            "chisub" => out.chi_sub = f,
                            "tiksub" => out.tik_sub = f,
                            "uxmsub" => out.uxm_sub = f,
                            "fp" => out.fp_bonus = f,
                            "skipd" => out.skip_day = f,
                            "topw" => out.top_worker = f,
                            "acap" => out.action_cap = f,
                            "ntop" => out.near_top = f,
                            "spend" => out.spend_fade = f,
                            "calib" => {
                                out.space = Space::Calib;
                                out.calib_alpha = f;
                            }
                            _ => return None,
                        }
                    }
                    None => match part {
                        "charge" => out.charge_placed = true,
                        "contend" => out.contention = true,
                        "corn" => out.corn_depth = true,
                        "colour" => out.building_colour = true,
                        "gates" => out.gates = true,
                        "hand" => out.hand_worker = true,
                        "ceiling" => out.ceiling = true,
                        "noceiling" => out.ceiling = false,
                        _ => return None,
                    },
                }
            }
            out
        }
    })
}

// =======================================================================
// A corpus of real positions
// =======================================================================

/// Turn roots from `n` self-played games, with the seat to move.
fn corpus(n: u64, sims: usize) -> Vec<(GameState, PlayerId)> {
    (0..n)
        .into_par_iter()
        .flat_map(|seed| {
            let a = GreedyAgent {
                ev: VEval(v::HEAD, "head"),
                cands: Candidates::Sampled(sims),
                record: true,
            };
            let agents: [&dyn Agent; N_PLAYERS] = [&a, &a, &a, &a];
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed ^ 0xC0FFEE);
            let r = play_game(seed, &agents, &GameConfig::evaluation(), &mut rng);
            r.nodes
                .into_iter()
                .filter(|nd| nd.flags & tzolkin::record::flags::TURN_ROOT != 0 && !nd.state.over)
                .map(|nd| (nd.state, nd.turn))
                .collect::<Vec<_>>()
        })
        .collect()
}

// =======================================================================
// Diagnostics that need no games
// =======================================================================

/// `V::HEAD` must be `eval::heuristic` exactly, or nothing measured here says
/// anything about the committed file.
fn cmd_eqcheck(pos: &[(GameState, PlayerId)]) {
    let mut worst = 0.0f32;
    let mut n = 0usize;
    for (g, _) in pos {
        for p in PlayerId::ALL {
            let a = tzolkin::eval::heuristic(g, p);
            let b = v::heuristic(&v::HEAD, g, p);
            worst = worst.max((a - b).abs());
            n += 1;
        }
    }
    println!("eqcheck: {n} (position, seat) pairs, max |V::HEAD - eval::heuristic| = {worst:e}");
    assert!(worst < 1e-4, "V::HEAD is not the committed evaluator");
}

/// ns per `heuristic` call, and where the time goes.
///
/// Every variant is timed three times round-robin and the **minimum** is
/// reported: a single pass over a 4,000-state corpus is memory-bound and the
/// first arm measured pays for warming the cache for the ones after it.
fn cmd_cost(pos: &[(GameState, PlayerId)]) {
    use std::hint::black_box;
    let states: Vec<(GameState, PlayerId)> = pos.iter().copied().take(1024).collect();
    let reps = 5;

    let arms: Vec<(&str, Box<dyn Fn(&GameState, PlayerId) -> f32>)> = vec![
        ("eval::heuristic (HEAD)", Box::new(|g, p| tzolkin::eval::heuristic(g, p))),
        ("v::heuristic head", Box::new(|g, p| v::heuristic(&v::HEAD, g, p))),
        ("v::heuristic hand", {
            let x = variant("hand").unwrap();
            Box::new(move |g, p| v::heuristic(&x, g, p))
        }),
        ("v::heuristic corn", {
            let x = variant("corn").unwrap();
            Box::new(move |g, p| v::heuristic(&x, g, p))
        }),
        ("v::heuristic colour", {
            let x = variant("colour").unwrap();
            Box::new(move |g, p| v::heuristic(&x, g, p))
        }),
        ("v::heuristic contend", {
            let x = variant("contend").unwrap();
            Box::new(move |g, p| v::heuristic(&x, g, p))
        }),
        ("eval::margin (4 calls)", Box::new(|g, p| tzolkin::eval::margin(g, p))),
        // What `HeuristicEvaluator` actually does at one search node: four
        // calls, one per seat, then a squash.
        ("all four seats", Box::new(|g, _p| {
            PlayerId::ALL.iter().map(|&q| tzolkin::eval::heuristic(g, q)).sum()
        })),
        ("heuristic, seat 0 only", Box::new(|g, _p| tzolkin::eval::heuristic(g, PlayerId(0)))),
        // Four calls to the *same* seat, to separate "four calls" from "four
        // different players' data".
        ("heuristic x4, seat 0", Box::new(|g, _p| {
            (0..4).map(|_| tzolkin::eval::heuristic(g, PlayerId(0))).sum()
        })),
        // What the search actually pays per node.
        ("HeuristicEvaluator::evaluate", Box::new(|g, p| {
            use tzolkin::phase::HeuristicEvaluator;
            HeuristicEvaluator.evaluate(g, Phase::Mode, p, 4).value[0]
        })),
    ];

    // Minimum over many short bursts. The box runs other people's
    // measurement jobs, so a single long pass measures whoever else was on the
    // core; the minimum of 25 bursts of 5 passes is the arm's own cost.
    let mut best = vec![f64::INFINITY; arms.len()];
    for _round in 0..25 {
        for (i, (_, f)) in arms.iter().enumerate() {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for _ in 0..reps {
                for (g, p) in &states {
                    acc += black_box(f(black_box(g), black_box(*p)));
                }
            }
            black_box(acc);
            let ns = t.elapsed().as_secs_f64() * 1e9 / (reps * states.len()) as f64;
            best[i] = best[i].min(ns);
        }
    }

    println!("\n-- cost per call, {} positions x {reps} passes, min of 25 bursts --", states.len());
    for (i, (name, _)) in arms.iter().enumerate() {
        println!("  {name:<28} {:>9.1} ns/call", best[i]);
    }

    // `derived` is thousands of times slower, so it gets a tiny sample of its
    // own rather than dragging the round-robin out.
    {
        let x = variant("derived").unwrap();
        let small: Vec<_> = states.iter().take(200).collect();
        let mut ns = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for (g, p) in &small {
                acc += black_box(v::heuristic(&x, g, *p));
            }
            black_box(acc);
            ns = ns.min(t.elapsed().as_secs_f64() * 1e9 / small.len() as f64);
        }
        println!("  {:<28} {ns:>9.0} ns/call   ({:.0}x head)", "v::heuristic derived", ns / best[1]);
    }

    // The one-ply probe `mcts::Priors::OnePly` and `Gradient` are measured
    // against, for calibration with the 123 ns / 16 ns figures in `mcts.rs`.
    let mut probe_ns = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        let mut acc = 0.0f32;
        let mut n = 0usize;
        for (g, p) in states.iter().take(256) {
            let steps = tzolkin::tree::legal_steps(g, Phase::Beg, *p, 0);
            for s in &steps {
                let mut probe = *g;
                tzolkin::tree::apply_step(&mut probe, Phase::Beg, *p, 0, s);
                acc += tzolkin::eval::heuristic(&probe, *p);
                n += 1;
            }
        }
        black_box(acc);
        probe_ns = probe_ns.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);
    }
    let mut step_ns = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        let mut n = 0usize;
        for (g, p) in states.iter().take(256) {
            let steps = tzolkin::tree::legal_steps(g, Phase::Beg, *p, 0);
            for s in &steps {
                let mut probe = *g;
                tzolkin::tree::apply_step(&mut probe, Phase::Beg, *p, 0, s);
                black_box(&probe);
                n += 1;
            }
        }
        step_ns = step_ns.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);
    }
    println!("  {:<28} {probe_ns:>9.1} ns/edge   (apply_step + heuristic)", "one-ply probe");
    println!("  {:<28} {step_ns:>9.1} ns/edge   (the apply_step half alone)", "  of which apply_step");
}

/// The hand table against the derived ones, space by space, over real states.
///
/// Three columns of price, because the two derived tables fail differently:
/// `list` is a hand price list over `Effect`, `probe` is `mcts::Gradient` —
/// the evaluator's own marginal for one more of each axis, taken against this
/// very position. `zero%` is how often the generator offered nothing worth
/// anything, which is where the hand table's floor lives.
fn cmd_spacetab(pos: &[(GameState, PlayerId)]) {
    let take = 1500.min(pos.len());
    println!("\n-- space value: hand table vs derived, over {take} positions --");
    println!(
        "{:>10} {:>4} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>6}",
        "gear", "pos", "table", "list", "probe", "t(nz)", "l(nz)", "p(nz)", "zero%"
    );
    let mut grand = [0.0f64; 3];
    for gear in Gear::ALL {
        for q in 0..gear.size() {
            let (mut ts, mut ls, mut ps) = (0.0f64, 0.0f64, 0.0f64);
            let (mut tz, mut lz, mut pz) = (0.0f64, 0.0f64, 0.0f64);
            let (mut zero, mut n) = (0usize, 0usize);
            for (g, p) in pos.iter().take(take) {
                let list = v::Pricer::List(v::derived_price(g, *p));
                let probe = v::Pricer::Probe(tzolkin::mcts::Gradient::new(g, *p));
                let t = v::table_space_value(&v::HEAD, g, *p, gear, q) as f64;
                let l = v::derived_space_value(g, *p, gear, q, &list) as f64;
                let pr = v::derived_space_value(g, *p, gear, q, &probe) as f64;
                ts += t;
                ls += l;
                ps += pr;
                n += 1;
                if l <= 0.0 {
                    zero += 1;
                } else {
                    tz += t;
                    lz += l;
                    pz += pr;
                }
            }
            let nz = (n - zero).max(1) as f64;
            grand[0] += ts;
            grand[1] += ls;
            grand[2] += ps;
            println!(
                "{:>10} {:>4} | {:>7.2} {:>7.2} {:>7.2} | {:>7.2} {:>7.2} {:>7.2} | {:>5.0}%",
                format!("{gear:?}"),
                q,
                ts / n as f64,
                ls / n as f64,
                ps / n as f64,
                tz / nz,
                lz / nz,
                pz / nz,
                100.0 * zero as f64 / n as f64,
            );
        }
    }
    println!(
        "  overall  list/table = {:.3}   probe/table = {:.3}",
        grand[1] / grand[0],
        grand[2] / grand[0]
    );

    // Where the shape difference actually bites: `board_position` does not
    // read a space value, it takes `max_j(value(j) - wait)` over everything a
    // worker can still ride to. Two tables that disagree about *which* space
    // that is send the worker somewhere else.
    let mut same = 0usize;
    let mut tot = 0usize;
    let mut same_static = 0usize;
    for (g, p) in pos.iter().take(take) {
        let list = v::Pricer::List(v::derived_price(g, *p));
        for gear in Gear::ALL {
            for from in 0..gear.size() {
                let reach = gear.size() - 1;
                let pick = |f: &dyn Fn(u8) -> f32| -> u8 {
                    let mut best = (f32::NEG_INFINITY, from);
                    for j in from..=reach {
                        let x = f(j) - (j - from) as f32 * 0.52;
                        if x > best.0 {
                            best = (x, j);
                        }
                    }
                    best.1
                };
                let t = pick(&|j| v::table_space_value(&v::HEAD, g, *p, gear, j));
                let d = pick(&|j| v::derived_space_value(g, *p, gear, j, &list));
                let st = pick(&|j| v::static_space_value(gear, j));
                tot += 1;
                if t == d {
                    same += 1;
                }
                if t == st {
                    same_static += 1;
                }
            }
        }
    }
    println!(
        "  the space a worker is aiming at agrees with the hand table:  derived {:.1}%   derived-static {:.1}%   ({tot} (position, start space) pairs)",
        100.0 * same as f64 / tot as f64,
        100.0 * same_static as f64 / tot as f64,
    );
}


/// The size of each term, and how each term moves *within* a turn.
///
/// `--midturn` says HEAD's estimate climbs +2.70 above the turn it will finish
/// in by depth 5. This says which of the eight terms does the climbing, which
/// is what turns "the evaluator drifts mid-turn" into a line to change.
fn cmd_terms(pos: &[(GameState, PlayerId)], take: usize) {
    use tzolkin::phase::Phase;
    let names = v::Components::NAMES;

    println!("\n-- term size over {} turn roots x 4 seats --", pos.len().min(take));
    println!("{:>12} {:>9} {:>9} {:>9}", "term", "mean", "sd", "mean|x|");
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); 8];
    for (g, _) in pos.iter().take(take) {
        for q in PlayerId::ALL {
            let t = v::components(&v::HEAD, g, q).terms();
            for i in 0..8 {
                cols[i].push(t[i] as f64);
            }
        }
    }
    for i in 0..8 {
        let n = cols[i].len() as f64;
        let m = cols[i].iter().sum::<f64>() / n;
        let sd = (cols[i].iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let ma = cols[i].iter().map(|x| x.abs()).sum::<f64>() / n;
        println!("{:>12} {m:>9.3} {sd:>9.3} {ma:>9.3}", names[i]);
    }

    // The same greedy descent `--midturn` walks, but recording every term.
    println!("\n-- term at depth k, less the same term of the completed turn --");
    let rows: Vec<Vec<[f32; 8]>> = pos
        .par_iter()
        .take(take)
        .filter_map(|(g0, p)| {
            let mut g = *g0;
            let mut ph = Phase::Beg;
            let mut done = 0u8;
            let mut out = vec![v::components(&v::HEAD, &g, *p).terms()];
            for _ in 0..64 {
                let steps = tzolkin::tree::legal_steps(&g, ph, *p, done);
                if steps.is_empty() {
                    return None;
                }
                let mut best = (f32::NEG_INFINITY, 0usize);
                for (i, s) in steps.iter().enumerate() {
                    let mut probe = g;
                    tzolkin::tree::apply_step(&mut probe, ph, *p, done, s);
                    let sc = v::heuristic(&v::HEAD, &probe, *p);
                    if sc > best.0 {
                        best = (sc, i);
                    }
                }
                let mut next = g;
                match tzolkin::tree::step_within_turn(&mut next, ph, *p, done, &steps[best.1]) {
                    Some((p2, d2)) => {
                        g = next;
                        ph = p2;
                        done = d2;
                        out.push(v::components(&v::HEAD, &g, *p).terms());
                    }
                    None => {
                        out.push(v::components(&v::HEAD, &next, *p).terms());
                        return Some(out);
                    }
                }
            }
            None
        })
        .collect();
    print!("{:>6} {:>7}", "depth", "n");
    for n in names {
        print!(" {n:>9}");
    }
    println!(" {:>9}", "total");
    let maxd = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for d in 0..maxd.min(11) {
        let sel: Vec<&Vec<[f32; 8]>> = rows.iter().filter(|r| r.len() > d + 1).collect();
        if sel.len() < 20 {
            continue;
        }
        print!("{d:>6} {:>7}", sel.len());
        let mut tot = 0.0f64;
        for i in 0..8 {
            let m = sel
                .iter()
                .map(|r| (r[d][i] - r.last().unwrap()[i]) as f64)
                .sum::<f64>()
                / sel.len() as f64;
            tot += m;
            print!(" {m:>+9.3}");
        }
        println!(" {tot:>+9.3}");
    }
}

/// What `board_position` promises a standing worker, against what taking its
/// action actually delivers — measured in the evaluator's own points.
///
/// This is the calibration F2's derived table could not do. `derived` priced
/// the `Effect`s a space emits in isolation and lost the interaction the hand
/// number was carrying (−27.9). This prices the *whole evaluator's* change when
/// the action is really taken, so every interaction is included by
/// construction: `delivery = h(after the Take) − h(before) + promise`, the
/// `+ promise` because retrieving the worker deletes its board credit.
///
/// Reported per gear and position against the hand table's own entry, so a
/// systematically mis-priced space is one row rather than a sweep.
fn cmd_promise(pos: &[(GameState, PlayerId)], take: usize) {
    use tzolkin::phase::{ModeChoice, Step};
    let vr = v::HEAD;

    // (gear, pos) -> (n, sum promise, sum delivery, sum sv_now, n dead)
    type Cell = (usize, f64, f64, f64, usize);
    let rows: Vec<(usize, u8, f64, f64, f64, bool)> = pos
        .par_iter()
        .take(take)
        .flat_map_iter(|(g0, p)| {
            let mut out: Vec<(usize, u8, f64, f64, f64, bool)> = Vec::new();
            let mut g = *g0;
            // Beg is a real decision; take the "don't beg" edge so the position
            // is the one the turn started from.
            if tzolkin::tree::step_within_turn(&mut g, Phase::Beg, *p, 0, &Step::Beg(None))
                .is_none()
            {
                return out.into_iter();
            }
            let steps = tzolkin::tree::legal_steps(&g, Phase::Mode, *p, 0);
            if !steps.contains(&Step::Mode(ModeChoice::Retrieve)) {
                return out.into_iter();
            }
            let before = v::heuristic(&vr, &g, *p);
            for st in tzolkin::tree::legal_steps(&g, Phase::PickWorker, *p, 0) {
                let Step::PickWorker(w) = st else { continue };
                let Some((gear, wp)) = g.loc(w).on_board() else {
                    continue;
                };
                let ph = Phase::Take { worker: w };
                let mut best = f32::NEG_INFINITY;
                for t in tzolkin::tree::legal_steps(&g, ph, *p, 0) {
                    let mut probe = g;
                    if tzolkin::tree::step_within_turn(&mut probe, ph, *p, 0, &t).is_some() {
                        best = best.max(v::heuristic(&vr, &probe, *p));
                    }
                }
                if best == f32::NEG_INFINITY {
                    continue;
                }
                let rounds_left = (tzolkin::state::LAST_DAY.saturating_sub(g.day)) as f32;
                let promise =
                    v::rider_worth_head(&vr, &g, *p, gear, wp.0, rounds_left) * vr.board_scale;
                let sv_now = v::space_value_head(&vr, &g, *p, gear, wp.0);
                out.push((
                    gear as usize,
                    wp.0,
                    promise as f64,
                    (best - before) as f64 + promise as f64,
                    sv_now as f64,
                    // "Dead": once the worker's board credit is added back,
                    // taking this action leaves the estimate where it was — the
                    // space hands over something the evaluator does not price.
                    // A mean delivery near zero can mean every instance is
                    // small or that the positive and negative ones cancel;
                    // only this column separates them.
                    ((best - before) + promise).abs() < 0.05,
                ));
            }
            out.into_iter()
        })
        .collect();

    let mut cells: std::collections::BTreeMap<(usize, u8), Cell> = Default::default();
    let (mut np, mut sp, mut sd) = (0usize, 0.0f64, 0.0f64);
    for (gi, pp, pr, de, sv, dead) in &rows {
        let c = cells.entry((*gi, *pp)).or_default();
        c.0 += 1;
        c.1 += pr;
        c.2 += de;
        c.3 += sv;
        c.4 += *dead as usize;
        np += 1;
        sp += pr;
        sd += de;
    }
    println!(
        "\n=== promise vs delivery, {np} standing workers over {} turn roots ===\n\
         promise: BOARD_SCALE x rider_worth, what `board_position` pays the worker.\n\
         delivery: h(best Take) - h(before) + promise, what taking the action is worth.\n\
         table:   space_value(gear, pos) as the hand table has it, act-now.\n\
         want:    delivery / BOARD_SCALE -- the table entry that would be honest.",
        pos.len().min(take)
    );
    println!(
        "  overall promise {:.3}  delivery {:.3}  ratio {:.3}",
        sp / np as f64,
        sd / np as f64,
        sd / sp
    );
    println!(
        "{:>10} {:>4} {:>7} {:>9} {:>9} {:>8} {:>8} {:>6}",
        "gear", "pos", "n", "promise", "deliver", "table", "want", "dead"
    );
    for ((gi, pp), c) in &cells {
        if c.0 < 10 {
            continue;
        }
        let n = c.0 as f64;
        println!(
            "{:>10} {pp:>4} {:>7} {:>9.3} {:>9.3} {:>8.2} {:>8.2} {:>6.2}",
            format!("{:?}", Gear::ALL[*gi]),
            c.0,
            c.1 / n,
            c.2 / n,
            c.3 / n,
            (c.2 / n) / vr.board_scale as f64,
            c.4 as f64 / n
        );
    }
    println!("-- machine readable: gear pos n table want --");
    for ((gi, pp), c) in &cells {
        let n = c.0 as f64;
        println!("RATIO {gi} {pp} {} {:.5} {:.5}", c.0, c.3 / n, (c.2 / n) / vr.board_scale as f64);
    }
}

// =======================================================================
// The mid-turn question
// =======================================================================

/// What a completed turn is worth, and what it did.
#[derive(Clone, Copy, Default)]
struct Done {
    /// The mover's estimate at the instant `apply_move` would have left the
    /// state — the honest value of the finished turn.
    val: f32,
    placed: u8,
    took: u8,
}

/// The best completion of this node under `vr`, over every completion there is.
///
/// `budget` bounds the walk; a position that exhausts it is dropped rather than
/// reported from a prefix of traversal order, which is systematically "place
/// one worker low on Palenque" and is not a sample of anything.
fn best_completion(
    vr: &v::V,
    g: &GameState,
    ph: Phase,
    turn: PlayerId,
    done: u8,
    placed: u8,
    took: u8,
    budget: &mut i64,
) -> Option<Done> {
    let mut best: Option<Done> = None;
    for s in tzolkin::tree::legal_steps(g, ph, turn, done) {
        *budget -= 1;
        if *budget < 0 {
            return None;
        }
        let (dp, dt) = step_counts(&s);
        let mut next = *g;
        let cand = match tzolkin::tree::step_within_turn(&mut next, ph, turn, done, &s) {
            Some((ph2, d2)) => best_completion(
                vr, &next, ph2, turn, d2, placed + dp, took + dt, budget,
            )?,
            None => Done {
                val: v::heuristic(vr, &next, turn),
                placed: placed + dp,
                took: took + dt,
            },
        };
        if best.map_or(true, |b| cand.val > b.val) {
            best = Some(cand);
        }
    }
    best
}

fn step_counts(s: &tzolkin::phase::Step) -> (u8, u8) {
    use tzolkin::phase::Step;
    match s {
        Step::Place(_) | Step::Pity(_) => (1, 0),
        Step::Take(_) => (0, 1),
        _ => (0, 0),
    }
}

/// One greedy descent through a turn, choosing each sub-decision by `vr`'s
/// estimate of the child **as the search sees it** — `tree::apply_step`, so a
/// commit edge carries the refill, the handoff and any round end with it,
/// exactly as `Mcts` would see them.
///
/// `traj` collects `vr`'s own estimate at each depth *within* the turn.
/// `mode` reports the `ModeChoice` the descent took, because a placing turn
/// and a retrieving turn move the estimate in opposite directions and averaging
/// them together is what made F3's trajectory look non-monotone.
fn greedy_descent(
    vr: &v::V,
    g0: &GameState,
    turn: PlayerId,
    traj: &mut Vec<f32>,
    mode: &mut u8,
) -> Option<Done> {
    let mut g = *g0;
    let mut ph = Phase::Beg;
    let mut done = 0u8;
    let (mut placed, mut took) = (0u8, 0u8);
    traj.push(v::heuristic(vr, &g, turn));
    for _ in 0..64 {
        let steps = tzolkin::tree::legal_steps(&g, ph, turn, done);
        if steps.is_empty() {
            return None;
        }
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (i, s) in steps.iter().enumerate() {
            let mut probe = g;
            tzolkin::tree::apply_step(&mut probe, ph, turn, done, s);
            let sc = v::heuristic(vr, &probe, turn);
            if sc > best.0 {
                best = (sc, i);
            }
        }
        let (dp, dt) = step_counts(&steps[best.1]);
        if let tzolkin::phase::Step::Mode(m) = &steps[best.1] {
            *mode = match m {
                tzolkin::phase::ModeChoice::Place => 0,
                tzolkin::phase::ModeChoice::Retrieve => 1,
                tzolkin::phase::ModeChoice::Pity => 2,
            };
        }
        let mut next = g;
        match tzolkin::tree::step_within_turn(&mut next, ph, turn, done, &steps[best.1]) {
            Some((p2, d2)) => {
                g = next;
                ph = p2;
                done = d2;
                placed += dp;
                took += dt;
                traj.push(v::heuristic(vr, &g, turn));
            }
            None => {
                return Some(Done {
                    val: v::heuristic(vr, &next, turn),
                    placed: placed + dp,
                    took: took + dt,
                })
            }
        }
    }
    None
}

struct MidRow {
    /// Each variant's estimate at each depth within the turn.
    trajs: Vec<Vec<f32>>,
    /// HEAD's estimate at each depth within the turn, and at the end.
    traj: Vec<f32>,
    head_end: f32,
    /// `ModeChoice` HEAD's descent took: 0 place, 1 retrieve, 2 pity.
    mode: u8,
    /// Per variant: the best completion, the greedy descent, and what the
    /// greedy descent is worth *under HEAD*.
    best: Vec<Done>,
    got: Vec<Done>,
    got_head: Vec<f32>,
}

fn cmd_midturn(pos: &[(GameState, PlayerId)], budget: i64, take: usize, vars_arg: Option<String>) {
    // `;`-separated because a variant spec is itself comma-separated. The
    // default list is the one F3 was measured with; pass `--vars` to put an
    // old evaluator on the same binary as the committed one, which is the only
    // way to tell a landing apart from a platform move (F14).
    let owned: Vec<String> = match &vars_arg {
        Some(s) => s.split(';').map(|x| x.to_string()).collect(),
        None => [
            "head", "charge", "hand", "hand2", "corn", "contend", "flat", "derived-static",
            "derived",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect(),
    };
    let names: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    let vars: Vec<v::V> = names
        .iter()
        .map(|n| variant(n).unwrap_or_else(|| panic!("unknown variant {n}")))
        .collect();


    let attempted = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<MidRow> = pos
        .par_iter()
        .take(take)
        .filter_map(|(g, p)| {
            attempted.fetch_add(1, Ordering::Relaxed);
            let mut traj = Vec::new();
            let mut trajs = Vec::new();
            let mut best = Vec::new();
            let mut got = Vec::new();
            let mut got_head = Vec::new();
            let mut head_end = 0.0;
            let mut mode = 255u8;
            for (i, vr) in vars.iter().enumerate() {
                let mut b = budget;
                best.push(best_completion(vr, g, Phase::Beg, *p, 0, 0, 0, &mut b)?);
                let mut t = Vec::new();
                let mut m = 255u8;
                let d = greedy_descent(vr, g, *p, &mut t, &mut m)?;
                if i == 0 {
                    mode = m;
                }
                // The same turn, re-scored by HEAD, so every variant is judged
                // on one yardstick as well as on its own.
                got_head.push(replay_under_head(g, *p, vr));
                if i == 0 {
                    traj = t.clone();
                    head_end = d.val;
                }
                t.push(d.val);
                trajs.push(t);
                got.push(d);
            }
            Some(MidRow { trajs, traj, head_end, mode, best, got, got_head })
        })
        .collect();

    assert!(!out.is_empty(), "no position finished inside the walk budget");
    println!(
        "\n=== mid-turn behaviour ===\n{} of {} turn roots completed inside a {budget}-node walk",
        out.len(),
        attempted.load(Ordering::Relaxed)
    );

    // ---- 1. the trajectory, paired against the turn it ends on ----------
    println!(
        "\n-- HEAD's estimate at depth k, less its estimate of the completed turn --\n\
         (same turn on both sides, so this is the mid-turn *bias*, not selection)"
    );
    println!("{:>6} {:>7} {:>11} {:>8}", "depth", "n", "v(k)-v(end)", "sd");
    let maxd = out.iter().map(|r| r.traj.len()).max().unwrap_or(0);
    for d in 0..maxd.min(12) {
        let xs: Vec<f64> = out
            .iter()
            .filter_map(|r| r.traj.get(d).map(|&x| (x - r.head_end) as f64))
            .collect();
        if xs.len() < 20 {
            continue;
        }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        let sd = (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt();
        println!("{d:>6} {:>7} {m:>+11.3} {sd:>8.3}", xs.len());
    }
    let root: Vec<f64> = out.iter().map(|r| (r.traj[0] - r.head_end) as f64).collect();
    println!(
        "\n  turn root under-states the completed turn by {:+.3} points on average",
        -(root.iter().sum::<f64>() / root.len() as f64)
    );

    // ---- 1a. the same trajectory for every variant, on one binary --------
    //
    // The point of this table is F14's lesson: the only honest way to ask
    // "did the landing move the mid-turn bias" is to run the old constants and
    // the new ones through the same build, on the same corpus, in one process.
    println!(
        "\n-- v(k)-v(end) by variant, each on its own descent and its own scale --"
    );
    print!("{:>28}", "variant");
    for d in 0..maxd.min(11) {
        print!(" {:>7}", format!("d{d}"));
    }
    println!(" {:>8}", "drift");
    for (i, name) in names.iter().enumerate() {
        print!("{name:>28}");
        for d in 0..maxd.min(11) {
            let xs: Vec<f64> = out
                .iter()
                .filter(|r| r.trajs[i].len() > d + 1)
                .map(|r| (r.trajs[i][d] - r.trajs[i].last().unwrap()) as f64)
                .collect();
            if xs.len() < 20 {
                print!(" {:>7}", "-");
            } else {
                print!(" {:>+7.3}", xs.iter().sum::<f64>() / xs.len() as f64);
            }
        }
        let (mut ds, mut dn) = (0.0f64, 0usize);
        for r in &out {
            let t = &r.trajs[i];
            let end = *t.last().unwrap();
            for k in 3..t.len().saturating_sub(1) {
                ds += (t[k] - end) as f64;
                dn += 1;
            }
        }
        println!(" {:>+8.3}", if dn == 0 { 0.0 } else { ds / dn as f64 });
    }

    // ---- 1b. split by what the turn actually was -------------------------
    //
    // A placing turn adds board value it has not paid for yet and a retrieving
    // turn converts board value into banked points, so the two move the
    // estimate in opposite directions. Averaged together they cancel into a
    // shape that looks like a hump and is really two lines crossing.
    for (tag, want) in [("PLACE", 0u8), ("RETRIEVE", 1u8)] {
        let sel: Vec<&MidRow> = out.iter().filter(|r| r.mode == want).collect();
        if sel.len() < 20 {
            continue;
        }
        println!("\n-- HEAD, {tag} turns only ({} of {}) --", sel.len(), out.len());
        println!("{:>6} {:>7} {:>11} {:>8}", "depth", "n", "v(k)-v(end)", "sd");
        for d in 0..maxd.min(12) {
            let xs: Vec<f64> = sel
                .iter()
                .filter_map(|r| r.traj.get(d).map(|&x| (x - r.head_end) as f64))
                .collect();
            if xs.len() < 20 {
                continue;
            }
            let m = xs.iter().sum::<f64>() / xs.len() as f64;
            let sd =
                (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt();
            println!("{d:>6} {:>7} {m:>+11.3} {sd:>8.3}", xs.len());
        }
    }

    // ---- 2. does the mid-turn signal find the good completed turn? -------
    println!(
        "\n-- regret of descending on the mid-turn estimate --\n\
         self:  best completion under the variant, less what its own greedy descent reaches.\n\
         head:  the same two turns re-scored by HEAD, so every row shares a yardstick.\n\
         place/take: workers placed and retrieved, greedy descent vs best completion."
    );
    println!(
        "{:>15} {:>7} {:>7} {:>7} {:>7} {:>13} {:>13}",
        "variant", "drift", "self", "head", "exact", "place g/best", "take g/best"
    );
    for (i, name) in names.iter().enumerate() {
        let n = out.len() as f64;
        let selfr: f64 = out.iter().map(|r| (r.best[i].val - r.got[i].val) as f64).sum::<f64>() / n;
        let headr: f64 = out
            .iter()
            .map(|r| (r.best[0].val - r.got_head[i]) as f64)
            .sum::<f64>()
            / n;
        let exact = out
            .iter()
            .filter(|r| (r.best[i].val - r.got[i].val).abs() < 1e-3)
            .count() as f64
            / n;
        let gp: f64 = out.iter().map(|r| r.got[i].placed as f64).sum::<f64>() / n;
        let bp: f64 = out.iter().map(|r| r.best[i].placed as f64).sum::<f64>() / n;
        let gt: f64 = out.iter().map(|r| r.got[i].took as f64).sum::<f64>() / n;
        let bt: f64 = out.iter().map(|r| r.best[i].took as f64).sum::<f64>() / n;
        // Mid-turn drift: how far the estimate at depths 3.. sits above the
        // completed turn it leads to, on the variant's own scale. Zero is an
        // evaluator whose mid-turn reading means what its end-of-turn reading
        // means; positive is one that rewards not stopping.
        let mut ds = 0.0f64;
        let mut dn = 0usize;
        for r in &out {
            let t = &r.trajs[i];
            let end = *t.last().unwrap();
            for k in 3..t.len().saturating_sub(1) {
                ds += (t[k] - end) as f64;
                dn += 1;
            }
        }
        let drift = if dn == 0 { 0.0 } else { ds / dn as f64 };
        println!(
            "{name:>15} {drift:>+7.3} {selfr:>7.3} {headr:>7.3} {exact:>7.3} {:>6.2}/{:<6.2} {:>6.2}/{:<6.2}",
            gp, bp, gt, bt
        );
    }
}

/// Replay a variant's greedy descent and score the turn it lands on with HEAD.
fn replay_under_head(g0: &GameState, turn: PlayerId, vr: &v::V) -> f32 {
    let mut g = *g0;
    let mut ph = Phase::Beg;
    let mut done = 0u8;
    for _ in 0..64 {
        let steps = tzolkin::tree::legal_steps(&g, ph, turn, done);
        if steps.is_empty() {
            return v::heuristic(&v::HEAD, &g, turn);
        }
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (i, s) in steps.iter().enumerate() {
            let mut probe = g;
            tzolkin::tree::apply_step(&mut probe, ph, turn, done, s);
            let sc = v::heuristic(vr, &probe, turn);
            if sc > best.0 {
                best = (sc, i);
            }
        }
        let mut next = g;
        match tzolkin::tree::step_within_turn(&mut next, ph, turn, done, &steps[best.1]) {
            Some((p2, d2)) => {
                g = next;
                ph = p2;
                done = d2;
            }
            None => return v::heuristic(&v::HEAD, &next, turn),
        }
    }
    v::heuristic(&v::HEAD, &g, turn)
}

// =======================================================================
// Rotation blocks
// =======================================================================

#[derive(Clone, Copy)]
struct Out {
    centred: f64,
    vs_base: f64,
    win: f64,
    cand: f64,
    base: f64,
}

/// How the two arms are played.
#[derive(Clone, Copy)]
enum Kind {
    /// `GreedyAgent`, one ply: `full` scores every legal move, `k` samples.
    Greedy(Candidates),
    /// `Mcts` at a fixed simulation count — the same count on both sides, so
    /// the comparison is the evaluator and not the search effort.
    ///
    /// The second field is `MctsConfig::c_puct_init`, negative for "leave the
    /// default alone". It is here because the committed default (2.0) stops the
    /// descent after about one turn, and an evaluator measured against a search
    /// that cannot look ahead is not necessarily the one that helps a search
    /// that can — every term about the *future* is exactly where the two differ.
    /// `docs/OVERNIGHT.md`, `mcts.rs::MctsConfig::c_puct_init`.
    Search(u32, f32),
}

fn agent_for(kind: Kind, vr: v::V, name: &'static str, seed: u64) -> Box<dyn Agent> {
    match kind {
        Kind::Greedy(c) => Box::new(GreedyAgent {
            ev: VEval(vr, name),
            cands: c,
            record: false,
        }),
        Kind::Search(sims, cp) => {
            let mut cfg = tzolkin::mcts::MctsConfig::default();
            if cp >= 0.0 {
                cfg.c_puct_init = cp;
            }
            Box::new(tzolkin::record::SearchAgent::with_config(
                Arc::new(VEval(vr, name)) as Arc<dyn Evaluator>,
                sims,
                false,
                tzolkin::record::Exploration::Off,
                cfg,
                seed,
            ))
        }
    }
}

/// One rotation block: the same seed played once per seating.
///
/// **One agent instance per seat, not one per side.** With a shared instance
/// the candidate plays one seat and the baseline plays three, so their trees
/// and their draw sequences diverge — and then the null (the same variant on
/// both sides) is not zero, because the four games of a block stop being the
/// same game. Per-seat instances restore it: under the null every seating is
/// bit-identical and the block mean is exactly 0, which is what makes the
/// common-random-numbers design worth having.
fn play_block(seed: u64, kind: Kind, cand: v::V, base: v::V) -> Vec<Out> {
    let cfg = GameConfig::evaluation();
    let mut out = Vec::new();
    for c in 0..N_PLAYERS {
        let held: Vec<Box<dyn Agent>> = (0..N_PLAYERS)
            .map(|s| {
                if s == c {
                    agent_for(kind, cand, "cand", seed)
                } else {
                    agent_for(kind, base, "base", seed)
                }
            })
            .collect();
        let agents: [&dyn Agent; N_PLAYERS] = std::array::from_fn(|s| held[s].as_ref());
        // Common random numbers: every seating of a block plays from the *same*
        // play RNG, not `bin/arena`'s per-seating offset. `bin/arena` offsets
        // because it wants four different games from one deal; this wants a
        // matched pair, so that under the null the only thing left in the block
        // mean is what the change under test did.
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            seed.wrapping_mul(0x9E37_79B9),
        );
        let r = play_game(seed, &agents, &cfg, &mut rng);
        let sc: [f64; N_PLAYERS] = std::array::from_fn(|s| r.scores[s] as f64);
        let table = sc.iter().sum::<f64>() / N_PLAYERS as f64;
        let cand_score = sc[c];
        let base_score = (0..N_PLAYERS)
            .filter(|&s| s != c)
            .map(|s| sc[s])
            .sum::<f64>()
            / (N_PLAYERS - 1) as f64;
        out.push(Out {
            centred: cand_score - table,
            vs_base: cand_score - base_score,
            win: r.win_share[c] as f64,
            cand: cand_score,
            base: base_score,
        });
    }
    out
}

fn mean(v: &[Out], f: impl Fn(&Out) -> f64) -> f64 {
    v.iter().map(&f).sum::<f64>() / v.len() as f64
}

#[allow(clippy::too_many_arguments)]
fn cmd_ab(
    cand_name: &str,
    base_name: &str,
    kind: Kind,
    blocks: u64,
    seed0: u64,
    out: PathBuf,
    resume: bool,
    label: &str,
) {
    let cand_v = variant(cand_name).unwrap_or_else(|| panic!("unknown variant {cand_name:?}"));
    let base_v = variant(base_name).unwrap_or_else(|| panic!("unknown variant {base_name:?}"));

    let done: std::collections::HashSet<u64> = if resume && out.exists() {
        std::fs::read_to_string(&out)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                let at = l.find("\"seed\":")? + 7;
                l[at..].split(',').next()?.trim_end_matches('}').parse().ok()
            })
            .collect()
    } else {
        if !resume {
            let _ = std::fs::remove_file(&out);
        }
        Default::default()
    };

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let file = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .unwrap(),
    );
    let n_done = AtomicUsize::new(0);
    let started = Instant::now();
    eprintln!("evalab: {cand_name} vs {base_name}, {label}, {blocks} blocks x 4 games -> {out:?}");

    (0..blocks).into_par_iter().for_each(|b| {
        let seed = seed0 + b;
        if done.contains(&seed) || stop.load(Ordering::SeqCst) {
            return;
        }
        // The search seed is the block's, so two arms of one block search the
        // same way and every block is a different search.
        let games = play_block(seed, kind, cand_v, base_v);
        let line = format!(
            r#"{{"seed":{seed},"centred":{:.4},"vs_base":{:.4},"win":{:.4},"cand":{:.3},"base":{:.3}}}"#,
            mean(&games, |o| o.centred),
            mean(&games, |o| o.vs_base),
            mean(&games, |o| o.win),
            mean(&games, |o| o.cand),
            mean(&games, |o| o.base),
        );
        {
            let mut f = file.lock().unwrap();
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
        let n = n_done.fetch_add(1, Ordering::SeqCst) + 1;
        if n % 25 == 0 {
            eprintln!("  {n}/{blocks} blocks, {:.0}s", started.elapsed().as_secs_f64());
        }
    });

    report(&out, cand_name, base_name, label, started.elapsed().as_secs_f64());
}

fn read_col(l: &str, k: &str) -> Option<f64> {
    let at = l.find(&format!("\"{k}\":"))? + k.len() + 3;
    let r = &l[at..];
    let e = r
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(r.len());
    r[..e].parse().ok()
}

fn report(out: &PathBuf, cand: &str, base: &str, label: &str, secs: f64) {
    let (mut centred, mut vs_base, mut win, mut ca, mut ba) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for l in std::fs::read_to_string(out).unwrap_or_default().lines() {
        if let (Some(c), Some(v), Some(w), Some(a), Some(b)) = (
            read_col(l, "centred"),
            read_col(l, "vs_base"),
            read_col(l, "win"),
            read_col(l, "cand"),
            read_col(l, "base"),
        ) {
            centred.push(c);
            vs_base.push(v);
            win.push(w);
            ca.push(a);
            ba.push(b);
        }
    }
    let s = Summary::of(&centred);
    let v = Summary::of(&vs_base);
    let w = Summary::of(&win);
    println!(
        "\n{cand} vs {base}   [{label}]\n\
         {} blocks ({} games), {secs:.0}s\n\
         mean centred score  {:+.2}  95% CI [{:+.2}, {:+.2}]   (null 0)\n\
         mean vs baseline    {:+.2}  95% CI [{:+.2}, {:+.2}]   (null 0)\n\
         win rate            {:.3}  95% CI [{:.3}, {:.3}]   (null 0.25)\n\
         mean score          candidate {:.1}   baseline {:.1}\n\
         {}",
        s.n,
        s.n * N_PLAYERS,
        s.mean,
        s.mean - s.ci,
        s.mean + s.ci,
        v.mean,
        v.mean - v.ci,
        v.mean + v.ci,
        w.mean,
        w.mean - w.ci,
        w.mean + w.ci,
        Summary::of(&ca).mean,
        Summary::of(&ba).mean,
        if s.mean.abs() > s.ci {
            "DISTINGUISHABLE from zero at 95%."
        } else {
            "not distinguishable from zero: the interval covers 0."
        }
    );
}

// =======================================================================

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let get = |n: &str| {
        argv.iter()
            .position(|a| a == n)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let has = |n: &str| argv.iter().any(|a| a == n);

    let n_pos: u64 = get("--corpus").and_then(|v| v.parse().ok()).unwrap_or(40);
    let need_corpus =
        has("--eqcheck") || has("--cost") || has("--midturn") || has("--spacetab") || has("--terms")
            || has("--promise");
    let pos = if need_corpus {
        let t = Instant::now();
        let c = corpus(n_pos, 8);
        eprintln!(
            "corpus: {} turn roots from {n_pos} games, {:.1}s",
            c.len(),
            t.elapsed().as_secs_f64()
        );
        c
    } else {
        Vec::new()
    };

    if has("--eqcheck") {
        cmd_eqcheck(&pos);
    }
    if has("--cost") {
        cmd_cost(&pos);
    }
    if has("--spacetab") {
        cmd_spacetab(&pos);
    }
    if has("--terms") {
        let take: usize = get("--take").and_then(|v| v.parse().ok()).unwrap_or(4000);
        cmd_terms(&pos, take);
    }
    if has("--promise") {
        let take: usize = get("--take").and_then(|v| v.parse().ok()).unwrap_or(4000);
        cmd_promise(&pos, take);
    }
    if has("--midturn") {
        let budget: i64 = get("--budget").and_then(|v| v.parse().ok()).unwrap_or(60_000);
        let take: usize = get("--take").and_then(|v| v.parse().ok()).unwrap_or(2000);
        cmd_midturn(&pos, budget, take, get("--vars"));
    }

    if let Some(cand) = get("--ab") {
        let base = get("--base").unwrap_or_else(|| "head".into());
        let spec = get("--agent").unwrap_or_else(|| "greedy:full".into());
        let kind = match spec.split_once(':') {
            Some(("greedy", "full")) => Kind::Greedy(Candidates::All),
            Some(("greedy", k)) => Kind::Greedy(Candidates::Sampled(k.parse().unwrap())),
            // `mcts:N` or `mcts:N:cp=X`. Both arms get the same `cp`, so the
            // comparison stays the evaluator; what changes is how deep the
            // search that is reading it looks.
            Some(("mcts", rest)) => match rest.split_once(":cp=") {
                Some((n, cp)) => Kind::Search(n.parse().unwrap(), cp.parse().unwrap()),
                None => Kind::Search(rest.parse().unwrap(), -1.0),
            },
            _ => panic!("--agent is greedy:full, greedy:K, mcts:N or mcts:N:cp=X"),
        };
        let games: u64 = get("--games").and_then(|v| v.parse().ok()).unwrap_or(400);
        let blocks = games.div_ceil(N_PLAYERS as u64);
        let seed0: u64 = get("--seed").and_then(|v| v.parse().ok()).unwrap_or(3_000_000);
        let out = PathBuf::from(get("--out").unwrap_or_else(|| {
            format!("evalab-{}-{}.jsonl", cand.replace(['=', ','], "_"), spec.replace(':', ""))
        }));
        let label: &'static str = Box::leak(spec.into_boxed_str());
        cmd_ab(&cand, &base, kind, blocks, seed0, out, has("--resume"), label);
    }
}
