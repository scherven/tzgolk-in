//! Setup, round flow and scoring.

use crate::data::buildings::{age1_ids, age2_ids};
use crate::data::monuments::monument_ids;
use crate::data::temples::STARTING_STEP;
use crate::data::tiles::{tile_ids, TILES};
use crate::ids::*;
use crate::invariants::validate;
use crate::moves::{apply_move, check_move, legal_moves, sample_legal_move, Move};
use crate::state::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

pub struct Game {
    pub state: GameState,
    pub rng: StdRng,
    /// Human-readable trace, only appended to when `trace` is on.
    pub log: Vec<String>,
    pub trace: bool,
    /// Largest legal-move list seen this game, and where it happened.
    pub max_branching: usize,
    pub max_branching_day: u8,
}

impl Game {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut a1 = age1_ids();
        let mut a2 = age2_ids();
        let mut mons = monument_ids();
        let mut tiles = tile_ids();
        a1.shuffle(&mut rng);
        a2.shuffle(&mut rng);
        mons.shuffle(&mut rng);
        tiles.shuffle(&mut rng);

        let mut palenque = [TileStack::default(); 8];
        // The corn-only space, then the three jungle spaces.
        palenque[2] = TileStack { corn: 4, wood: 0 };
        for i in 3..=5 {
            palenque[i] = TileStack { corn: 4, wood: 4 };
        }

        let mut workers = [WorkerLoc::Locked; N_WORKERS];
        for p in PlayerId::ALL {
            for (n, w) in GameState::worker_ids(p).enumerate() {
                if n < STARTING_WORKERS {
                    workers[w.idx()] = WorkerLoc::Available;
                }
            }
        }

        let mut state = GameState {
            players: std::array::from_fn(|i| Player::new(Color::ALL[i])),
            workers,
            gears: [GearState::empty(); 5],
            temples: [[STARTING_STEP; N_PLAYERS]; 3],
            research: [[0; 4]; N_PLAYERS],
            palenque,
            chichen_filled: 0,
            age1: Deck::new(a1),
            age2: Deck::new(a2),
            monument_deck: Deck::new(mons),
            buildings_up: [None; N_DISPLAY],
            monuments_up: [None; N_DISPLAY],
            first_player_space: None,
            accumulated_corn: 0,
            skulls_remaining: N_SKULLS,
            current: PlayerId(0),
            first_player: PlayerId(0),
            age: 1,
            day: 0,
            over: false,
        };

        state.refill_buildings();
        for i in 0..N_DISPLAY {
            state.monuments_up[i] = state.monument_deck.draw().map(MonumentId);
        }

        // Each player is dealt four starting tiles and keeps two.
        //
        // Which two is a real decision; with no agent yet it is drawn at
        // random. It belongs in the search once there is one.
        let mut t = 0usize;
        for p in PlayerId::ALL {
            let dealt: [u8; 4] = std::array::from_fn(|i| tiles[t + i]);
            t += 4;
            let mut idx = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                idx.swap(i, rng.gen_range(0..=i));
            }
            for &k in &idx[..2] {
                for e in TILES[dealt[k] as usize] {
                    e.apply(&mut state, p);
                }
            }
        }

        Game {
            state,
            rng,
            log: Vec::new(),
            trace: false,
            max_branching: 0,
            max_branching_day: 0,
        }
    }

    fn note(&mut self, s: impl Into<String>) {
        if self.trace {
            self.log.push(s.into());
        }
    }

    // ---- flow ----------------------------------------------------------

    /// Play the whole game with uniformly random legal moves.
    pub fn run_random(&mut self) {
        while !self.state.over {
            self.play_round();
        }
    }

    /// Same, but validating every generated move and the whole state after
    /// every turn. Returns the first violation with the log leading up to it.
    pub fn run_checked(&mut self) -> Result<(), String> {
        let mut guard = 0;
        while !self.state.over {
            self.state.current = self.state.first_player;
            for _ in 0..N_PLAYERS {
                let p = self.state.current;
                let moves = legal_moves(&self.state, p);
                if moves.len() > self.max_branching {
                    self.max_branching = moves.len();
                    self.max_branching_day = self.state.day;
                }
                // "You cannot skip your turn" -- with the pity rule in place a
                // player always has something, so an empty list is a bug.
                let Some(m) = self.pick(&moves) else {
                    return Err(format!(
                        "day {}: {:?} has no legal move at all",
                        self.state.day, p
                    ));
                };
                check_move(&self.state, p, &m).map_err(|e| format!("day {}: {e}", self.state.day))?;
                self.play(p, &m);
                validate(&self.state).map_err(|e| format!("day {}: {e}", self.state.day))?;
                self.state.current = self.state.current.next(1);
            }

            self.end_round();
            validate(&self.state).map_err(|e| format!("day {} (after rotate): {e}", self.state.day))?;

            guard += 1;
            if guard > 200 {
                return Err("game did not terminate within 200 rounds".into());
            }
        }
        Ok(())
    }

    fn pick(&mut self, moves: &[Move]) -> Option<Move> {
        if moves.is_empty() {
            return None;
        }
        let i = self.rng.gen_range(0..moves.len());
        Some(moves[i].clone())
    }

    pub fn play_round(&mut self) {
        self.state.current = self.state.first_player;
        for _ in 0..N_PLAYERS {
            self.take_turn();
            self.state.current = self.state.current.next(1);
        }

        // The marker only moves when somebody claims the space; an unclaimed
        // first player space leaves the starting player where they are.
        self.end_round();
    }

    /// One turn using the sampling rollout policy: no enumeration.
    pub fn take_turn_sampled(&mut self) {
        let p = self.state.current;
        match sample_legal_move(&self.state, p, &mut self.rng) {
            Some(m) => self.play(p, &m),
            None => debug_assert!(false, "sampler found no move for {p:?}"),
        }
    }

    /// A full playout, for MCTS rollouts.
    pub fn run_sampled(&mut self) {
        while !self.state.over {
            self.state.current = self.state.first_player;
            for _ in 0..N_PLAYERS {
                self.take_turn_sampled();
                self.state.current = self.state.current.next(1);
            }
            self.end_round();
        }
    }

    pub fn take_turn(&mut self) {
        let p = self.state.current;
        let moves = legal_moves(&self.state, p);
        if moves.is_empty() {
            // You cannot skip your turn. With the pity rule in place there is
            // always something, so an empty list means generation is broken.
            debug_assert!(false, "no legal move for {p:?} on day {}", self.state.day);
            self.note(format!("[BUG] no legal move for {}", self.state.players[p.idx()].color));
            return;
        }
        let idx = self.rng.gen_range(0..moves.len());
        let m = moves[idx].clone();
        self.play(p, &m);
    }

    pub fn play(&mut self, p: PlayerId, m: &Move) {
        if self.trace {
            let c = self.state.players[p.idx()].color;
            self.log.push(format!("d{} {c}: {m}", self.state.day));
        }
        apply_move(&mut self.state, p, m);
        self.state.refill_buildings();
    }

    pub fn resolve_first_player(&mut self) -> Option<PlayerId> {
        let before = self.state.accumulated_corn;
        let claimer = self.state.resolve_first_player();
        if let Some(p) = claimer {
            let c = self.state.players[p.idx()].color;
            self.note(format!("{c} takes first player, +{before} corn"));
        }
        claimer
    }

    pub fn may_take_extra_day(&self, p: PlayerId) -> bool {
        self.state.may_take_extra_day(p)
    }

    /// Resolve the round end: hand over the first player space, let the claimer
    /// decide on the extra day, then advance the calendar once.
    pub fn end_round_public(&mut self) {
        self.end_round();
    }

    fn end_round(&mut self) {
        let claimer = self.resolve_first_player();
        let mut days = 1;
        if let Some(p) = claimer {
            if self.state.may_take_extra_day(p) && self.rng.gen_bool(0.5) {
                self.note("calendar advances an extra day");
                self.state.spend_extra_day(p);
                days = 2;
            }
        }
        let before = self.state.day;
        self.state.advance_days(days);
        if self.trace && self.state.day != before {
            if RESOURCE_DAYS.contains(&self.state.day) {
                self.log.push(format!("day {}: resources", self.state.day));
            } else if POINT_DAYS.contains(&self.state.day) {
                self.log.push(format!("day {}: points", self.state.day));
            }
        }
    }

    pub(crate) fn rotate(&mut self) {
        let day = self.state.day;
        self.state.advance_day();
        if self.state.day != day && self.trace {
            if RESOURCE_DAYS.contains(&self.state.day) {
                self.log.push(format!("day {}: resources", self.state.day));
            } else if POINT_DAYS.contains(&self.state.day) {
                self.log.push(format!("day {}: points", self.state.day));
            }
        }
    }

    pub fn food_day(&mut self) {
        self.state.food_day();
    }

    pub fn winners(&self) -> Vec<PlayerId> {
        self.state.winners()
    }

    pub fn scores(&self) -> [i16; N_PLAYERS] {
        self.state.scores()
    }

    /// Advance the calendar one day. For tests that need to reach a specific
    /// day without playing turns.
    pub fn rotate_for_test(&mut self) {
        self.rotate();
    }
}
