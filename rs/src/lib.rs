//! A Tzolk'in engine: base game only, no expansions.
//!
//! The design in one line: board spaces are *generators* (closures that answer
//! "what can this player do here, right now?"), and what they generate is
//! *data* (a closed 13-variant `Effect` vocabulary). Generation stays
//! expressive; execution stays comparable, hashable and rollback-free.

/// Bumped whenever a rule changes.
///
/// Every self-play record carries this. A rules change invalidates the learned
/// value function even when every tensor shape survives, so without a stamp the
/// whole replay buffer becomes suspect instead of just the records before the
/// change.
pub const RULES_VERSION: u32 = 1;

pub mod data;
pub mod effect;
pub mod encode;
pub mod eval;
pub mod ffi;
pub mod game;
pub mod ids;
pub mod invariants;
pub mod mcts;
pub mod moves;
pub mod net;
pub mod options;
pub mod phase;
pub mod plan;
pub mod record;
pub mod tree;
pub mod research;
pub mod search;
pub mod spaces;
pub mod state;
pub mod ui;

pub use effect::{Choice, Effect};
pub use game::Game;
pub use ids::*;
pub use phase::{Evaluation, Evaluator, Phase, Step};
pub use state::{GameState, Player, WorkerLoc};
