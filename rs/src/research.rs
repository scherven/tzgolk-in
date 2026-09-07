//! Queries over the four science tracks.
//!
//! These are read-only derivations from `GameState::research`. Every one of them
//! is called during *generation*, so its result gets baked into an `Effect`.

use crate::ids::*;
use crate::state::GameState;

impl GameState {
    /// Extra corn on a corn space, by that space's colour.
    ///
    /// The Go version tested level 1 before level 3 with a `>=` predicate, so
    /// the level-3 arm was unreachable and a maxed agriculture track paid 1
    /// corn instead of 3. Highest tier is tested first here.
    pub fn corn_bonus(&self, p: PlayerId, space: Color) -> i8 {
        match space {
            Color::Blue => i8::from(self.has_level(p, Science::Agriculture, 2)),
            Color::Green => {
                if self.has_level(p, Science::Agriculture, 3) {
                    3
                } else if self.has_level(p, Science::Agriculture, 1) {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Extra units when gathering a block. Skulls are never boosted; the Go
    /// version reached that answer only by asking for a level-4 that cannot
    /// exist, so it is stated outright here.
    pub fn resource_bonus(&self, p: PlayerId, r: Resource) -> i8 {
        let needed = match r {
            Resource::Wood => 1,
            Resource::Stone => 2,
            Resource::Gold => 3,
            Resource::Skull => return 0,
        };
        i8::from(self.has_level(p, Science::Extraction, needed))
    }

    /// Take corn from Palenque even with no tiles left.
    pub fn irrigation(&self, p: PlayerId) -> bool {
        self.has_level(p, Science::Agriculture, 2)
    }

    /// Use the next Chichen space up when yours is taken.
    pub fn foresight(&self, p: PlayerId) -> bool {
        self.has_level(p, Science::Theology, 1)
    }

    /// Pay a block alongside a skull for an extra temple step.
    pub fn devout(&self, p: PlayerId) -> bool {
        self.has_level(p, Science::Theology, 2)
    }

    /// Build for one block less.
    pub fn builder(&self, p: PlayerId) -> bool {
        self.has_level(p, Science::Architecture, 3)
    }

    /// The bonus for constructing a building, resolved now rather than at
    /// execution time.
    pub fn build_bonus(&self, p: PlayerId) -> crate::effect::Effects {
        let mut out = crate::effect::Effects::new();
        if self.has_level(p, Science::Architecture, 1) {
            out.push(crate::effect::Effect::Corn(1));
        }
        if self.has_level(p, Science::Architecture, 2) {
            out.push(crate::effect::Effect::Points(2));
        }
        out
    }
}
