//! Temple track layout. Values transcribed from the Go `impl/temple.go`.

use crate::ids::Resource;

pub struct TempleDef {
    pub steps: u8,
    pub age1_prize: i16,
    pub age2_prize: i16,
    /// Points for standing on each step. Only the first `steps` entries are used.
    pub points: [i16; 9],
    /// Resources granted on a resource day for reaching this step or higher.
    pub resources: &'static [(u8, Resource)],
}

pub static TEMPLES: [TempleDef; 3] = [
    // Brown
    TempleDef {
        steps: 7,
        age1_prize: 6,
        age2_prize: 2,
        points: [-1, 0, 2, 4, 6, 7, 8, 0, 0],
        resources: &[(2, Resource::Stone), (4, Resource::Stone)],
    },
    // Yellow
    TempleDef {
        steps: 9,
        age1_prize: 2,
        age2_prize: 6,
        points: [-2, 0, 1, 2, 4, 6, 9, 12, 13],
        resources: &[(3, Resource::Gold), (5, Resource::Gold)],
    },
    // Green
    TempleDef {
        steps: 8,
        age1_prize: 4,
        age2_prize: 4,
        points: [-3, 0, 1, 3, 5, 7, 9, 10, 0],
        resources: &[(2, Resource::Wood), (4, Resource::Wood), (5, Resource::Skull)],
    },
];

/// Every player starts one step up from the bottom.
pub const STARTING_STEP: u8 = 1;
