//! TEMPORARY width probe. Delete.
use tzolkin::effect::{Choice, Effect};
use tzolkin::game::Game;
use tzolkin::ids::*;

fn n30(v: &[Choice]) -> usize {
    v.iter().filter(|c| c.0.contains(&Effect::Build(BuildingId(30)))).count()
}

#[test]
fn width() {
    let p = PlayerId(0);
    println!("{:>5} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10}",
             "corn", "bc_all", "bc_30", "cat_T2", "cat30_T2", "cfw_T2", "cfw30_T2");
    for (corn, arch) in [(0u8,0u8),(1,0),(2,0),(5,0),(5,1),(5,2),(5,3)] {
        let mut g = Game::new(42);
        g.state.buildings_up = [None; tzolkin::state::N_DISPLAY];
        g.state.buildings_up[0] = Some(BuildingId(30));
        g.state.research[p.idx()] = [0; 4];
        g.state.research[p.idx()][Science::Architecture.idx()] = arch;
        g.state.players[p.idx()].res = [4, 4, 4, 0];
        g.state.players[p.idx()].corn = corn;

        let bc = tzolkin::options::building_choices(&g.state, p, None, true, 1);
        let raw2 = tzolkin::spaces::raw_at(&g.state, p, Gear::Tikal, Pos(2));
        let cat2 = tzolkin::spaces::choices_at(&g.state, p, Gear::Tikal, Pos(2));
        let cfw2 = tzolkin::moves::choices_for_worker(&g.state, p, Gear::Tikal, Pos(2));
        let cat4 = tzolkin::spaces::choices_at(&g.state, p, Gear::Tikal, Pos(4));
        println!("{corn:>2}/a{arch} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10}   raw_T2={} cat_T4={}",
                 bc.len(), n30(&bc), cat2.len(), n30(&cat2), cfw2.len(), n30(&cfw2),
                 raw2.len(), cat4.len());
    }

    // The mirror's other caller is untouched: these must not move.
    let mut g = Game::new(42);
    g.state.players[p.idx()].corn = 20;
    g.state.players[p.idx()].res = [5, 5, 5, 2];
    for pos in [5u8, 6, 7] {
        println!("uxmal{pos}: raw={} cat={} cfw={}",
                 tzolkin::spaces::raw_at(&g.state, p, Gear::Uxmal, Pos(pos)).len(),
                 tzolkin::spaces::choices_at(&g.state, p, Gear::Uxmal, Pos(pos)).len(),
                 tzolkin::moves::choices_for_worker(&g.state, p, Gear::Uxmal, Pos(pos)).len());
    }
}
