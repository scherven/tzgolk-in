//! Every variant must print a different name.
//!
//! A naming bug cost this project a whole experiment once: `mcts_label`
//! omitted a flag, both sides of a race printed `mcts2048/heuristic`, and the
//! result table said an agent had beaten itself by nine points. `AgentSpec`
//! parses sugar (`quality`, `nocap`) that the label deliberately does *not*
//! print back as sugar -- it prints the flags the config actually holds -- so
//! this checks the round trip that matters: two specs that search differently
//! must not collide, and two that search identically must.
//!
//!     cargo run --release --example nametest            # the fixed list
//!     cargo run --release --example nametest -- SPEC... # anything else

use std::collections::HashMap;

const SPECS: &[&str] = &[
    "mcts:2048",
    "mcts:2048:heuristic:ord=prior",
    "mcts:2048:heuristic:ord=grad",
    "mcts:2048:heuristic:nocap",
    "mcts:2048:heuristic:wcap=512",
    "mcts:2048:heuristic:k=64",
    "mcts:2048:heuristic:priors=1ply",
    "mcts:2048:heuristic:priors=grad",
    "mcts:2048:heuristic:priors=mixed",
    "mcts:2048:heuristic:priors=mixed,pt=0.5,pmin=2",
    "mcts:2048:heuristic:priors=eval",
    "mcts:2048:heuristic:q0=0.5",
    "mcts:2048:heuristic:q0=1.0,qn=4",
    "mcts:2048:heuristic:q0=1.0,qn=4,cp=0.5",
    "mcts:2048:heuristic:pick=q",
    "mcts:2048:heuristic:pick=q,q0=1.0",
    "mcts:2048:heuristic:pmarg",
    "mcts:2048:heuristic:pmarg,pt=4",
    "mcts:2048:heuristic:pmarg,q0=1.0",
    "mcts:2048:heuristic:q0=1.0",
    "mcts:2048:heuristic:priors=mixed,q0=0.5",
    "mcts:2048:heuristic:cp=1.0",
    "mcts:2048:heuristic:fpu=0.4",
    "mcts:2048:heuristic:cp=1.0,fpu=0.4",
    "mcts:2048:heuristic:cpb=2000",
    "mcts:2048:heuristic:quality",
    "mcts:8192:heuristic:deep",
    "mcts:8192:heuristic:deeper",
    "mcts:8192:heuristic:cp=0.02,pmin=2",
    "mcts:32768:heuristic:deeper",
    "mcts:8192:heuristic:cp=0.02",
    "mcts:8192:deep",
    "mcts:1533:heuristic:quality",
    "mcts:2048:heuristic:priors=1ply,pt=1",
    "mcts:2048:heuristic:priors=1ply,nocap,k=64",
    "mcts:2048:heuristic:wc=1.0,wa=0.75",
    "mcts:512",
    "mcts:8192",
    "minimax:8:600000::greedy:capw=25",
];

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let specs: Vec<&str> = if argv.is_empty() {
        SPECS.to_vec()
    } else {
        argv.iter().map(String::as_str).collect()
    };

    let mut seen: HashMap<String, &str> = HashMap::new();
    let mut clashes = 0;
    for spec in specs {
        match tzolkin::record::AgentSpec::parse(spec, false) {
            Ok(s) => {
                let name = s.instance().name();
                let note = match seen.insert(name.clone(), spec) {
                    // Sugar and its expansion are the same agent, so the same
                    // name is the right answer; anything else is the bug above.
                    Some(prev) => {
                        clashes += 1;
                        format!("   <-- same name as {prev}")
                    }
                    None => String::new(),
                };
                println!("{spec:<44} -> {name}{note}");
            }
            Err(e) => println!("{spec:<44} -> ERR {e}"),
        }
    }
    if clashes > 0 {
        println!("\n{clashes} name collision(s); check each is a genuine alias.");
    }
}
