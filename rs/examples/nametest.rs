fn main() {
    for spec in [
        "mcts:2048",
        "mcts:2048:heuristic:ord=prior",
        "mcts:2048:heuristic:ord=grad",
        "mcts:2048:heuristic:nocap",
        "mcts:2048:heuristic:wcap=512",
        "mcts:2048:heuristic:k=64",
        "mcts:2048:heuristic:priors=1ply",
        "mcts:2048:heuristic:priors=1ply,nocap,k=64",
        "mcts:2048:heuristic:wc=1.0,wa=0.75",
        "mcts:512", "mcts:8192",
    ] {
        match tzolkin::record::AgentSpec::parse(spec, false) {
            Ok(s) => println!("{:<44} -> {}", spec, s.instance().name()),
            Err(e) => println!("{:<44} -> ERR {}", spec, e),
        }
    }
}
