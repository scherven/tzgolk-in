fn main() {
    for spec in std::env::args().skip(1) {
        match tzolkin::record::AgentSpec::parse(&spec, false) {
            Ok(s) => println!("{:<52} -> {}", spec, s.instance().name()),
            Err(e) => println!("{:<52} -> ERR {}", spec, e),
        }
    }
}
