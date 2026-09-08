//! Terminal rendering for watching a game.
//!
//! Read-only: it draws live `GameState`, never mutates it. The move preview
//! works by applying a candidate to a copy of the state and diffing — which is
//! only cheap because the state is `Copy`.

use crate::data::buildings::def as bdef;
use crate::data::monuments::def as mdef;
use crate::data::temples::TEMPLES;
use crate::effect::Choice;
use crate::ids::*;
use crate::moves::{Move, MoveKind};
use crate::state::{GameState, WorkerLoc};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use std::time::Duration;

// Both preludes export a `Color`. An explicit import beats the globs.
use crate::ids::Color;

pub fn colour(c: Color) -> ratatui::style::Color {
    match c {
        Color::Red => ratatui::style::Color::LightRed,
        Color::Green => ratatui::style::Color::LightGreen,
        Color::Blue => ratatui::style::Color::LightBlue,
        Color::Yellow => ratatui::style::Color::LightYellow,
    }
}

fn dim() -> Style {
    Style::default().fg(ratatui::style::Color::DarkGray)
}

fn label() -> Style {
    Style::default().fg(ratatui::style::Color::Gray)
}

/// Where the move list comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MoveSource {
    /// Drawn from the `sample_legal_move` rollout policy. Cheap, and the only
    /// option that stays instant in a position with a six-figure move list.
    Sampled,
    /// The best of **every** legal move, by `eval::margin`. Nothing is capped
    /// and nothing is sampled.
    Full,
    /// The best of every legal move as the configured `--agent` ranks them —
    /// for `minimax`, that is the search's own backed-up values, not a one-ply
    /// score. Falls back to [`MoveSource::Full`] with no agent.
    Agent,
}

impl MoveSource {
    pub fn label(self) -> &'static str {
        match self {
            MoveSource::Sampled => "sampled",
            MoveSource::Full => "scored, every move",
            MoveSource::Agent => "agent's own ranking",
        }
    }

    /// The order `t` cycles through.
    pub fn next(self) -> MoveSource {
        match self {
            MoveSource::Sampled => MoveSource::Full,
            MoveSource::Full => MoveSource::Agent,
            MoveSource::Agent => MoveSource::Sampled,
        }
    }
}

/// What the number in the Moves column means.
///
/// `Ranking` has no field for this and four producers put three different
/// scales into it (see `docs/FINDINGS-tui.md` T1), so the only statement of the
/// scale is the prose the producer wrote into `Ranking::note`. This reads that
/// back. Every caller keeps the note on screen beside the number, so a wrong
/// inference is contradicted in view rather than quietly believed — and
/// [`ScoreUnit::Unstated`] exists so the panel can decline to name a unit it
/// cannot establish instead of guessing "points".
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScoreUnit {
    /// `eval::margin`: this player's projected final score less the best
    /// opponent's. Computed by the viewer itself, so this arm is never a guess.
    Margin,
    /// Share of the search's *finished* simulations that played this turn.
    VisitShare,
    /// An agent's own number on a scale it did not name — the evaluator value
    /// a `greedy:` agent returns is squashed into (-1, 1) and is emphatically
    /// not points, so the column stays unlabelled rather than mislabelled.
    Unstated,
}

impl ScoreUnit {
    pub fn infer(source: MoveSource, note: &str) -> ScoreUnit {
        match source {
            // The viewer built these lists itself out of `eval::margin`.
            MoveSource::Sampled | MoveSource::Full => ScoreUnit::Margin,
            // `SearchAgent::ranked_moves` and `Search::search` are the two
            // agent rankings whose notes say what they are; anything else
            // declines to claim.
            MoveSource::Agent if note.contains("visit share") => ScoreUnit::VisitShare,
            MoveSource::Agent if note.contains("root moves") => ScoreUnit::Margin,
            MoveSource::Agent => ScoreUnit::Unstated,
        }
    }

    /// Column heading. Six columns wide so the numbers under it line up.
    pub fn heading(self) -> &'static str {
        match self {
            ScoreUnit::Margin => "margin",
            ScoreUnit::VisitShare => "visits",
            ScoreUnit::Unstated => " agent",
        }
    }

    fn render(self, v: f32) -> String {
        match self {
            ScoreUnit::Margin => format!("{v:>+6.1}"),
            ScoreUnit::VisitShare => format!("{v:>5.1}%"),
            ScoreUnit::Unstated => format!("{v:>6.2}"),
        }
    }
}

/// A live search, as the screen needs to describe it.
///
/// Plain data on purpose: the thread, the channel and the agent handle all live
/// in `bin/tui.rs`, so this module stays a pure renderer and the size-sweep test
/// can build a thinking screen without starting anything.
#[derive(Clone)]
pub struct Thinking {
    /// What is running, in the present tense: "searching R's turn".
    pub what: String,
    /// One honest line of detail — a stage count, or what is already known.
    pub detail: String,
    pub elapsed: Duration,
}

impl Thinking {
    /// A spinner frame that advances with `elapsed`, so the screen is visibly
    /// alive even when the search has nothing new to report. Derived from the
    /// clock rather than a counter: a redraw the event loop skipped must not
    /// leave the animation frozen and imply the search stopped too.
    fn tick(&self) -> char {
        const FRAMES: [char; 8] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
        FRAMES[(self.elapsed.as_millis() / 110) as usize % FRAMES.len()]
    }
}

pub struct App {
    pub game: crate::game::Game,
    /// The agent driving `n` and autoplay, if one was named. `None` falls back
    /// to the rollout policy, which is what the viewer did before checkpoints
    /// could be loaded.
    ///
    /// `Arc`, not `Box`: `Agent` is `Send + Sync`, and the viewer hands a clone
    /// of this to a worker thread so a multi-second turn does not block the
    /// draw loop.
    pub agent: Option<std::sync::Arc<dyn crate::record::Agent>>,
    pub agent_name: String,
    /// Set while a search is running on the worker thread. The screen must
    /// never simply stop redrawing: at the champion's budget a turn is ~200 ms,
    /// and the deliverable may be run at a budget where it is seconds.
    pub thinking: Option<Thinking>,
    /// The sub-decisions of the last turn the agent played: label, what it
    /// chose, and the share of visits that went there. This is the search
    /// talking, not a heuristic ranking of it.
    pub last_decisions: Vec<(String, String, f32)>,
    /// The move list on show, best first, together with how many moves it was
    /// drawn from and what produced it.
    pub ranking: crate::eval::Ranking,
    pub selected: usize,
    pub source: MoveSource,
    pub autoplay: bool,
    pub status: String,
}

impl App {
    pub fn candidates(&self) -> &[(Move, f32)] {
        &self.ranking.moves
    }

    /// The shortlist as the panel shows it: one entry per distinct outcome,
    /// restatements folded. `selected` indexes *this*, not `ranking.moves`.
    pub fn rows(&self) -> Vec<Row> {
        fold_rows(&self.ranking.moves, &self.game.state, self.game.state.current)
    }

    pub fn selected_move(&self) -> Option<&Move> {
        let rows = self.rows();
        let i = rows.get(self.selected)?.pick;
        self.ranking.moves.get(i).map(|(m, _)| m)
    }

    pub fn unit(&self) -> ScoreUnit {
        ScoreUnit::infer(self.source, &self.ranking.note)
    }
}

// ---- folding restatements ----------------------------------------------

/// One line of the Moves panel: a distinct outcome, and every spelling of it.
pub struct Row {
    /// Index into `Ranking::moves` of the spelling that represents the group.
    pub pick: usize,
    /// The agent's number, summed across the spellings — the search split its
    /// visits between them, and the reader is being told how much went to *this
    /// outcome*.
    pub score: f32,
    /// One-ply `eval::margin` of the successor. Computed here, so unlike
    /// `score` its unit is known for certain; it is the second opinion that
    /// makes a disagreement with the search visible.
    pub margin: f32,
    /// How many spellings folded into this row. 1 means nothing was hidden.
    pub spellings: usize,
    pub text: String,
}

/// Whether two moves reach the same position, allowing for a `skip` pickup
/// having been written in a different place in the sequence.
///
/// `Move::same_effect` already collapses interchangeable *placements*, and it
/// is right not to collapse these: `retrieve w0[..] w1[..]` and the same with
/// `w2[skip]` inserted differ, because picking w2 up returns it to hand. What
/// it cannot see is that the three positions `w2[skip]` can occupy in that
/// sequence are one move written three ways.
///
/// `apply_move` walks a retrieval as `choice.apply(); retrieve_worker()`. A
/// `skip` applies nothing, and a `Choice` is fully-resolved data that never
/// reads the board (`effect.rs` module note), so sliding a `skip` past an
/// effectful pickup cannot change the state reached. The effectful pickups
/// therefore keep their order — a contested building or a Palenque tile makes
/// *that* order load-bearing — and only the idle ones are treated as a set.
///
/// Display only. Nothing here feeds generation, search or `same_effect`.
pub fn same_outcome(a: &Move, b: &Move) -> bool {
    match (&a.kind, &b.kind) {
        (MoveKind::Retrieve(x), MoveKind::Retrieve(y)) => {
            if a.beg != b.beg || a.corn_cost != b.corn_cost || x.len() != y.len() {
                return false;
            }
            let acted = |v: &crate::moves::Retrievals| -> Vec<(WorkerId, Choice)> {
                v.iter().filter(|(_, c)| !c.is_skip()).cloned().collect()
            };
            let mut ia: Vec<u8> = idle(x);
            let mut ib: Vec<u8> = idle(y);
            ia.sort_unstable();
            ib.sort_unstable();
            ia == ib && acted(x) == acted(y)
        }
        _ => a.same_effect(b),
    }
}

/// Workers a retrieval takes off the board without doing anything with them.
fn idle(v: &crate::moves::Retrievals) -> Vec<u8> {
    v.iter().filter(|(_, c)| c.is_skip()).map(|(w, _)| w.0).collect()
}

/// How the panel spells one move: the same text as `Move`'s `Display`, except
/// that the no-op pickups leave the body and become a suffix naming who came
/// back. That is the whole readability fix — the body is then what the move
/// actually *does*, and it is the same body for every row that does it.
pub fn row_text(m: &Move) -> String {
    let MoveKind::Retrieve(v) = &m.kind else {
        return m.to_string();
    };
    let mut s = String::new();
    if let Some(t) = m.beg {
        s.push_str(&format!("[beg {}] ", t.letter()));
    }
    s.push_str("retrieve");
    for (w, c) in v.iter().filter(|(_, c)| !c.is_skip()) {
        s.push_str(&format!(" w{}[{c}]", w.0));
    }
    let idle = idle(v);
    if !idle.is_empty() {
        let who: Vec<String> = idle.iter().map(|w| format!("w{w}")).collect();
        // "+ ... to hand" rather than "[skip]": the worker really does come
        // back, which is the entire difference between these rows.
        let plus = if v.len() == idle.len() { " " } else { " + " };
        s.push_str(&format!("{plus}{} to hand", who.join(",")));
    }
    s
}

/// Group a ranking into one row per distinct outcome, best first.
///
/// Order is preserved from the input, which is already best-first, so folding
/// cannot promote a row above one that outranked every one of its spellings.
pub fn fold_rows(moves: &[(Move, f32)], g: &GameState, p: PlayerId) -> Vec<Row> {
    let mut out: Vec<Row> = Vec::with_capacity(moves.len());
    for (i, (m, score)) in moves.iter().enumerate() {
        match out
            .iter_mut()
            .find(|r| same_outcome(&moves[r.pick].0, m))
        {
            Some(r) => {
                r.score += score;
                r.spellings += 1;
            }
            None => out.push(Row {
                pick: i,
                score: *score,
                margin: crate::eval::margin(&crate::eval::successor(g, p, m), p),
                spellings: 1,
                text: row_text(m),
            }),
        }
    }
    out
}

// ---- top-level layout ---------------------------------------------------

pub fn draw(f: &mut Frame, app: &App) {
    let root = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        // One line, above the pane it belongs to, saying what the numbers in
        // the Moves column mean. It is only one row because it was previously
        // zero and the panel was unreadable for want of it.
        Constraint::Length(1),
        Constraint::Length(8),
        Constraint::Length(1),
    ])
    .split(f.area());

    header(f, root[0], app);

    let body = Layout::horizontal([Constraint::Percentage(54), Constraint::Percentage(46)])
        .split(root[1]);

    let left = Layout::vertical([
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Min(5),
    ])
    .split(body[0]);
    gears(f, left[0], app);
    players(f, left[1], app);
    cards(f, left[2], app);

    let right = Layout::vertical([
        Constraint::Length(13),
        Constraint::Length(7),
        Constraint::Min(4),
    ])
    .split(body[1]);
    temples(f, right[0], app);
    research(f, right[1], app);
    moves(f, right[2], app);

    units(f, root[2], app);
    preview(f, root[3], app);
    help(f, root[4], app);
}

fn boxed(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(dim())
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(ratatui::style::Color::White),
        ))
}

// ---- panels -------------------------------------------------------------

fn header(f: &mut Frame, area: Rect, app: &App) {
    let g = &app.game.state;
    let cur = g.players[g.current.idx()].color;
    let first = g.players[g.first_player.idx()].color;

    let mut spans = vec![
        Span::styled("Tzolk'in", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("   day ", label()),
        Span::styled(
            format!("{}/{}", g.day, crate::state::LAST_DAY),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled("   age ", label()),
        Span::raw(g.age.to_string()),
        Span::styled("   turn ", label()),
        Span::styled(
            cur.to_string(),
            Style::default().fg(colour(cur)).add_modifier(Modifier::BOLD),
        ),
        Span::styled("   first ", label()),
        Span::styled(first.to_string(), Style::default().fg(colour(first))),
        Span::styled("   fp space ", label()),
    ];
    match g.first_player_space {
        Some(w) => {
            let c = g.players[w.owner().idx()].color;
            spans.push(Span::styled(c.to_string(), Style::default().fg(colour(c))));
        }
        None => spans.push(Span::styled("·", dim())),
    }
    spans.push(Span::styled("   pot ", label()));
    spans.push(Span::raw(format!("{} corn", g.accumulated_corn)));
    spans.push(Span::styled("   skull bank ", label()));
    spans.push(Span::raw(g.skulls_remaining.to_string()));
    if !app.agent_name.is_empty() {
        spans.push(Span::styled("   agent ", label()));
        spans.push(Span::styled(
            short_agent(&app.agent_name),
            Style::default().fg(ratatui::style::Color::Cyan),
        ));
    }
    // Visible from the top line, so a long search reads as working rather than
    // as a terminal that stopped responding.
    if let Some(t) = &app.thinking {
        spans.push(Span::styled(
            format!("   {} thinking {:.1}s", t.tick(), t.elapsed.as_secs_f64()),
            Style::default()
                .fg(ratatui::style::Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }

    // The calendar, one character per round.
    let mut cal = String::new();
    for d in 1..=crate::state::LAST_DAY {
        cal.push(if d == g.day {
            '#'
        } else if crate::state::POINT_DAYS.contains(&d) {
            'P'
        } else if crate::state::RESOURCE_DAYS.contains(&d) {
            'R'
        } else {
            '-'
        });
    }

    let text = vec![Line::from(spans), Line::from(vec![Span::styled(cal, label())])];
    f.render_widget(Paragraph::new(text).block(boxed("")), area);
}

fn gears(f: &mut Frame, area: Rect, app: &App) {
    let g = &app.game.state;
    let mut lines = Vec::new();

    for gear in Gear::ALL {
        let mut spans = vec![Span::styled(
            format!("{:<13}", gear.name()),
            Style::default().fg(ratatui::style::Color::White),
        )];
        for pos in 0..gear.size() {
            let p = Pos(pos);
            let last = pos == gear.size() - 1;
            match g.gears[gear.idx()].at(p) {
                Some(w) => {
                    let c = g.players[w.owner().idx()].color;
                    spans.push(Span::styled(
                        format!(" {c} "),
                        Style::default()
                            .fg(colour(c))
                            .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                    ));
                }
                None => {
                    // A used-up Chichen space, the mirror space, or a free space.
                    let (txt, st) = if gear == Gear::Chichen && g.chichen_is_full(p) {
                        (" x ".to_string(), dim())
                    } else if last && gear != Gear::Chichen {
                        (" * ".to_string(), label())
                    } else {
                        (format!("{pos:^3}"), dim())
                    };
                    spans.push(Span::styled(txt, st));
                }
            }
        }
        if gear == Gear::Palenque {
            spans.push(Span::styled("  tiles ", label()));
            for i in 2..=5usize {
                let t = g.palenque[i];
                spans.push(Span::styled(format!("{}/{} ", t.wood, t.corn), dim()));
            }
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(vec![Span::styled(
        "  * = any-action space · x = skull space used · tiles are wood/corn",
        dim(),
    )]));

    f.render_widget(Paragraph::new(lines).block(boxed("Gears")), area);
}

fn temples(f: &mut Frame, area: Rect, app: &App) {
    let g = &app.game.state;
    let tallest = TEMPLES.iter().map(|t| t.steps).max().unwrap_or(0);
    // Each temple column below is 11 wide, after a 3-wide step gutter.
    let mut head = String::from("   ");
    for name in ["Brown", "Yellow", "Green"] {
        head.push_str(&format!("{name:^11}"));
    }
    let mut lines = vec![Line::from(vec![Span::styled(head, label())])];

    for step in (0..tallest).rev() {
        let mut spans = vec![Span::styled(format!("{step:>2} "), dim())];
        for t in Temple::ALL {
            let d = &TEMPLES[t.idx()];
            if step >= d.steps {
                spans.push(Span::raw(" ".repeat(11)));
                continue;
            }
            // Resource icon for this step, if any.
            let res = d
                .resources
                .iter()
                .find(|(at, _)| *at == step)
                .map(|(_, r)| r.letter())
                .unwrap_or(' ');
            spans.push(Span::styled(format!("{res}"), label()));
            spans.push(Span::styled(format!("{:>3}", d.points[step as usize]), dim()));
            spans.push(Span::raw(" "));

            let mut here = String::new();
            for q in PlayerId::ALL {
                if g.temple_pos(q, t) == step {
                    here.push(g.players[q.idx()].color.letter());
                }
            }
            for _ in here.len()..4 {
                here.push(' ');
            }
            for ch in here.chars() {
                let c = Color::ALL.iter().find(|c| c.letter() == ch);
                match c {
                    Some(&c) => spans.push(Span::styled(
                        ch.to_string(),
                        Style::default().fg(colour(c)).add_modifier(Modifier::BOLD),
                    )),
                    None => spans.push(Span::raw(" ")),
                }
            }
            spans.push(Span::raw("  "));
        }
        lines.push(Line::from(spans));
    }

    f.render_widget(Paragraph::new(lines).block(boxed("Temples")), area);
}

fn research(f: &mut Frame, area: Rect, app: &App) {
    let g = &app.game.state;
    let names = ["Agriculture", "Extraction", "Architecture", "Theology"];
    let mut lines = vec![Line::from(vec![Span::styled(
        "                 0     1     2     3",
        label(),
    )])];

    for (i, s) in Science::ALL.iter().enumerate() {
        let mut spans = vec![Span::styled(format!("{:<14}", names[i]), label())];
        for lvl in 0..4u8 {
            let mut cell = String::new();
            for q in PlayerId::ALL {
                if g.level(q, *s) == lvl {
                    cell.push(g.players[q.idx()].color.letter());
                }
            }
            for _ in cell.len()..4 {
                cell.push(' ');
            }
            spans.push(Span::raw(" "));
            for ch in cell.chars() {
                match Color::ALL.iter().find(|c| c.letter() == ch) {
                    Some(&c) => spans.push(Span::styled(
                        ch.to_string(),
                        Style::default().fg(colour(c)).add_modifier(Modifier::BOLD),
                    )),
                    None => spans.push(Span::styled("·", dim())),
                }
            }
            spans.push(Span::raw(" "));
        }
        lines.push(Line::from(spans));
    }

    f.render_widget(Paragraph::new(lines).block(boxed("Research")), area);
}

fn players(f: &mut Frame, area: Rect, app: &App) {
    let g = &app.game.state;
    let mut lines = vec![Line::from(vec![Span::styled(
        "   corn   W  S  G  sk   pts   wk  free  disc  bld mon  tiles",
        label(),
    )])];

    for p in PlayerId::ALL {
        let pl = &g.players[p.idx()];
        let c = pl.color;
        let is_turn = g.current == p;
        let marker = if is_turn { ">" } else { " " };
        let avail = g.available(p).count();
        let board = g.on_board(p).count();

        lines.push(Line::from(vec![
            Span::styled(
                format!("{marker}{c} "),
                Style::default()
                    .fg(colour(c))
                    .add_modifier(if is_turn { Modifier::BOLD } else { Modifier::empty() }),
            ),
            Span::raw(format!("{:>4}  ", pl.corn)),
            Span::raw(format!(
                "{:>2} {:>2} {:>2} {:>3}  ",
                pl.get(Resource::Wood),
                pl.get(Resource::Stone),
                pl.get(Resource::Gold),
                pl.get(Resource::Skull)
            )),
            Span::styled(
                format!("{:>5}  ", pl.points),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{avail}+{board}   ")),
            Span::raw(format!("{:>2}    {:>2}   ", pl.free_workers, pl.worker_discount)),
            Span::raw(format!("{:>3} {:>3}  ", pl.n_buildings(), pl.n_monuments())),
            Span::styled(
                format!("{}c {}w", pl.corn_tiles, pl.wood_tiles),
                dim(),
            ),
        ]));
    }
    lines.push(Line::from(vec![Span::styled(
        "   wk = in hand + on gears",
        dim(),
    )]));

    f.render_widget(Paragraph::new(lines).block(boxed("Players")), area);
}

fn cards(f: &mut Frame, area: Rect, app: &App) {
    let g = &app.game.state;
    let mut lines = Vec::new();

    let mut spans = vec![Span::styled("buildings ", label())];
    for slot in &g.buildings_up {
        match slot {
            Some(id) => {
                let d = bdef(*id);
                spans.push(Span::styled(
                    format!("[{:>2} {}]", id.0, cost_str(d.cost)),
                    Style::default().fg(colour(d.color)),
                ));
            }
            None => spans.push(Span::styled("[     ]", dim())),
        }
        spans.push(Span::raw(" "));
    }
    lines.push(Line::from(spans));
    lines.push(Line::from(vec![Span::styled(
        format!(
            "          age1 deck {}   age2 deck {}",
            g.age1.remaining(),
            g.age2.remaining()
        ),
        dim(),
    )]));

    for row in 0..2 {
        let mut spans = vec![Span::styled(
            if row == 0 { "monuments " } else { "          " },
            label(),
        )];
        for slot in &g.monuments_up[row * 3..row * 3 + 3] {
            match slot {
                Some(id) => {
                    let d = mdef(*id);
                    spans.push(Span::styled(
                        format!("[{:>2} {}]", id.0, cost_wide(d.cost)),
                        Style::default().fg(colour(d.color)),
                    ));
                }
                None => spans.push(Span::styled("[       ]", dim())),
            }
            spans.push(Span::raw(" "));
        }
        lines.push(Line::from(spans));
    }

    f.render_widget(Paragraph::new(lines).block(boxed("Cards")), area);
}

/// Monument costs run to six blocks.
fn cost_wide(cost: Bundle) -> String {
    let mut s = String::new();
    for r in Resource::BLOCKS {
        for _ in 0..cost[r.idx()] {
            s.push(r.letter());
        }
    }
    if cost[Resource::Skull.idx()] > 0 {
        s.push('!');
    }
    while s.chars().count() < 6 {
        s.push(' ');
    }
    s
}

fn cost_str(cost: Bundle) -> String {
    let mut s = String::new();
    for r in Resource::BLOCKS {
        for _ in 0..cost[r.idx()] {
            s.push(r.letter());
        }
    }
    while s.chars().count() < 4 {
        s.push(' ');
    }
    s
}

fn moves(f: &mut Frame, area: Rect, app: &App) {
    let r = &app.ranking;
    let rows = app.rows();
    let unit = app.unit();
    // Folded rows, not raw ones: the count in the title has to be the count of
    // lines below it, or the panel is describing a different list.
    let shown = rows.len();
    let title = if r.total == 0 {
        format!("Moves — {}", app.source.label())
    } else if r.exhaustive {
        // The claim the panel is making: this shortlist came out of the whole
        // move space, not a sample of it. Say how big that space was.
        let restated = if r.distinct < r.total {
            format!(", {} distinct", r.distinct)
        } else {
            String::new()
        };
        format!(
            "Moves — top {shown} of every one of {}{restated} · {}",
            r.total,
            app.source.label()
        )
    } else if unit == ScoreUnit::VisitShare {
        // `total` here is turns the tree *reached*, not legal turns, and the
        // panel has to not imply otherwise. See `SearchAgent::ranked_moves`.
        format!("Moves — {shown} of {} turns the search reached", r.total)
    } else {
        format!("Moves — {shown} of {} · {}", r.total, app.source.label())
    };

    // The old panel truncated at a hard 58 columns, which cut the tail off
    // every row at 132x44 -- exactly the rows whose tails were the difference
    // between them. Budget from the real width instead.
    let inner = area.width.saturating_sub(2) as usize;
    let wide = inner >= 46;
    let gutter = if wide { 2 + 7 + 8 } else { 2 + 7 };
    let room = inner.saturating_sub(gutter).max(8);

    let mut items: Vec<ListItem> = Vec::with_capacity(rows.len() + 1);
    if !rows.is_empty() {
        // A header row inside the list rather than in the title: the title is
        // already carrying the provenance, and the units belong over the
        // numbers they label.
        let head = if wide {
            format!("  {:>6}  {:>6}  {}", unit.heading(), "1-ply", "move")
        } else {
            format!("  {:>6}  {}", unit.heading(), "move")
        };
        items.push(ListItem::new(Line::from(Span::styled(head, dim()))));
    }

    for (i, row) in rows.iter().enumerate() {
        let sel = i == app.selected;
        let mut text = row.text.clone();
        // The fold hides nothing: say how many spellings went into the row.
        if row.spellings > 1 {
            text.push_str(&format!("  x{}", row.spellings));
        }
        if text.chars().count() > room {
            text = text.chars().take(room.saturating_sub(3)).collect::<String>() + "...";
        }
        let mut spans = vec![
            Span::styled(
                if sel { "> " } else { "  " },
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                unit.render(row.score),
                Style::default().fg(ratatui::style::Color::Cyan),
            ),
        ];
        if wide {
            // The one number on this line whose scale is not in doubt: the
            // viewer computed it. Where it disagrees with the agent's column,
            // that gap is the lookahead, and it is the most interesting thing
            // on the screen.
            spans.push(Span::styled(
                format!("  {:>+6.1}", row.margin),
                dim(),
            ));
        }
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            text,
            if sel {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            },
        ));
        items.push(ListItem::new(Line::from(spans)));
    }

    let mut state = ListState::default();
    // The header occupies row 0, so the highlight is one below the selection.
    state.select(Some(app.selected + 1));
    f.render_stateful_widget(
        List::new(items)
            .block(boxed(&title))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED)),
        area,
        &mut state,
    );
}

/// One dim full-width line saying what the numbers above it mean.
///
/// The units used to live only in `Ranking::note`, which the status line
/// truncated off the right edge — so the single place the scale was written
/// down was the one place not on screen. This says it in the viewer's own words
/// first (short, and definitely true of the column it labels) and then quotes
/// the producer's note as the evidence, so the two can be checked against each
/// other.
fn units(f: &mut Frame, area: Rect, app: &App) {
    let mine = match app.unit() {
        ScoreUnit::VisitShare => {
            "visits = share of the search's finished simulations that played this turn"
        }
        ScoreUnit::Margin => "margin = projected final score less the best opponent's",
        ScoreUnit::Unstated => "agent = this agent's own scale, which it does not name",
    };
    // The label is already in the header; repeating it here costs 40 columns of
    // a line that is about to be truncated.
    let note = app
        .ranking
        .note
        .strip_prefix(&app.agent_name)
        .map(|s| s.trim_start_matches(" ·").trim())
        .unwrap_or(&app.ranking.note);
    let mut text = format!(" {mine} · 1-ply = eval::margin of the successor · {note}");
    let w = area.width as usize;
    if text.chars().count() > w {
        text = text.chars().take(w.saturating_sub(1)).collect();
    }
    f.render_widget(Paragraph::new(Span::styled(text, dim())), area);
}

fn preview(f: &mut Frame, area: Rect, app: &App) {
    if let Some(t) = &app.thinking {
        thinking(f, area, app, t);
        return;
    }

    // Both halves, not one or the other. The diff answers "what does this move
    // do"; the sub-decisions answer "how did the search arrive at it". The old
    // pane showed the second and, once an agent had moved, could never be made
    // to show the first again.
    let split = area.width >= 100 && !app.last_decisions.is_empty();
    let (l, r) = if split {
        let cols =
            Layout::horizontal([Constraint::Percentage(54), Constraint::Percentage(46)]).split(area);
        (cols[0], Some(cols[1]))
    } else {
        (area, None)
    };

    if let Some(r) = r {
        search_pane(f, r, app);
        why_pane(f, l, app);
    } else if app.last_decisions.is_empty() {
        why_pane(f, l, app);
    } else {
        search_pane(f, l, app);
    }
}

/// What the highlighted candidate does, and what the two columns said about it.
fn why_pane(f: &mut Frame, area: Rect, app: &App) {
    let rows = app.rows();
    let unit = app.unit();
    let lines = match (rows.get(app.selected), app.selected_move()) {
        (Some(row), Some(m)) => {
            let before = app.game.state;
            let mut after = before;
            crate::moves::apply_move(&mut after, before.current, m);
            let mut head = vec![
                Span::styled(unit.heading().trim().to_string(), label()),
                Span::styled(
                    format!(" {}", unit.render(row.score).trim()),
                    Style::default().fg(ratatui::style::Color::Cyan),
                ),
                Span::styled("   1-ply ", label()),
                Span::styled(format!("{:+.1}", row.margin), Style::default()),
            ];
            if let Some(top) = rows.first() {
                if app.selected > 0 {
                    // The comparison is the point of a shortlist: this row is
                    // being shown *because* the top one beat it.
                    head.push(Span::styled("   vs top ", label()));
                    head.push(Span::styled(
                        format!(
                            "{} / {:+.1}",
                            unit.render(top.score).trim(),
                            top.margin
                        ),
                        dim(),
                    ));
                }
            }
            if row.spellings > 1 {
                head.push(Span::styled(
                    format!("   {} spellings folded", row.spellings),
                    dim(),
                ));
            }
            let mut out = vec![
                Line::from(vec![Span::styled(
                    row.text.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                )]),
                Line::from(head),
            ];
            out.extend(diff_lines(&before, &after));
            out
        }
        _ => vec![Line::from(Span::styled("no candidate selected", dim()))],
    };

    f.render_widget(
        Paragraph::new(lines)
            .block(boxed("Why — what the highlighted move changes"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The search's own account of the turn it just played.
fn search_pane(f: &mut Frame, area: Rect, app: &App) {
    let short = short_agent(&app.agent_name);
    let n = app.last_decisions.len();
    let mut out = vec![Line::from(vec![Span::styled(
        // Forced sub-decisions carry no distribution and are not recorded, so
        // this is never the whole chain and must not read as if it were.
        format!("{short} — {n} sub-decisions it had a choice at"),
        Style::default().add_modifier(Modifier::BOLD),
    )])];
    // Room for the bar shrinks with the pane; at 46% of 80 columns there is
    // none, and a bar clipped mid-way misreports a share.
    let bar_w = (area.width as usize).saturating_sub(46).min(20);
    for (phase, chosen, share) in app.last_decisions.iter().take(area.height.saturating_sub(3) as usize) {
        let mut spans = vec![
            Span::styled(format!("{phase:<11}"), label()),
            Span::styled(
                format!("{:>4.0}% ", share * 100.0),
                Style::default().fg(ratatui::style::Color::Cyan),
            ),
        ];
        if bar_w > 0 {
            let bar = "#".repeat(((share * bar_w as f32) as usize).min(bar_w));
            spans.push(Span::styled(format!("{bar:<w$} ", w = bar_w), dim()));
        }
        spans.push(Span::raw(chosen.clone()));
        out.push(Line::from(spans));
    }
    if app.last_decisions.is_empty() {
        out.push(Line::from(Span::styled(
            "nothing recorded — this agent does not build a tree",
            dim(),
        )));
    }
    f.render_widget(
        Paragraph::new(out).block(boxed("Search — visit share per sub-decision")),
        area,
    );
}

/// The screen while a search is running.
///
/// Deliberately indeterminate: nothing here knows how many simulations have
/// finished (see `docs/FINDINGS-tui.md` T3), so there is no progress *fraction*
/// to draw and a bar that filled up would be inventing one. A sweep and a
/// ticking clock say "running" without claiming to say how far.
fn thinking(f: &mut Frame, area: Rect, app: &App, t: &Thinking) {
    let secs = t.elapsed.as_secs_f64();
    let w = (area.width as usize).saturating_sub(4).clamp(8, 100);
    // Position derives from the clock, so a slow frame shifts the marker
    // further rather than stalling it.
    let span = (w * 2).max(2);
    let at = (t.elapsed.as_millis() / 40) as usize % span;
    let at = if at < w { at } else { span - at - 1 };
    let mut sweep = String::new();
    for i in 0..w {
        sweep.push(if i.abs_diff(at) < 2 { '#' } else { '·' });
    }

    let out = vec![
        Line::from(vec![
            Span::styled(
                format!("{} {}", t.tick(), t.what),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("   {secs:.1}s"), Style::default().fg(ratatui::style::Color::Cyan)),
        ]),
        Line::from(Span::styled(t.detail.clone(), label())),
        Line::from(Span::styled(sweep, dim())),
        Line::from(Span::styled(
            "the board above is the position it is thinking about · q quits".to_string(),
            dim(),
        )),
    ];
    f.render_widget(
        Paragraph::new(out)
            .block(boxed(&format!("Thinking — {}", short_agent(&app.agent_name))))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn help(f: &mut Frame, area: Rect, app: &App) {
    let auto = if app.autoplay { "on" } else { "off" };
    // While a search runs the mutating keys are refused, so offering them would
    // be a lie about what the terminal will do.
    let text = match &app.thinking {
        Some(t) => format!(
            " {} {} · {:.1}s · j/k move · q quit    {}",
            t.tick(),
            t.what,
            t.elapsed.as_secs_f64(),
            app.status
        ),
        None => format!(
            " j/k move · enter play · n next turn · r redraw · t {} · a autoplay [{auto}] · q quit    {}",
            app.source.next().label(),
            app.status
        ),
    };
    f.render_widget(Paragraph::new(Span::styled(text, label())), area);
}

/// Turn an agent's decision nodes into the search panel's rows.
///
/// `Node::visits` stores `(edge index, visit count)`, not the step itself, so
/// the enumeration has to be regenerated to say what was chosen. That is also
/// why `done` rides along: at `PickWorker` the edge set depends on it, and a
/// list one edge short would mislabel everything after the gap.
///
/// Only about a quarter of turns carry nodes — the rest run a reduced search
/// budget and record nothing — so a caller wanting a populated panel should
/// keep playing until this returns something.
/// A displayable agent name. Checkpoint specs carry a whole path.
pub fn short_agent(name: &str) -> String {
    match name.rsplit_once('/') {
        Some((_, file)) => {
            let head = name.split('/').next().unwrap_or("");
            format!("{head}/{}", file.trim_end_matches(".safetensors"))
        }
        None => name.to_string(),
    }
}

/// What one edge out of a search node was, in the game's vocabulary.
///
/// `{step:?}` printed `Take(Choice([Res(Wood, -1), Res(Gold, -1), Build(...` —
/// the wrapper repeats the phase already in the left column, and the payload is
/// then truncated before it reaches the part a reader wants. `Choice` and
/// `Placement` both have a Display that says what the action *is*, so use it.
fn step_label(s: &crate::phase::Step) -> String {
    use crate::phase::Step;
    match s {
        Step::Take(c) => c.to_string(),
        Step::Beg(Some(t)) => format!("beg {}", t.letter()),
        Step::Beg(None) => "no beg".into(),
        Step::Mode(m) => format!("{m:?}").to_lowercase(),
        Step::Place(crate::moves::Placement::Gear(g, pos)) => format!("{}:{}", g.name(), pos.0),
        Step::Place(crate::moves::Placement::FirstPlayer) => "first player space".into(),
        Step::PickWorker(w) => format!("w{}", w.0),
        Step::StopPlacing | Step::StopRetrieving => "stop".into(),
        Step::ExtraDay(b) => if *b { "take the extra day" } else { "decline" }.into(),
        other => format!("{other:?}"),
    }
}

pub fn decisions_from(nodes: &[crate::record::Node]) -> Vec<(String, String, f32)> {
    let mut out = Vec::new();
    for node in nodes {
        // Only `TREE_EDGE` nodes index the step enumeration. A one-ply or
        // minimax agent stores indices into its *own* candidate list under
        // `policy_kind::NONE`, and reading those through `legal_steps` would
        // print a confident label for a step the agent never considered.
        if node.policy_kind != crate::record::policy_kind::TREE_EDGE {
            continue;
        }
        let total: u32 = node.visits.iter().map(|(_, n)| *n as u32).sum();
        if total == 0 {
            continue;
        }
        let Some(&(idx, n)) = node.visits.iter().max_by_key(|(_, n)| *n) else {
            continue;
        };
        let steps = crate::tree::legal_steps(&node.state, node.phase, node.turn, node.done);
        let chosen = steps
            .get(idx as usize)
            .map(step_label)
            .unwrap_or_else(|| format!("edge {idx}"));
        out.push((
            format!("{:?}", node.phase)
                .split(['{', ' '])
                .next()
                .unwrap_or("?")
                .to_string(),
            chosen,
            n as f32 / total as f32,
        ));
    }
    out
}

// ---- the diff -----------------------------------------------------------

/// Human-readable description of what changed between two states.
///
/// This is the whole point of the preview pane, and it is trivial only because
/// `GameState` is `Copy`: apply the candidate to a throwaway copy and compare.
pub fn diff_lines(a: &GameState, b: &GameState) -> Vec<Line<'static>> {
    let mut out = Vec::new();

    for p in PlayerId::ALL {
        let (x, y) = (&a.players[p.idx()], &b.players[p.idx()]);
        let c = x.color;
        let mut parts: Vec<String> = Vec::new();

        if x.corn != y.corn {
            parts.push(format!("corn {}→{}", x.corn, y.corn));
        }
        for r in Resource::ALL {
            if x.get(r) != y.get(r) {
                parts.push(format!("{} {}→{}", r.letter(), x.get(r), y.get(r)));
            }
        }
        if x.points != y.points {
            parts.push(format!("pts {}→{}", x.points, y.points));
        }
        if x.free_workers != y.free_workers {
            parts.push(format!("free {}→{}", x.free_workers, y.free_workers));
        }
        if x.worker_discount != y.worker_discount {
            parts.push(format!("disc {}→{}", x.worker_discount, y.worker_discount));
        }
        if x.corn_tiles != y.corn_tiles || x.wood_tiles != y.wood_tiles {
            parts.push(format!(
                "tiles {}c{}w→{}c{}w",
                x.corn_tiles, x.wood_tiles, y.corn_tiles, y.wood_tiles
            ));
        }
        let new_b = y.buildings & !x.buildings;
        if new_b != 0 {
            let ids: Vec<String> = (0..32u8)
                .filter(|i| new_b & (1 << i) != 0)
                .map(|i| format!("#{}", i + 1))
                .collect();
            parts.push(format!("built {}", ids.join(",")));
        }
        let new_m = y.monuments & !x.monuments;
        if new_m != 0 {
            let ids: Vec<String> = (0..16u8)
                .filter(|i| new_m & (1 << i) != 0)
                .map(|i| format!("#{}", i + 1))
                .collect();
            parts.push(format!("monument {}", ids.join(",")));
        }
        for t in Temple::ALL {
            let (u, v) = (a.temple_pos(p, t), b.temple_pos(p, t));
            if u != v {
                parts.push(format!("{} temple {u}→{v}", t.letter()));
            }
        }
        for s in Science::ALL {
            let (u, v) = (a.level(p, s), b.level(p, s));
            if u != v {
                parts.push(format!("{} research {u}→{v}", s.letter()));
            }
        }

        if !parts.is_empty() {
            out.push(Line::from(vec![
                Span::styled(
                    format!("{c}  "),
                    Style::default().fg(colour(c)).add_modifier(Modifier::BOLD),
                ),
                Span::raw(parts.join(",  ")),
            ]));
        }
    }

    // Workers.
    let mut moved: Vec<String> = Vec::new();
    for i in 0..N_WORKERS {
        let (u, v) = (a.workers[i], b.workers[i]);
        if u != v {
            moved.push(format!(
                "{}{} {}→{}",
                a.players[WorkerId(i as u8).owner().idx()].color,
                i,
                loc_str(u),
                loc_str(v)
            ));
        }
    }
    if !moved.is_empty() {
        out.push(Line::from(vec![
            Span::styled("wk ", label()),
            Span::raw(moved.join(",  ")),
        ]));
    }

    // Board-level bookkeeping.
    let mut board: Vec<String> = Vec::new();
    if a.skulls_remaining != b.skulls_remaining {
        board.push(format!(
            "skull bank {}→{}",
            a.skulls_remaining, b.skulls_remaining
        ));
    }
    if a.chichen_filled != b.chichen_filled {
        let newly = b.chichen_filled & !a.chichen_filled;
        let spots: Vec<String> = (0..16u8)
            .filter(|i| newly & (1 << i) != 0)
            .map(|i| i.to_string())
            .collect();
        board.push(format!("chichen {} used", spots.join(",")));
    }
    for i in 2..=5usize {
        if a.palenque[i] != b.palenque[i] {
            board.push(format!(
                "palenque {i} {}/{}→{}/{}",
                a.palenque[i].wood, a.palenque[i].corn, b.palenque[i].wood, b.palenque[i].corn
            ));
        }
    }
    if a.first_player_space != b.first_player_space {
        board.push("first player space claimed".into());
    }
    if !board.is_empty() {
        out.push(Line::from(vec![
            Span::styled("bd ", label()),
            Span::raw(board.join(",  ")),
        ]));
    }

    if out.is_empty() {
        out.push(Line::from(Span::styled("(no change)", dim())));
    }
    out
}

fn loc_str(l: WorkerLoc) -> String {
    match l {
        WorkerLoc::Locked => "locked".into(),
        WorkerLoc::Available => "hand".into(),
        WorkerLoc::FirstPlayerSpace => "first".into(),
        WorkerLoc::OnGear { gear, pos } => format!("{}:{}", &gear.name()[..3], pos.0),
    }
}
