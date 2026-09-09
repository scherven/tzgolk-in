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
///
/// # Why there is no percentage in here
///
/// Nothing outside `mcts.rs` can see a simulation counter, so a bar that filled
/// from 0 to 1 would be inventing the only number on the screen. Every field
/// below is something the viewer genuinely knows: which search of how many is
/// running, what the finished ones returned, and what the *previous* turn cost.
#[derive(Clone, Default)]
pub struct Thinking {
    /// What is running, in the present tense: "searching R's turn".
    pub what: String,
    /// One honest line of detail — a stage count, or what is already known.
    pub detail: String,
    pub elapsed: Duration,
    /// Search `k` of `n`. Pressing `n` costs two — one to choose the move, one
    /// to rank what it passed over — and this is the one fraction on the screen
    /// that is a fraction of something counted.
    pub stage: Option<(u32, u32)>,
    /// What the run has already settled, oldest first: after stage 1 this holds
    /// the move that was chosen, so a slow stage 2 is not a bare clock.
    pub known: Vec<String>,
    /// What the last comparable search cost. The only defensible scale for "how
    /// much longer", and the bar below is labelled as a comparison to it rather
    /// than as progress, because it can and does run past 100%.
    pub prior: Option<Duration>,
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
    /// The sub-decisions of the last turn the agent played. This is the search
    /// talking, not a heuristic ranking of it.
    pub last_decisions: Vec<Decision>,
    /// Who played that turn and what they played, e.g. `R played retrieve …`.
    ///
    /// Not decoration. The search pane sits beside a shortlist belonging to a
    /// *different* player in the position *after* the move — press `n` and the
    /// left half is R's reasoning while the right half is G's options — and
    /// with nothing saying so the two read as one account of one turn.
    pub last_played: Option<String>,
    /// The move list on show, best first, together with how many moves it was
    /// drawn from and what produced it.
    pub ranking: crate::eval::Ranking,
    pub selected: usize,
    pub source: MoveSource,
    pub autoplay: bool,
    pub status: String,
}

impl App {
    /// The raw ranking, spellings and all.
    ///
    /// Almost never what a caller wants: the panel, the selection and the
    /// preview all work over [`App::rows`], which is this list with the
    /// restatements folded, and a caller that sized a walk on *this* count
    /// walked the selection off the end of the panel. Kept for the arithmetic
    /// the fold itself needs.
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

    /// Move the selection by `d` rows, wrapping.
    ///
    /// Over [`App::rows`], **not** `ranking.moves`. The two differ by exactly
    /// the fold — ten spellings become three outcomes — and a caller that
    /// wrapped on the raw count walked the selection off the end of the panel,
    /// where `selected_move` returns `None` and the preview goes blank.
    pub fn step_selection(&mut self, d: isize) {
        let n = self.rows().len();
        if n == 0 {
            self.selected = 0;
            return;
        }
        let n = n as isize;
        self.selected = (((self.selected as isize + d) % n + n) % n) as usize;
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
    let mut idle = idle(v);
    // Sorted, because `same_outcome` treats these as a set and folds the
    // orderings together: leaving them in sequence order printed `w3,w2` for
    // one row and `w2,w3` for the next, which is a difference the fold has
    // already decided is not one.
    idle.sort_unstable();
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
    // The bottom pane grows with the terminal. A turn is a chain of up to a
    // dozen sub-decisions and the pane that lists them was pinned at eight
    // rows, so on a tall terminal the extra height went to a Moves panel with
    // three rows in it while the search's own account of the turn was cut off
    // at two.
    let deep = (f.area().height / 4).clamp(8, 16);
    let root = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        // One line, above the pane it belongs to, saying what the numbers in
        // the Moves column mean. It is only one row because it was previously
        // zero and the panel was unreadable for want of it.
        Constraint::Length(1),
        Constraint::Length(deep),
        Constraint::Length(1),
    ])
    .split(f.area());

    // Folded once and threaded down. Both the shortlist and the preview need
    // this list, and each row costs a `successor` and a `margin`; recomputing
    // it per panel spent that twice per frame, on the draw loop, at the exact
    // moment the other thread is saturating a core with the search.
    let rows = app.rows();

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

    // Three fixed lengths over-subscribed this column below ~44 rows, and the
    // solver resolved it by starving whichever panel had the weakest
    // constraint — which was the move list, the one panel this viewer exists
    // for: at 80x30 the shortlist was a single row. Allocated explicitly
    // instead, moves first. Temples and Research are reference panels a reader
    // can read past; the shortlist is the answer.
    let h = body[1].height;
    let mv = (h / 3).clamp(6, 12).min(h);
    let rs = 7.min((h - mv).div_ceil(2));
    // Temples never wants more than its own height, so anything above that
    // falls through to the shortlist, which takes the remainder below.
    let tp = (h - mv - rs).min(13);
    let right = Layout::vertical([
        Constraint::Length(tp),
        Constraint::Length(rs),
        Constraint::Min(0),
    ])
    .split(body[1]);
    temples(f, right[0], app);
    research(f, right[1], app);
    moves(f, right[2], app, &rows);

    units(f, root[2], app);
    preview(f, root[3], app, &rows);
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

/// Shorten `s` to `room` columns by dropping the **middle**.
///
/// Cutting from the right is what the panel used to do, and after the fold it
/// removes precisely the characters that make one row different from another:
/// the shared body `retrieve w0[+3 corn] w1[-3 corn, G+1]` survives and the
/// distinguishing tail `+ w2 to hand` becomes `+..`, so three genuinely
/// different outcomes render as three copies of one line. Head and tail are
/// both load-bearing; the middle of a long retrieval is the part a reader
/// skims.
pub fn fit(s: &str, room: usize) -> String {
    let cs: Vec<char> = s.chars().collect();
    if cs.len() <= room {
        return s.to_string();
    }
    if room <= 4 {
        return cs.iter().take(room).collect();
    }
    // A third to the tail: enough for `+ w2,w3 to hand`, not so much that the
    // verb and the first pickup are lost from the head.
    let keep = room - 1;
    let tail = (keep / 3).max(3);
    let head = keep - tail;
    let mut out: String = cs[..head].iter().collect();
    out.push('…');
    out.extend(&cs[cs.len() - tail..]);
    out
}

/// [`fit`], but keeping three quarters at the **end**.
///
/// For a producer's note, where the closing clause is a caveat and the opening
/// one is a label already on screen. Right-truncating `... — NOT the whole move
/// space` removed the warning; a centred elision left ` move space`, which
/// reads as the opposite of what it says. Losing the head costs a duplicate.
pub fn fit_tail(s: &str, room: usize) -> String {
    let cs: Vec<char> = s.chars().collect();
    if cs.len() <= room || room <= 8 {
        return fit(s, room);
    }
    let keep = room - 1;
    let head = keep / 4;
    let mut out: String = cs[..head].iter().collect();
    out.push('…');
    out.extend(&cs[cs.len() - (keep - head)..]);
    out
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

fn moves(f: &mut Frame, area: Rect, app: &App, rows: &[Row]) {
    let r = &app.ranking;
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
        // Two different truncations happen before this list, and a reader who
        // is told about neither will read the shares as a distribution that
        // ought to sum to 100. `total` is turns the tree *reached*, not legal
        // turns (see `SearchAgent::ranked_moves`); and the shortlist is the top
        // few of those, so say what fraction of the search's own visits the
        // rows below actually account for.
        let held: f32 = rows.iter().map(|x| x.score).sum();
        format!("Moves — {shown} outcomes, {held:.0}% of visits, of {} reached", r.total)
    } else {
        format!("Moves — {shown} of {} · {}", r.total, app.source.label())
    };

    // The old panel truncated at a hard 58 columns, which cut the tail off
    // every row at 132x44 -- exactly the rows whose tails were the difference
    // between them. Budget from the real width instead.
    let inner = area.width.saturating_sub(2) as usize;
    let wide = inner >= 46;
    // Every `ScoreUnit::render` arm is six columns wide, deliberately; the rest
    // of this is the selection marker, the separator and the 1-ply column. It
    // has to be exact — a gutter one short clips the `xN` marker that the fold
    // reserved room for, which is the one mark saying a row stands for several.
    let gutter = 2 + 6 + 2 + if wide { 8 } else { 0 };
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
        // The fold hides nothing: say how many spellings went into the row.
        // Budgeted *before* the move text rather than appended after it — as a
        // suffix it was the first thing the truncation ate, so the one marker
        // saying the row stands for several was never on screen.
        let mark = if row.spellings > 1 {
            format!("  x{}", row.spellings)
        } else {
            String::new()
        };
        let text = format!(
            "{}{mark}",
            fit(&row.text, room.saturating_sub(mark.chars().count()))
        );
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
        ScoreUnit::VisitShare => "visits = share of finished sims that played this turn",
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

    // Assembled shortest-first and dropped from the right, because the pieces
    // are in falling order of certainty: the viewer's own definition of its own
    // column, then of the column it computed, then the producer's prose. The
    // note is *elided in the middle* rather than cut off, since its tail is
    // where a producer puts its caveat — "NOT the whole move space" is the
    // sentence that stops the shortlist being read as the move list, and
    // truncating from the right removed exactly that.
    let w = area.width as usize;
    let mut text = format!(" {mine}");
    let ply = " · 1-ply = eval::margin of the successor";
    // The note goes on before the 1-ply gloss, and only the gloss is dropped
    // when they will not both fit: the column is already headed `1-ply`, while
    // the note is the only line that can contradict this panel's own guess at
    // what its numbers are.
    let left = w.saturating_sub(text.chars().count() + 3);
    if left >= 20 && !note.is_empty() {
        text.push_str(" · ");
        text.push_str(&fit_tail(note, left.min(76)));
    }
    if text.chars().count() + ply.chars().count() <= w {
        text.push_str(ply);
    }
    if text.chars().count() > w {
        text = fit(&text, w);
    }
    f.render_widget(Paragraph::new(Span::styled(text, dim())), area);
}

fn preview(f: &mut Frame, area: Rect, app: &App, rows: &[Row]) {
    if let Some(t) = &app.thinking {
        thinking(f, area, app, t);
        return;
    }

    // Both halves, not one or the other. The diff answers "what does this move
    // do"; the sub-decisions answer "how did the search arrive at it". Showing
    // only the second — which is what a narrow terminal used to get, since the
    // split needed 100 columns — meant that under an agent the move preview
    // was unreachable, and `j`/`k` moved a selection nothing then described.
    // Below 100 columns the two stack instead, which every plausible terminal
    // has the height for.
    let both = !app.last_decisions.is_empty();
    let (l, r) = if both && area.width >= 100 {
        let c = Layout::horizontal([Constraint::Percentage(54), Constraint::Percentage(46)])
            .split(area);
        (c[0], Some(c[1]))
    } else if both && area.height >= 8 {
        // The diff gets the larger half: it is what the selection keys act on,
        // and the search pane's first line already carries its headline number.
        let c = Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);
        (c[0], Some(c[1]))
    } else {
        (area, None)
    };

    match r {
        Some(r) => {
            why_pane(f, l, app, rows);
            search_pane(f, r, app);
        }
        None if both => search_pane(f, l, app),
        None => why_pane(f, l, app, rows),
    }
}

/// What the highlighted candidate does, and what the two columns said about it.
fn why_pane(f: &mut Frame, area: Rect, app: &App, rows: &[Row]) {
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
            out.extend(lookahead_line(rows, unit, app.selected));
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

/// Why the shortlist is in the order it is in, in one line.
///
/// The two numeric columns are a search's opinion and a one-ply evaluator's,
/// and where they disagree the gap *is* the lookahead — priced in the unit the
/// second column is already in. Saying that out loud is the difference between
/// a reader seeing two numbers and a reader seeing an argument: the search is
/// giving up points now, and the amount is on the line.
///
/// Only under [`ScoreUnit::VisitShare`]. Under `Margin` both columns are the
/// same evaluator and any disagreement would be a bug, not a plan.
fn lookahead_line(rows: &[Row], unit: ScoreUnit, selected: usize) -> Option<Line<'static>> {
    // Only under the top row. It is a statement about the search's *pick*, and
    // read three rows down it looked like a claim about the row highlighted
    // there — which already has `vs top` on the line above saying what it cost.
    if unit != ScoreUnit::VisitShare || rows.len() < 2 || selected != 0 {
        return None;
    }
    let top = rows.first()?;
    let (i, best) = rows
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.margin.total_cmp(&b.1.margin))?;
    if i == 0 {
        return Some(Line::from(vec![
            Span::styled("why ", label()),
            Span::styled(
                "the pick also tops the one-ply column".to_string(),
                dim(),
            ),
        ]));
    }
    Some(Line::from(vec![
        Span::styled("why ", label()),
        Span::raw(format!(
            "the pick is {:.1} pts behind row {} right now; the search is buying \
             something one ply cannot see",
            best.margin - top.margin,
            i + 1
        )),
    ]))
}

/// The search's own account of the turn it just played.
///
/// The Moves panel says *which* turn won; this says how it was assembled. Each
/// row is one link of the chain: what the search settled on, how hard, out of
/// how many, and — the part that carries the reasoning — the best thing it
/// declined. A 51/49 row is where the game was actually close.
fn search_pane(f: &mut Frame, area: Rect, app: &App) {
    let short = short_agent(&app.agent_name);
    let d = &app.last_decisions;
    // The value at the first node is the value of the position the turn started
    // from, which is the search's one-number verdict on how the game is going.
    let inner = area.width.saturating_sub(2) as usize;
    let mut out = Vec::new();
    if let Some(played) = &app.last_played {
        out.push(Line::from(Span::styled(
            fit(played, inner),
            Style::default().add_modifier(Modifier::BOLD),
        )));
    }
    // The agent's name is already in the top bar; spending 38 columns repeating
    // it here cost the value its place on the line.
    let head = match d.first() {
        Some(x) => format!(
            // "choice points", not "sub-decisions": forced links of the chain
            // carry no distribution and are never recorded, so this is a count
            // of where the search had something to decide, not of turn length.
            "{} choice point{} · position ~{:+.0} pts vs field",
            d.len(),
            if d.len() == 1 { "" } else { "s" },
            x.points()
        ),
        // Forced sub-decisions carry no distribution and are not recorded, so
        // this is never the whole chain and must not read as if it were.
        None => format!("{short} — no recorded sub-decisions"),
    };
    out.push(Line::from(vec![Span::styled(fit(&head, inner), label())]));

    // Everything before the step text is fixed width, so the step and the
    // rejected alternative share whatever is left. Bars go first when the pane
    // is narrow: a bar clipped mid-way misreports a share, and the number it
    // duplicates is already there.
    let fixed = 9 + 5 + 6;
    let rest = inner.saturating_sub(fixed);
    // The bar is the first thing cut. It restates the percentage two columns to
    // its left, whereas the rejected alternative beside it is the only thing on
    // the row that is not already somewhere else on the screen.
    let bar_w = if rest >= 56 { 12 } else { 0 };
    let room = rest - bar_w;

    for x in d.iter().take((area.height as usize).saturating_sub(2 + out.len())) {
        let mut spans = vec![
            Span::styled(format!("{:<9}", fit(&x.phase, 8)), label()),
            Span::styled(
                format!("{:>3.0}% ", x.share * 100.0),
                Style::default().fg(ratatui::style::Color::Cyan),
            ),
            // Out of how many. `97%` is a different statement at 2 edges and at
            // 40, and without this the reader cannot tell the two apart.
            Span::styled(format!("/{:<4} ", x.edges), dim()),
        ];
        if bar_w > 0 {
            let bar = "#".repeat(((x.share * bar_w as f32) as usize).min(bar_w));
            spans.push(Span::styled(format!("{bar:<w$} ", w = bar_w), dim()));
        }
        // The runner-up is the *reason* text: it is what the position offered
        // and the search turned down. It is elided before the chosen step is,
        // because a row that lost its chosen step says nothing at all.
        match &x.runner_up {
            Some((alt, share)) if room >= 26 => {
                let half = room / 2;
                spans.push(Span::raw(format!("{:<w$}", fit(&x.chosen, half), w = half)));
                spans.push(Span::styled(
                    fit(
                        &format!(" · not {alt} ({:.0}%)", share * 100.0),
                        room - half,
                    ),
                    dim(),
                ));
            }
            _ => spans.push(Span::raw(fit(&x.chosen, room))),
        }
        out.push(Line::from(spans));
    }
    if d.is_empty() {
        out.push(Line::from(Span::styled(
            "this agent builds no tree, so it has no visit counts to show",
            dim(),
        )));
    }
    f.render_widget(
        Paragraph::new(out).block(boxed("Search — the turn just played, step by step")),
        area,
    );
}

/// The screen while a search is running.
///
/// Nothing outside `mcts.rs` can see a simulation counter, so there is no
/// completion fraction to draw and a bar that filled from 0 to 1 would be
/// inventing the only number on the screen. What is drawn instead is measured:
/// which search of how many, what the finished ones returned, and elapsed
/// against **what the last turn cost** — which can and does run past the end of
/// the bar, and is labelled as a comparison rather than as progress so that
/// overrun reads as information instead of as a stuck widget. With no prior to
/// compare against, a sweep says "running" and claims nothing else.
fn thinking(f: &mut Frame, area: Rect, app: &App, t: &Thinking) {
    let secs = t.elapsed.as_secs_f64();
    let w = (area.width as usize).saturating_sub(4).clamp(8, 100);

    let mut top = vec![
        Span::styled(
            format!("{} {}", t.tick(), t.what),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("   {secs:.1}s"),
            Style::default().fg(ratatui::style::Color::Cyan),
        ),
    ];
    if let Some((k, n)) = t.stage {
        top.push(Span::styled(format!("   search {k} of {n}"), label()));
    }
    let mut out = vec![Line::from(top), Line::from(Span::styled(t.detail.clone(), label()))];

    // What is already settled. Two lines at most: the pane is eight rows tall
    // and the bar and the footer have to fit under whatever this takes.
    for k in t.known.iter().rev().take(2).rev() {
        out.push(Line::from(vec![
            Span::styled("· ", dim()),
            Span::raw(fit(k, w.saturating_sub(2))),
        ]));
    }

    match t.prior {
        Some(d) if d.as_secs_f64() > 0.05 => {
            let frac = secs / d.as_secs_f64();
            let filled = ((frac * w as f64) as usize).min(w);
            let bar: String = (0..w)
                .map(|i| if i < filled { '#' } else { '·' })
                .collect();
            out.push(Line::from(Span::styled(bar, dim())));
            out.push(Line::from(Span::styled(
                format!(
                    "{secs:.1}s against {:.1}s for the last search — a scale, not a deadline",
                    d.as_secs_f64()
                ),
                dim(),
            )));
        }
        _ => {
            // Position derives from the clock, so a slow frame shifts the
            // marker further rather than stalling it.
            let span = (w * 2).max(2);
            let at = (t.elapsed.as_millis() / 40) as usize % span;
            let at = if at < w { at } else { span - at - 1 };
            let sweep: String = (0..w)
                .map(|i| if i.abs_diff(at) < 2 { '#' } else { '·' })
                .collect();
            out.push(Line::from(Span::styled(sweep, dim())));
            out.push(Line::from(Span::styled(
                "no earlier search to compare against, so this shows only that it is running"
                    .to_string(),
                dim(),
            )));
        }
    }
    // Orientation, and the first thing to go: the footer already says `q quit`
    // while a search runs, and what stage 2 has to report is worth more than a
    // hint. Dropped rather than clipped, so nothing renders as a half-sentence.
    if out.len() + 1 <= area.height.saturating_sub(2) as usize {
        out.push(Line::from(Span::styled(
            "the board above is the position it is thinking about · q quits".to_string(),
            dim(),
        )));
    }

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

/// One sub-decision of a turn, as the search left it.
///
/// The share alone does not say much: 97% out of two edges is a coin that came
/// up the same way twice, and 97% out of forty is a conclusion. So the width of
/// the choice and what came second travel with it.
#[derive(Clone)]
pub struct Decision {
    /// The `Phase` variant, without its payload — the payload is the step, and
    /// it is already in `chosen`.
    pub phase: String,
    pub chosen: String,
    /// Share of the node's visits that went to `chosen`.
    pub share: f32,
    /// The best step the search declined, and its share. `None` at a node with
    /// one edge, where there was nothing to decline.
    pub runner_up: Option<(String, f32)>,
    /// Legal edges at this node, before the widening cap.
    pub edges: usize,
    pub visits: u32,
    /// Backed-up value for the player to move, on `record::z_rel`'s scale.
    pub value: f32,
}

impl Decision {
    /// `value` read back as points ahead of the table average.
    ///
    /// `z_rel` is `tanh((score - mean) / Z_SCALE)`, so this inverts it. It is
    /// approximate and the panel says so with a `~`: the backed-up number is a
    /// mean of `tanh`, not the `tanh` of a mean, so the magnitude is pulled in
    /// towards zero. The sign and the ordering — which is what a reader takes
    /// from it — survive the transform exactly, because it is monotone.
    pub fn points(&self) -> f32 {
        let v = self.value.clamp(-0.999, 0.999);
        v.atanh() * crate::record::Z_SCALE
    }
}

/// Turn an agent's decision nodes into the search panel's rows.
///
/// `Node::visits` stores `(edge index, visit count)`, not the step itself, so
/// the enumeration has to be regenerated to say what was chosen. That is also
/// why `done` rides along: at `PickWorker` the edge set depends on it, and a
/// list one edge short would mislabel everything after the gap.
///
/// The denominator is `total_visits`, not the sum of the vector: `visits` is
/// truncated to `record::MAX_VISITS` before it is written, so summing it would
/// quietly renormalise a long tail away and inflate every share on screen.
pub fn decisions_from(nodes: &[crate::record::Node]) -> Vec<Decision> {
    let mut out = Vec::new();
    for node in nodes {
        // Only `TREE_EDGE` nodes index the step enumeration. A one-ply or
        // minimax agent stores indices into its *own* candidate list under
        // `policy_kind::NONE`, and reading those through `legal_steps` would
        // print a confident label for a step the agent never considered.
        if node.policy_kind != crate::record::policy_kind::TREE_EDGE {
            continue;
        }
        let total = node.total_visits;
        if total == 0 || node.visits.is_empty() {
            continue;
        }
        let steps = crate::tree::legal_steps(&node.state, node.phase, node.turn, node.done);
        let name = |idx: u16| {
            steps
                .get(idx as usize)
                .map(step_label)
                .unwrap_or_else(|| format!("edge {idx}"))
        };
        // `search_node` sorts descending before it truncates, but a caller
        // could hand us anything, so take the top two rather than assume.
        let mut top: Vec<(u16, u16)> = node.visits.clone();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        let (best, n) = top[0];
        out.push(Decision {
            phase: format!("{:?}", node.phase)
                .split(['{', ' '])
                .next()
                .unwrap_or("?")
                .to_string(),
            chosen: name(best),
            share: n as f32 / total as f32,
            runner_up: top
                .get(1)
                .map(|&(i, m)| (name(i), m as f32 / total as f32)),
            edges: node.n_edges as usize,
            visits: total,
            value: node.root_value[node.turn.idx()],
        });
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
