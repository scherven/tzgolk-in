//! Terminal rendering for watching a game.
//!
//! Read-only: it draws live `GameState`, never mutates it. The move preview
//! works by applying a candidate to a copy of the state and diffing — which is
//! only cheap because the state is `Copy`.

use crate::data::buildings::def as bdef;
use crate::data::monuments::def as mdef;
use crate::data::temples::TEMPLES;
use crate::ids::*;
use crate::moves::Move;
use crate::state::{GameState, WorkerLoc};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

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

/// What the move list is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MoveSource {
    /// Ten drawn from the rollout policy.
    Sampled,
    /// The ten best by the placeholder heuristic.
    Ranked,
}

impl MoveSource {
    pub fn label(self) -> &'static str {
        match self {
            MoveSource::Sampled => "10 sampled",
            MoveSource::Ranked => "top 10 by heuristic",
        }
    }
}

pub struct App {
    pub game: crate::game::Game,
    /// The agent driving `n` and autoplay, if one was named. `None` falls back
    /// to the rollout policy, which is what the viewer did before checkpoints
    /// could be loaded.
    pub agent: Option<Box<dyn crate::record::Agent>>,
    pub agent_name: String,
    /// The sub-decisions of the last turn the agent played: label, what it
    /// chose, and the share of visits that went there. This is the search
    /// talking, not a heuristic ranking of it.
    pub last_decisions: Vec<(String, String, f32)>,
    pub candidates: Vec<(Move, f32)>,
    pub selected: usize,
    pub source: MoveSource,
    pub autoplay: bool,
    pub total_moves: Option<usize>,
    pub status: String,
}

// ---- top-level layout ---------------------------------------------------

pub fn draw(f: &mut Frame, app: &App) {
    let root = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(9),
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

    preview(f, root[2], app);
    help(f, root[3], app);
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
    let title = match app.total_moves {
        Some(n) => format!("Moves — {} of {n}", app.source.label()),
        None => format!("Moves — {}", app.source.label()),
    };

    let items: Vec<ListItem> = app
        .candidates
        .iter()
        .enumerate()
        .map(|(i, (m, score))| {
            let sel = i == app.selected;
            let mut text = m.to_string();
            if text.chars().count() > 58 {
                text = text.chars().take(55).collect::<String>() + "...";
            }
            ListItem::new(Line::from(vec![
                Span::styled(
                    if sel { "> " } else { "  " },
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{score:>7.1}  "),
                    Style::default().fg(ratatui::style::Color::Cyan),
                ),
                Span::styled(
                    text,
                    if sel {
                        Style::default().add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    },
                ),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected));
    f.render_stateful_widget(
        List::new(items)
            .block(boxed(&title))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED)),
        area,
        &mut state,
    );
}

fn preview(f: &mut Frame, area: Rect, app: &App) {
    // When an agent has just moved, its own search is more interesting than a
    // heuristic ranking of the alternatives.
    if !app.last_decisions.is_empty() {
        let short = short_agent(&app.agent_name);
        let mut out = vec![Line::from(vec![Span::styled(
            format!("last turn, as {short} searched it"),
            Style::default().add_modifier(Modifier::BOLD),
        )])];
        for (phase, chosen, share) in app.last_decisions.iter().take(7) {
            let bar = "#".repeat(((share * 20.0) as usize).min(20));
            out.push(Line::from(vec![
                Span::styled(format!("{phase:<12}"), label()),
                Span::styled(
                    format!("{:>5.0}% ", share * 100.0),
                    Style::default().fg(ratatui::style::Color::Cyan),
                ),
                Span::styled(format!("{bar:<20} "), dim()),
                Span::raw(chosen.clone()),
            ]));
        }
        f.render_widget(
            Paragraph::new(out)
                .block(boxed("Search — visit share at each sub-decision"))
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    let lines = match app.candidates.get(app.selected) {
        Some((m, _)) => {
            let before = app.game.state;
            let mut after = before;
            crate::moves::apply_move(&mut after, before.current, m);
            let mut out = vec![Line::from(vec![Span::styled(
                m.to_string(),
                Style::default().add_modifier(Modifier::BOLD),
            )])];
            out.extend(diff_lines(&before, &after));
            out
        }
        None => vec![Line::from(Span::styled("no candidate selected", dim()))],
    };

    f.render_widget(
        Paragraph::new(lines)
            .block(boxed("Preview — what this move changes"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn help(f: &mut Frame, area: Rect, app: &App) {
    let auto = if app.autoplay { "on" } else { "off" };
    let text = format!(
        " j/k move · enter play selected · n next turn · r reroll · t {} · a autoplay [{auto}] · q quit    {}",
        match app.source {
            MoveSource::Sampled => "ranked",
            MoveSource::Ranked => "sampled",
        },
        app.status
    );
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

pub fn decisions_from(nodes: &[crate::record::Node]) -> Vec<(String, String, f32)> {
    let mut out = Vec::new();
    for node in nodes {
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
            .map(|s| {
                // The variant name alone just repeats the phase, so keep the
                // payload and trim to what the pane can show.
                let d = format!("{s:?}");
                if d.chars().count() > 52 {
                    d.chars().take(49).collect::<String>() + "..."
                } else {
                    d
                }
            })
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
