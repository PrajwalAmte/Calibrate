use std::io;

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame, Terminal,
};

use crate::output::OutputRenderer;
use crate::session::state::SessionSnapshot;

/// Full-screen ratatui TUI that refreshes each sampling interval.
///
/// Layout:
/// ```
/// ┌─ Header: GPU, cost/hr, elapsed ──────────────────────┐
/// ├─ MFU gauge ──────────────────────────────────────────┤
/// ├─ Time breakdown (horizontal bar chart) ──────────────┤
/// ├─ Memory / thermal row ───────────────────────────────┤
/// └─ Recommendation box ─────────────────────────────────┘
/// ```
pub struct TerminalRenderer {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalRenderer {
    pub fn new() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    fn draw(&mut self, snapshot: &SessionSnapshot) {
        let _ = self.terminal.draw(|f| render_frame(f, snapshot));
    }
}

impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
    }
}

impl OutputRenderer for TerminalRenderer {
    fn render(&mut self, snapshot: &SessionSnapshot) {
        // Drain any pending keyboard events; Ctrl+C is handled by the signal
        // handler in the command layer, not here.
        while event::poll(std::time::Duration::ZERO).unwrap_or(false) {
            let _ = event::read();
        }
        self.draw(snapshot);
    }

    fn finish(&mut self, snapshot: Option<&SessionSnapshot>) {
        // Leave alternate screen, then print the summary to the normal screen.
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        if let Some(s) = snapshot {
            crate::output::summary::SummaryReport::print(s);
        }
    }
}

// ── Rendering helpers ─────────────────────────────────────────────────────────

fn render_frame(f: &mut Frame, snap: &SessionSnapshot) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(3),  // MFU gauge
            Constraint::Length(7),  // time breakdown
            Constraint::Length(3),  // memory / thermal
            Constraint::Min(4),     // recommendation
        ])
        .split(f.area());

    render_header(f, chunks[0], snap);
    render_mfu_gauge(f, chunks[1], snap);
    render_breakdown(f, chunks[2], snap);
    render_thermal_memory(f, chunks[3], snap);
    render_recommendation(f, chunks[4], snap);
}

fn render_header(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let elapsed = snap.elapsed;
    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;

    let cost_str = snap
        .cost_impact
        .as_ref()
        .map(|c| format!("  •  ${:.2}/hr", c.cost_per_hour))
        .unwrap_or_default();

    let text = format!(
        "GPU: {}{}  •  Elapsed: {:02}:{:02}:{:02}",
        snap.gpu_name, cost_str, h, m, s
    );

    let p = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(" calibrate watch "))
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(p, area);
}

fn render_mfu_gauge(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let mfu = snap.mfu.mfu_pct.0;
    let color = if mfu >= 45.0 {
        Color::Green
    } else if mfu >= 25.0 {
        Color::Yellow
    } else {
        Color::Red
    };

    let confidence = if matches!(snap.mfu.confidence, crate::metrics::mfu::Confidence::Low) {
        " (approx)"
    } else {
        ""
    };

    let label = format!(
        "MFU: {:.1}%{}  •  {:.1} / {:.1} TFLOPS  (target >45%)",
        mfu,
        confidence,
        snap.mfu.actual_tflops.0,
        snap.mfu.peak_tflops.0,
    );

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL))
        .gauge_style(Style::default().fg(color))
        .ratio((mfu / 100.0).clamp(0.0, 1.0) as f64)
        .label(label);

    f.render_widget(gauge, area);
}

fn render_breakdown(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let bd = &snap.breakdown;

    let lines = vec![
        breakdown_line("Forward/backward", bd.forward_backward_pct, Color::Green),
        breakdown_line("Data loader wait", bd.data_loader_pct, Color::Yellow),
        breakdown_line("CUDA sync        ", bd.cuda_sync_pct, Color::Red),
        breakdown_line("Optimizer step   ", bd.optimizer_pct, Color::Blue),
        breakdown_line("Memory alloc     ", bd.memory_alloc_pct, Color::Magenta),
    ];

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Time Breakdown "));
    f.render_widget(p, area);
}

fn breakdown_line(label: &str, pct: f32, color: Color) -> Line<'static> {
    let bar_width = 30usize;
    let filled = ((pct / 100.0) * bar_width as f32).round() as usize;
    let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
    let label = label.to_string();
    let text = format!("  {label}  {bar}  {pct:>5.1}%");

    Line::from(vec![Span::styled(
        text,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )])
}

fn render_thermal_memory(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let throttle = if snap.throttle_thermal {
        "⚠ THERMAL THROTTLE"
    } else {
        "OK"
    };

    let text = format!(
        "  Temp: {}  •  Power: {} / {}  •  Throttle: {}  •  VRAM: {} / {}",
        snap.temperature,
        snap.power_draw,
        snap.power_limit,
        throttle,
        snap.vram_used_mib,
        snap.vram_total_mib,
    );

    let style = if snap.throttle_thermal {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };

    let p = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(" Hardware "))
        .style(style);
    f.render_widget(p, area);
}

fn render_recommendation(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let rec = &snap.recommendation;
    let title = rec.title;
    let gain = rec.expected_mfu_gain_ppt;
    let action = rec.action;

    let text = if gain > 0.0 {
        format!("  [+{gain:.0} ppt MFU expected]\n\n  {action}")
    } else {
        format!("  {action}")
    };

    let title_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);

    let p = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Recommendation: {title} "))
                .title_style(title_style),
        )
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(p, area);
}
