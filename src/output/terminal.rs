use std::io;

use crossterm::{
    event::{self},
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
pub struct TerminalRenderer {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    /// Set to `true` in `finish()` so the `Drop` impl does not double-restore
    /// the terminal.
    cleaned_up: bool,
}

impl TerminalRenderer {
    pub fn new() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            cleaned_up: false,
        })
    }

    fn draw(&mut self, snapshot: &SessionSnapshot) {
        let _ = self.terminal.draw(|f| render_frame(f, snapshot));
    }
}

impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        if !self.cleaned_up {
            let _ = disable_raw_mode();
            let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        }
    }
}

impl OutputRenderer for TerminalRenderer {
    fn render(&mut self, snapshot: &SessionSnapshot) {
        while event::poll(std::time::Duration::ZERO).unwrap_or(false) {
            let _ = event::read();
        }
        self.draw(snapshot);
    }

    fn finish(&mut self, snapshot: Option<&SessionSnapshot>) {
        self.cleaned_up = true;
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        if let Some(s) = snapshot {
            crate::output::summary::SummaryReport::print(s);
        }
    }
}

// ── Rendering helpers ─────────────────────────────────────────────────────────

fn render_frame(f: &mut Frame, snap: &SessionSnapshot) {
    let status_rows = u16::from(
        !snap.nvml_available
            || snap.mfu_divergent
            || snap.vram_growing
            || snap.elapsed.as_secs() < 30,
    );
    let multi_gpu_rows = if snap.per_gpu.len() > 1 {
        snap.per_gpu.len() as u16 + 2
    } else {
        0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),               // header
            Constraint::Length(3 + status_rows), // status badges (optional)
            Constraint::Length(5),               // MFU gauge + percentile row
            Constraint::Length(7),               // time breakdown
            Constraint::Length(3),               // memory / thermal
            Constraint::Length(multi_gpu_rows),  // per-GPU table (optional)
            Constraint::Min(4),                  // recommendation
        ])
        .split(f.area());

    render_header(f, chunks[0], snap);
    render_status(f, chunks[1], snap);
    render_mfu_gauge(f, chunks[2], snap);
    render_breakdown(f, chunks[3], snap);
    render_thermal_memory(f, chunks[4], snap);
    if snap.per_gpu.len() > 1 {
        render_per_gpu(f, chunks[5], snap);
    }
    render_recommendation(f, chunks[6], snap);
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

    let warmup = if elapsed.as_secs() < 30 {
        "  (warming up)"
    } else {
        ""
    };
    let text = format!(
        "GPU: {}{}  •  Elapsed: {:02}:{:02}:{:02}{}",
        snap.gpu_name, cost_str, h, m, s, warmup
    );

    let p = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" calibrate watch "),
        )
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(p, area);
}

/// Status badges row: NVML warning and/or GPU divergence alert.
fn render_status(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    if area.height == 0 {
        return;
    }
    let mut lines: Vec<Line<'static>> = Vec::new();

    if !snap.nvml_available {
        lines.push(Line::from(Span::styled(
            "  ⚠  NVML unavailable — GPU metrics disabled (non-NVIDIA GPU or missing driver)",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
    }
    if snap.mfu_divergent {
        lines.push(Line::from(Span::styled(
            format!(
                "  ⚠  GPU MFU divergence detected: {:.0} ppt spread across devices",
                snap.gpu_mfu_divergence_ppt
            ),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
    }
    if snap.vram_growing {
        lines.push(Line::from(Span::styled(
            "  ⚠  VRAM growing every tick — possible memory leak in training loop",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
    }
    if snap.elapsed.as_secs() < 30 {
        lines.push(Line::from(Span::styled(
            "  ℹ  Warming up — MFU estimate is approximate until 30 s of data are collected",
            Style::default().fg(Color::DarkGray),
        )));
    }

    if lines.is_empty() {
        return;
    }
    let p = Paragraph::new(lines).block(Block::default().borders(Borders::TOP));
    f.render_widget(p, area);
}

fn render_mfu_gauge(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    // Split the area: 3 lines for the gauge, 2 for the percentile row.
    let sub = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(2)])
        .split(area);

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
        mfu, confidence, snap.mfu.actual_tflops.0, snap.mfu.peak_tflops.0,
    );

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL))
        .gauge_style(Style::default().fg(color))
        .ratio((mfu / 100.0).clamp(0.0, 1.0) as f64)
        .label(label);

    f.render_widget(gauge, sub[0]);

    // Percentile row: shown once enough samples have been collected.
    let pct_text = match &snap.mfu_percentiles {
        Some(p) => format!(
            "  p50: {:.1}%   p75: {:.1}%   p95: {:.1}%   peak: {:.1}%",
            p.p50, p.p75, p.p95, snap.peak_mfu_pct
        ),
        None => "  MFU percentiles: collecting samples…".to_string(),
    };
    let pct_style = Style::default().fg(Color::DarkGray);
    let pct_paragraph = Paragraph::new(pct_text).style(pct_style);
    f.render_widget(pct_paragraph, sub[1]);
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

    let p = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Time Breakdown "),
    );
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

    let erratic = if snap.step_time_erratic {
        "  •  Steps: ERRATIC"
    } else {
        ""
    };
    let step_str = if snap.step_time_ms_mean > 0.0 {
        format!("  •  Step: {:.0} ms avg{}", snap.step_time_ms_mean, erratic)
    } else {
        String::new()
    };

    let text = format!(
        "  Temp: {}  •  Power: {} / {}  •  Throttle: {}  •  VRAM: {} / {}{}",
        snap.temperature,
        snap.power_draw,
        snap.power_limit,
        throttle,
        snap.vram_used_mib,
        snap.vram_total_mib,
        step_str,
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

/// Per-GPU table — only rendered when per_gpu.len() > 1.
fn render_per_gpu(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let mut lines = vec![Line::from(Span::styled(
        "  GPU   SM util   VRAM used   Temp",
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    ))];

    for g in &snap.per_gpu {
        let diverge_marker = if snap.mfu_divergent { " ⚠" } else { "" };
        lines.push(Line::from(Span::styled(
            format!(
                "  GPU{:<3} {:>6.1}%{}   {:>8}   {}",
                g.gpu_index, g.sm_utilization.0, diverge_marker, g.vram_used_mib, g.temperature,
            ),
            Style::default().fg(Color::Cyan),
        )));
    }

    let p = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Per-GPU "));
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
