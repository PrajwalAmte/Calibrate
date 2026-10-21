use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Wrap},
    Frame, Terminal,
};

use crate::metrics::breakdown::BreakdownSource;
use crate::metrics::mfu::{Confidence, MfuSource};
use crate::metrics::units::{fmt_hms, fmt_mib, fmt_opt, fmt_pct, fmt_temp, fmt_usd, fmt_watts};
use crate::output::OutputRenderer;
use crate::session::state::SessionSnapshot;

static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();

fn restore_terminal() {
    if TERMINAL_ACTIVE.swap(false, Ordering::SeqCst) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

/// Full-screen ratatui TUI. Quit with `q`, `Esc` or `Ctrl+C`.
pub struct TerminalRenderer {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    quit: bool,
}

impl TerminalRenderer {
    pub fn new() -> anyhow::Result<Self> {
        // Restore the terminal before the default panic message is printed,
        // otherwise a panic leaves the shell in raw mode.
        PANIC_HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                restore_terminal();
                prev(info);
            }));
        });
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        TERMINAL_ACTIVE.store(true, Ordering::SeqCst);
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Self {
            terminal,
            quit: false,
        })
    }

    /// Drain pending input; raw mode turns Ctrl+C into a key event rather
    /// than SIGINT, so it must be handled here.
    fn poll_input(&mut self) {
        while event::poll(std::time::Duration::ZERO).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                if is_quit_key(k.code, k.modifiers) {
                    self.quit = true;
                }
            }
        }
    }
}

pub fn is_quit_key(code: KeyCode, mods: KeyModifiers) -> bool {
    matches!(code, KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc)
        || (mods.contains(KeyModifiers::CONTROL)
            && matches!(code, KeyCode::Char('c') | KeyCode::Char('d')))
}

impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        restore_terminal();
    }
}

impl OutputRenderer for TerminalRenderer {
    fn render(&mut self, snapshot: &SessionSnapshot) {
        self.poll_input();
        let _ = self.terminal.draw(|f| render_frame(f, snapshot));
    }

    fn finish(&mut self, snapshot: Option<&SessionSnapshot>) {
        restore_terminal();
        if let Some(s) = snapshot {
            crate::output::summary::SummaryReport::print(s);
        }
    }

    fn quit_requested(&mut self) -> bool {
        self.poll_input();
        self.quit
    }
}

//  Rendering

fn render_frame(f: &mut Frame, snap: &SessionSnapshot) {
    let status = status_lines(snap);
    let per_gpu_rows = if snap.gpus.len() > 1 {
        snap.gpus.len() as u16 + 3
    } else {
        0
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(if status.is_empty() {
                0
            } else {
                status.len() as u16 + 1
            }),
            Constraint::Length(5),
            Constraint::Length(8),
            Constraint::Length(4),
            Constraint::Length(per_gpu_rows),
            Constraint::Min(4),
        ])
        .split(f.area());

    render_header(f, chunks[0], snap);
    if !status.is_empty() {
        f.render_widget(
            Paragraph::new(status)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::TOP)),
            chunks[1],
        );
    }
    render_mfu(f, chunks[2], snap);
    render_breakdown(f, chunks[3], snap);
    render_hardware(f, chunks[4], snap);
    if snap.gpus.len() > 1 {
        render_per_gpu(f, chunks[5], snap);
    }
    render_recommendation(f, chunks[6], snap);
}

fn render_header(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let cost = snap
        .cost
        .as_ref()
        .map(|c| format!("  •  ${:.2}/hr", c.cost_per_hour))
        .unwrap_or_default();
    let target = match (snap.target.mode.as_str(), snap.target.pid) {
        ("process", Some(pid)) if snap.target.process_count > 1 => {
            format!("PID {pid} (+{} children)", snap.target.process_count - 1)
        }
        ("process", Some(pid)) => format!("PID {pid}"),
        (m, _) => m.to_string(),
    };
    let text = format!(
        "{}  •  {}{}  •  {}  •  source: {}   [q] quit",
        snap.gpu_name,
        target,
        cost,
        fmt_hms(snap.elapsed_secs),
        snap.metrics_source
    );
    f.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" calibrate watch "),
            )
            .style(Style::default().fg(Color::Cyan)),
        area,
    );
}

fn status_lines(snap: &SessionSnapshot) -> Vec<Line<'static>> {
    let warn = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let alert = Style::default().fg(Color::Red).add_modifier(Modifier::BOLD);
    let mut lines: Vec<Line<'static>> = snap
        .warnings
        .iter()
        .map(|w| Line::from(Span::styled(format!("  ⚠  {w}"), warn)))
        .collect();
    if snap.gpu_divergent {
        lines.push(Line::from(Span::styled(
            format!(
                "  ⚠  GPU utilisation diverges by {:.0} ppt across devices — check for a straggler rank",
                snap.gpu_divergence_ppt.unwrap_or(0.0)
            ),
            alert,
        )));
    }
    if snap.vram_growing {
        lines.push(Line::from(Span::styled(
            "  ⚠  GPU memory has grown steadily for 30+ samples — possible leak",
            alert,
        )));
    }
    lines
}

fn render_mfu(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let sub = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(2)])
        .split(area);
    let m = &snap.mfu;
    let label_kind = m.source.label();
    let (ratio, color, label) = match m.mfu_pct {
        Some(pct) => {
            let color = if m.upper_bound {
                Color::Gray
            } else if pct >= snap.target_mfu_pct {
                Color::Green
            } else if pct >= snap.target_mfu_pct * 0.55 {
                Color::Yellow
            } else {
                Color::Red
            };
            let bound = if m.upper_bound { "≤ " } else { "" };
            let approx = if m.confidence == Confidence::Low {
                " (warming up)"
            } else {
                ""
            };
            let tflops = match (m.achieved_tflops, m.peak_tflops) {
                (Some(a), Some(p)) => format!("  •  {a:.1} / {p:.1} TFLOPS"),
                (Some(a), None) => format!("  •  {a:.1} TFLOPS (peak unknown)"),
                _ => String::new(),
            };
            (
                (pct / 100.0).clamp(0.0, 1.0) as f64,
                color,
                format!(
                    "{label_kind}: {bound}{pct:.1}%{approx}{tflops}  (target ≥{:.0}%)",
                    snap.target_mfu_pct
                ),
            )
        }
        None => (0.0, Color::DarkGray, format!("{label_kind}: N/A")),
    };
    f.render_widget(
        Gauge::default()
            .block(Block::default().borders(Borders::ALL))
            .gauge_style(Style::default().fg(color))
            .ratio(ratio)
            .label(label),
        sub[0],
    );
    let mut row = match &snap.mfu_percentiles {
        Some(p) => format!(
            "  p50 {:.1}%   p75 {:.1}%   p95 {:.1}%   peak {}",
            p.p50,
            p.p75,
            p.p95,
            fmt_pct(snap.peak_mfu_pct)
        ),
        None => "  percentiles: collecting samples…".to_string(),
    };
    if let Some(s) = &snap.steps {
        row.push_str(&format!(
            "   •   step {:.0} ms (p95 {:.0}){}",
            s.mean_ms,
            s.p95_ms,
            s.tokens_per_sec
                .map(|t| format!("  •  {t:.0} tok/s"))
                .unwrap_or_default()
        ));
    }
    if m.source == MfuSource::KernelBusy {
        row.push_str("   •   busy-time bounds MFU from above");
    }
    f.render_widget(
        Paragraph::new(row).style(Style::default().fg(Color::DarkGray)),
        sub[1],
    );
}

fn render_breakdown(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let b = &snap.breakdown;
    let (title, lines) = match b.source {
        BreakdownSource::Unavailable => (
            " Time Breakdown — unavailable ".to_string(),
            vec![Line::from(
                "  Needs GPU activity and process CPU data (not available in this mode).",
            )],
        ),
        BreakdownSource::Instrumented => (
            " Time Breakdown — measured by the Python hook ".to_string(),
            vec![
                bar("Forward/backward ", b.compute_pct, Color::Green),
                bar("Waiting for data ", b.data_wait_pct, Color::Yellow),
                bar("Optimizer step   ", b.optimizer_pct, Color::Blue),
                bar("Other            ", b.other_pct, Color::DarkGray),
            ],
        ),
        BreakdownSource::Heuristic => (
            " Time Breakdown — inferred from sampling (approximate) ".to_string(),
            vec![
                bar("GPU busy         ", b.compute_pct, Color::Green),
                bar("GPU idle, CPU busy", b.data_wait_pct, Color::Yellow),
                bar("GPU & CPU idle   ", b.sync_stall_pct, Color::Red),
                bar("Memory-bound     ", b.memory_bound_pct, Color::Magenta),
                bar("Partial GPU use  ", b.other_pct, Color::DarkGray),
            ],
        ),
    };
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

fn bar(label: &str, pct: f32, color: Color) -> Line<'static> {
    let width = 30usize;
    let filled = ((pct.clamp(0.0, 100.0) / 100.0) * width as f32).round() as usize;
    let text = format!(
        "  {label}  {}{}  {pct:>5.1}%",
        "█".repeat(filled),
        "░".repeat(width - filled)
    );
    Line::from(Span::styled(
        text,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))
}

fn render_hardware(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let throttled = snap.gpus.iter().any(|g| g.throttle.any());
    let first = snap.gpus.first();
    let line1 = match first {
        Some(g) => {
            let mut reasons = vec![];
            if g.throttle.thermal() {
                reasons.push("THERMAL");
            }
            if g.throttle.power_cap {
                reasons.push("POWER CAP");
            }
            if g.throttle.hw_slowdown {
                reasons.push("HW SLOWDOWN");
            }
            let throttle = if reasons.is_empty() {
                "none".to_string()
            } else {
                reasons.join(", ")
            };
            format!(
                "  Temp {}  •  Power {} / {}  •  Clock {}  •  Throttle: {}",
                fmt_temp(g.temperature_c),
                fmt_watts(g.power_w),
                fmt_watts(g.power_limit_w),
                fmt_opt(g.clock_pct_of_boost, |c| format!("{c:.0}% of boost")),
                throttle
            )
        }
        None => "  No GPU in use by the target".to_string(),
    };
    let vram = first
        .map(|g| {
            format!(
                "  VRAM {} / {} (peak {})",
                fmt_opt(g.vram_used_mib, fmt_mib),
                fmt_opt(g.vram_total_mib, fmt_mib),
                fmt_opt(g.peak_vram_mib, fmt_mib)
            )
        })
        .unwrap_or_default();
    let line2 = format!(
        "{vram}  •  CPU (process tree) {}{}",
        fmt_opt(snap.cpu_pct, |c| format!("{c:.0}%")),
        if snap.hook_peers > 0 {
            format!("  •  hook: {} process(es)", snap.hook_peers)
        } else {
            String::new()
        }
    );
    let style = if throttled {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };
    f.render_widget(
        Paragraph::new(vec![Line::from(line1), Line::from(line2)])
            .block(Block::default().borders(Borders::ALL).title(" Hardware "))
            .style(style),
        area,
    );
}

fn render_per_gpu(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let mut lines = vec![Line::from(Span::styled(
        "  GPU   busy    tensor   VRAM used    Temp   Power",
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    ))];
    for g in &snap.gpus {
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<4} {:>6}  {:>7}  {:>10}  {:>6}  {:>6}{}",
                g.index,
                fmt_pct(g.sm_active_pct.or(g.util_pct)),
                fmt_pct(g.tensor_active_pct),
                fmt_opt(g.vram_used_mib, fmt_mib),
                fmt_temp(g.temperature_c),
                fmt_watts(g.power_w),
                if g.vram_growing {
                    "  ⚠ VRAM growing"
                } else {
                    ""
                }
            ),
            Style::default().fg(Color::Cyan),
        )));
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Per-GPU ")),
        area,
    );
}

fn render_recommendation(f: &mut Frame, area: Rect, snap: &SessionSnapshot) {
    let rec = &snap.recommendation;
    let mut text = String::new();
    if let Some(g) = rec.expected_mfu_gain_ppt {
        text.push_str(&format!("  [up to +{g:.0} ppt MFU]\n\n"));
    }
    text.push_str("  ");
    text.push_str(&rec.action);
    if let Some(c) = &snap.cost {
        let lb = if c.is_lower_bound { "≥ " } else { "" };
        text.push_str(&format!(
            "\n\n  Cost: {} spent, {lb}{} wasted below target ({lb}{}/hr)",
            fmt_usd(c.spent_usd),
            fmt_usd(c.wasted_usd),
            fmt_usd(c.waste_per_hour)
        ));
    }
    f.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} ", rec.title))
                    .title_style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_to_text(snap: &SessionSnapshot, w: u16, h: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(w, h);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render_frame(f, snap)).unwrap();
        let buf = term.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn renders_labels_na_and_source() {
        let snap = crate::export::tests::sample_snapshot();
        let text = render_to_text(&snap, 140, 45);
        assert!(text.contains("MFU upper bound (GPU busy time)"), "{text}");
        assert!(text.contains("source: nvml"));
        assert!(text.contains("[q] quit"));
        assert!(text.contains("inferred from sampling"));
    }

    #[test]
    fn small_terminal_and_many_gpus_do_not_panic() {
        let mut snap = crate::export::tests::sample_snapshot();
        let g = snap.gpus[0].clone();
        snap.gpus = (0..8)
            .map(|i| {
                let mut x = g.clone();
                x.index = i;
                x.temperature_c = None;
                x
            })
            .collect();
        snap.warnings = vec!["a very long warning ".repeat(10); 3];
        for (w, h) in [(80, 24), (40, 10), (200, 60), (1, 1)] {
            render_to_text(&snap, w, h);
        }
        let text = render_to_text(&snap, 160, 70);
        assert!(text.contains("Per-GPU"));
        assert!(text.contains("N/A"), "missing temperature renders as N/A");
    }

    #[test]
    fn quit_keys() {
        assert!(is_quit_key(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(is_quit_key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(is_quit_key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!is_quit_key(KeyCode::Char('c'), KeyModifiers::NONE));
        assert!(!is_quit_key(KeyCode::Char('x'), KeyModifiers::NONE));
    }
}
