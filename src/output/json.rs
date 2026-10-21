use std::io::Write;

use crate::output::OutputRenderer;
use crate::session::state::{SessionSnapshot, SCHEMA_VERSION};

/// Newline-delimited JSON on stdout: one `{"event":"snapshot",...}` object per
/// tick and a final `{"event":"session_end",...}`.
pub struct JsonRenderer;

/// Serialize a snapshot as a tagged JSON line.
pub fn snapshot_line(snapshot: &SessionSnapshot) -> serde_json::Result<String> {
    let mut v = serde_json::to_value(snapshot)?;
    if let Some(obj) = v.as_object_mut() {
        obj.insert("event".into(), "snapshot".into());
    }
    serde_json::to_string(&v)
}

pub fn session_end_line(snapshot: Option<&SessionSnapshot>) -> String {
    let body = match snapshot {
        Some(s) => serde_json::json!({
            "event": "session_end",
            "schema_version": SCHEMA_VERSION,
            "elapsed_secs": s.elapsed_secs,
            "ticks_observed": s.ticks_observed,
            "mfu_pct": s.mfu.mfu_pct,
            "mfu_source": s.mfu.source,
            "mfu_upper_bound": s.mfu.upper_bound,
            "peak_mfu_pct": s.peak_mfu_pct,
            "mfu_percentiles": s.mfu_percentiles,
            "primary_bottleneck": s.bottleneck,
            "cost": s.cost,
        }),
        None => serde_json::json!({
            "event": "session_end",
            "schema_version": SCHEMA_VERSION,
            "ticks_observed": 0,
        }),
    };
    body.to_string()
}

impl OutputRenderer for JsonRenderer {
    fn render(&mut self, snapshot: &SessionSnapshot) {
        match snapshot_line(snapshot) {
            Ok(line) => {
                let mut out = std::io::stdout().lock();
                // A closed pipe (e.g. `| head`) is not an error worth reporting.
                let _ = writeln!(out, "{line}").and_then(|_| out.flush());
            }
            Err(e) => tracing::error!("JSON serialization error: {e}"),
        }
    }

    fn finish(&mut self, snapshot: Option<&SessionSnapshot>) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{}", session_end_line(snapshot)).and_then(|_| out.flush());
    }
}
