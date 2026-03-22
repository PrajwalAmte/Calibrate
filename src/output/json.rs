use crate::output::OutputRenderer;
use crate::session::state::SessionSnapshot;

/// Renders each snapshot as a newline-delimited JSON object to stdout.
///
/// Suitable for piping into `jq`, logging pipelines, or other tooling.
pub struct JsonRenderer;

impl OutputRenderer for JsonRenderer {
    fn render(&mut self, snapshot: &SessionSnapshot) {
        match serde_json::to_string(snapshot) {
            Ok(json) => println!("{json}"),
            Err(e) => tracing::error!("JSON serialization error: {e}"),
        }
    }

    fn finish(&mut self, snapshot: Option<&SessionSnapshot>) {
        if let Some(s) = snapshot {
            let summary = serde_json::json!({
                "event": "session_end",
                "final_mfu_pct": s.mfu.mfu_pct.0,
                "elapsed_secs": s.elapsed.as_secs(),
                "steps_observed": s.steps_observed,
                "primary_bottleneck": s.bottleneck,
            });
            println!("{summary}");
        }
    }
}
