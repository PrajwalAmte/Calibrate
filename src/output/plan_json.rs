use anyhow::Result;

use crate::plan::PlanReport;

/// Serialize the plan report to stdout as pretty-printed JSON.
pub fn render(report: &PlanReport) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}
