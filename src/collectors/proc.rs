use std::path::PathBuf;
use std::time::Instant;

use crate::error::CalibrateError;
use crate::metrics::units::Percent;

/// Reads and diffs `/proc/{pid}/stat` to compute the CPU utilization of a
/// specific process between two successive calls.
///
/// This is created once and polled on each sampling interval.  The delta
/// between consecutive reads is divided by elapsed wall-clock time to yield
/// a percentage (may exceed 100 on multi-core systems, mirroring `top`).
pub struct ProcCollector {
    pid: u32,
    stat_path: PathBuf,
    prev_cpu_ticks: Option<u64>,
    prev_instant: Option<Instant>,
    /// Kernel clock ticks per second (cached from `sysconf(_SC_CLK_TCK)`).
    ticks_per_sec: u64,
}

impl ProcCollector {
    /// Create a new collector for the given PID.
    ///
    /// Returns [`CalibrateError::ProcessNotFound`] immediately if the process
    /// does not exist.
    pub fn new(pid: u32) -> Result<Self, CalibrateError> {
        let stat_path = PathBuf::from(format!("/proc/{pid}/stat"));
        if !stat_path.exists() {
            return Err(CalibrateError::ProcessNotFound { pid });
        }

        // SAFETY: `sysconf` is a POSIX call with a well-defined return value.
        let ticks_per_sec = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as u64;
        let ticks_per_sec = if ticks_per_sec == 0 { 100 } else { ticks_per_sec };

        Ok(Self {
            pid,
            stat_path,
            prev_cpu_ticks: None,
            prev_instant: None,
            ticks_per_sec,
        })
    }

    /// Poll once and return the CPU utilization since the last call.
    ///
    /// On the first call, returns `Percent(0.0)` because there is no
    /// previous sample to diff against.
    pub fn sample(&mut self) -> Result<Percent, CalibrateError> {
        let raw =
            std::fs::read_to_string(&self.stat_path).map_err(|e| CalibrateError::ProcRead {
                pid: self.pid,
                source: e,
            })?;

        let ticks = parse_cpu_ticks(&raw, self.pid)?;
        let now = Instant::now();

        let cpu_pct = if let (Some(prev_ticks), Some(prev_instant)) =
            (self.prev_cpu_ticks, self.prev_instant)
        {
            let elapsed_secs = prev_instant.elapsed().as_secs_f64();
            let tick_delta = ticks.saturating_sub(prev_ticks) as f64;
            let pct = (tick_delta / self.ticks_per_sec as f64 / elapsed_secs) * 100.0;
            Percent::clamped(pct as f32)
        } else {
            Percent(0.0)
        };

        self.prev_cpu_ticks = Some(ticks);
        self.prev_instant = Some(now);

        Ok(cpu_pct)
    }

    /// Returns `true` if the process is still alive (its `/proc` entry exists).
    pub fn is_alive(&self) -> bool {
        self.stat_path.exists()
    }
}

/// Parse fields 14 + 15 (utime + stime) from `/proc/{pid}/stat`.
///
/// The format has the process name in parentheses at position 2, which may
/// contain spaces — we find matching parens and skip past them before
/// parsing the numeric fields.
fn parse_cpu_ticks(raw: &str, pid: u32) -> Result<u64, CalibrateError> {
    // Find the closing paren of the comm field, then split the remainder.
    let after_comm = raw
        .rfind(')')
        .map(|i| &raw[i + 1..])
        .ok_or(CalibrateError::ProcFormat { pid })?;

    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    // After ')' the fields are: state(0), ppid(1), ... utime(11), stime(12)
    // (0-indexed in the remainder, so fields[11] = utime, fields[12] = stime).
    let utime: u64 = fields
        .get(11)
        .and_then(|s| s.parse().ok())
        .ok_or(CalibrateError::ProcFormat { pid })?;
    let stime: u64 = fields
        .get(12)
        .and_then(|s| s.parse().ok())
        .ok_or(CalibrateError::ProcFormat { pid })?;

    Ok(utime + stime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stat_simple() {
        // Simulated /proc/PID/stat with utime=100, stime=50 at fields 14+15.
        // We construct the string with the right structure.
        let line = "1234 (python3) S 0 0 0 0 -1 0 0 0 0 0 100 50 0 0 ...";
        let ticks = parse_cpu_ticks(line, 1234).unwrap();
        assert_eq!(ticks, 150);
    }

    #[test]
    fn parse_stat_comm_with_spaces() {
        // Process names can contain spaces — ensure rfind(')') handles it.
        let line = "5678 (my weird process) R 0 0 0 0 -1 0 0 0 0 0 200 80 0 0 ...";
        let ticks = parse_cpu_ticks(line, 5678).unwrap();
        assert_eq!(ticks, 280);
    }
}
