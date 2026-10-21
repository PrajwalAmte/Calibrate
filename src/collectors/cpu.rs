//! CPU usage of a process tree, as a percentage of one core.
//!
//! Summing over the tree matters: with `num_workers > 0` the DataLoader runs
//! in child processes, so the main process alone looks idle exactly when the
//! input pipeline is the bottleneck.

use std::collections::HashMap;
use std::time::Instant;

/// Cumulative CPU time (user + system) consumed by `pid`, in nanoseconds.
pub fn process_cpu_ns(pid: u32) -> Option<u64> {
    platform::process_cpu_ns(pid)
}

/// Differences cumulative CPU time across calls.
#[derive(Debug, Default)]
pub struct TreeCpuSampler {
    prev: HashMap<u32, u64>,
    prev_at: Option<Instant>,
}

impl TreeCpuSampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// CPU use of `pids` since the previous call, in percent of one core.
    /// Returns `None` on the first call or when nothing could be read.
    pub fn sample(&mut self, pids: &[u32]) -> Option<f32> {
        let now = Instant::now();
        let current: HashMap<u32, u64> = pids
            .iter()
            .filter_map(|&p| process_cpu_ns(p).map(|ns| (p, ns)))
            .collect();
        let result = self.prev_at.and_then(|prev_at| {
            let wall_ns = now.duration_since(prev_at).as_nanos() as f64;
            delta_pct(&self.prev, &current, wall_ns)
        });
        self.prev = current;
        self.prev_at = Some(now);
        result
    }
}

/// Percent of one core used between two readings. Processes that appeared
/// or exited in between are excluded (their delta is unknown), which keeps
/// the figure from going negative or spiking.
pub fn delta_pct(prev: &HashMap<u32, u64>, cur: &HashMap<u32, u64>, wall_ns: f64) -> Option<f32> {
    if wall_ns <= 0.0 || cur.is_empty() {
        return None;
    }
    let busy_ns: u64 = cur
        .iter()
        .filter_map(|(pid, &now)| prev.get(pid).map(|&before| now.saturating_sub(before)))
        .sum();
    if !cur.keys().any(|p| prev.contains_key(p)) {
        return None;
    }
    Some((busy_ns as f64 / wall_ns * 100.0).max(0.0) as f32)
}

#[cfg(target_os = "linux")]
mod platform {
    use std::sync::OnceLock;

    fn ticks_per_sec() -> u64 {
        static TCK: OnceLock<u64> = OnceLock::new();
        *TCK.get_or_init(|| {
            // SAFETY: sysconf has no preconditions.
            let t = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
            if t > 0 {
                t as u64
            } else {
                100
            }
        })
    }

    pub fn process_cpu_ns(pid: u32) -> Option<u64> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let ticks = super::parse_stat_cpu_ticks(&stat)?;
        Some(ticks * 1_000_000_000 / ticks_per_sec())
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::sync::OnceLock;

    /// Mach time → nanoseconds. On Apple Silicon `pti_total_*` are reported in
    /// Mach ticks (24 MHz timebase, numer/denom = 125/3), on Intel in ns.
    #[allow(deprecated)] // libc's mach_timebase_info is stable ABI; avoid a mach2 dependency
    fn timebase() -> (u64, u64) {
        static TB: OnceLock<(u64, u64)> = OnceLock::new();
        *TB.get_or_init(|| {
            let mut info = libc::mach_timebase_info { numer: 0, denom: 0 };
            // SAFETY: writes two u32s into `info`.
            let r = unsafe { libc::mach_timebase_info(&mut info) };
            if r == 0 && info.numer > 0 && info.denom > 0 {
                (info.numer as u64, info.denom as u64)
            } else {
                (1, 1)
            }
        })
    }

    pub fn process_cpu_ns(pid: u32) -> Option<u64> {
        // SAFETY: proc_taskinfo is plain-old-data; proc_pidinfo writes at most
        // `size` bytes and returns the number written.
        let ticks = unsafe {
            let mut info: libc::proc_taskinfo = std::mem::zeroed();
            let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
            let r = libc::proc_pidinfo(
                pid as libc::pid_t,
                libc::PROC_PIDTASKINFO,
                0,
                &mut info as *mut _ as *mut libc::c_void,
                size,
            );
            if r != size {
                return None;
            }
            info.pti_total_user + info.pti_total_system
        };
        let (numer, denom) = timebase();
        Some((ticks as u128 * numer as u128 / denom as u128) as u64)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    pub fn process_cpu_ns(_pid: u32) -> Option<u64> {
        None
    }
}

/// utime + stime (fields 14 and 15) from /proc/PID/stat, in clock ticks.
/// The comm field may contain spaces and parentheses, so split after the
/// last ')'.
#[allow(dead_code)] // used on Linux; tested everywhere
pub fn parse_stat_cpu_ticks(stat: &str) -> Option<u64> {
    let after = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stat_simple() {
        let line = "1234 (python3) S 0 0 0 0 -1 0 0 0 0 0 100 50 0 0 20 0";
        assert_eq!(parse_stat_cpu_ticks(line), Some(150));
    }

    #[test]
    fn parses_stat_comm_with_spaces_and_parens() {
        let line = "5678 (my (weird) proc) R 0 0 0 0 -1 0 0 0 0 0 200 80 0 0";
        assert_eq!(parse_stat_cpu_ticks(line), Some(280));
    }

    #[test]
    fn malformed_stat_is_none() {
        assert_eq!(parse_stat_cpu_ticks("garbage"), None);
        assert_eq!(parse_stat_cpu_ticks("1 (x) S 1 2"), None);
    }

    #[test]
    fn delta_sums_across_processes_and_can_exceed_100() {
        let prev = HashMap::from([(1, 0u64), (2, 0)]);
        let cur = HashMap::from([(1, 1_000_000_000u64), (2, 500_000_000)]);
        // 1.5 s of CPU in 1 s of wall time = 150 % of one core.
        let pct = delta_pct(&prev, &cur, 1e9).unwrap();
        assert!((pct - 150.0).abs() < 0.01, "{pct}");
    }

    #[test]
    fn new_and_exited_processes_are_ignored() {
        let prev = HashMap::from([(1, 100u64), (9, 5_000)]);
        let cur = HashMap::from([(1, 100 + 250_000_000u64), (2, 9_999_999_999)]);
        let pct = delta_pct(&prev, &cur, 1e9).unwrap();
        assert!((pct - 25.0).abs() < 0.01, "{pct}");
    }

    #[test]
    fn no_overlap_is_none() {
        let prev = HashMap::from([(1, 0u64)]);
        let cur = HashMap::from([(2, 10u64)]);
        assert_eq!(delta_pct(&prev, &cur, 1e9), None);
    }

    #[test]
    fn first_sample_is_none_then_measures_own_cpu() {
        let me = std::process::id();
        let mut s = TreeCpuSampler::new();
        assert_eq!(s.sample(&[me]), None);
        // Burn some CPU so the second reading is non-trivial.
        let t = Instant::now();
        let mut x = 0u64;
        while t.elapsed().as_millis() < 150 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        std::hint::black_box(x);
        let pct = s.sample(&[me]);
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            let pct = pct.expect("own CPU time must be readable");
            assert!(pct > 20.0 && pct < 1000.0, "implausible CPU% {pct}");
        }
    }
}
