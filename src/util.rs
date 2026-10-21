//! Small shared helpers.

use std::io::Read;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

/// Run a command to completion, killing it if it exceeds `timeout`.
///
/// stdout/stderr are drained on background threads so a chatty child cannot
/// deadlock on a full pipe while we wait.
pub fn output_with_timeout(cmd: &mut Command, timeout: Duration) -> Result<Output> {
    let program = format!("{:?}", cmd.get_program());
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch {program}"))?;
    let mut out = child.stdout.take().expect("piped");
    let mut err = child.stderr.take().expect("piped");
    let out_t = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = out.read_to_end(&mut b);
        b
    });
    let err_t = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = err.read_to_end(&mut b);
        b
    });
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "{program} timed out after {:.0}s and was killed",
                timeout.as_secs_f64()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Ok(Output {
        status,
        stdout: out_t.join().unwrap_or_default(),
        stderr: err_t.join().unwrap_or_default(),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn captures_output() {
        let o = output_with_timeout(
            Command::new("sh").args(["-c", "echo hi; echo err >&2"]),
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(o.status.success());
        assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "hi");
        assert_eq!(String::from_utf8_lossy(&o.stderr).trim(), "err");
    }

    #[test]
    fn kills_on_timeout() {
        let t = Instant::now();
        let e = output_with_timeout(Command::new("sleep").arg("30"), Duration::from_millis(300))
            .unwrap_err();
        assert!(e.to_string().contains("timed out"));
        assert!(t.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn large_output_does_not_deadlock() {
        let o = output_with_timeout(
            Command::new("sh").args(["-c", "head -c 2000000 /dev/zero"]),
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(o.stdout.len(), 2_000_000);
    }
}
