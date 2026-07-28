use regex::Regex;

/// A running process that looks like a training job.
#[derive(Debug, Clone)]
pub struct TrainingProcess {
    pub pid: u32,
    pub command: String,
}

impl std::fmt::Display for TrainingProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let short = if self.command.len() > 80 {
            format!("{}…", &self.command[..80])
        } else {
            self.command.clone()
        };
        write!(f, "PID {:6}  {}", self.pid, short)
    }
}

/// Resolve a PID from the three possible watch entry-points.
///
/// Precedence:
///   1. `--pid N`                → used directly
///   2. `--process-name PATTERN` → regex scan; error if 0 or >1 match
///   3. `--auto`                 → heuristic scan; interactive picker if >1 match
pub fn resolve_pid(
    pid: Option<u32>,
    process_name: Option<&str>,
    auto: bool,
) -> anyhow::Result<u32> {
    if let Some(p) = pid {
        return Ok(p);
    }

    if let Some(pattern) = process_name {
        return resolve_by_pattern(pattern);
    }

    if auto {
        return resolve_auto();
    }

    // clap's arg group guarantees one of the three is set, so this is unreachable.
    anyhow::bail!("no target process specified — use --pid, --process-name, or --auto");
}

//  --process-name

fn resolve_by_pattern(pattern: &str) -> anyhow::Result<u32> {
    let re = Regex::new(pattern).map_err(|e| {
        anyhow::anyhow!(
            "invalid regex '{pattern}': {e}\n\
             \n\
             Examples of valid patterns:\n\
               --process-name \"train\\.py\"\n\
               --process-name \"torch|lightning\"\n\
               --process-name \"minerva\""
        )
    })?;

    let candidates = all_processes()
        .into_iter()
        .filter(|p| re.is_match(&p.command))
        .collect::<Vec<_>>();

    match candidates.len() {
        0 => anyhow::bail!(
            "no running process matches the pattern '{pattern}'.\n\
             \n\
             Check currently visible processes with:\n\
               ps aux | grep python\n\
             Or use --auto to scan for training jobs automatically."
        ),
        1 => Ok(candidates[0].pid),
        _ => {
            let list = candidates
                .iter()
                .map(|p| format!("  {p}"))
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!(
                "{} processes match '{pattern}' — be more specific:\n\n\
                 {list}\n\n\
                 Use --pid <N> to attach to a specific one, or refine your pattern.\n\
                 Example: --process-name \"train\\.py --data\"",
                candidates.len()
            )
        }
    }
}

//  --auto

fn resolve_auto() -> anyhow::Result<u32> {
    let candidates = all_processes()
        .into_iter()
        .filter(|p| is_training_cmd(&p.command))
        .collect::<Vec<_>>();

    match candidates.len() {
        0 => anyhow::bail!(
            "no active training processes detected.\n\
             \n\
             calibrate looks for Python processes that use torch, tensorflow,\n\
             jax, keras, or lightning, or whose command line contains 'train',\n\
             'finetune', or 'pretrain'.\n\
             \n\
             If your process is running, try:\n\
               calibrate watch --process-name \"<script_name>\"\n\
               calibrate watch --pid $(pgrep -f train.py)"
        ),
        1 => {
            eprintln!(
                "[calibrate] Auto-detected training process:\n  {}",
                candidates[0]
            );
            Ok(candidates[0].pid)
        }
        _ => {
            eprintln!("[calibrate] Multiple training processes found. Select one:\n");
            for (i, p) in candidates.iter().enumerate() {
                eprintln!("  [{:>2}]  {p}", i + 1);
            }
            eprintln!();
            eprint!("Enter number [1-{}]: ", candidates.len());

            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| anyhow::anyhow!("failed to read selection: {e}"))?;

            let choice: usize = input.trim().parse().map_err(|_| {
                anyhow::anyhow!(
                    "invalid selection '{}' — expected a number between 1 and {}",
                    input.trim(),
                    candidates.len()
                )
            })?;

            if choice < 1 || choice > candidates.len() {
                anyhow::bail!(
                    "selection {choice} is out of range (1–{})",
                    candidates.len()
                );
            }

            Ok(candidates[choice - 1].pid)
        }
    }
}

//  Process enumeration ─

/// Return all visible processes as (pid, full_cmdline) pairs.
fn all_processes() -> Vec<TrainingProcess> {
    let mut results = Vec::new();

    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Ok(pid) = name.parse::<u32>() {
                    if pid == std::process::id() {
                        continue;
                    }
                    let cmdline_path = format!("/proc/{pid}/cmdline");
                    if let Ok(bytes) = std::fs::read(cmdline_path) {
                        let cmd = bytes
                            .split(|&b| b == 0)
                            .filter(|s| !s.is_empty())
                            .map(|s| String::from_utf8_lossy(s).into_owned())
                            .collect::<Vec<_>>()
                            .join(" ");
                        if !cmd.is_empty() {
                            results.push(TrainingProcess { pid, command: cmd });
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(out) = Command::new("ps").args(["-eo", "pid,command"]).output() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            for line in stdout.lines().skip(1) {
                let trimmed = line.trim();
                let mut parts = trimmed.splitn(2, ' ');
                if let (Some(pid_str), Some(cmd)) = (parts.next(), parts.next()) {
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        if pid == std::process::id() {
                            continue;
                        }
                        results.push(TrainingProcess {
                            pid,
                            command: cmd.trim().to_string(),
                        });
                    }
                }
            }
        }
    }

    results
}

/// Returns true if `cmd` looks like a Python-based training job.
pub fn is_training_cmd(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let is_python = lower.contains("python") || lower.contains("python3");
    let has_training_keyword = ["train", "finetune", "fine_tune", "pretrain"]
        .iter()
        .any(|k| lower.contains(k));
    let has_framework = ["torch", "tensorflow", "jax", "keras", "lightning"]
        .iter()
        .any(|f| lower.contains(f));
    is_python && (has_training_keyword || has_framework)
}

//  Tests ─

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_pid_returns_given_pid_directly() {
        assert_eq!(resolve_pid(Some(42), None, false).unwrap(), 42);
    }

    #[test]
    fn resolve_by_pattern_invalid_regex_gives_actionable_error() {
        let err = resolve_pid(None, Some("[invalid"), false).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("invalid regex"),
            "expected 'invalid regex' in error; got: {msg}"
        );
        assert!(
            msg.contains("Examples"),
            "error should include usage examples; got: {msg}"
        );
    }

    #[test]
    fn is_training_cmd_positive_cases() {
        assert!(is_training_cmd("python3 train.py --data /data"));
        assert!(is_training_cmd("python finetune.py --model llama"));
        assert!(is_training_cmd("python3 run.py --backend torch"));
        assert!(is_training_cmd(
            "python3 -m lightning fit --config cfg.yaml"
        ));
    }

    #[test]
    fn is_training_cmd_negative_cases() {
        assert!(!is_training_cmd("python3 serve.py --port 8080"));
        assert!(!is_training_cmd("node train.js"));
        assert!(!is_training_cmd(""));
    }

    #[test]
    fn is_training_cmd_case_insensitive() {
        assert!(is_training_cmd("PYTHON3 TRAIN.PY"));
    }
}
