use regex::Regex;

use crate::collectors::proc_tree;

/// A running process that looks like a training job.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingProcess {
    pub pid: u32,
    pub ppid: u32,
    pub command: String,
}

/// Truncate to at most `max` characters, on a char boundary.
pub fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

impl std::fmt::Display for TrainingProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PID {:6}  {}",
            self.pid,
            truncate_chars(&self.command, 80)
        )
    }
}

/// Resolve a PID from the watch entry-points.
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
    anyhow::bail!("no target process specified — use --pid, --process-name, --auto or --gpus");
}

/// Keep only candidates none of whose ancestors are also candidates, so a
/// torchrun launcher and its ranks collapse into the launcher (calibrate
/// follows children automatically).
pub fn collapse_trees(cands: Vec<TrainingProcess>, table: &[(u32, u32)]) -> Vec<TrainingProcess> {
    let pids: Vec<u32> = cands.iter().map(|c| c.pid).collect();
    cands
        .into_iter()
        .filter(|c| {
            !proc_tree::ancestors_in(table, c.pid)
                .iter()
                .any(|a| pids.contains(a))
        })
        .collect()
}

/// Exclude calibrate itself and its ancestors (`sudo`, the invoking shell),
/// whose command lines contain the user's pattern.
fn exclude_self(procs: Vec<TrainingProcess>, table: &[(u32, u32)]) -> Vec<TrainingProcess> {
    let me = std::process::id();
    let mut skip = proc_tree::ancestors_in(table, me);
    skip.push(me);
    procs
        .into_iter()
        .filter(|p| !skip.contains(&p.pid))
        .collect()
}

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
    let table = proc_tree::process_table();
    let matches: Vec<TrainingProcess> = exclude_self(all_processes(&table), &table)
        .into_iter()
        .filter(|p| re.is_match(&p.command))
        .collect();
    let n_raw = matches.len();
    let candidates = collapse_trees(matches, &table);

    match candidates.len() {
        0 => anyhow::bail!(
            "no running process matches the pattern '{pattern}'.\n\
             \n\
             Check currently visible processes with:\n\
               ps aux | grep python\n\
             Or use --auto to scan for training jobs automatically."
        ),
        1 => {
            if n_raw > 1 {
                eprintln!(
                    "[calibrate] {n_raw} matching processes belong to one tree — attaching to its \
                     root (children are included):\n  {}",
                    candidates[0]
                );
            }
            Ok(candidates[0].pid)
        }
        _ => {
            let list = candidates
                .iter()
                .map(|p| format!("  {p}"))
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!(
                "{} unrelated processes match '{pattern}' — be more specific:\n\n\
                 {list}\n\n\
                 Use --pid <N> to attach to a specific one, or refine your pattern.",
                candidates.len()
            )
        }
    }
}

fn resolve_auto() -> anyhow::Result<u32> {
    let table = proc_tree::process_table();
    let found: Vec<TrainingProcess> = exclude_self(all_processes(&table), &table)
        .into_iter()
        .filter(|p| is_training_cmd(&p.command))
        .collect();
    let candidates = collapse_trees(found, &table);

    match candidates.len() {
        0 => anyhow::bail!(
            "no active training processes detected.\n\
             \n\
             calibrate looks for Python processes and launchers (torchrun, accelerate,\n\
             deepspeed) that use torch, tensorflow, jax, keras or lightning, or whose\n\
             command line contains 'train', 'finetune' or 'pretrain'.\n\
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
            use std::io::IsTerminal;
            if !std::io::stdin().is_terminal() {
                let list = candidates
                    .iter()
                    .map(|p| format!("  {p}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                anyhow::bail!(
                    "{} training processes found and stdin is not a terminal, so no menu \
                     can be shown:\n\n{list}\n\nUse --pid <N>.",
                    candidates.len()
                );
            }
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

/// Training-like processes, one per process tree (for `diagnose`).
pub fn training_processes() -> Vec<TrainingProcess> {
    let table = proc_tree::process_table();
    let found = exclude_self(all_processes(&table), &table)
        .into_iter()
        .filter(|p| is_training_cmd(&p.command))
        .collect();
    collapse_trees(found, &table)
}

/// All visible processes with their full command lines.
fn all_processes(table: &[(u32, u32)]) -> Vec<TrainingProcess> {
    let ppid_of = |pid: u32| {
        table
            .iter()
            .find(|(p, _)| *p == pid)
            .map(|(_, pp)| *pp)
            .unwrap_or(0)
    };
    let mut results = Vec::new();

    #[cfg(target_os = "linux")]
    for &(pid, ppid) in table {
        if let Ok(bytes) = std::fs::read(format!("/proc/{pid}/cmdline")) {
            let cmd = bytes
                .split(|&b| b == 0)
                .filter(|s| !s.is_empty())
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect::<Vec<_>>()
                .join(" ");
            if !cmd.is_empty() {
                results.push(TrainingProcess {
                    pid,
                    ppid,
                    command: cmd,
                });
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(out) = Command::new("ps").args(["-axo", "pid=,command="]).output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let trimmed = line.trim_start();
                let Some((pid_str, cmd)) = trimmed.split_once(' ') else {
                    continue;
                };
                if let Ok(pid) = pid_str.parse::<u32>() {
                    results.push(TrainingProcess {
                        pid,
                        ppid: ppid_of(pid),
                        command: cmd.trim().to_string(),
                    });
                }
            }
        }
    }

    let _ = ppid_of;
    results
}

/// Returns true if `cmd` looks like a training job.
pub fn is_training_cmd(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let launcher = [
        "torchrun",
        "accelerate launch",
        "deepspeed ",
        "torch.distributed.run",
    ]
    .iter()
    .any(|k| lower.contains(k));
    let is_python = lower.contains("python");
    let has_training_keyword = ["train", "finetune", "fine_tune", "pretrain", "sft"]
        .iter()
        .any(|k| lower.contains(k));
    let has_framework = ["torch", "tensorflow", "jax", "keras", "lightning"]
        .iter()
        .any(|f| lower.contains(f));
    launcher || (is_python && (has_training_keyword || has_framework))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_pid_returns_given_pid_directly() {
        assert_eq!(resolve_pid(Some(42), None, false).unwrap(), 42);
    }

    #[test]
    fn invalid_regex_gives_actionable_error() {
        let msg = resolve_pid(None, Some("[invalid"), false)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("invalid regex"), "{msg}");
        assert!(msg.contains("Examples"), "{msg}");
    }

    #[test]
    fn truncation_is_utf8_safe() {
        // 79 ASCII bytes then multi-byte characters straddling byte 80.
        let cmd = format!("{}日本語のパス/train.py", "a".repeat(79));
        let p = TrainingProcess {
            pid: 1,
            ppid: 0,
            command: cmd,
        };
        let shown = p.to_string();
        assert!(shown.ends_with('…'));
        assert_eq!(truncate_chars("short", 80), "short");
        assert_eq!(truncate_chars("héllo", 2), "hé…");
    }

    #[test]
    fn launcher_and_ranks_collapse_to_launcher() {
        let table = vec![(1, 0), (100, 1), (101, 100), (102, 100), (300, 1)];
        let mk = |pid, ppid| TrainingProcess {
            pid,
            ppid,
            command: "python train.py".into(),
        };
        let out = collapse_trees(vec![mk(100, 1), mk(101, 100), mk(102, 100)], &table);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pid, 100);
        let out = collapse_trees(vec![mk(101, 100), mk(300, 1)], &table);
        assert_eq!(out.len(), 2, "unrelated jobs stay separate");
    }

    #[test]
    fn calibrate_does_not_match_itself_or_its_ancestors() {
        let table = proc_tree::process_table();
        let me = std::process::id();
        let procs = vec![TrainingProcess {
            pid: me,
            ppid: 0,
            command: "calibrate watch --process-name train".into(),
        }];
        assert!(exclude_self(procs, &table).is_empty());
    }

    #[test]
    fn training_command_detection() {
        assert!(is_training_cmd("python3 train.py --data /data"));
        assert!(is_training_cmd("python finetune.py --model llama"));
        assert!(is_training_cmd("python3 run.py --backend torch"));
        assert!(is_training_cmd(
            "python3 -m lightning fit --config cfg.yaml"
        ));
        assert!(is_training_cmd(
            "/usr/bin/torchrun --nproc-per-node 8 main.py"
        ));
        assert!(is_training_cmd("accelerate launch sft.py"));
        assert!(!is_training_cmd("python3 serve.py --port 8080"));
        assert!(!is_training_cmd("node train.js"));
        assert!(!is_training_cmd(""));
        assert!(is_training_cmd("PYTHON3 TRAIN.PY"));
    }
}
