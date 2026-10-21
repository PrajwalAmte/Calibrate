//! End-to-end tests against the compiled binary.
//!
//! Every test runs offline with an isolated HOME so it never touches the
//! network or the developer's caches/config.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

struct Env {
    home: tempfile::TempDir,
}

impl Env {
    fn new() -> Self {
        Self {
            home: tempfile::tempdir().unwrap(),
        }
    }
    fn cmd(&self) -> Command {
        let mut c = Command::cargo_bin("calibrate").unwrap();
        c.env("HOME", self.home.path())
            .env("XDG_CACHE_HOME", self.home.path().join(".cache"))
            .env("XDG_CONFIG_HOME", self.home.path().join(".config"))
            .env("CALIBRATE_OFFLINE", "1")
            .env_remove("CALIBRATE_CONFIG")
            .env_remove("HF_TOKEN")
            .env("RUST_LOG", "error");
        c
    }
}

fn json_lines(out: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(out)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

fn replay(env: &Env, name: &str) -> (Vec<Value>, Value) {
    let out = env
        .cmd()
        .args(["watch", "--replay"])
        .arg(fixture(name))
        .args(["--output", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let lines = json_lines(&out);
    let last = lines
        .iter()
        .rev()
        .find(|l| l["event"] == "snapshot")
        .expect("at least one snapshot")
        .clone();
    (lines, last)
}

//  Global CLI

#[test]
fn help_and_version() {
    let env = Env::new();
    env.cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("watch").and(predicate::str::contains("plan")));
    env.cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn watch_options_combine_with_a_target() {
    // Regression: `--pid` used to conflict with every other option.
    let env = Env::new();
    env.cmd()
        .args([
            "watch",
            "--pid",
            "999999999",
            "--cost-per-hour",
            "0.34",
            "--interval",
            "1",
        ])
        .args(["--output", "json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with").not())
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn watch_argument_validation() {
    let env = Env::new();
    env.cmd()
        .args(["watch", "--interval", "1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
    env.cmd()
        .args(["watch", "--pid", "1", "--interval", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("between 0.1 and 3600"));
    env.cmd()
        .args(["watch", "--pid", "1", "--auto"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn terminal_output_requires_a_tty() {
    let env = Env::new();
    env.cmd()
        .args(["watch", "--replay"])
        .arg(fixture("healthy_h100_gpm.jsonl"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("interactive terminal"));
}

//  Replay of recorded traces through the full analysis

#[test]
fn replay_healthy_h100() {
    let env = Env::new();
    let (lines, s) = replay(&env, "healthy_h100_gpm.jsonl");
    assert_eq!(s["schema_version"], 1);
    assert_eq!(s["mfu"]["source"], "tensor_active");
    assert_eq!(s["mfu"]["upper_bound"], false);
    let mfu = s["mfu"]["mfu_pct"].as_f64().unwrap();
    assert!((50.0..60.0).contains(&mfu), "{mfu}");
    assert_eq!(s["bottleneck"], "none");
    assert_eq!(s["gpus"][0]["peak_tflops"], 989.0);
    let end = lines.last().unwrap();
    assert_eq!(end["event"], "session_end");
    assert_eq!(end["ticks_observed"], 40);
}

#[test]
fn replay_dataloader_bound() {
    let env = Env::new();
    let (_, s) = replay(&env, "dataloader_bound.jsonl");
    assert_eq!(s["bottleneck"], "data_loader");
    assert_eq!(s["breakdown"]["source"], "heuristic");
    assert!(s["recommendation"]["action"]
        .as_str()
        .unwrap()
        .contains("num_workers"));
}

#[test]
fn replay_kernel_busy_is_upper_bound_and_unverified() {
    let env = Env::new();
    let (_, s) = replay(&env, "kernel_busy_only_a10g.jsonl");
    assert_eq!(s["mfu"]["source"], "kernel_busy");
    assert_eq!(s["mfu"]["upper_bound"], true);
    assert_eq!(
        s["bottleneck"], "unverified",
        "a busy GPU is not proof of efficiency"
    );
}

#[test]
fn replay_thermal_straggler_leak_cpu_only() {
    let env = Env::new();
    assert_eq!(
        replay(&env, "thermal_throttle.jsonl").1["bottleneck"],
        "thermal_throttle"
    );
    let (_, s) = replay(&env, "multi_gpu_straggler.jsonl");
    assert_eq!(s["bottleneck"], "gpu_imbalance");
    assert_eq!(s["gpus"].as_array().unwrap().len(), 4);
    assert_eq!(s["gpu_divergent"], true);
    let (_, s) = replay(&env, "vram_leak.jsonl");
    assert_eq!(s["vram_growing"], true);
    let (_, s) = replay(&env, "cpu_only.jsonl");
    assert_eq!(s["bottleneck"], "no_gpu_metrics");
    assert!(
        s["mfu"]["mfu_pct"].is_null(),
        "no GPU data must not be reported as 0 %"
    );
}

#[test]
fn replay_hooked_job_reports_measured_mfu() {
    let env = Env::new();
    let (_, s) = replay(&env, "hooked_instrumented.jsonl");
    assert_eq!(s["mfu"]["source"], "instrumented");
    let mfu = s["mfu"]["mfu_pct"].as_f64().unwrap();
    assert!((mfu - 40.0).abs() < 0.5, "{mfu}");
    assert_eq!(s["breakdown"]["source"], "instrumented");
    let wait = s["breakdown"]["data_wait_pct"].as_f64().unwrap();
    assert!((wait - 20.0).abs() < 0.5, "{wait}");
    assert_eq!(s["hook_peers"], 1);
    assert!(s["steps"]["tokens_per_sec"].as_f64().unwrap() > 0.0);
}

#[test]
fn cost_tracking_through_replay() {
    let env = Env::new();
    let out = env
        .cmd()
        .args(["watch", "--replay"])
        .arg(fixture("dataloader_bound.jsonl"))
        .args(["--output", "json", "--cost-per-hour", "3.6"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let end = json_lines(&out).pop().unwrap();
    let c = &end["cost"];
    // 39 intervals × 2 s = 78 s at $3.60/h = $0.078
    assert!((c["spent_usd"].as_f64().unwrap() - 0.078).abs() < 1e-6);
    assert!(c["wasted_usd"].as_f64().unwrap() > 0.0);
}

#[test]
fn record_and_record_raw_round_trip() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("raw.jsonl");
    let snaps = dir.path().join("snaps.jsonl");
    env.cmd()
        .args(["watch", "--replay"])
        .arg(fixture("multi_gpu_straggler.jsonl"))
        .args(["--output", "none", "--record-raw"])
        .arg(&raw)
        .arg("--record")
        .arg(&snaps)
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
    let recorded = json_lines(&std::fs::read(&snaps).unwrap());
    assert_eq!(
        recorded.iter().filter(|l| l["event"] == "snapshot").count(),
        30
    );
    assert_eq!(recorded.last().unwrap()["event"], "session_end");
    // The re-recorded raw trace replays to the same result.
    let (_, again) = replay(&env, raw.to_str().unwrap());
    let original = recorded
        .iter()
        .rev()
        .find(|l| l["event"] == "snapshot")
        .unwrap();
    assert_eq!(again["mfu"]["mfu_pct"], original["mfu"]["mfu_pct"]);
    assert_eq!(again["bottleneck"], original["bottleneck"]);
}

#[test]
fn invalid_trace_is_rejected() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.jsonl");
    std::fs::write(&bad, "{not json}\n").unwrap();
    env.cmd()
        .args(["watch", "--output", "json", "--replay"])
        .arg(&bad)
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid tick"));
}

//  Config file

#[test]
fn config_file_is_applied_and_validated() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.toml");
    std::fs::write(&good, "[watch]\ntarget_mfu = 60.0\n").unwrap();
    let out = env
        .cmd()
        .arg("--config")
        .arg(&good)
        .args(["watch", "--output", "json", "--replay"])
        .arg(fixture("healthy_h100_gpm.jsonl"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = json_lines(&out)
        .into_iter()
        .rev()
        .find(|l| l["event"] == "snapshot")
        .unwrap();
    assert_eq!(s["target_mfu_pct"], 60.0);
    assert_eq!(
        s["bottleneck"], "below_target_mfu",
        "55 % is below a 60 % target"
    );

    let bad = dir.path().join("bad.toml");
    std::fs::write(&bad, "[watch]\nintervall = 2\n").unwrap();
    env.cmd()
        .arg("--config")
        .arg(&bad)
        .args(["watch", "--gpus", "all"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown field"));
    env.cmd()
        .args([
            "--config",
            "/no/such/file.toml",
            "plan",
            "-m",
            "x",
            "--params-b",
            "7",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("config file not found"));
}

//  plan

fn prices_file(dir: &Path) -> PathBuf {
    let p = dir.join("prices.json");
    std::fs::write(
        &p,
        r#"[
          {"provider":"onprem","gpu_model":"H100 SXM","gpu_count":8,"vram_per_gpu_gib":80,"hourly_usd":20.0},
          {"provider":"cloud","gpu_model":"RTX 4090","vram_per_gpu_gib":24,"hourly_usd":0.44},
          {"provider":"cloud","gpu_model":"RTX 3080","vram_per_gpu_gib":10,"hourly_usd":0.10}
        ]"#,
    )
    .unwrap();
    p
}

#[test]
fn plan_offline_golden() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let out = env
        .cmd()
        .args([
            "plan",
            "-m",
            "my-org/my-8b",
            "--params-b",
            "8.03",
            "--method",
            "lora",
            "--dataset-rows",
            "50000",
            "--output",
            "json",
            "--prices-file",
        ])
        .arg(prices_file(dir.path()))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let r: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(r["schema_version"], 1);
    assert_eq!(r["price_source"], "file");
    assert_eq!(r["excluded_too_small"], 1, "10 GB card is excluded");
    let req = r["workload"]["required_vram_gib"].as_f64().unwrap();
    assert!((16.0..24.0).contains(&req), "{req}");
    let listings = r["listings"].as_array().unwrap();
    assert_eq!(listings.len(), 2);
    for l in listings {
        let d = &l["duration_range"];
        let hours = d["low_secs"].as_f64().unwrap() / 3600.0;
        assert!(
            hours > 0.5,
            "8B × 1e8 tokens is hours of work, got {hours} h"
        );
    }
    assert!(r["recommendation"]["listing"]["gpu_model"].is_string());
}

#[test]
fn plan_rejects_bad_input() {
    let env = Env::new();
    env.cmd()
        .args([
            "plan",
            "-m",
            "x",
            "--params-b",
            "7",
            "--method",
            "full",
            "--quantization",
            "4bit",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("qlora"));
    env.cmd()
        .args(["plan", "-m", "some/model"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--offline"));
    env.cmd()
        .args(["plan", "-m", "x", "--mfu", "1.5"])
        .assert()
        .failure();
}

#[test]
fn plan_terminal_output_is_readable() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    env.cmd()
        .args([
            "plan",
            "-m",
            "x",
            "--params-b",
            "7",
            "--dataset-rows",
            "1000",
            "--prices-file",
        ])
        .arg(prices_file(dir.path()))
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Recommendation:")
                .and(predicate::str::contains("Assumptions:"))
                .and(predicate::str::contains("8× H100 SXM")),
        );
}

//  bench / probe / diagnose

#[test]
fn bench_error_paths() {
    let env = Env::new();
    env.cmd()
        .args(["bench", "--model", "/no/such/model.onnx"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
    let dir = tempfile::tempdir().unwrap();
    let m = dir.path().join("m.onnx");
    std::fs::write(&m, b"x").unwrap();
    env.cmd()
        .args(["bench", "--runtimes", "candle", "--model"])
        .arg(&m)
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown runtime"));
}

#[test]
fn probe_requires_a_target() {
    Env::new().cmd().arg("probe").assert().failure();
}

#[test]
fn diagnose_runs_and_reports_sections() {
    let out = Env::new().cmd().arg("diagnose").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("[ Platform ]"));
    assert!(text.contains("[ Process visibility ]"));
    assert!(text.contains("Can observe own PID"));
    assert!(text.contains("Result:"));
}

//  Live collection against a real child process

#[cfg(unix)]
#[test]
fn live_watch_of_a_process_tree_emits_versioned_json() {
    let env = Env::new();
    let mut child = std::process::Command::new("sh")
        .args(["-c", "sleep 30 & sleep 30 & wait"])
        .spawn()
        .unwrap();
    let out = env
        .cmd()
        .args(["watch", "--pid", &child.id().to_string()])
        .args(["--interval", "0.3", "--duration", "2", "--output", "json"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let _ = child.kill();
    let _ = child.wait();
    let lines = json_lines(&out);
    let snaps: Vec<&Value> = lines.iter().filter(|l| l["event"] == "snapshot").collect();
    assert!(snaps.len() >= 3, "got {} snapshots", snaps.len());
    let s = snaps.last().unwrap();
    assert_eq!(s["schema_version"], 1);
    assert_eq!(s["target"]["mode"], "process");
    assert!(
        s["target"]["process_count"].as_u64().unwrap() >= 3,
        "children are included: {}",
        s["target"]["process_count"]
    );
    assert_eq!(lines.last().unwrap()["event"], "session_end");
}

#[cfg(unix)]
#[test]
fn watch_stops_when_the_target_exits() {
    let env = Env::new();
    let mut child = std::process::Command::new("sleep")
        .arg("1")
        .spawn()
        .unwrap();
    let pid = child.id().to_string();
    let reaper = std::thread::spawn(move || child.wait());
    let start = std::time::Instant::now();
    env.cmd()
        .args([
            "watch",
            "--pid",
            &pid,
            "--interval",
            "0.2",
            "--output",
            "json",
        ])
        .timeout(std::time::Duration::from_secs(30))
        .assert()
        .success();
    assert!(start.elapsed().as_secs() < 15);
    reaper.join().unwrap().unwrap();
}

/// Traces recorded on real GPUs by `scripts/gpu-validation.sh` are committed
/// under tests/fixtures/real/ and must keep replaying cleanly.
#[test]
fn real_gpu_traces_replay_cleanly() {
    let root = fixture("real");
    let Ok(gpus) = std::fs::read_dir(&root) else {
        return; // none recorded yet
    };
    let env = Env::new();
    for gpu in gpus.flatten() {
        for f in std::fs::read_dir(gpu.path()).unwrap().flatten() {
            let p = f.path();
            if p.to_string_lossy().ends_with(".trace.jsonl") {
                let out = env
                    .cmd()
                    .args(["watch", "--output", "json", "--replay"])
                    .arg(&p)
                    .assert()
                    .success()
                    .get_output()
                    .stdout
                    .clone();
                let lines = json_lines(&out);
                assert!(
                    lines.iter().any(|l| l["event"] == "snapshot"),
                    "{} produced no snapshots",
                    p.display()
                );
            }
        }
    }
}
