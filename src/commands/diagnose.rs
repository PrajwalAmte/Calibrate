use std::io::Write;
use std::process::Command;

use crate::cli::DiagnoseArgs;

/// Outcome of a single check.
#[derive(Debug)]
enum Status {
    Pass(String),
    Warn(String),
    Fail(String),
    Info(String),
}

impl Status {
    fn prefix(&self) -> &'static str {
        match self {
            Status::Pass(_) => "✓",
            Status::Warn(_) => "⚠",
            Status::Fail(_) => "✗",
            Status::Info(_) => "·",
        }
    }
    fn message(&self) -> &str {
        match self {
            Status::Pass(m) | Status::Warn(m) | Status::Fail(m) | Status::Info(m) => m,
        }
    }
    fn is_fail(&self) -> bool {
        matches!(self, Status::Fail(_))
    }
    fn is_warn(&self) -> bool {
        matches!(self, Status::Warn(_))
    }
}

pub async fn run(_args: DiagnoseArgs) -> anyhow::Result<()> {
    println!("calibrate diagnose — system capability check\n");
    let mut sections: Vec<(&str, Vec<Status>)> = vec![("Platform", check_platform())];

    if cfg!(target_os = "linux") {
        sections.push(("NVIDIA GPU", check_nvidia()));
        sections.push(("GPU profiling (MFU accuracy)", check_profiling()));
    }
    #[cfg(target_os = "macos")]
    sections.push(("Apple GPU", check_apple_gpu()));

    sections.push(("Process visibility", check_proc_visibility()));
    sections.push(("Configuration & network", check_config_and_network().await));

    let (mut fail, mut warn) = (false, false);
    for (title, checks) in &sections {
        println!("[ {title} ]");
        for s in checks {
            println!("  {}  {}", s.prefix(), s.message());
            fail |= s.is_fail();
            warn |= s.is_warn();
        }
        println!();
    }

    if fail {
        println!("Result: FAIL — one or more required components are missing.");
        println!("        Fix the items marked ✗ above, then re-run `calibrate diagnose`.");
        let _ = std::io::stdout().flush();
        std::process::exit(1);
    } else if warn {
        println!("Result: WARN — calibrate will work with limited functionality.");
        println!("        Review the items marked ⚠ above for full support.");
    } else {
        println!("Result: OK — all checks passed. calibrate is ready to use.");
    }
    Ok(())
}

fn check_platform() -> Vec<Status> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let mut out = vec![Status::Info(format!(
        "OS: {os}  arch: {arch}  •  calibrate {}",
        env!("CARGO_PKG_VERSION")
    ))];
    out.push(match os {
        "linux" => Status::Pass("Linux — NVIDIA NVML support available".into()),
        "macos" => Status::Pass("macOS — Apple GPU support via IOKit (system-wide figures)".into()),
        other => Status::Warn(format!(
            "Platform '{other}' — only `calibrate bench` and `calibrate plan` are supported"
        )),
    });
    let ctx = crate::process::container::detect();
    if ctx != crate::process::attach::ContainerContext::Host {
        out.push(Status::Info(format!(
            "Running inside a container ({ctx:?}) — use `watch --gpus all` if PIDs cannot be mapped"
        )));
    }
    out
}

fn check_nvidia() -> Vec<Status> {
    use std::path::Path;
    let mut out = Vec::new();

    match which("nvidia-smi") {
        Some(p) => out.push(Status::Pass(format!("nvidia-smi found: {}", p.display()))),
        None => out.push(Status::Warn(
            "nvidia-smi not in PATH (not required, but usually ships with the driver)\n\
             \n\
             To install the driver:\n\
               Ubuntu/Debian: sudo ubuntu-drivers autoinstall\n\
               RHEL/CentOS:   sudo dnf install -y nvidia-driver"
                .into(),
        )),
    }

    if Path::new("/dev/nvidiactl").exists() {
        out.push(
            match std::fs::OpenOptions::new()
                .read(true)
                .open("/dev/nvidiactl")
            {
                Ok(_) => Status::Pass("/dev/nvidiactl is accessible".into()),
                Err(e) => Status::Fail(format!(
                    "/dev/nvidiactl is not accessible ({e})\n\
                 \n\
                 To fix:\n\
                   sudo usermod -aG video $USER && newgrp video"
                )),
            },
        );
    } else {
        out.push(Status::Fail(
            "/dev/nvidiactl not found — NVIDIA kernel module not loaded\n\
             \n\
             To fix:\n\
               sudo modprobe nvidia\n\
               Check for errors: sudo dmesg | grep -i nvidia\n\
               In Docker: run with --gpus all (nvidia-container-toolkit)"
                .into(),
        ));
    }

    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            let count = nvml.device_count().unwrap_or(0);
            let driver = nvml
                .sys_driver_version()
                .unwrap_or_else(|_| "unknown".into());
            out.push(Status::Pass(format!(
                "NVML initialized — {count} device(s), driver {driver}"
            )));
            let db = crate::gpu_specs::SpecDb::builtin();
            for i in 0..count {
                if let Ok(d) = nvml.device_by_index(i) {
                    let name = d.name().unwrap_or_else(|_| "Unknown".into());
                    let mem = d
                        .memory_info()
                        .map(|m| crate::metrics::units::fmt_mib(m.total / 1024 / 1024))
                        .unwrap_or_else(|_| "unknown VRAM".into());
                    out.push(Status::Info(format!("  [{i}] {name}  •  {mem}")));
                    if !db.resolve(&name).known() {
                        out.push(Status::Warn(format!(
                            "  '{name}' is not in the built-in GPU spec database — TFLOPS figures \
                             will be unavailable (please open an issue with this name)"
                        )));
                    }
                }
            }
        }
        Err(e) => out.push(Status::Fail(
            crate::error::CalibrateError::NvmlInit(e.to_string()).to_string(),
        )),
    }
    out
}

fn check_profiling() -> Vec<Status> {
    let mut out = Vec::new();
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        out.push(Status::Info("skipped — NVML unavailable".into()));
        return out;
    };
    let count = nvml.device_count().unwrap_or(0);
    let gpm = crate::collectors::gpm::Gpm::load();
    let mut gpm_ok = 0;
    match &gpm {
        Some(g) => {
            for i in 0..count {
                if let Ok(d) = nvml.device_by_index(i) {
                    if g.is_supported(&d) {
                        gpm_ok += 1;
                    }
                }
            }
            if gpm_ok == count && count > 0 {
                out.push(Status::Pass(format!(
                    "NVML GPM supported on all {count} GPU(s) — tensor-core activity available"
                )));
            } else {
                out.push(Status::Info(format!(
                    "NVML GPM supported on {gpm_ok}/{count} GPU(s) (requires Hopper or newer)"
                )));
            }
        }
        None => out.push(Status::Info(
            "NVML GPM symbols not present (driver older than R520)".into(),
        )),
    }
    let dcgm = crate::collectors::dcgm::DcgmReader::available();
    if dcgm {
        out.push(Status::Pass(
            "DCGM (`dcgmi`) available — profiling metrics usable".into(),
        ));
    } else {
        out.push(Status::Info(
            "DCGM not available (install datacenter-gpu-manager and start nv-hostengine)".into(),
        ));
    }
    if gpm_ok < count && !dcgm {
        out.push(Status::Warn(
            "Without GPM or DCGM, `watch` can only show an MFU *upper bound* (GPU busy time). \
             Use the calibrate Python hook for measured MFU — see docs/hook.md"
                .into(),
        ));
    }
    out
}

#[cfg(target_os = "macos")]
fn check_apple_gpu() -> Vec<Status> {
    let mut out = vec![Status::Info(format!(
        "Chip: {}",
        crate::process::attach::apple_gpu_name()
    ))];
    if let Ok(ver) = run_cmd("sw_vers", &["-productVersion"]) {
        let ver = ver.trim().to_string();
        let major: u64 = ver
            .split('.')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        out.push(if major >= 12 {
            Status::Pass(format!("macOS {ver} — IOKit GPU statistics supported"))
        } else {
            Status::Warn(format!(
                "macOS {ver} — IOKit GPU statistics require macOS 12 or later"
            ))
        });
    }
    match crate::collectors::apple_gpu::AppleBackend::probe() {
        Ok(_) => out.push(Status::Pass(
            "IOKit GPU service found (AGXAccelerator / IOAccelerator)".into(),
        )),
        Err(e) => out.push(Status::Fail(format!(
            "IOKit GPU probe failed: {e}\n\
             \n\
             To fix:\n\
               • Run from a standard Terminal session (not a sandbox)\n\
               • Upgrade to macOS 12 (Monterey) or later"
        ))),
    }
    out.push(Status::Info(
        "GPU temperature/power are not available without root (powermetrics); shown as N/A".into(),
    ));
    out
}

fn check_proc_visibility() -> Vec<Status> {
    let mut out = Vec::new();
    let me = std::process::id();
    if crate::collectors::proc_tree::is_alive(me) {
        out.push(Status::Pass(format!(
            "Can observe own PID ({me}) — process visibility OK"
        )));
    } else {
        out.push(Status::Fail(format!(
            "Cannot observe own PID ({me}) — /proc may be restricted (hidepid)"
        )));
    }
    if crate::collectors::cpu::process_cpu_ns(me).is_some() {
        out.push(Status::Pass("Process CPU accounting readable".into()));
    } else {
        out.push(Status::Warn(
            "Process CPU accounting not readable — input-pipeline analysis disabled".into(),
        ));
    }

    let found = crate::process::discover::training_processes();
    if found.is_empty() {
        out.push(Status::Info(
            "No active training processes detected (python/torch/launchers)".into(),
        ));
    } else {
        out.push(Status::Info(format!(
            "Detected {} likely training job(s):",
            found.len()
        )));
        for p in &found {
            out.push(Status::Info(format!(
                "  {p}\n        → calibrate watch --pid {}",
                p.pid
            )));
        }
    }
    out
}

async fn check_config_and_network() -> Vec<Status> {
    let s = crate::config::get();
    let mut out = Vec::new();
    match crate::config::default_config_path() {
        Some(p) if p.exists() => out.push(Status::Pass(format!("Config file: {}", p.display()))),
        Some(p) => out.push(Status::Info(format!(
            "No config file (optional): {}",
            p.display()
        ))),
        None => {}
    }
    let dir = crate::collectors::hook::default_dir();
    out.push(
        match tempfile::Builder::new()
            .prefix(".calibrate-probe")
            .tempfile_in(&dir)
        {
            Ok(_) => Status::Pass(format!("Hook socket directory writable: {}", dir.display())),
            Err(e) => Status::Warn(format!(
                "Hook socket directory {} is not writable ({e}) — set CALIBRATE_HOOK_DIR",
                dir.display()
            )),
        },
    );
    out.push(Status::Info(format!(
        "Credentials: HF_TOKEN {}, LAMBDA_API_KEY {}, RUNPOD_API_KEY {}",
        if s.hf_token.is_some() {
            "set"
        } else {
            "not set"
        },
        if s.lambda_api_key.is_some() {
            "set"
        } else {
            "not set"
        },
        if s.runpod_api_key.is_some() {
            "set"
        } else {
            "not set"
        },
    )));
    if s.offline {
        out.push(Status::Info("--offline: network checks skipped".into()));
        return out;
    }
    let client = match crate::net::client(Some(std::time::Duration::from_secs(5))) {
        Ok(c) => c,
        Err(e) => {
            out.push(Status::Warn(format!("HTTP client unavailable: {e}")));
            return out;
        }
    };
    match client.head(&s.specs_url).send().await {
        Ok(r) if r.status().is_success() => out.push(Status::Pass(format!(
            "GPU spec database reachable ({})",
            s.specs_url
        ))),
        Ok(r) => out.push(Status::Warn(format!(
            "GPU spec database returned HTTP {} — the built-in table will be used",
            r.status()
        ))),
        Err(e) => out.push(Status::Warn(format!(
            "Network unavailable ({e}) — built-in GPU specs are used; `plan` needs --prices-file. \
             Behind a proxy? Set HTTPS_PROXY."
        ))),
    }
    out
}

fn which(program: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|d| d.join(program))
            .find(|c| c.is_file())
    })
}

#[allow(dead_code)]
fn run_cmd(program: &str, args: &[&str]) -> anyhow::Result<String> {
    let output = Command::new(program).args(args).output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        anyhow::bail!("{}", String::from_utf8_lossy(&output.stderr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_prefixes() {
        assert_eq!(Status::Pass("x".into()).prefix(), "✓");
        assert_eq!(Status::Fail("x".into()).prefix(), "✗");
        assert_eq!(Status::Warn("x".into()).prefix(), "⚠");
        assert_eq!(Status::Info("x".into()).prefix(), "·");
        assert!(Status::Fail("x".into()).is_fail());
        assert!(!Status::Warn("x".into()).is_fail());
        assert!(Status::Warn("x".into()).is_warn());
    }

    #[test]
    fn platform_check_reports_os() {
        let s = check_platform();
        assert!(s.iter().any(|x| x.message().contains(std::env::consts::OS)));
    }

    #[test]
    fn proc_visibility_passes_for_own_pid() {
        let s = check_proc_visibility();
        assert!(s
            .iter()
            .any(|x| matches!(x, Status::Pass(m) if m.contains("own PID"))));
    }

    #[test]
    fn which_finds_sh() {
        assert!(which("sh").is_some());
        assert!(which("definitely-not-a-real-binary-xyz").is_none());
    }
}
