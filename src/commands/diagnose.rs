use std::process::Command;

use crate::cli::DiagnoseArgs;

/// Represents the pass/fail/skip state of a single check.
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
    let mut any_fail = false;
    let mut any_warn = false;

    println!("calibrate diagnose — system capability check\n");

    // Section 1: Platform
    println!("[ Platform ]");
    let platform = check_platform();
    for s in &platform {
        print_status(s);
        if s.is_fail() {
            any_fail = true;
        }
        if s.is_warn() {
            any_warn = true;
        }
    }
    println!();

    // Section 2: NVIDIA GPU (Linux)
    #[cfg(target_os = "linux")]
    {
        println!("[ NVIDIA GPU ]");
        let nvidia = check_nvidia();
        for s in &nvidia {
            print_status(s);
            if s.is_fail() {
                any_fail = true;
            }
            if s.is_warn() {
                any_warn = true;
            }
        }
        println!();
    }

    // Section 3: Apple GPU (macOS)
    #[cfg(target_os = "macos")]
    {
        println!("[ Apple GPU ]");
        let apple = check_apple_gpu();
        for s in &apple {
            print_status(s);
            if s.is_fail() {
                any_fail = true;
            }
            if s.is_warn() {
                any_warn = true;
            }
        }
        println!();
    }

    // Section 4: Process visibility
    println!("[ Process visibility ]");
    let proc = check_proc_visibility();
    for s in &proc {
        print_status(s);
        if s.is_fail() {
            any_fail = true;
        }
        if s.is_warn() {
            any_warn = true;
        }
    }
    println!();

    // Summary
    if any_fail {
        println!("Result: FAIL — one or more required components are missing.");
        println!("        Fix the items marked ✗ above, then re-run `calibrate diagnose`.");
        std::process::exit(1);
    } else if any_warn {
        println!("Result: WARN — calibrate will work with limited functionality.");
        println!("        Review the items marked ⚠ above for full support.");
    } else {
        println!("Result: OK — all checks passed. calibrate is ready to use.");
    }

    Ok(())
}

fn print_status(s: &Status) {
    println!("  {}  {}", s.prefix(), s.message());
}

// Platform checks

fn check_platform() -> Vec<Status> {
    let mut out = Vec::new();

    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    out.push(Status::Info(format!("OS: {os}  arch: {arch}")));

    match os {
        "linux" => out.push(Status::Pass(
            "Linux — full NVIDIA NVML support available".into(),
        )),
        "macos" => out.push(Status::Pass(
            "macOS — Apple Silicon GPU support via IOKit available".into(),
        )),
        other => out.push(Status::Warn(format!(
            "Platform '{other}' — only `calibrate bench` and `calibrate plan` are supported"
        ))),
    }

    out
}

// NVIDIA checks (Linux)

#[cfg(target_os = "linux")]
fn check_nvidia() -> Vec<Status> {
    use std::path::Path;

    let mut out = Vec::new();

    // 1. nvidia-smi presence
    let smi_path = which("nvidia-smi");
    match &smi_path {
        Some(p) => out.push(Status::Pass(format!("nvidia-smi found: {}", p.display()))),
        None => {
            out.push(Status::Fail(
                "nvidia-smi not found in PATH\n\
                 \n\
                 To fix:\n\
                 \n\
                 Ubuntu/Debian:\n\
                   sudo apt-get install -y nvidia-driver-555\n\
                   (or latest: ubuntu-drivers autoinstall)\n\
                 \n\
                 RHEL/CentOS:\n\
                   sudo dnf install -y nvidia-driver\n\
                 \n\
                 After install: reboot, then re-run calibrate diagnose"
                    .into(),
            ));
            // No point continuing if smi is missing
            return out;
        }
    }

    // 2. nvidia-smi execution
    match run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
    ) {
        Ok(output) if !output.trim().is_empty() => {
            for line in output.trim().lines() {
                out.push(Status::Pass(format!("GPU detected: {line}")));
            }
        }
        Ok(_) => {
            out.push(Status::Warn(
                "nvidia-smi ran but reported no GPUs — driver may not be loaded".into(),
            ));
        }
        Err(e) => {
            out.push(Status::Fail(format!(
                "nvidia-smi failed: {e}\n\
                 \n\
                 To fix:\n\
                   sudo modprobe nvidia\n\
                   If that fails, check dmesg: sudo dmesg | grep -i nvidia"
            )));
        }
    }

    // 3. NVML library
    let nvml_paths = [
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
    ];
    let nvml_found = nvml_paths.iter().find(|p| Path::new(p).exists());
    match nvml_found {
        Some(p) => out.push(Status::Pass(format!("NVML library: {p}"))),
        None => {
            out.push(Status::Fail(
                "NVML library (libnvidia-ml.so.1) not found\n\
                 \n\
                 To fix:\n\
                   Ubuntu/Debian: sudo apt-get install -y libnvidia-ml1\n\
                   RHEL/CentOS:   sudo dnf install -y nvidia-driver-cuda-libs\n\
                   Then reload:   sudo ldconfig"
                    .into(),
            ));
        }
    }

    // 4. /dev/nvidiactl permissions
    let dev_path = "/dev/nvidiactl";
    if Path::new(dev_path).exists() {
        match std::fs::metadata(dev_path) {
            Ok(meta) => {
                use std::os::unix::fs::PermissionsExt;
                let mode = meta.permissions().mode();
                // Check world-readable (o+r) = mode & 0o004 != 0
                if mode & 0o004 != 0 {
                    out.push(Status::Pass(format!("{dev_path} is readable")));
                } else {
                    out.push(Status::Warn(format!(
                        "{dev_path} is not world-readable (current mode: {:o})\n\
                         \n\
                         To fix:\n\
                           sudo chmod o+rw {dev_path}\n\
                           Or add your user to the 'video' group:\n\
                           sudo usermod -aG video $USER && newgrp video",
                        mode & 0o777
                    )));
                }
            }
            Err(e) => out.push(Status::Warn(format!("Could not stat {dev_path}: {e}"))),
        }
    } else {
        out.push(Status::Fail(
            "/dev/nvidiactl not found — NVIDIA kernel module not loaded\n\
             \n\
             To fix:\n\
               sudo modprobe nvidia\n\
               Check for errors: sudo dmesg | tail -20"
                .into(),
        ));
    }

    // 5. NVML init test via nvml-wrapper
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            let count = nvml.device_count().unwrap_or(0);
            out.push(Status::Pass(format!(
                "NVML initialized successfully — {count} device(s) visible"
            )));
            for i in 0..count {
                if let Ok(device) = nvml.device_by_index(i) {
                    let name = device.name().unwrap_or_else(|_| "Unknown".into());
                    let driver = nvml
                        .sys_driver_version()
                        .unwrap_or_else(|_| "Unknown".into());
                    let mem = device
                        .memory_info()
                        .map(|m| format!("{} MiB VRAM", m.total / 1024 / 1024))
                        .unwrap_or_else(|_| "unknown VRAM".into());
                    out.push(Status::Info(format!(
                        "  [{i}] {name}  •  {mem}  •  driver {driver}"
                    )));
                }
            }
        }
        Err(e) => {
            out.push(Status::Fail(format!(
                "NVML init failed: {e}\n\
                 \n\
                 Common causes and fixes:\n\
                 \n\
                 1. Driver not loaded:\n\
                    sudo modprobe nvidia\n\
                 \n\
                 2. libnvidia-ml not found by linker:\n\
                    sudo ldconfig\n\
                    ldd $(which nvidia-smi) | grep nvidia-ml\n\
                 \n\
                 3. Running inside Docker without GPU passthrough:\n\
                    docker run --gpus all ...\n\
                    (requires nvidia-container-toolkit)\n\
                 \n\
                 4. Permission denied:\n\
                    sudo usermod -aG video $USER && newgrp video"
            )));
        }
    }

    out
}

// Apple GPU checks (macOS)

#[cfg(target_os = "macos")]
fn check_apple_gpu() -> Vec<Status> {
    use crate::collectors::apple_gpu::AppleGpuCollector;

    let mut out = Vec::new();

    // 1. Chip identification
    let mut buf = [0u8; 256];
    let mut len = buf.len();
    let ret = unsafe {
        libc::sysctlbyname(
            b"machdep.cpu.brand_string\0".as_ptr() as *const libc::c_char,
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 && len > 1 {
        let chip = std::str::from_utf8(&buf[..len - 1])
            .unwrap_or("Unknown")
            .trim()
            .to_string();
        out.push(Status::Info(format!("Chip: {chip}")));
    }

    // 2. macOS version
    if let Ok(ver) = run_cmd("sw_vers", &["-productVersion"]) {
        let ver = ver.trim().to_string();
        // IOKit GPU stats require macOS 12+
        let parts: Vec<u64> = ver.split('.').filter_map(|s| s.parse().ok()).collect();
        if parts.first().copied().unwrap_or(0) >= 12 {
            out.push(Status::Pass(format!(
                "macOS {ver} — IOKit GPU stats supported"
            )));
        } else {
            out.push(Status::Warn(format!(
                "macOS {ver} — IOKit GPU stats require macOS 12 (Monterey) or later"
            )));
        }
    }

    // 3. IOKit probe
    match AppleGpuCollector::probe() {
        Ok(_) => out.push(Status::Pass(
            "IOKit GPU service found (AGXAccelerator / IOAccelerator)".into(),
        )),
        Err(e) => out.push(Status::Fail(format!(
            "IOKit GPU probe failed: {e}\n\
             \n\
             To fix:\n\
               • Ensure System Preferences → Privacy → Full Disk Access is not blocking the terminal\n\
               • If running in a sandboxed environment, use a standard Terminal.app session\n\
               • Upgrade to macOS 12 (Monterey) or later"
        ))),
    }

    out
}

// Process visibility checks

fn check_proc_visibility() -> Vec<Status> {
    let mut out = Vec::new();

    // Check that we can see our own PID
    let own_pid = std::process::id();
    let exists = {
        #[cfg(target_os = "linux")]
        {
            Path::new(&format!("/proc/{own_pid}")).exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            unsafe { libc::kill(own_pid as libc::pid_t, 0) == 0 }
        }
    };

    if exists {
        out.push(Status::Pass(format!(
            "Can observe own PID ({own_pid}) — process visibility OK"
        )));
    } else {
        out.push(Status::Fail(format!(
            "Cannot observe own PID ({own_pid}) — likely running inside a container with PID namespace isolation\n\
             \n\
             To fix:\n\
               Run calibrate on the host, not inside a container.\n\
               Or use: docker run --pid=host ..."
        )));
    }

    // Suggest how to find a training process PID
    let training_pids = find_training_pids();
    if training_pids.is_empty() {
        out.push(Status::Info(
            "No active training processes detected (python/torch not found in process list)".into(),
        ));
        out.push(Status::Info(
            "When training starts, find the PID with: pgrep -f train.py".into(),
        ));
    } else {
        out.push(Status::Info(format!(
            "Detected {} likely training process(es):",
            training_pids.len()
        )));
        for (pid, cmd) in &training_pids {
            out.push(Status::Info(format!(
                "  PID {pid}  →  calibrate watch --pid {pid}"
            )));
            out.push(Status::Info(format!("           cmd: {cmd}")));
        }
    }

    out
}

// Helpers

#[cfg(target_os = "linux")]
fn which(program: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path).find_map(|dir| {
            let candidate = dir.join(program);
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

fn run_cmd(program: &str, args: &[&str]) -> anyhow::Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        Err(anyhow::anyhow!("{stderr}"))
    }
}

/// Return (pid, truncated_cmdline) for processes that look like training jobs.
fn find_training_pids() -> Vec<(u32, String)> {
    let mut results = Vec::new();

    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Ok(pid) = name.parse::<u32>() {
                    let cmdline_path = format!("/proc/{pid}/cmdline");
                    if let Ok(bytes) = std::fs::read(&cmdline_path) {
                        let cmd = bytes
                            .split(|&b| b == 0)
                            .filter(|s| !s.is_empty())
                            .map(|s| String::from_utf8_lossy(s).into_owned())
                            .collect::<Vec<_>>()
                            .join(" ");
                        if is_training_cmd(&cmd) {
                            results.push((pid, truncate(&cmd, 80)));
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("ps").args(["-eo", "pid,command"]).output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().skip(1) {
                let mut parts = line.trim().splitn(2, ' ');
                if let (Some(pid_str), Some(cmd)) = (parts.next(), parts.next()) {
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        if is_training_cmd(cmd) {
                            results.push((pid, truncate(cmd, 80)));
                        }
                    }
                }
            }
        }
    }

    results
}

fn is_training_cmd(cmd: &str) -> bool {
    let keywords = ["train", "finetune", "fine_tune", "pretrain"];
    let frameworks = ["torch", "tensorflow", "jax", "keras", "lightning"];
    let lower = cmd.to_lowercase();
    (lower.contains("python") || lower.contains("python3"))
        && (keywords.iter().any(|k| lower.contains(k))
            || frameworks.iter().any(|f| lower.contains(f)))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

// Tests ─

#[cfg(test)]
mod tests {
    use super::*;

    // is_training_cmd

    #[test]
    fn recognises_train_py() {
        assert!(is_training_cmd(
            "python3 training/train.py --data /data/features.bin"
        ));
    }

    #[test]
    fn recognises_finetune_script() {
        assert!(is_training_cmd("python finetune.py --model llama"));
    }

    #[test]
    fn recognises_torch_in_command() {
        assert!(is_training_cmd("python3 run.py --backend torch"));
    }

    #[test]
    fn recognises_pytorch_lightning() {
        assert!(is_training_cmd(
            "python3 -m lightning fit --config config.yaml"
        ));
    }

    #[test]
    fn does_not_match_plain_python() {
        assert!(!is_training_cmd("python3 serve.py --port 8080"));
    }

    #[test]
    fn does_not_match_non_python_process() {
        assert!(!is_training_cmd("node train.js"));
    }

    #[test]
    fn does_not_match_empty_string() {
        assert!(!is_training_cmd(""));
    }

    #[test]
    fn case_insensitive_match() {
        assert!(is_training_cmd("Python3 TRAIN.PY"));
    }

    // truncate

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact_length_unchanged() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_long_string_appends_ellipsis() {
        let result = truncate("abcdefghij", 5);
        assert!(result.starts_with("abcde"), "got: {result}");
        assert!(result.ends_with('…'), "got: {result}");
    }

    #[test]
    fn truncate_zero_max_still_appends_ellipsis() {
        let result = truncate("hello", 0);
        assert_eq!(result, "…");
    }

    // Status helpers

    #[test]
    fn status_pass_prefix_is_checkmark() {
        assert_eq!(Status::Pass("ok".into()).prefix(), "✓");
    }

    #[test]
    fn status_fail_prefix_is_cross() {
        assert_eq!(Status::Fail("bad".into()).prefix(), "✗");
    }

    #[test]
    fn status_warn_prefix_is_triangle() {
        assert_eq!(Status::Warn("meh".into()).prefix(), "⚠");
    }

    #[test]
    fn status_info_prefix_is_dot() {
        assert_eq!(Status::Info("note".into()).prefix(), "·");
    }

    #[test]
    fn only_fail_variant_is_fail() {
        assert!(Status::Fail("x".into()).is_fail());
        assert!(!Status::Pass("x".into()).is_fail());
        assert!(!Status::Warn("x".into()).is_fail());
        assert!(!Status::Info("x".into()).is_fail());
    }

    #[test]
    fn only_warn_variant_is_warn() {
        assert!(Status::Warn("x".into()).is_warn());
        assert!(!Status::Pass("x".into()).is_warn());
        assert!(!Status::Fail("x".into()).is_warn());
        assert!(!Status::Info("x".into()).is_warn());
    }

    // Error message content (actionability)

    #[test]
    fn nvml_init_error_mentions_modprobe() {
        let err = crate::error::CalibrateError::NvmlInit("some error".into());
        let msg = err.to_string();
        assert!(
            msg.contains("modprobe"),
            "NvmlInit error must mention 'modprobe'; got: {msg}"
        );
    }

    #[test]
    fn nvml_init_error_mentions_diagnose() {
        let err = crate::error::CalibrateError::NvmlInit("some error".into());
        let msg = err.to_string();
        assert!(
            msg.contains("diagnose"),
            "NvmlInit error must point users to 'calibrate diagnose'; got: {msg}"
        );
    }

    #[test]
    fn process_not_found_error_mentions_pgrep() {
        let err = crate::error::CalibrateError::ProcessNotFound { pid: 99 };
        let msg = err.to_string();
        assert!(
            msg.contains("pgrep"),
            "ProcessNotFound must suggest pgrep; got: {msg}"
        );
    }

    #[test]
    fn permission_denied_error_mentions_sudo() {
        let err = crate::error::CalibrateError::PermissionDenied { pid: 42 };
        let msg = err.to_string();
        assert!(
            msg.contains("sudo"),
            "PermissionDenied must suggest sudo; got: {msg}"
        );
    }

    #[test]
    fn no_gpu_process_error_is_actionable() {
        let err = crate::error::CalibrateError::NoGpuProcess { pid: 1 };
        let msg = err.to_string();
        // Should explain possible causes, not just "not using GPU"
        assert!(
            msg.contains("CPU") || msg.contains("device"),
            "NoGpuProcess should explain causes; got: {msg}"
        );
    }

    // check_platform

    #[test]
    fn platform_check_returns_at_least_two_entries() {
        // Platform + OS-specific result
        let results = check_platform();
        assert!(
            results.len() >= 2,
            "Expected at least 2 platform check results, got {}",
            results.len()
        );
    }

    #[test]
    fn platform_check_includes_os_info() {
        let results = check_platform();
        let has_os_info = results
            .iter()
            .any(|s| s.message().contains("OS:") && s.message().contains("arch:"));
        assert!(has_os_info, "Platform check should report OS and arch");
    }

    // check_proc_visibility

    #[test]
    fn proc_visibility_passes_for_own_pid() {
        let results = check_proc_visibility();
        let own_pid = std::process::id();
        let pass = results
            .iter()
            .any(|s| matches!(s, Status::Pass(_)) && s.message().contains(&own_pid.to_string()));
        assert!(
            pass,
            "Own PID ({own_pid}) should be visible; got: {results:?}"
        );
    }
}
