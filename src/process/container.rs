use crate::process::attach::ContainerContext;

/// Detect whether calibrate itself is running inside a container.
///
/// Checks `/proc/self/cgroup` for Docker/Kubernetes markers and the
/// presence of `/.dockerenv`.  Result is best-effort — host environments
/// always have access to all /proc entries.
pub fn detect() -> ContainerContext {
    if std::path::Path::new("/.dockerenv").exists() {
        return ContainerContext::Docker;
    }

    if let Ok(cgroup) = std::fs::read_to_string("/proc/self/cgroup") {
        if cgroup.contains("/kubepods") {
            return ContainerContext::Kubernetes;
        }
        if cgroup.contains("/docker") {
            return ContainerContext::Docker;
        }
        // cgroup v2 unified hierarchy: non-root cgroup path suggests container.
        if cgroup.lines().any(|l| {
            let path = l.split(':').nth(2).unwrap_or("");
            !path.is_empty() && path != "/" && !path.starts_with("/system.slice")
        }) {
            return ContainerContext::Unknown;
        }
    }

    ContainerContext::Host
}
