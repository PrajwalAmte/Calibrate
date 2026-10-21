use crate::process::attach::ContainerContext;

/// Detect whether calibrate itself is running inside a container.
///
/// Only positive markers count: `/.dockerenv`, Podman's `/run/.containerenv`,
/// the Kubernetes service-account mount, and runtime-specific cgroup paths.
/// An arbitrary non-root cgroup is *not* evidence of a container — systemd
/// puts every login session under `/user.slice/...`.
pub fn detect() -> ContainerContext {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let k8s_sa = std::path::Path::new("/var/run/secrets/kubernetes.io").exists();
    classify(
        &cgroup,
        std::path::Path::new("/.dockerenv").exists(),
        std::path::Path::new("/run/.containerenv").exists(),
        k8s_sa || std::env::var_os("KUBERNETES_SERVICE_HOST").is_some(),
    )
}

/// Pure classification (testable).
pub fn classify(cgroup: &str, dockerenv: bool, containerenv: bool, k8s: bool) -> ContainerContext {
    if k8s || cgroup.contains("kubepods") {
        return ContainerContext::Kubernetes;
    }
    if dockerenv || cgroup.contains("/docker") || cgroup.contains("docker-") {
        return ContainerContext::Docker;
    }
    if containerenv
        || ["libpod", "/lxc/", "crio-", "containerd", "/actions_job/"]
            .iter()
            .any(|m| cgroup.contains(m))
    {
        return ContainerContext::Unknown;
    }
    ContainerContext::Host
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemd_login_session_is_host() {
        // Regression: this used to be reported as a container.
        let cg = "0::/user.slice/user-1000.slice/session-2.scope\n";
        assert_eq!(classify(cg, false, false, false), ContainerContext::Host);
        let svc = "0::/system.slice/sshd.service\n";
        assert_eq!(classify(svc, false, false, false), ContainerContext::Host);
        assert_eq!(
            classify("0::/\n", false, false, false),
            ContainerContext::Host
        );
    }

    #[test]
    fn docker_markers() {
        assert_eq!(
            classify("0::/system.slice/docker-abc.scope", false, false, false),
            ContainerContext::Docker
        );
        assert_eq!(
            classify("0::/", true, false, false),
            ContainerContext::Docker
        );
        assert_eq!(
            classify("12:pids:/docker/abc", false, false, false),
            ContainerContext::Docker
        );
    }

    #[test]
    fn kubernetes_markers() {
        assert_eq!(
            classify(
                "0::/kubepods.slice/kubepods-burstable.slice/x",
                false,
                false,
                false
            ),
            ContainerContext::Kubernetes
        );
        assert_eq!(
            classify("0::/", false, false, true),
            ContainerContext::Kubernetes
        );
    }

    #[test]
    fn podman_and_others() {
        assert_eq!(
            classify("0::/", false, true, false),
            ContainerContext::Unknown
        );
        assert_eq!(
            classify("0::/machine.slice/libpod-123.scope", false, false, false),
            ContainerContext::Unknown
        );
    }
}
