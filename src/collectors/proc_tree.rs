//! Process-tree enumeration.
//!
//! Training jobs are rarely a single process: DataLoader workers, torchrun /
//! DeepSpeed / accelerate ranks and NCCL helpers are all children of the
//! process a user points calibrate at. Every collector works on the tree.

use std::collections::{HashMap, HashSet, VecDeque};

/// `(pid, ppid)` for every visible process.
pub fn process_table() -> Vec<(u32, u32)> {
    platform::process_table()
}

/// `root` plus all of its descendants (root first). Empty if root is gone.
pub fn descendants(root: u32) -> Vec<u32> {
    descendants_in(&process_table(), root)
}

/// Pure tree walk over a process table (testable).
pub fn descendants_in(table: &[(u32, u32)], root: u32) -> Vec<u32> {
    if !table.iter().any(|&(p, _)| p == root) {
        return vec![];
    }
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(pid, ppid) in table {
        if pid != ppid {
            children.entry(ppid).or_default().push(pid);
        }
    }
    let mut out = vec![root];
    let mut seen: HashSet<u32> = HashSet::from([root]);
    let mut queue = VecDeque::from([root]);
    while let Some(p) = queue.pop_front() {
        if let Some(kids) = children.get(&p) {
            for &k in kids {
                if seen.insert(k) {
                    out.push(k);
                    queue.push_back(k);
                }
            }
        }
    }
    out
}

/// Ancestors of `pid` (parent first), stopping at pid 1 / 0.
pub fn ancestors_in(table: &[(u32, u32)], pid: u32) -> Vec<u32> {
    let parent: HashMap<u32, u32> = table.iter().copied().collect();
    let mut out = Vec::new();
    let mut cur = pid;
    while let Some(&pp) = parent.get(&cur) {
        if pp == 0 || pp == cur || out.contains(&pp) {
            break;
        }
        out.push(pp);
        if pp == 1 {
            break;
        }
        cur = pp;
    }
    out
}

/// `true` if the process exists (and is not a zombie on Linux).
pub fn is_alive(pid: u32) -> bool {
    platform::is_alive(pid)
}

#[cfg(target_os = "linux")]
mod platform {
    pub fn process_table() -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return out;
        };
        for entry in entries.flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                if let Some(ppid) = parse_ppid(&stat) {
                    out.push((pid, ppid));
                }
            }
        }
        out
    }

    /// Field 4 of /proc/PID/stat (after the parenthesised comm).
    pub(super) fn parse_ppid(stat: &str) -> Option<u32> {
        let after = &stat[stat.rfind(')')? + 1..];
        after.split_whitespace().nth(1)?.parse().ok()
    }

    pub fn is_alive(pid: u32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => {
                let state = stat
                    .rfind(')')
                    .and_then(|i| stat[i + 1..].split_whitespace().next());
                !matches!(state, Some("Z") | Some("X"))
            }
            Err(_) => false,
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    pub fn process_table() -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        // SAFETY: proc_listallpids with a null buffer returns the count; we then
        // pass a buffer sized with headroom for processes spawned in between.
        unsafe {
            let n = libc::proc_listallpids(std::ptr::null_mut(), 0);
            if n <= 0 {
                return out;
            }
            let cap = n as usize + 64;
            let mut pids = vec![0 as libc::pid_t; cap];
            let n = libc::proc_listallpids(
                pids.as_mut_ptr() as *mut libc::c_void,
                (cap * std::mem::size_of::<libc::pid_t>()) as libc::c_int,
            );
            if n <= 0 {
                return out;
            }
            for &pid in pids.iter().take(n as usize) {
                if pid <= 0 {
                    continue;
                }
                if let Some(ppid) = ppid_of(pid) {
                    out.push((pid as u32, ppid));
                }
            }
        }
        out
    }

    fn ppid_of(pid: libc::pid_t) -> Option<u32> {
        // SAFETY: proc_bsdinfo is plain-old-data; proc_pidinfo writes at most
        // `size` bytes into it and returns the number written.
        unsafe {
            let mut info: libc::proc_bsdinfo = std::mem::zeroed();
            let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
            let r = libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTBSDINFO,
                0,
                &mut info as *mut _ as *mut libc::c_void,
                size,
            );
            (r == size).then_some(info.pbi_ppid)
        }
    }

    /// SZOMB from <sys/proc.h>.
    const SZOMB: u32 = 5;

    pub fn is_alive(pid: u32) -> bool {
        // kill(pid, 0) succeeds for live processes we may signal; EPERM still
        // means the process exists. Zombies also pass kill(), so check the
        // BSD process status when it is readable.
        let r = unsafe { libc::kill(pid as libc::pid_t, 0) };
        let exists = r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
        if !exists {
            return false;
        }
        // SAFETY: see ppid_of.
        unsafe {
            let mut info: libc::proc_bsdinfo = std::mem::zeroed();
            let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
            let n = libc::proc_pidinfo(
                pid as libc::pid_t,
                libc::PROC_PIDTBSDINFO,
                0,
                &mut info as *mut _ as *mut libc::c_void,
                size,
            );
            if n == size {
                return info.pbi_status != SZOMB;
            }
            // Unreadable (another user's process without root): trust kill().
            std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    pub fn process_table() -> Vec<(u32, u32)> {
        vec![]
    }
    pub fn is_alive(_pid: u32) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &[(u32, u32)] = &[
        (1, 0),
        (100, 1),   // torchrun
        (101, 100), // rank 0
        (102, 100), // rank 1
        (201, 101), // dataloader worker of rank 0
        (300, 1),   // unrelated
    ];

    #[test]
    fn walks_whole_subtree() {
        let mut d = descendants_in(TABLE, 100);
        assert_eq!(d[0], 100);
        d.sort_unstable();
        assert_eq!(d, vec![100, 101, 102, 201]);
    }

    #[test]
    fn missing_root_is_empty() {
        assert!(descendants_in(TABLE, 999).is_empty());
    }

    #[test]
    fn leaf_is_just_itself() {
        assert_eq!(descendants_in(TABLE, 201), vec![201]);
    }

    #[test]
    fn ancestors_stop_at_init() {
        assert_eq!(ancestors_in(TABLE, 201), vec![101, 100, 1]);
    }

    #[test]
    fn cycles_do_not_hang() {
        let t = &[(5, 6), (6, 5)];
        assert_eq!(descendants_in(t, 5).len(), 2);
        assert!(ancestors_in(t, 5).len() <= 2);
    }

    #[test]
    fn own_process_is_visible_and_alive() {
        let me = std::process::id();
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            assert!(process_table().iter().any(|&(p, _)| p == me));
            assert!(is_alive(me));
            assert_eq!(descendants(me)[0], me);
        }
    }

    #[test]
    fn zombies_are_not_alive() {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        // Give it time to exit without reaping it.
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            !is_alive(pid),
            "an exited, unreaped child must not count as alive"
        );
        child.wait().unwrap();
        assert!(!is_alive(pid));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_ppid_with_spaces_in_comm() {
        assert_eq!(
            platform::parse_ppid("42 (my (weird) proc) S 7 42 42 0"),
            Some(7)
        );
    }
}
