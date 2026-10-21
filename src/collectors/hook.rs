//! Client for the `calibrate_hook` Python package.
//!
//! A training process that imports the hook publishes a Unix socket at
//! `<hook_dir>/calibrate-hook-<pid>.sock` and streams JSON lines:
//!
//! ```json
//! {"v":1,"type":"hello","pid":4242,"rank":0,"world_size":8,"framework":"pytorch","device":"cuda:0"}
//! {"v":1,"type":"step","pid":4242,"rank":0,"step":17,"t_end_ms":1730000000000,
//!  "step_s":0.412,"data_s":0.051,"compute_s":0.301,"optimizer_s":0.042,
//!  "tokens":65536,"samples":32,"flops":3.1e15}
//! ```
//!
//! Real step timings and declared FLOPs turn the heuristic breakdown and the
//! utilization proxies into measured values. The protocol is versioned (`v`);
//! unknown message types and fields are ignored for forward compatibility.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::{debug, warn};

use crate::collectors::{HookPeer, StepEvent};

pub const PROTOCOL_VERSION: u32 = 1;
const MAX_LINE: usize = 64 * 1024;

/// Default socket directory, shared with the Python side.
pub fn default_dir() -> PathBuf {
    std::env::var_os("CALIBRATE_HOOK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

pub fn socket_path(dir: &Path, pid: u32) -> PathBuf {
    dir.join(format!("calibrate-hook-{pid}.sock"))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Message {
    Hello {
        #[serde(default)]
        v: Option<u32>,
        #[serde(flatten)]
        peer: HookPeer,
    },
    Step {
        #[serde(default)]
        v: Option<u32>,
        #[serde(flatten)]
        step: StepEvent,
    },
    #[serde(other)]
    Unknown,
}

/// Parsed content of one protocol line.
#[derive(Debug, PartialEq)]
pub enum Parsed {
    Hello(HookPeer),
    Step(StepEvent),
    Ignored,
}

/// Parse a single JSON line from the hook.
pub fn parse_line(line: &str) -> Parsed {
    match serde_json::from_str::<Message>(line) {
        Ok(Message::Hello { v, peer }) if v.unwrap_or(1) <= PROTOCOL_VERSION => Parsed::Hello(peer),
        Ok(Message::Step { v, step })
            if v.unwrap_or(1) <= PROTOCOL_VERSION
                && step.step_s.is_finite()
                && step.step_s > 0.0 =>
        {
            Parsed::Step(step)
        }
        Ok(_) => Parsed::Ignored,
        Err(e) => {
            debug!("ignoring malformed hook line: {e}");
            Parsed::Ignored
        }
    }
}

#[cfg(unix)]
struct Conn {
    stream: std::os::unix::net::UnixStream,
    buf: Vec<u8>,
}

/// Non-blocking reader over all hook sockets of a process tree.
pub struct HookClient {
    dir: PathBuf,
    #[cfg(unix)]
    conns: HashMap<u32, Conn>,
    peers: HashMap<u32, HookPeer>,
}

impl HookClient {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            #[cfg(unix)]
            conns: HashMap::new(),
            peers: HashMap::new(),
        }
    }

    pub fn peers(&self) -> Vec<HookPeer> {
        let mut v: Vec<HookPeer> = self.peers.values().cloned().collect();
        v.sort_by_key(|p| (p.rank, p.pid));
        v
    }

    /// Connect to new sockets for `pids`, drop dead ones, and drain all
    /// complete lines. Never blocks.
    #[cfg(unix)]
    pub fn poll(&mut self, pids: &[u32]) -> Vec<StepEvent> {
        use std::io::{ErrorKind, Read};
        use std::os::unix::net::UnixStream;

        for &pid in pids {
            if self.conns.contains_key(&pid) {
                continue;
            }
            let path = socket_path(&self.dir, pid);
            if !path.exists() {
                continue;
            }
            match UnixStream::connect(&path) {
                Ok(stream) => {
                    if stream.set_nonblocking(true).is_ok() {
                        debug!(pid, "connected to calibrate hook");
                        self.conns.insert(
                            pid,
                            Conn {
                                stream,
                                buf: Vec::new(),
                            },
                        );
                    }
                }
                Err(e) => debug!(pid, "hook socket not connectable: {e}"),
            }
        }

        let mut events = Vec::new();
        let mut dead = Vec::new();
        let mut chunk = [0u8; 8192];
        for (&pid, conn) in self.conns.iter_mut() {
            loop {
                match conn.stream.read(&mut chunk) {
                    Ok(0) => {
                        dead.push(pid);
                        break;
                    }
                    Ok(n) => conn.buf.extend_from_slice(&chunk[..n]),
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(_) => {
                        dead.push(pid);
                        break;
                    }
                }
            }
            while let Some(nl) = conn.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = conn.buf.drain(..=nl).collect();
                let Ok(text) = std::str::from_utf8(&line) else {
                    continue;
                };
                match parse_line(text.trim()) {
                    Parsed::Hello(peer) => {
                        self.peers.insert(pid, peer);
                    }
                    Parsed::Step(mut s) => {
                        s.pid = pid;
                        events.push(s);
                    }
                    Parsed::Ignored => {}
                }
            }
            if conn.buf.len() > MAX_LINE {
                warn!(pid, "hook line too long — dropping connection");
                dead.push(pid);
            }
        }
        for pid in dead {
            self.conns.remove(&pid);
            self.peers.remove(&pid);
        }
        events
    }

    #[cfg(not(unix))]
    pub fn poll(&mut self, _pids: &[u32]) -> Vec<StepEvent> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hello_and_step() {
        let h = parse_line(
            r#"{"v":1,"type":"hello","pid":7,"rank":0,"world_size":2,"framework":"pytorch"}"#,
        );
        match h {
            Parsed::Hello(p) => {
                assert_eq!(p.pid, 7);
                assert_eq!(p.world_size, Some(2));
            }
            other => panic!("{other:?}"),
        }
        let s = parse_line(
            r#"{"v":1,"type":"step","pid":7,"step":3,"t_end_ms":10,"step_s":0.5,"data_s":0.1,"flops":1e12,"extra":"ignored"}"#,
        );
        match s {
            Parsed::Step(e) => {
                assert_eq!(e.step, 3);
                assert_eq!(e.data_s, Some(0.1));
                assert_eq!(e.compute_s, None);
                assert_eq!(e.flops, Some(1e12));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ignores_future_versions_unknown_types_and_garbage() {
        assert_eq!(
            parse_line(r#"{"v":99,"type":"step","pid":1,"step":1,"t_end_ms":1,"step_s":1}"#),
            Parsed::Ignored
        );
        assert_eq!(parse_line(r#"{"v":1,"type":"bye"}"#), Parsed::Ignored);
        assert_eq!(parse_line("not json"), Parsed::Ignored);
        assert_eq!(
            parse_line(r#"{"v":1,"type":"step","pid":1,"step":1,"t_end_ms":1,"step_s":0}"#),
            Parsed::Ignored,
            "zero-length steps are rejected"
        );
    }

    #[cfg(unix)]
    #[test]
    fn reads_events_from_a_live_socket() {
        use std::io::Write;
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let pid = 424242;
        let listener = UnixListener::bind(socket_path(dir.path(), pid)).unwrap();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            s.write_all(b"{\"v\":1,\"type\":\"hello\",\"pid\":424242,\"rank\":1}\n")
                .unwrap();
            s.write_all(b"{\"v\":1,\"type\":\"step\",\"pid\":0,\"step\":1,\"t_end_ms\":5,\"step_s\":0.25}\n{\"v\":1,\"type\":\"st")
                .unwrap();
            s.write_all(b"ep\",\"pid\":0,\"step\":2,\"t_end_ms\":9,\"step_s\":0.5}\n")
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        });

        let mut client = HookClient::new(dir.path().to_path_buf());
        let mut events = Vec::new();
        let start = std::time::Instant::now();
        while events.len() < 2 && start.elapsed().as_secs() < 5 {
            events.extend(client.poll(&[pid]));
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        server.join().unwrap();
        assert_eq!(events.len(), 2, "split line must be reassembled");
        assert!(
            events.iter().all(|e| e.pid == pid),
            "pid comes from the socket"
        );
        assert_eq!(client.peers().len(), 1);
        assert_eq!(client.peers()[0].rank, Some(1));
    }
}
