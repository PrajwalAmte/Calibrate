"""Non-blocking Unix-socket publisher used by the hook.

Design constraints: never block the training thread, never raise into it,
never leak a socket file, and behave under ``fork`` (DataLoader workers).
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import select
import socket
import threading
from typing import Any, Deque, Dict, List, Optional

_BACKLOG = 64  # recent events replayed to a newly connected watcher


def socket_path(pid: Optional[int] = None, hook_dir: Optional[str] = None) -> str:
    d = hook_dir or os.environ.get("CALIBRATE_HOOK_DIR") or "/tmp"
    return os.path.join(d, f"calibrate-hook-{pid or os.getpid()}.sock")


class HookServer:
    def __init__(self, hello: Dict[str, Any], hook_dir: Optional[str] = None) -> None:
        self._hello = (json.dumps(hello) + "\n").encode()
        self._backlog: Deque[bytes] = collections.deque(maxlen=_BACKLOG)
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()
        self._closed = False
        self.path: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        if not hasattr(socket, "AF_UNIX"):
            return
        try:
            path = socket_path(hook_dir=hook_dir)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            old = os.umask(0o177)  # socket file mode 0600: owner (and root) only
            try:
                s.bind(path)
            except OSError:
                s.close()
                raise
            finally:
                os.umask(old)
            s.listen(8)
            s.setblocking(False)
            self._sock = s
            self.path = path
            atexit.register(self.close)
            t = threading.Thread(target=self._accept_loop, name="calibrate-hook", daemon=True)
            t.start()
        except Exception:
            self._sock = None
            self.path = None

    def _accept_loop(self) -> None:
        while not self._closed and self._sock is not None:
            try:
                r, _, _ = select.select([self._sock], [], [], 0.5)
                if not r:
                    continue
                conn, _ = self._sock.accept()
                conn.setblocking(False)
                with self._lock:
                    payload = self._hello + b"".join(self._backlog)
                    if self._send(conn, payload):
                        self._clients.append(conn)
            except Exception:
                if self._closed:
                    return

    @staticmethod
    def _send(conn: socket.socket, data: bytes) -> bool:
        try:
            conn.sendall(data)
            return True
        except (BlockingIOError, InterruptedError):
            # A watcher that cannot keep up is dropped rather than stalling training.
            try:
                conn.close()
            except Exception:
                pass
            return False
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return False

    def publish(self, msg: Dict[str, Any]) -> None:
        if self._closed or os.getpid() != self._owner_pid:
            return
        try:
            line = (json.dumps(msg) + "\n").encode()
        except Exception:
            return
        with self._lock:
            self._backlog.append(line)
            self._clients = [c for c in self._clients if self._send(c, line)]

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        # Forked children inherit atexit handlers; only the creator cleans up.
        if self._closed or os.getpid() != self._owner_pid:
            return
        self._closed = True
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except Exception:
                    pass
            self._clients = []
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self.path:
            try:
                os.unlink(self.path)
            except OSError:
                pass
