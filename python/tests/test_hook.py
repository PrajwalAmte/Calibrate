import json
import os
import socket
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import calibrate_hook  # noqa: E402
from calibrate_hook import Hook, transformer_flops_per_token  # noqa: E402
from calibrate_hook._server import HookServer, socket_path  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def read_lines(sock, n, timeout=5.0):
    sock.settimeout(timeout)
    buf = b""
    deadline = time.time() + timeout
    while buf.count(b"\n") < n and time.time() < deadline:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return [json.loads(l) for l in buf.decode().splitlines() if l.strip()]


def connect(path, timeout=5.0):
    deadline = time.time() + timeout
    while True:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(path)
            return s
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.05)


class FlopsTest(unittest.TestCase):
    def test_full_and_lora(self):
        self.assertEqual(transformer_flops_per_token(1e9), 6e9)
        self.assertEqual(transformer_flops_per_token(1e9, mode="lora"), 4e9)
        with_attn = transformer_flops_per_token(1e9, n_layers=2, hidden_size=4, seq_len=8)
        self.assertEqual(with_attn, 6e9 + 12 * 2 * 4 * 8)
        with self.assertRaises(ValueError):
            transformer_flops_per_token(1, mode="bogus")


class HookTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cal-hook-")

    def make(self, **kw):
        clock = FakeClock()
        hook = Hook(flops_per_token=10.0, sync="none", hook_dir=self.dir, clock=clock, **kw)
        self.addCleanup(hook.close)
        return hook, clock

    def test_socket_path_and_permissions(self):
        hook, _ = self.make()
        self.assertEqual(hook.socket_path, socket_path(os.getpid(), self.dir))
        mode = os.stat(hook.socket_path).st_mode & 0o777
        self.assertEqual(mode & 0o077, 0, "socket must not be group/world accessible")

    def test_steps_are_published_with_data_wait_and_flops(self):
        hook, clock = self.make(rank=3, world_size=8)
        s = connect(hook.socket_path)
        self.addCleanup(s.close)
        hello = read_lines(s, 1)[0]
        self.assertEqual(hello["type"], "hello")
        self.assertEqual(hello["pid"], os.getpid())
        self.assertEqual(hello["rank"], 3)
        self.assertEqual(hello["v"], 1)

        for _ in range(3):
            clock.advance(0.2)  # data loading before the step
            with hook.step(tokens=100):
                clock.advance(0.8)
        # Wait until the accept loop has registered us before checking.
        steps = read_lines(s, 3)
        self.assertEqual([m["step"] for m in steps], [1, 2, 3])
        first, second = steps[0], steps[1]
        self.assertNotIn("data_s", first, "first step has no previous end")
        self.assertAlmostEqual(second["data_s"], 0.2)
        self.assertAlmostEqual(second["step_s"], 1.0, msg="step_s includes data wait")
        self.assertEqual(second["flops"], 1000.0)
        self.assertNotIn("compute_s", second, "phases need sync='phases'")

    def test_backlog_is_replayed_to_late_watchers(self):
        hook, clock = self.make()
        for _ in range(5):
            with hook.step(tokens=1):
                clock.advance(0.1)
        s = connect(hook.socket_path)
        self.addCleanup(s.close)
        msgs = read_lines(s, 6)
        self.assertEqual(msgs[0]["type"], "hello")
        self.assertEqual([m["step"] for m in msgs[1:]], [1, 2, 3, 4, 5])

    def test_phases_reported_only_with_phase_sync(self):
        clock = FakeClock()
        hook = Hook(sync="phases", hook_dir=self.dir, clock=clock)
        self.addCleanup(hook.close)
        s = connect(hook.socket_path)
        self.addCleanup(s.close)
        read_lines(s, 1)
        with hook.step():
            with hook.phase("compute"):
                clock.advance(0.6)
            with hook.phase("optimizer"):
                clock.advance(0.1)
        msg = read_lines(s, 1)[0]
        self.assertAlmostEqual(msg["compute_s"], 0.6)
        self.assertAlmostEqual(msg["optimizer_s"], 0.1)

    def test_never_raises_into_training(self):
        hook = Hook(hook_dir="/definitely/not/a/dir", sync="none")
        self.assertIsNone(hook.socket_path)
        with hook.step(tokens=1):
            pass
        hook.step_end()  # unmatched end is harmless
        hook.close()
        with self.assertRaises(ValueError):
            Hook(sync="sometimes")

    def test_close_removes_socket(self):
        hook, _ = self.make()
        path = hook.socket_path
        self.assertTrue(os.path.exists(path))
        hook.close()
        self.assertFalse(os.path.exists(path))

    @unittest.skipUnless(hasattr(os, "fork"), "needs fork")
    def test_forked_child_does_not_remove_parent_socket(self):
        hook, _ = self.make()
        pid = os.fork()
        if pid == 0:  # DataLoader-worker-like child
            hook.close()
            hook._server.publish({"type": "step"})
            os._exit(0)
        os.waitpid(pid, 0)
        self.assertTrue(os.path.exists(hook.socket_path))


class GlobalStartTest(unittest.TestCase):
    def tearDown(self):
        calibrate_hook.stop()

    def test_start_is_idempotent(self):
        d = tempfile.mkdtemp()
        a = calibrate_hook.start(flops_per_token=1.0, hook_dir=d, sync="none")
        b = calibrate_hook.start(flops_per_token=2.0, hook_dir=d, sync="none")
        self.assertIs(a, b)
        self.assertIs(calibrate_hook.get(), a)

    def test_flops_from_torch_model(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        m = torch.nn.Linear(10, 10)  # 110 params
        h = calibrate_hook.start(model=m, hook_dir=tempfile.mkdtemp(), sync="none")
        self.assertEqual(h.flops_per_token, 6 * 110)


if __name__ == "__main__":
    unittest.main()
