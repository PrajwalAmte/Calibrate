"""calibrate_hook — measured MFU and step-time breakdown for `calibrate watch`.

Usage::

    import calibrate_hook

    hook = calibrate_hook.start(model=model)          # FLOPs/token from the model
    for batch in loader:                              # time spent here = data wait
        with hook.step(tokens=batch["input_ids"].numel()):
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

Then run ``calibrate watch --pid <training pid>`` (or ``--process-name``);
calibrate finds the hook automatically and reports *measured* MFU instead of
a hardware estimate.

The hook publishes JSON lines on a Unix socket at
``$CALIBRATE_HOOK_DIR/calibrate-hook-<pid>.sock`` (default ``/tmp``). It has no
dependencies, never blocks the training loop, and every failure inside it is
swallowed — instrumentation must not break training.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Any, Callable, Iterator, Optional

from ._server import HookServer, socket_path

__all__ = [
    "Hook",
    "start",
    "get",
    "stop",
    "transformer_flops_per_token",
    "socket_path",
    "PROTOCOL_VERSION",
]
__version__ = "0.2.0"

PROTOCOL_VERSION = 1

_SYNC_MODES = ("step", "phases", "none")


def transformer_flops_per_token(
    n_params: float,
    n_layers: int = 0,
    hidden_size: int = 0,
    seq_len: int = 0,
    mode: str = "full",
) -> float:
    """Model FLOPs per trained token (PaLM appendix B convention).

    ``6N + 12·L·H·S`` for full fine-tuning / pre-training. With ``mode="lora"``
    frozen weights get no weight-gradient matmul, so the parameter term is
    ``4N``. Activation recomputation is *not* counted — that is what makes
    this MFU rather than HFU.
    """
    if mode not in ("full", "lora"):
        raise ValueError("mode must be 'full' or 'lora'")
    per_param = 6.0 if mode == "full" else 4.0
    attn = 12.0 * n_layers * hidden_size * seq_len
    if mode == "lora":
        attn *= 4.0 / 6.0
    return per_param * float(n_params) + attn


def _count_params(model: Any) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def _device_name() -> Optional[str]:
    torch = __import__("sys").modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return f"cuda:{torch.cuda.current_device()}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _synchronize() -> None:
    torch = __import__("sys").modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def _env_int(*names: str) -> Optional[int]:
    for n in names:
        v = os.environ.get(n)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return None


class Hook:
    """Times training steps and publishes them to calibrate.

    ``sync`` controls device synchronisation, which is what makes GPU timings
    trustworthy with asynchronous CUDA/MPS execution:

    * ``"step"`` (default): one synchronise at the end of each step. Step time
      and data-wait time are exact; overhead is negligible for steps longer
      than a few tens of milliseconds.
    * ``"phases"``: also synchronise after each ``phase()`` block, giving an
      exact forward/backward vs optimizer split at some throughput cost.
    * ``"none"``: never synchronise. Step time is right on average in steady
      state, but data wait may overlap GPU work, so treat it as approximate.
    """

    def __init__(
        self,
        flops_per_token: Optional[float] = None,
        sync: str = "step",
        framework: str = "pytorch",
        hook_dir: Optional[str] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        clock: Callable[[], float] = time.perf_counter,
        server: Optional[HookServer] = None,
    ) -> None:
        if sync not in _SYNC_MODES:
            raise ValueError(f"sync must be one of {_SYNC_MODES}")
        self.flops_per_token = flops_per_token
        self.sync = sync
        self._clock = clock
        self._step = 0
        self._last_end: Optional[float] = None
        self._begin: Optional[float] = None
        self._phases: dict = {}
        self._lock = threading.Lock()
        self.rank = rank if rank is not None else _env_int("RANK", "LOCAL_RANK")
        self.world_size = (
            world_size if world_size is not None else _env_int("WORLD_SIZE")
        )
        self.framework = framework
        hello = {
            "v": PROTOCOL_VERSION,
            "type": "hello",
            "pid": os.getpid(),
            "rank": self.rank,
            "world_size": self.world_size,
            "framework": framework,
            "device": _device_name(),
            "sync": sync,
            "hook_version": __version__,
        }
        self._server = server if server is not None else HookServer(hello, hook_dir)

    # Explicit API -----------------------------------------------------------

    def step_begin(self) -> None:
        try:
            self._begin = self._clock()
            self._phases = {}
        except Exception:
            pass

    def step_end(
        self,
        tokens: Optional[int] = None,
        samples: Optional[int] = None,
        flops: Optional[float] = None,
    ) -> None:
        """Finish a step. ``flops`` overrides ``flops_per_token × tokens``."""
        try:
            if self.sync != "none":
                _synchronize()
            end = self._clock()
            begin = self._begin if self._begin is not None else end
            # Data wait = gap between the previous step's end and this begin.
            data_s = (begin - self._last_end) if self._last_end is not None else None
            start = self._last_end if self._last_end is not None else begin
            step_s = end - start
            self._last_end = end
            self._begin = None
            if step_s <= 0:
                return
            if flops is None and self.flops_per_token and tokens:
                flops = float(self.flops_per_token) * float(tokens)
            self._step += 1
            msg = {
                "v": PROTOCOL_VERSION,
                "type": "step",
                "pid": os.getpid(),
                "rank": self.rank,
                "step": self._step,
                "t_end_ms": int(time.time() * 1000),
                "step_s": step_s,
                "data_s": max(0.0, data_s) if data_s is not None else None,
                "tokens": tokens,
                "samples": samples,
                "flops": flops,
            }
            if self.sync == "phases" and self._phases:
                msg["compute_s"] = self._phases.get("compute")
                msg["optimizer_s"] = self._phases.get("optimizer")
            self._server.publish({k: v for k, v in msg.items() if v is not None})
        except Exception:
            pass

    # Context managers -------------------------------------------------------

    @contextlib.contextmanager
    def step(
        self,
        tokens: Optional[int] = None,
        samples: Optional[int] = None,
        flops: Optional[float] = None,
    ) -> Iterator["Hook"]:
        self.step_begin()
        try:
            yield self
        finally:
            self.step_end(tokens=tokens, samples=samples, flops=flops)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a sub-phase: ``"compute"`` (forward+backward) or ``"optimizer"``.

        Only reported with ``sync="phases"``; otherwise asynchronous execution
        makes CPU-side phase timings misleading.
        """
        t0 = self._clock()
        try:
            yield
        finally:
            try:
                if self.sync == "phases":
                    _synchronize()
                dt = self._clock() - t0
                self._phases[name] = self._phases.get(name, 0.0) + dt
            except Exception:
                pass

    def close(self) -> None:
        self._server.close()

    @property
    def socket_path(self) -> Optional[str]:
        return self._server.path


_GLOBAL: Optional[Hook] = None
_GLOBAL_LOCK = threading.Lock()


def start(
    model: Any = None,
    flops_per_token: Optional[float] = None,
    mode: str = "full",
    seq_len: int = 0,
    sync: str = "step",
    framework: str = "pytorch",
    hook_dir: Optional[str] = None,
) -> Hook:
    """Create (once per process) and return the global hook.

    Pass ``model`` to derive FLOPs per token from its parameter count (and,
    for Hugging Face models, its layer count and width when ``seq_len`` is
    given), or pass ``flops_per_token`` explicitly. Without either, step
    timings are still reported but MFU falls back to hardware counters.
    """
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            return _GLOBAL
        if flops_per_token is None and model is not None:
            n = _count_params(model)
            cfg = getattr(model, "config", None)
            layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
            hidden = int(getattr(cfg, "hidden_size", 0) or 0)
            if n:
                flops_per_token = transformer_flops_per_token(
                    n, layers, hidden, seq_len, mode=mode
                )
        _GLOBAL = Hook(
            flops_per_token=flops_per_token,
            sync=sync,
            framework=framework,
            hook_dir=hook_dir,
        )
        return _GLOBAL


def get() -> Optional[Hook]:
    return _GLOBAL


def stop() -> None:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            _GLOBAL.close()
            _GLOBAL = None
