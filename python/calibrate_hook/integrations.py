"""Framework callbacks. Import only the one you use; each imports its
framework lazily so this module has no hard dependencies."""

from __future__ import annotations

from typing import Any, Callable, Optional

from . import start


def hf_callback(model: Any = None, mode: str = "full", seq_len: int = 0, sync: str = "step"):
    """A ``transformers.TrainerCallback`` that reports every optimizer step.

    Tokens per step come from ``state.num_input_tokens_seen`` (enable
    ``TrainingArguments(include_num_input_tokens_seen=True)``) so FLOPs and
    MFU can be computed. With gradient accumulation a "step" is an optimizer
    step, so data wait between micro-batches is counted as compute.
    """
    from transformers import TrainerCallback  # noqa: PLC0415

    class CalibrateCallback(TrainerCallback):
        def __init__(self) -> None:
            self._hook = None
            self._tokens_seen = 0

        def on_train_begin(self, args, state, control, model=None, **kw):
            self._hook = start(model=model, mode=mode, seq_len=seq_len, sync=sync,
                               framework="transformers")

        def on_step_begin(self, args, state, control, **kw):
            if self._hook:
                self._hook.step_begin()

        def on_step_end(self, args, state, control, **kw):
            if not self._hook:
                return
            seen = int(getattr(state, "num_input_tokens_seen", 0) or 0)
            tokens = seen - self._tokens_seen if seen else None
            self._tokens_seen = seen
            self._hook.step_end(tokens=tokens if tokens and tokens > 0 else None)

    return CalibrateCallback()


def lightning_callback(
    tokens_fn: Optional[Callable[[Any], int]] = None,
    mode: str = "full",
    seq_len: int = 0,
    sync: str = "step",
):
    """A Lightning ``Callback``. ``tokens_fn(batch) -> int`` supplies tokens per step."""
    try:
        from lightning.pytorch import Callback  # noqa: PLC0415
    except ImportError:  # pragma: no cover - older package name
        from pytorch_lightning import Callback  # noqa: PLC0415

    class CalibrateLightningCallback(Callback):
        def __init__(self) -> None:
            self._hook = None

        def on_train_start(self, trainer, pl_module):
            self._hook = start(model=pl_module, mode=mode, seq_len=seq_len, sync=sync,
                               framework="lightning")

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self._hook:
                self._hook.step_begin()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._hook:
                tokens = None
                if tokens_fn is not None:
                    try:
                        tokens = int(tokens_fn(batch))
                    except Exception:
                        tokens = None
                self._hook.step_end(tokens=tokens)

    return CalibrateLightningCallback()
