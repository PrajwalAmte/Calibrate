# The calibrate Python hook

`calibrate_hook` (in `python/`) turns estimates into measurements: real step
times, an exact data-wait / compute / optimizer split, and MFU from declared
FLOPs. It has no dependencies and cannot break training — every error inside
it is swallowed.

```bash
pip install ./python
```

## Plain PyTorch

```python
import calibrate_hook

hook = calibrate_hook.start(model=model, seq_len=2048)   # mode="lora" for LoRA/QLoRA
for batch in loader:                                     # time here = data wait
    with hook.step(tokens=batch["input_ids"].numel()):
        with hook.phase("compute"):
            loss = model(**batch).loss
            loss.backward()
        with hook.phase("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
```

`start(model=...)` derives FLOPs per token as `6N + 12·L·H·S` (PaLM appendix B;
`4N` for `mode="lora"`), using the model's parameter count and — for Hugging
Face models — `config.num_hidden_layers`/`hidden_size`. Pass
`flops_per_token=` (or `flops=` per step) to override, e.g. for MoE models where
only active parameters count.

## Hugging Face Trainer / Lightning

```python
from calibrate_hook.integrations import hf_callback, lightning_callback

Trainer(..., callbacks=[hf_callback(seq_len=2048)],
        args=TrainingArguments(..., include_num_input_tokens_seen=True))

L.Trainer(callbacks=[lightning_callback(tokens_fn=lambda b: b["input_ids"].numel())])
```

With gradient accumulation the HF callback reports optimizer steps, so data
loading between micro-batches is counted as compute.

## Synchronisation modes

CUDA and MPS execute asynchronously, so CPU-side timestamps alone can be
wrong. `start(sync=...)`:

| Mode | Behaviour | Use when |
|---|---|---|
| `"step"` (default) | one device synchronise at each step end | always, unless steps are < ~20 ms |
| `"phases"` | also after each `phase()` block → exact compute/optimizer split | diagnosing; costs some throughput |
| `"none"` | never synchronise; data wait may overlap GPU work | you cannot afford any sync |

## Discovery and multi-process jobs

Each process with the hook listens on
`$CALIBRATE_HOOK_DIR/calibrate-hook-<pid>.sock` (default `/tmp`, mode `0600`).
`calibrate watch` connects to the sockets of every process in the target tree,
so with `torchrun` each rank that calls `start()` is picked up; MFU sums FLOPs
over ranks and divides by the combined peak of their GPUs. Rank and world
size come from `RANK`/`LOCAL_RANK`/`WORLD_SIZE`.

The socket is owner-only: run calibrate as the same user (or root). In
Kubernetes, share an `emptyDir` between trainer and sidecar and run both with
the same UID — see `deploy/kubernetes/sidecar-example.yaml`.

Forked DataLoader workers inherit the hook object but never publish or remove
the parent's socket. The last 64 events are replayed to a newly connected
watcher, so attaching mid-run shows data immediately. A watcher that cannot
keep up is disconnected rather than ever blocking the training loop.

## Protocol (v1)

Newline-delimited JSON over a Unix stream socket, server → client only.

```json
{"v":1,"type":"hello","pid":4242,"rank":0,"world_size":8,"framework":"pytorch",
 "device":"cuda:0","sync":"step","hook_version":"0.2.0"}
{"v":1,"type":"step","pid":4242,"rank":0,"step":17,"t_end_ms":1760000000000,
 "step_s":0.412,"data_s":0.051,"compute_s":0.301,"optimizer_s":0.042,
 "tokens":65536,"samples":32,"flops":3.1e15}
```

| Field | Meaning |
|---|---|
| `step_s` | wall time from the previous step's end to this step's end (includes data wait) |
| `data_s` | gap between the previous step's end and this step's start |
| `compute_s`, `optimizer_s` | phase times (only with `sync="phases"`) |
| `flops` | model FLOPs performed in this step |

Readers ignore unknown fields and message types; a message with a `v` newer
than the reader supports is ignored. Anything that speaks this protocol (JAX,
custom frameworks, other languages) works with calibrate.
