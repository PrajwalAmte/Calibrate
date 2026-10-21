# calibrate-hook

Instrumentation for [calibrate](https://github.com/PrajwalAmte/Calibrate). Add
three lines to a training loop and `calibrate watch` reports **measured** MFU,
step times and an exact data-wait / compute / optimizer breakdown, instead of
hardware-counter estimates.

```bash
pip install ./python        # from a calibrate checkout
```

```python
import calibrate_hook

hook = calibrate_hook.start(model=model, seq_len=2048)   # mode="lora" for LoRA
for batch in loader:
    with hook.step(tokens=batch["input_ids"].numel()):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Hugging Face Trainer:

```python
from calibrate_hook.integrations import hf_callback
trainer = Trainer(..., callbacks=[hf_callback(seq_len=2048)],
                  args=TrainingArguments(..., include_num_input_tokens_seen=True))
```

Lightning:

```python
from calibrate_hook.integrations import lightning_callback
trainer = L.Trainer(callbacks=[lightning_callback(tokens_fn=lambda b: b["input_ids"].numel())])
```

See `docs/hook.md` in the calibrate repository for the protocol, the `sync`
modes and multi-rank behaviour.
