"""Minimal training loop instrumented with calibrate_hook.

Trains a small transformer on random tokens. Use it to try the hook, to
reproduce input-pipeline stalls (`--data-delay`), and as the reference
workload in scripts/gpu-validation.sh.

    python examples/train_with_hook.py --steps 200 --data-delay 0.02 &
    calibrate watch --pid $!
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch  # noqa: E402

import calibrate_hook  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--data-delay", type=float, default=0.0,
                    help="seconds of artificial input-pipeline latency per batch")
    ap.add_argument("--sync", default="step", choices=["step", "phases", "none"])
    args = ap.parse_args()

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    vocab = 8192
    layer = torch.nn.TransformerEncoderLayer(
        args.dim, nhead=8, dim_feedforward=4 * args.dim, batch_first=True
    )
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab, args.dim),
        torch.nn.TransformerEncoder(layer, args.layers),
        torch.nn.Linear(args.dim, vocab),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    hook = calibrate_hook.start(model=model, sync=args.sync)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pid={os.getpid()} device={device} params={n_params/1e6:.1f}M "
          f"socket={hook.socket_path}", flush=True)

    for step in range(args.steps):
        if args.data_delay:
            time.sleep(args.data_delay)  # simulated slow DataLoader
        x = torch.randint(0, vocab, (args.batch, args.seq), device=device)
        with hook.step(tokens=x.numel()):
            with hook.phase("compute"):
                with torch.autocast(device_type=device, dtype=dtype, enabled=device != "cpu"):
                    out = model(x)
                    loss = loss_fn(out.view(-1, vocab).float(), x.view(-1))
                loss.backward()
            with hook.phase("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
