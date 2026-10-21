#!/usr/bin/env bash
# gpu-validation.sh — calibrate the estimators on a real NVIDIA machine.
#
# Runs the reference workload (examples/train_with_hook.py, which reports
# exact FLOPs through calibrate_hook) and compares, on the same run:
#
#   measured MFU (hook)  vs  tensor-active estimate (GPM/DCGM)  vs  busy-time upper bound
#
# and checks that an injected input-pipeline stall is diagnosed. Raw traces
# are saved under tests/fixtures/real/<gpu>/ so they can be committed and
# replayed by the test-suite as regression fixtures.
#
# Requirements: NVIDIA driver, PyTorch with CUDA, python3. DCGM optional.
#
#   ./scripts/gpu-validation.sh                 # uses target/release/calibrate
#   ./scripts/gpu-validation.sh --bin ./calibrate --steps 400
#
# Pass criteria (printed at the end):
#   * |tensor-active − measured| ≤ 10 ppt     (when GPM or DCGM is available)
#   * busy-time upper bound ≥ measured MFU
#   * the --data-delay run is diagnosed as data_loader

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CAL="$REPO/target/release/calibrate"
DURATION=40
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) CAL="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "unknown flag $1"; exit 2 ;;
    esac
done
[ -x "$CAL" ] || (cd "$REPO" && cargo build --release --locked)
python3 -c "import torch; assert torch.cuda.is_available()" \
    || { echo "PyTorch with CUDA is required"; exit 1; }

GPU="$(python3 -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '_'))")"
OUT="$REPO/tests/fixtures/real/$GPU"
mkdir -p "$OUT"
echo "GPU: $GPU   results: $OUT"
"$CAL" diagnose | sed -n '/GPU profiling/,/^$/p' || true

run_case() {  # <name> <extra train args...>
    local name="$1"; shift
    CUDA_VISIBLE_DEVICES=0 python3 "$REPO/examples/train_with_hook.py" \
        --steps 1000000 --batch 16 --seq 1024 --dim 1024 --layers 8 "$@" \
        >"$OUT/$name.train.log" 2>&1 &
    local pid=$!
    sleep 20   # model build + CUDA warm-up
    "$CAL" watch --pid "$pid" --interval 1 --duration "$DURATION" --output json \
        --record-raw "$OUT/$name.trace.jsonl" >"$OUT/$name.snapshots.jsonl"
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true

    # Replay without hook events to obtain the hardware-only estimate.
    python3 - "$OUT/$name.trace.jsonl" "$OUT/$name.nohook.jsonl" <<'EOF'
import json, sys
with open(sys.argv[1]) as f, open(sys.argv[2], "w") as g:
    for i, line in enumerate(f):
        d = json.loads(line)
        if i > 0:
            d["steps"], d["hook_peers"] = [], []
        g.write(json.dumps(d) + "\n")
EOF
    "$CAL" watch --replay "$OUT/$name.nohook.jsonl" --output json >"$OUT/$name.nohook.snapshots.jsonl"
    # Busy-time-only view: strip profiling counters too.
    python3 - "$OUT/$name.nohook.jsonl" "$OUT/$name.busy.jsonl" <<'EOF'
import json, sys
with open(sys.argv[1]) as f, open(sys.argv[2], "w") as g:
    for i, line in enumerate(f):
        d = json.loads(line)
        if i > 0:
            for gpu in d["gpus"]:
                gpu["prof"] = None
        g.write(json.dumps(d) + "\n")
EOF
    "$CAL" watch --replay "$OUT/$name.busy.jsonl" --output json >"$OUT/$name.busy.snapshots.jsonl"
}

run_case well_fed
run_case data_stall --data-delay 0.15

python3 - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
def last(path):
    snaps = [json.loads(l) for l in open(path) if '"snapshot"' in l]
    return snaps[-1] if snaps else None
ok = True
print(f"\n{'case':<12} {'measured':>9} {'tensor-act':>11} {'busy≤':>7}  bottleneck")
for case in ("well_fed", "data_stall"):
    m, h, b = (last(os.path.join(out, f"{case}{s}.jsonl"))
               for s in (".snapshots", ".nohook.snapshots", ".busy.snapshots"))
    mm = m["mfu"]["mfu_pct"] if m and m["mfu"]["source"] == "instrumented" else None
    hh = h["mfu"]["mfu_pct"] if h and h["mfu"]["source"] == "tensor_active" else None
    bb = b["mfu"]["mfu_pct"] if b else None
    fmt = lambda v: "n/a" if v is None else f"{v:.1f}%"
    print(f"{case:<12} {fmt(mm):>9} {fmt(hh):>11} {fmt(bb):>7}  {m['bottleneck'] if m else '?'}")
    if mm is None:
        print("  ✗ hook did not report — measured MFU missing"); ok = False; continue
    if hh is not None and abs(hh - mm) > 10:
        print(f"  ✗ tensor-active estimate off by {abs(hh-mm):.1f} ppt"); ok = False
    if bb is not None and bb + 1 < mm:
        print("  ✗ busy-time bound is below measured MFU"); ok = False
    if case == "data_stall" and m["bottleneck"] != "data_loader":
        print("  ✗ stall not diagnosed as data_loader"); ok = False
print("\nRESULT:", "PASS" if ok else "FAIL")
print("Commit tests/fixtures/real/ to keep these traces as regression fixtures.")
sys.exit(0 if ok else 1)
EOF
