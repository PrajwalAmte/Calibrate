#!/usr/bin/env bash
# test-all.sh — release smoke test for a built calibrate binary.
#
# Exercises every subcommand against the real binary, offline and
# deterministically. Core checks never skip: if something cannot run on a
# supported platform, that is a failure. Only optional extras skip (PyTorch
# hook demo, live network, Docker).
#
# Usage:
#   ./scripts/test-all.sh                    # build release binary, run everything
#   ./scripts/test-all.sh --no-build         # use target/release/calibrate
#   ./scripts/test-all.sh --bin <path>       # test a specific binary (e.g. a release artifact)
#   ./scripts/test-all.sh --network          # also hit live HF Hub / pricing APIs
#   ./scripts/test-all.sh --docker           # also run `diagnose` in an Ubuntu container
#
# Exit code: 0 when every non-skipped check passed, 1 otherwise.

set -uo pipefail

BUILD=true
BIN=""
NETWORK=false
DOCKER=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=false; shift ;;
        --bin)      BIN="$2"; shift 2 ;;
        --network)  NETWORK=true; shift ;;
        --docker)   DOCKER=true; shift ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; DIM='\033[2m'; NC='\033[0m'
[ -t 1 ] || { RED=''; GRN=''; YLW=''; BLU=''; DIM=''; NC=''; }
PASS=0; FAIL=0; SKIP=0
pass() { echo -e "  ${GRN}✓${NC}  $*"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${NC}  $*"; FAIL=$((FAIL + 1)); }
skip() { echo -e "  ${YLW}·${NC}  ${DIM}SKIP${NC}  $*"; SKIP=$((SKIP + 1)); }
section() { echo -e "\n${BLU}[ $* ]${NC}"; }

# expect <desc> <regex> <text>
expect() {
    if printf '%s' "$3" | grep -qE -- "$2"; then pass "$1"; else
        fail "$1"; echo -e "       ${DIM}expected /$2/, got:${NC}"
        printf '%s\n' "$3" | head -5 | sed 's/^/         /'
    fi
}
# expect_exit <desc> <code> <cmd...>
expect_exit() {
    local desc="$1" want="$2"; shift 2
    "$@" >/dev/null 2>&1; local got=$?
    if [ "$got" = "$want" ]; then pass "$desc (exit $got)"; else fail "$desc — expected exit $want, got $got"; fi
}
# json_get <json> <python expression over d> — prints the result
json_get() { python3 -c "import json,sys; d=json.loads(sys.argv[1]); print($2)" "$1" 2>/dev/null; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "$BIN" ]; then CAL="$BIN"
elif $BUILD; then
    echo "Building release binary…"
    (cd "$REPO" && cargo build --release --locked 2>&1 | tail -2)
    CAL="$REPO/target/release/calibrate"
else CAL="$REPO/target/release/calibrate"; fi
[ -x "$CAL" ] || { echo -e "${RED}binary not found: $CAL${NC}"; exit 1; }

# Isolate from the developer's config, caches and network.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; jobs -p | xargs -r kill 2>/dev/null' EXIT
export HOME="$WORK/home" XDG_CACHE_HOME="$WORK/home/.cache" XDG_CONFIG_HOME="$WORK/home/.config"
mkdir -p "$HOME"
unset CALIBRATE_CONFIG HF_TOKEN
export CALIBRATE_OFFLINE=1 RUST_LOG=error
OS="$(uname -s)"
FIX="$REPO/tests/fixtures"
echo -e "${GRN}Testing:${NC} $CAL  ${DIM}($OS $(uname -m))${NC}"

section "Global CLI"
expect "--help lists subcommands" "diagnose.*" "$("$CAL" --help 2>&1)"
expect "--version" "calibrate [0-9]+\.[0-9]+\.[0-9]+" "$("$CAL" --version 2>&1)"
expect_exit "no arguments is a usage error" 2 "$CAL"

section "diagnose"
OUT="$("$CAL" diagnose 2>&1)"
expect "platform section" "\[ Platform \]" "$OUT"
expect "own PID visible" "Can observe own PID" "$OUT"
expect "summary line" "Result: (OK|WARN|FAIL)" "$OUT"
if [ "$OS" = "Darwin" ]; then expect "Apple GPU section" "\[ Apple GPU \]" "$OUT"; fi
if [ "$OS" = "Linux" ]; then expect "NVIDIA section" "\[ NVIDIA GPU \]" "$OUT"; fi

section "watch — argument handling"
OUT="$("$CAL" watch --pid 999999999 --cost-per-hour 0.34 --interval 1 --output json 2>&1)"
expect "--pid combines with other options (regression)" "not found" "$OUT"
expect "interval validation" "between 0.1 and 3600" "$("$CAL" watch --pid 1 --interval 0 2>&1)"
expect "target required" "required" "$("$CAL" watch --interval 1 2>&1)"

section "watch — replay of recorded traces"
check_replay() {  # <fixture> <python-assert-expression> <desc>
    local out last
    out="$("$CAL" watch --replay "$FIX/$1" --output json 2>&1)" || { fail "$3 (exit $?)"; return; }
    last="$(printf '%s\n' "$out" | grep '"event":"snapshot"' | tail -1)"
    if [ "$(json_get "$last" "$2")" = "True" ]; then pass "$3"; else
        fail "$3"; echo "         $(json_get "$last" "d['bottleneck'], d['mfu']")"; fi
}
check_replay healthy_h100_gpm.jsonl "d['bottleneck']=='none' and d['mfu']['source']=='tensor_active'" "healthy H100 → no bottleneck, tensor-active MFU"
check_replay dataloader_bound.jsonl "d['bottleneck']=='data_loader'" "input-pipeline bound job detected"
check_replay kernel_busy_only_a10g.jsonl "d['mfu']['upper_bound'] and d['bottleneck']=='unverified'" "busy-time only → upper bound, unverified"
check_replay thermal_throttle.jsonl "d['bottleneck']=='thermal_throttle'" "thermal throttling detected"
check_replay multi_gpu_straggler.jsonl "d['bottleneck']=='gpu_imbalance'" "straggler GPU detected"
check_replay vram_leak.jsonl "d['vram_growing']" "VRAM leak detected"
check_replay hooked_instrumented.jsonl "d['mfu']['source']=='instrumented' and abs(d['mfu']['mfu_pct']-40)<1" "hook → measured MFU"
check_replay cpu_only.jsonl "d['mfu']['mfu_pct'] is None and d['bottleneck']=='no_gpu_metrics'" "no GPU → N/A, never 0 %"

section "watch — live process tree, exporters, recording"
sh -c 'sleep 60 & sleep 60 & wait' & TREE=$!
sleep 0.3
"$CAL" watch --pid "$TREE" --interval 0.3 --duration 3 --output json \
    --prometheus-listen 127.0.0.1:0 --record "$WORK/snaps.jsonl" --record-raw "$WORK/raw.jsonl" \
    >"$WORK/live.jsonl" 2>"$WORK/live.err" &
WATCH=$!
ADDR=""
for _ in $(seq 1 30); do
    ADDR="$(grep -oE 'http://[0-9.:]+/metrics' "$WORK/live.err" | head -1)"
    [ -n "$ADDR" ] && break; sleep 0.1
done
sleep 1.5
if [ -n "$ADDR" ] && command -v curl >/dev/null; then
    METRICS="$(curl -fsS "$ADDR" 2>&1)"
    expect "Prometheus endpoint serves metrics" "^calibrate_up 1" "$METRICS"
    expect "Prometheus exposition has TYPE lines" "# TYPE calibrate_info gauge" "$METRICS"
else
    fail "Prometheus endpoint did not start ($(head -1 "$WORK/live.err"))"
fi
wait "$WATCH"; RC=$?
pkill -P "$TREE" 2>/dev/null; kill "$TREE" 2>/dev/null; wait "$TREE" 2>/dev/null
[ "$RC" = 0 ] && pass "watch exits 0 after --duration" || fail "watch exited $RC: $(tail -3 "$WORK/live.err")"
LAST="$(grep '"event":"snapshot"' "$WORK/live.jsonl" | tail -1)"
expect "live JSON is versioned" '"schema_version":1' "$LAST"
[ "$(json_get "$LAST" "d['target']['process_count']>=3")" = "True" ] \
    && pass "child processes are included" || fail "child processes missing: $LAST"
expect "session_end emitted" '"event":"session_end"' "$(tail -1 "$WORK/live.jsonl")"
expect "--record wrote snapshots" '"event":"snapshot"' "$(head -1 "$WORK/snaps.jsonl")"
expect "--record-raw wrote a trace header" '"calibrate_trace":1' "$(head -1 "$WORK/raw.jsonl")"
expect_exit "recorded raw trace replays" 0 "$CAL" watch --replay "$WORK/raw.jsonl" --output none

section "probe"
sleep 30 & SL=$!
OUT="$("$CAL" probe --pid "$SL" --count 2 --interval 0.3 2>/dev/null)"
kill "$SL" 2>/dev/null; wait "$SL" 2>/dev/null
expect "probe streams raw ticks" '"timestamp_ms"' "$OUT"
expect_exit "probe needs a target" 2 "$CAL" probe

section "plan (offline)"
cat >"$WORK/prices.json" <<'EOF'
[{"provider":"onprem","gpu_model":"H100 SXM","gpu_count":8,"vram_per_gpu_gib":80,"hourly_usd":20.0},
 {"provider":"cloud","gpu_model":"RTX 4090","vram_per_gpu_gib":24,"hourly_usd":0.44}]
EOF
OUT="$("$CAL" plan -m test/8b --params-b 8 --method lora --dataset-rows 50000 --prices-file "$WORK/prices.json" --output json 2>/dev/null)"
[ "$(json_get "$OUT" "d['schema_version']==1 and d['listings'][0]['duration_range']['low_secs']>3600")" = "True" ] \
    && pass "plan JSON has realistic durations (sequence length included)" \
    || fail "plan JSON unexpected: $(printf '%s' "$OUT" | head -3)"
expect "plan terminal shows recommendation" "Recommendation:" \
    "$("$CAL" plan -m t --params-b 1 --dataset-rows 100 --prices-file "$WORK/prices.json" 2>&1)"
expect "invalid method/quantization rejected" "qlora" \
    "$("$CAL" plan -m t --params-b 7 --method full --quantization 4bit 2>&1)"
expect "offline requires --params-b for hub IDs" "offline" "$("$CAL" plan -m org/model 2>&1)"

section "bench"
expect "missing model file" "not found" "$("$CAL" bench --model "$WORK/none.onnx" 2>&1)"
touch "$WORK/m.onnx"
expect "unknown runtime rejected" "unknown runtime" "$("$CAL" bench --runtimes candle --model "$WORK/m.onnx" 2>&1)"
if python3 -c "import torch" 2>/dev/null; then
    python3 - "$WORK/mlp.pt" <<'EOF'
import sys, torch
class M(torch.nn.Module):
    def __init__(s): super().__init__(); s.l = torch.nn.Linear(64, 64)
    def forward(s, x): return s.l(x)
torch.jit.trace(M().eval(), torch.randn(1, 64)).save(sys.argv[1])
EOF
    OUT="$("$CAL" bench --model "$WORK/mlp.pt" -b 1 --iterations 12 --warmup 2 --throughput-window 0.5 --output json 2>/dev/null)"
    [ "$(json_get "$OUT" "d['results'][0]['stats']['sample_count']==12")" = "True" ] \
        && pass "torchscript bench honours --iterations" || fail "torchscript bench: $(printf '%s' "$OUT" | head -2)"
else
    skip "torchscript bench (PyTorch not installed)"
fi

section "Python hook"
if (cd "$REPO/python" && python3 -m unittest discover -s tests >"$WORK/py.log" 2>&1); then
    pass "calibrate_hook unit tests"
else
    fail "calibrate_hook unit tests: $(tail -5 "$WORK/py.log")"
fi
if python3 -c "import torch" 2>/dev/null; then
    python3 "$REPO/examples/train_with_hook.py" --steps 100000 --data-delay 0.05 \
        --dim 128 --layers 1 --seq 64 >"$WORK/train.log" 2>&1 & TR=$!
    sleep 4
    OUT="$("$CAL" watch --pid "$TR" --interval 0.5 --duration 6 --output json 2>/dev/null | grep '"event":"snapshot"' | tail -1)"
    kill "$TR" 2>/dev/null; wait "$TR" 2>/dev/null
    [ "$(json_get "$OUT" "d['mfu']['source']=='instrumented' and d['breakdown']['source']=='instrumented'")" = "True" ] \
        && pass "hooked PyTorch job → measured MFU and breakdown" \
        || fail "hook end-to-end: $(json_get "$OUT" "d['mfu'], d['hook_peers']")"
else
    skip "hook end-to-end (PyTorch not installed)"
fi

section "Binary"
command -v file >/dev/null && expect "is an executable" "executable|Mach-O|ELF" "$(file "$CAL")"
if [ "$OS" = "Linux" ] && command -v ldd >/dev/null; then
    LDD="$(ldd "$CAL" 2>&1)"
    if printf '%s' "$LDD" | grep -q libnvidia-ml; then fail "must not link libnvidia-ml (dlopen at runtime)"
    else pass "NVML is loaded at runtime, not linked"; fi
    if printf '%s' "$LDD" | grep -qE "libssl|libcrypto"; then fail "must not depend on OpenSSL"
    else pass "no OpenSSL dependency (rustls)"; fi
fi

section "Network (optional)"
if $NETWORK; then
    OUT="$(CALIBRATE_OFFLINE=0 "$CAL" plan -m Qwen/Qwen2.5-0.5B --dataset-rows 1000 --providers vastai,runpod --output json 2>/dev/null)"
    [ "$(json_get "$OUT" "d['workload']['model']['param_source']=='hub_safetensors'")" = "True" ] \
        && pass "HF Hub exact parameter count" || fail "HF Hub resolution: $(printf '%s' "$OUT" | tail -2)"
else
    skip "live network checks (pass --network)"
fi

section "Linux diagnose in Docker (optional)"
if $DOCKER && command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    OUT="$(docker build -q -f "$REPO/Dockerfile" "$REPO" 2>&1 | tail -1)"
    D="$(docker run --rm "$OUT" diagnose 2>&1 || true)"
    expect "container diagnose runs" "\[ NVIDIA GPU \]" "$D"
    expect "missing driver is explained" "modprobe|nvidiactl" "$D"
    docker rmi "$OUT" >/dev/null 2>&1 || true
else
    skip "Docker diagnose (pass --docker)"
fi

echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e " ${GRN}passed $PASS${NC}   ${RED}failed $FAIL${NC}   ${YLW}skipped $SKIP${NC}"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$FAIL" -gt 0 ]; then echo -e "${RED}RESULT: FAIL${NC}"; exit 1; fi
echo -e "${GRN}RESULT: PASS${NC}"
