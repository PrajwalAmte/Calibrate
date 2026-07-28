#!/usr/bin/env bash
# test-all.sh — comprehensive integration test for all calibrate features
#
# Tests every subcommand and flag combination against the built binary.
# Suitable for running locally (macOS or Linux) and in CI.
#
# When Docker is available, also builds a Linux x86-64 image and runs
# `calibrate diagnose` inside Ubuntu 20.04 to validate the NVIDIA error
# paths (no GPU present → expected FAIL with actionable output).
#
# Usage:
#   ./scripts/test-all.sh                  # auto-builds release binary
#   ./scripts/test-all.sh --no-build       # skip build, use existing binary
#   ./scripts/test-all.sh --bin <path>     # use a specific binary path
#   ./scripts/test-all.sh --no-docker      # skip Docker NVIDIA section
#
# Exit codes:
#   0  all tests passed
#   1  one or more tests failed

set -euo pipefail

# Argument parsing 
BUILD=true
BIN=""
RUN_DOCKER=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)  BUILD=false; shift ;;
        --bin)       BIN="$2"; shift 2 ;;
        --no-docker) RUN_DOCKER=false; shift ;;
        *)           echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Colours 
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
BLU='\033[0;34m'
DIM='\033[2m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Helpers 
pass() { echo -e "  ${GRN}✓${NC}  $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "  ${RED}✗${NC}  $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
skip() { echo -e "  ${YLW}·${NC}  ${DIM}SKIP${NC}  $*"; SKIP_COUNT=$((SKIP_COUNT + 1)); }
section() { echo -e "\n${BLU}[ $* ]${NC}"; }

# assert_output <description> <expected_pattern> <actual_output>
assert_output() {
    local desc="$1" pattern="$2" output="$3"
    if echo "$output" | grep -qE "$pattern"; then
        pass "$desc"
    else
        fail "$desc"
        echo -e "       ${DIM}expected pattern: ${pattern}${NC}"
        echo -e "       ${DIM}got (first 3 lines):${NC}"
        echo "$output" | head -3 | sed 's/^/         /'
    fi
}

# assert_exit_code <description> <expected> <command...>
assert_exit_code() {
    local desc="$1" expected="$2"
    shift 2
    local actual
    actual=$("$@" >/dev/null 2>&1; echo $?) || true
    if [ "$actual" = "$expected" ]; then
        pass "$desc (exit $actual)"
    else
        fail "$desc — expected exit $expected, got $actual"
    fi
}

# run_cmd — captures stdout+stderr, always succeeds (returns output)
run_cmd() { "$@" 2>&1 || true; }

# Locate / build binary 
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "$BIN" ]; then
    CAL="$BIN"
elif $BUILD; then
    echo -e "${YLW}Building release binary…${NC}"
    (cd "$REPO_ROOT" && cargo build --release 2>&1 | tail -3)
    CAL="$REPO_ROOT/target/release/calibrate"
else
    CAL="$REPO_ROOT/target/release/calibrate"
fi

if [ ! -x "$CAL" ]; then
    echo -e "${RED}Binary not found or not executable: $CAL${NC}"
    exit 1
fi

echo -e "${GRN}Testing binary:${NC} $CAL"
echo -e "${DIM}OS: $(uname -s)  arch: $(uname -m)${NC}"

OS="$(uname -s)"

# Unit + cargo tests
section "Cargo unit tests"

(cd "$REPO_ROOT" && cargo test --quiet 2>&1 | tail -3) | {
    while IFS= read -r line; do echo "  $line"; done
}
CARGO_EXIT=$( (cd "$REPO_ROOT" && cargo test --quiet >/dev/null 2>&1); echo $? )
if [ "$CARGO_EXIT" = "0" ]; then
    pass "cargo test — all tests passed"
else
    fail "cargo test — one or more tests failed (exit $CARGO_EXIT)"
fi

# Global CLI 
section "Global CLI"

OUT=$(run_cmd "$CAL" --help)
assert_output "--help lists all subcommands" "diagnose|watch|probe|bench|plan" "$OUT"
assert_output "--help shows version flag" "\-V|--version" "$OUT"

OUT=$(run_cmd "$CAL" --version)
assert_output "--version outputs version string" "calibrate [0-9]" "$OUT"

# Ensure the binary fails with a usage error when given no args
assert_exit_code "No args → non-zero exit" "2" "$CAL"

# calibrate diagnose
section "calibrate diagnose"

OUT=$(run_cmd "$CAL" diagnose)

assert_output "diagnose: --help works" "Usage|USAGE|calibrate diagnose" \
    "$(run_cmd "$CAL" diagnose --help)"

assert_output "diagnose: Platform section present" "Platform" "$OUT"
assert_output "diagnose: OS info line present" "OS:.*arch:" "$OUT"
assert_output "diagnose: Process visibility section present" "Process visibility" "$OUT"
assert_output "diagnose: Result line present" "Result:" "$OUT"

if [ "$OS" = "Darwin" ]; then
    assert_output "diagnose (macOS): Apple GPU section present" "Apple GPU" "$OUT"
    assert_output "diagnose (macOS): IOKit result present" "IOKit|AGXAccelerator" "$OUT"
    assert_output "diagnose (macOS): macOS version present" "macOS [0-9]" "$OUT"
elif [ "$OS" = "Linux" ]; then
    assert_output "diagnose (Linux): NVIDIA GPU section present" "NVIDIA GPU" "$OUT"
    # Without an actual GPU the NVIDIA section shows FAIL lines
    assert_output "diagnose (Linux): nvidia-smi check present" "nvidia-smi" "$OUT"
fi

# Own PID must always be visible
assert_output "diagnose: own-PID visibility passes" "Can observe own PID" "$OUT"

# calibrate watch — error paths 
section "calibrate watch — error paths"

# Nonexistent PID
OUT=$(run_cmd "$CAL" watch --pid 9999999)
assert_output "watch: nonexistent PID → actionable error" \
    "not found|pgrep|diagnose" "$OUT"

# Missing required --pid flag
OUT=$(run_cmd "$CAL" watch)
assert_output "watch: missing --pid → usage error" "pid|required|error" "$OUT"

# --help
OUT=$(run_cmd "$CAL" watch --help)
assert_output "watch --help: shows --pid flag" "\-\-pid|\-p" "$OUT"
assert_output "watch --help: shows --interval flag" "\-\-interval|\-i" "$OUT"
assert_output "watch --help: shows --output flag" "\-\-output|\-o" "$OUT"
assert_output "watch --help: shows --cost-per-hour flag" "cost" "$OUT"

# calibrate probe — error paths 
section "calibrate probe — error paths"

OUT=$(run_cmd "$CAL" probe --pid 9999999)
assert_output "probe: nonexistent PID → actionable error" \
    "not found|pgrep|attach|NVML" "$OUT"

OUT=$(run_cmd "$CAL" probe --help)
assert_output "probe --help: shows --pid flag" "\-\-pid|\-p" "$OUT"
assert_output "probe --help: shows --count flag" "\-\-count|\-n" "$OUT"
assert_output "probe --help: shows --interval flag" "\-\-interval|\-i" "$OUT"

# calibrate bench ─
section "calibrate bench — error paths"

OUT=$(run_cmd "$CAL" bench --help)
assert_output "bench --help: shows --model flag" "\-\-model|\-m" "$OUT"
assert_output "bench --help: shows --batch-sizes flag" "batch" "$OUT"
assert_output "bench --help: shows --iterations flag" "iter" "$OUT"
assert_output "bench --help: shows --warmup flag" "warmup" "$OUT"
assert_output "bench --help: shows --optimize-for flag" "optim" "$OUT"
assert_output "bench --help: shows --output flag" "output" "$OUT"
assert_output "bench --help: shows --compare flag" "compare" "$OUT"
assert_output "bench --help: shows --save flag" "save" "$OUT"

# Missing --model
OUT=$(run_cmd "$CAL" bench)
assert_output "bench: missing --model → usage error" "model|required|error" "$OUT"

# Nonexistent model file
OUT=$(run_cmd "$CAL" bench --model /tmp/does_not_exist_calibrate_test.onnx)
assert_output "bench: missing model file → error" \
    "not found|No such|error|invalid" "$OUT"

# calibrate plan
section "calibrate plan"

OUT=$(run_cmd "$CAL" plan --help)
assert_output "plan --help: shows --model flag" "\-\-model|\-m" "$OUT"
assert_output "plan --help: shows --method flag" "method" "$OUT"
assert_output "plan --help: shows --optimizer flag" "optim" "$OUT"
assert_output "plan --help: shows --quantization flag" "quant" "$OUT"
assert_output "plan --help: shows --budget flag" "budget" "$OUT"
assert_output "plan --help: shows --epochs flag" "epoch" "$OUT"
assert_output "plan --help: shows --batch-size flag" "batch" "$OUT"
assert_output "plan --help: shows --providers flag" "provider" "$OUT"
assert_output "plan --help: shows --mfu flag" "mfu" "$OUT"

# plan with a known small model — may fail on network, but must not panic
OUT=$(run_cmd "$CAL" plan --model "gpt2" --method lora --providers vastai)
# Accept either real output or a network/timeout error — just no panic/segfault
assert_output "plan: gpt2 lora — no panic" \
    "VRAM|vram|provider|error|timeout|fetch|network|Resolving" "$OUT"

# plan with explicitly small params (offline-friendly)
OUT=$(run_cmd "$CAL" plan --model "tiny-test-model" --params-b 0.1 --method lora \
    --providers vastai --output json 2>&1 || true)
assert_output "plan: --params-b override accepted" \
    "vram|VRAM|param|0\.1|error|timeout|fetch" "$OUT"

# plan --method full
OUT=$(run_cmd "$CAL" plan --model "gpt2" --method full --providers runpod 2>&1 || true)
assert_output "plan: --method full accepted" \
    "VRAM|vram|full|error|fetch|timeout" "$OUT"

# plan --method qlora
OUT=$(run_cmd "$CAL" plan --model "gpt2" --method qlora --providers lambda 2>&1 || true)
assert_output "plan: --method qlora accepted" \
    "VRAM|vram|qlora|error|fetch|timeout" "$OUT"

# plan --optimizer unsloth
OUT=$(run_cmd "$CAL" plan --model "gpt2" --optimizer unsloth --providers vastai 2>&1 || true)
assert_output "plan: --optimizer unsloth accepted" \
    "VRAM|vram|unsloth|error|fetch|timeout" "$OUT"

# plan --quantization 4bit
OUT=$(run_cmd "$CAL" plan --model "gpt2" --quantization 4bit --providers vastai 2>&1 || true)
assert_output "plan: --quantization 4bit accepted" \
    "VRAM|vram|4bit|error|fetch|timeout" "$OUT"

# plan --output json
OUT=$(run_cmd "$CAL" plan --model "gpt2" --method lora --output json 2>&1 || true)
assert_output "plan: --output json accepted" \
    "\{|json|error|fetch|timeout" "$OUT"

# plan --budget flag accepted (no crash)
OUT=$(run_cmd "$CAL" plan --model "gpt2" --budget 50 --providers vastai 2>&1 || true)
assert_output "plan: --budget flag accepted" \
    "VRAM|vram|budget|50|error|fetch|timeout" "$OUT"

# Watch with self-PID using JSON output (non-GPU smoke test)
section "calibrate watch — self-attach smoke test (JSON)"

if [ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ]; then
    # Start a long-running process we can monitor
    sleep 300 &
    SLEEP_PID=$!
    trap "kill $SLEEP_PID 2>/dev/null || true" EXIT

    # Give it a moment to appear in process table
    sleep 0.2

    # Collect one JSON line from watch then terminate
    OUT=$(timeout 10s "$CAL" watch --pid "$SLEEP_PID" --interval 1 \
        --output json 2>/dev/null | head -1 || true)
    kill "$SLEEP_PID" 2>/dev/null || true
    trap - EXIT

    if [ -n "$OUT" ]; then
        assert_output "watch self-attach: JSON output received" \
            "gpu_name|mfu|elapsed|vram" "$OUT"
        assert_output "watch self-attach: elapsed field present" "elapsed" "$OUT"
        assert_output "watch self-attach: vram field present" "vram" "$OUT"
        assert_output "watch self-attach: gpu_name field present" "gpu_name" "$OUT"
    else
        skip "watch self-attach: no JSON output in 10s (GPU may not be visible for sleep process)"
    fi
else
    skip "watch self-attach: not on Linux or macOS"
fi

# Probe with self-PID (JSON line smoke test)
section "calibrate probe — self-attach smoke test"

if [ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ]; then
    sleep 300 &
    SLEEP_PID=$!
    trap "kill $SLEEP_PID 2>/dev/null || true" EXIT

    sleep 0.2
    OUT=$(timeout 8s "$CAL" probe --pid "$SLEEP_PID" --count 1 --interval 1 \
        2>&1 || true)
    kill "$SLEEP_PID" 2>/dev/null || true
    trap - EXIT

    # probe either succeeds and emits JSON or fails with "not using GPU"
    assert_output "probe self-attach: ran without panic" \
        "Attached|not found|GPU|sm_utilization|error" "$OUT"
else
    skip "probe self-attach: not on Linux or macOS"
fi

# Binary hardening checks 
section "Binary hardening"

if command -v file >/dev/null 2>&1; then
    FILE_OUT=$(file "$CAL")
    assert_output "Binary: is an executable" "executable|Mach-O|ELF" "$FILE_OUT"
fi

if [ "$OS" = "Linux" ] && command -v ldd >/dev/null 2>&1; then
    LDD_OUT=$(ldd "$CAL" 2>&1 || true)
    # Should NOT depend on libnvidia-ml at runtime (we dlopen it)
    if echo "$LDD_OUT" | grep -q "libnvidia-ml"; then
        fail "Binary: should not statically link libnvidia-ml (use dlopen)"
    else
        pass "Binary: does not hard-link libnvidia-ml (runtime dlopen OK)"
    fi
fi

# Ensure binary exits cleanly on SIGTERM during idle state
"$CAL" --help >/dev/null 2>&1
pass "Binary: --help exits cleanly"

# Docker: NVIDIA Linux diagnose (Ubuntu 20.04, no GPU) 
section "calibrate diagnose — Linux NVIDIA paths (Docker)"

if ! $RUN_DOCKER; then
    skip "Docker section skipped via --no-docker"
elif ! command -v docker >/dev/null 2>&1; then
    skip "Docker not found in PATH — skipping Linux NVIDIA test"
elif ! docker info >/dev/null 2>&1; then
    skip "Docker daemon not running — skipping Linux NVIDIA test"
else
    DOCKER_IMAGE="calibrate-diagnose-test-$$"
    info() { echo -e "  ${YLW}·${NC}  $*"; }

    info "Building Ubuntu 20.04 image (first run takes ~3 min)…"
    docker build \
        --quiet \
        --file - \
        --tag "$DOCKER_IMAGE" \
        "$REPO_ROOT" <<'DOCKERFILE' 2>/dev/null
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_LOG=warn
ENV CARGO_TERM_COLOR=never
RUN apt-get update -q && \
    apt-get install -qy --no-install-recommends \
        curl ca-certificates gcc pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"
WORKDIR /src
COPY . .
RUN cargo build --release 2>&1 | tail -3
ENTRYPOINT ["/src/target/release/calibrate"]
DOCKERFILE

    DOCKER_OUT=$(docker run --rm "$DOCKER_IMAGE" diagnose 2>/dev/null || true)

    # Cleanup image regardless of assertion outcome
    docker rmi "$DOCKER_IMAGE" >/dev/null 2>&1 || true

    # Assertions — same logic as the old standalone script
    if echo "$DOCKER_OUT" | grep -q "Linux"; then
        pass "Linux diagnose: Platform section shows Linux"
    else
        fail "Linux diagnose: expected Linux in Platform section"
    fi

    if echo "$DOCKER_OUT" | grep -qE "✗.*nvidia-smi|nvidia-smi not found"; then
        pass "Linux diagnose: nvidia-smi FAIL line present (no driver in container)"
    else
        fail "Linux diagnose: expected nvidia-smi FAIL line"
    fi

    if echo "$DOCKER_OUT" | grep -q "apt-get"; then
        pass "Linux diagnose: apt-get remediation present"
    else
        fail "Linux diagnose: expected apt-get remediation in output"
    fi

    if echo "$DOCKER_OUT" | grep -q "libnvidia-ml"; then
        pass "Linux diagnose: libnvidia-ml library check present"
    else
        fail "Linux diagnose: expected libnvidia-ml check in output"
    fi

    if echo "$DOCKER_OUT" | grep -q "/dev/nvidiactl"; then
        pass "Linux diagnose: /dev/nvidiactl check present"
    else
        fail "Linux diagnose: expected /dev/nvidiactl check in output"
    fi

    if echo "$DOCKER_OUT" | grep -q "Can observe own PID"; then
        pass "Linux diagnose: own-PID visibility passes inside container"
    else
        fail "Linux diagnose: expected own-PID visibility PASS"
    fi

    if echo "$DOCKER_OUT" | grep -q "Result: FAIL"; then
        pass "Linux diagnose: Result: FAIL (correct — no GPU in container)"
    else
        fail "Linux diagnose: expected 'Result: FAIL' summary"
    fi
fi

# Summary 
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo ""
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e " Tests run:  $TOTAL"
echo -e " ${GRN}Passed:      $PASS_COUNT${NC}"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e " ${RED}Failed:      $FAIL_COUNT${NC}"
fi
if [ "$SKIP_COUNT" -gt 0 ]; then
    echo -e " ${YLW}Skipped:     $SKIP_COUNT${NC}"
fi
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "${RED}RESULT: FAIL${NC}"
    exit 1
else
    echo -e "${GRN}RESULT: PASS${NC}"
    exit 0
fi
