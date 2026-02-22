#!/usr/bin/env bash
# run_e2e_kvcache_test.sh
# End-to-end test: llama-server + IOWarp KV cache
#
# What this proves:
#   1st request  — prefill runs, KV saved to CTE ("IOWarp: saved KV ...")
#   2nd request  — KV restored from CTE, prefill skipped ("IOWarp: restored KV ...")
#
# Usage (from inside the container):
#   sed 's/\r//' /workspace/run_e2e_kvcache_test.sh > /tmp/run_e2e.sh
#   /bin/bash /tmp/run_e2e.sh /workspace/build_llm_cpu /path/to/model.gguf
#
# Or on Windows host:
#   docker exec iowarp-llm-test bash -c "..."

set -e

BUILD_DIR="${1:-/workspace/build_llm_cpu}"
MODEL="${2:-}"
# Chimaera runtime lives in the original build dir (built with CUDA preset)
CHI_BUILD_DIR="${3:-/workspace/build}"
COMPOSE_CFG=/workspace/cte_kvcache_compose.yaml
PORT=8088
LOG=/tmp/llama_server_e2e.log

# ─── helpers ─────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

# Put Chimaera + kvcache libs on PATH/LD_LIBRARY_PATH
export PATH="$CHI_BUILD_DIR/bin:$BUILD_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CHI_BUILD_DIR/bin:$BUILD_DIR/bin:$LD_LIBRARY_PATH"

# ─── check prereqs ───────────────────────────────────────────────────────────
[ -f "$BUILD_DIR/bin/llama-server" ]           || die "llama-server not found in $BUILD_DIR/bin"
[ -f "$BUILD_DIR/bin/libwrp_llm_kvcache.so" ]  || die "libwrp_llm_kvcache.so not found in $BUILD_DIR/bin"
[ -f "$CHI_BUILD_DIR/bin/chimaera_start_runtime" ] || die "chimaera_start_runtime not found in $CHI_BUILD_DIR/bin"
[ -f "$COMPOSE_CFG" ]                           || die "CTE compose YAML not found: $COMPOSE_CFG"

if [ -z "$MODEL" ]; then
    # try to auto-detect a gguf in /workspace/models
    MODEL=$(find /workspace/models -name '*.gguf' -not -name '*vocab*' -size +10M 2>/dev/null | head -1)
    [ -n "$MODEL" ] || die "No model specified and none found in /workspace/models/
Please download a small GGUF model, e.g.:
  # On Windows, download to: C:\\Users\\rajni\\Documents\\GPU_OS\\core\\models\\
  # Recommended: Qwen2-0.5B-Instruct-Q4_K_M.gguf (~350 MB)
  #   https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF
  # Then re-run this script."
    echo "Auto-detected model: $MODEL"
fi
[ -f "$MODEL" ] || die "Model file not found: $MODEL"

# ─── CTE runtime ─────────────────────────────────────────────────────────────
echo "=== Starting Chimaera CTE runtime ==="
export CHI_SERVER_CONF=$COMPOSE_CFG
"$CHI_BUILD_DIR/bin/chimaera_start_runtime" > /tmp/chimaera_e2e.log 2>&1 &
CTE_PID=$!
sleep 4
"$CHI_BUILD_DIR/bin/chimaera_compose" $COMPOSE_CFG >> /tmp/chimaera_e2e.log 2>&1
echo "CTE runtime PID=$CTE_PID"

cleanup() {
    echo "=== Stopping llama-server and CTE ==="
    kill $SERVER_PID 2>/dev/null || true
    "$CHI_BUILD_DIR/bin/chimaera_stop_runtime" 2>/dev/null || true
    wait $CTE_PID 2>/dev/null || true
}
trap cleanup EXIT

# ─── llama-server ────────────────────────────────────────────────────────────
echo "=== Starting llama-server (CPU, port $PORT) ==="
LD_LIBRARY_PATH="$BUILD_DIR/bin:$LD_LIBRARY_PATH" \
    "$BUILD_DIR/bin/llama-server" \
    --model "$MODEL" \
    --port  $PORT \
    --ctx-size 2048 \
    --n-predict 32 \
    --cache-ram 0 \
    --slot-prompt-similarity 0.0 \
    --log-prefix \
    --verbose \
    > $LOG 2>&1 &
SERVER_PID=$!

# wait for server to be ready
echo -n "Waiting for server"
for i in $(seq 1 60); do
    sleep 1
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo " DIED. Last log:"
        tail -20 $LOG
        exit 1
    fi
done

SHARED_PROMPT="You are a helpful assistant. Answer concisely."

MSG_A="What is 2+2? Answer briefly."
MSG_B="What is the capital of France? Answer briefly."

# ─── Request 1: cold — prefill A, save A's KV to CTE ────────────────────────
echo ""
echo "=== Request 1 (cold A — expect: IOWarp: saved KV) ==="
curl -sf http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"system\",\"content\":\"$SHARED_PROMPT\"},{\"role\":\"user\",\"content\":\"$MSG_A\"}],\"max_tokens\":32}" \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1 | sed 's/^/Response: /'

sleep 1

# ─── Request 2: evict — different prompt forces slot clear, saves B's KV ─────
echo ""
echo "=== Request 2 (evict — different prompt, evicts A from GPU slot) ==="
curl -sf http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"system\",\"content\":\"$SHARED_PROMPT\"},{\"role\":\"user\",\"content\":\"$MSG_B\"}],\"max_tokens\":32}" \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1 | sed 's/^/Response: /'

sleep 1

# ─── Request 3: restore — same as R1, GPU has B not A → IOWarp restore fires ─
echo ""
echo "=== Request 3 (restore A — expect: IOWarp: restored KV) ==="
curl -sf http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"system\",\"content\":\"$SHARED_PROMPT\"},{\"role\":\"user\",\"content\":\"$MSG_A\"}],\"max_tokens\":32}" \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1 | sed 's/^/Response: /'

sleep 1

# ─── results ─────────────────────────────────────────────────────────────────
echo ""
echo "=== IOWarp KV cache log lines ==="
grep -i "iowarp" $LOG || echo "(none found — check $LOG for details)"

echo ""
echo "=== Summary ==="
SAVED=$(grep -c "IOWarp: saved KV" $LOG 2>/dev/null || true)
RESTORED=$(grep -c "IOWarp: restored KV" $LOG 2>/dev/null || true)
INIT_OK=$(grep -c "IOWarp: KV cache manager initialized" $LOG 2>/dev/null || true)

if [ "$INIT_OK" -gt 0 ]; then
    echo "  IOWarp manager: INITIALIZED"
else
    echo "  IOWarp manager: NOT INITIALIZED (check CTE)"
fi
echo "  KV saves    : $SAVED"
echo "  KV restores : $RESTORED"

if [ "$SAVED" -gt 0 ] && [ "$RESTORED" -gt 0 ]; then
    echo ""
    echo "PASS: IOWarp prefix KV cache is working end-to-end."
elif [ "$SAVED" -gt 0 ]; then
    echo ""
    echo "PARTIAL: KV saved but not restored. Shared prompt may be too short to trigger restore."
    echo "  Try a longer system prompt or check iowarp_restore_kv logic."
else
    echo ""
    echo "FAIL: No IOWarp KV activity detected."
    echo "  Check $LOG for details."
fi
