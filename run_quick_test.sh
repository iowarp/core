#!/bin/bash
set -e
export LD_LIBRARY_PATH="/workspace/build_llm_gpu/bin:/usr/local/cuda-12.8/lib64"
export CHI_SERVER_CONF=/workspace/cte_kvcache_compose.yaml
CHI=/workspace/build_llm_gpu/bin/chimaera

echo "=== Starting runtime ==="
$CHI runtime start > /tmp/chi_rt.log 2>&1 &
CTE_PID=$!
sleep 6
echo "CTE runtime PID=$CTE_PID"

# New Chimaera runs compose during runtime start (ServerInit),
# so no separate compose command needed.

echo "=== Starting llama-server ==="
/workspace/build_llm_gpu/bin/llama-server \
    --model /workspace/models/qwen2-0_5b-instruct-q4_k_m.gguf \
    --port 8089 --ctx-size 2048 --n-predict 32 --n-gpu-layers 999 \
    --cache-ram 0 --slot-prompt-similarity 0.0 --verbose \
    > /tmp/llama_e2e.log 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server"
for i in $(seq 1 90); do
    sleep 1
    if curl -sf http://localhost:8089/health > /dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo " DIED."
        tail -20 /tmp/llama_e2e.log
        exit 1
    fi
done

echo "=== Request 1 ==="
curl -sf http://localhost:8089/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is 2+2?"}],"max_tokens":32}' \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1

sleep 1
echo "=== Request 2 ==="
curl -sf http://localhost:8089/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Capital of France?"}],"max_tokens":32}' \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1

sleep 1
echo "=== Request 3 ==="
curl -sf http://localhost:8089/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is 2+2?"}],"max_tokens":32}' \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1

echo ""
echo "=== IOWarp log lines ==="
grep -i "iowarp" /tmp/llama_e2e.log 2>/dev/null || echo "none"
grep "act FlexGen" /tmp/llama_e2e.log 2>/dev/null | head -3 || true
grep "ping-pong" /tmp/llama_e2e.log 2>/dev/null | head -2 || true

echo ""
SAVED=$(grep -c "IOWarp: saved KV" /tmp/llama_e2e.log 2>/dev/null || echo 0)
RESTORED=$(grep -c "IOWarp: restored KV" /tmp/llama_e2e.log 2>/dev/null || echo 0)
INIT_OK=$(grep -c "IOWarp: KV cache manager initialized" /tmp/llama_e2e.log 2>/dev/null || echo 0)
echo "IOWarp manager: $INIT_OK"
echo "KV saves: $SAVED"
echo "KV restores: $RESTORED"

if [ "$SAVED" -gt 0 ] && [ "$RESTORED" -gt 0 ]; then
    echo "PASS"
else
    echo "FAIL"
fi

kill $SERVER_PID 2>/dev/null || true
$CHI runtime stop 2>/dev/null || true
wait $CTE_PID 2>/dev/null || true
