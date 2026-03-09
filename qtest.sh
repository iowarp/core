#\!/bin/bash
BUILD_DIR=/workspace/build_llm_gpu
MODEL=/workspace/models/qwen2-0_5b-instruct-q4_k_m.gguf
LOG=/workspace/qtest.log
echo "=== Starting ===" > $LOG

GGML_CUDA_NO_GRAPHS=1 LD_LIBRARY_PATH="$BUILD_DIR/bin:/usr/local/cuda-12.8/lib64"   $BUILD_DIR/bin/llama-server --model $MODEL --port 8093 --ctx-size 512   --n-predict 8 --n-gpu-layers 999 --cache-ram 0 --log-prefix >> $LOG 2>&1 &
SERVER_PID=$\!

for i in $(seq 1 20); do
  sleep 1
  if curl -s http://127.0.0.1:8093/health 2>/dev/null | grep -q "ok"; then
    echo "SERVER_STARTED after ${i}s" >> $LOG
    break
  fi
  if \! kill -0 $SERVER_PID 2>/dev/null; then
    echo "SERVER_DIED after ${i}s" >> $LOG
    break
  fi
done

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "DONE" >> $LOG
