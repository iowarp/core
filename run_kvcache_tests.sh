#!/bin/bash
set -e

BIN_DIR=/workspace/build/bin
COMPOSE_CFG=/workspace/cte_kvcache_compose.yaml
export LD_LIBRARY_PATH=${BIN_DIR}:/usr/local/lib:/home/iowarp/miniconda3/lib:${LD_LIBRARY_PATH}
export CHI_SERVER_CONF=${COMPOSE_CFG}

stop_runtime() {
    if [ -n "${RUNTIME_PID}" ] && kill -0 ${RUNTIME_PID} 2>/dev/null; then
        ${BIN_DIR}/chimaera_stop_runtime 2>/dev/null || true
        sleep 2
        kill -9 ${RUNTIME_PID} 2>/dev/null || true
        wait ${RUNTIME_PID} 2>/dev/null || true
    fi
}
trap stop_runtime EXIT

# Clean slate
rm -f /dev/shm/chimaera_* 2>/dev/null || true
rm -rf /tmp/cte_kvcache_test && mkdir -p /tmp/cte_kvcache_test

echo '=== Starting Chimaera runtime ==='
${BIN_DIR}/chimaera_start_runtime &
RUNTIME_PID=$!
sleep 4

kill -0 ${RUNTIME_PID} 2>/dev/null || { echo 'FATAL: runtime died immediately'; exit 1; }
echo "Runtime alive (PID=${RUNTIME_PID})"

echo ''
echo '=== Composing CTE (wrp_cte_core pool) ==='
${BIN_DIR}/chimaera_compose ${COMPOSE_CFG}
echo 'Compose complete'
sleep 1

echo ''
echo '=== Running test_iowarp_kvcache ==='
${BIN_DIR}/test_iowarp_kvcache
TEST_EXIT=$?

echo ''
echo '=== Stopping runtime ==='
stop_runtime
trap - EXIT

exit ${TEST_EXIT}
