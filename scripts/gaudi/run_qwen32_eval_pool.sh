#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
NETWORK=${NETWORK:-polaris_net}
BASE_STATE=${BASE_STATE:-${HOME}/.cache/polaris-gaudi}
SHARED_HF_CACHE=${SHARED_HF_CACHE:-${BASE_STATE}/shared-hf-cache}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-true}
LB_HOST_PORT=${LB_HOST_PORT:-18080}
START_LB=${START_LB:-true}

: "${HF_TOKEN:?Set HF_TOKEN in your environment (export HF_TOKEN=...)}"

if [[ ! -d "${SHARED_HF_CACHE}" ]]; then
  echo "ERROR: shared HF cache directory does not exist: ${SHARED_HF_CACHE}" >&2
  echo "Create it first, then pre-populate it with a completed model download." >&2
  exit 1
fi

mkdir -p "${BASE_STATE}"

start_replica() {
  local replica_num="$1"
  local device_id="$2"
  local host_port="$3"
  local state_dir="${BASE_STATE}/qwen-eval-${replica_num}"

  mkdir -p "${state_dir}"

  echo "Starting qwen-eval-${replica_num} on HPU ${device_id} -> localhost:${host_port}"
  MODEL="${MODEL}" \
  DEVICES="${device_id}" \
  CONTAINER_NAME="qwen-eval-${replica_num}" \
  NETWORK_ALIAS="qwen-eval-${replica_num}" \
  HOST_PORT="${host_port}" \
  VLLM_SKIP_WARMUP="${VLLM_SKIP_WARMUP}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  NETWORK="${NETWORK}" \
  VOLUME="${state_dir}" \
  HF_CACHE_DIR="${SHARED_HF_CACHE}" \
  VLLM_HOME_DIR="${state_dir}/vllm-home" \
  RECIPE_CACHE_BASE_DIR="${state_dir}/recipe-cache" \
  "${SCRIPT_DIR}/run_vllm_server.sh"
}

start_replica 1 0 18081
start_replica 2 1 18082
start_replica 3 2 18083
start_replica 4 3 18084

if [[ "${START_LB,,}" == "true" || "${START_LB}" == "1" || "${START_LB,,}" == "yes" ]]; then
  echo "Starting HAProxy load balancer on localhost:${LB_HOST_PORT}"
  NETWORK="${NETWORK}" HOST_PORT="${LB_HOST_PORT}" "${SCRIPT_DIR}/run_haproxy_eval_lb.sh"
fi

echo
echo "Pool startup complete."
echo "Shared HF cache:    ${SHARED_HF_CACHE}"
echo "Replica state root: ${BASE_STATE}"
echo "Replicas:"
echo "  qwen-eval-1 -> http://localhost:18081/v1"
echo "  qwen-eval-2 -> http://localhost:18082/v1"
echo "  qwen-eval-3 -> http://localhost:18083/v1"
echo "  qwen-eval-4 -> http://localhost:18084/v1"
if [[ "${START_LB,,}" == "true" || "${START_LB}" == "1" || "${START_LB,,}" == "yes" ]]; then
  echo "Load balancer:      http://localhost:${LB_HOST_PORT}/v1"
fi
