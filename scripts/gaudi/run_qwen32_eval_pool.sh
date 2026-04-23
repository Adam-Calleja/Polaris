#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
NETWORK=${NETWORK:-polaris_net}
BASE_STATE=${BASE_STATE:-${HOME}/.cache/polaris-gaudi}
SHARED_HF_CACHE=${SHARED_HF_CACHE:-${BASE_STATE}/shared-hf-cache}
DEVICES=${DEVICES:-0,1,2,3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-true}
LB_HOST_PORT=${LB_HOST_PORT:-18080}
FIRST_HOST_PORT=${FIRST_HOST_PORT:-18081}
START_LB=${START_LB:-true}

: "${HF_TOKEN:?Set HF_TOKEN in your environment (export HF_TOKEN=...)}"

if [[ ! -d "${SHARED_HF_CACHE}" ]]; then
  echo "ERROR: shared HF cache directory does not exist: ${SHARED_HF_CACHE}" >&2
  echo "Create it first, then pre-populate it with a completed model download." >&2
  exit 1
fi

mkdir -p "${BASE_STATE}"

declare -a DEVICE_IDS=()

parse_devices() {
  local devices_csv="$1"
  local raw_devices=()
  local device_id
  local old_ifs="${IFS}"

  if [[ -z "${devices_csv//[[:space:]]/}" ]]; then
    echo "ERROR: DEVICES must contain at least one device id." >&2
    exit 1
  fi

  IFS=',' read -r -a raw_devices <<< "${devices_csv}"
  IFS="${old_ifs}"

  DEVICE_IDS=()
  for device_id in "${raw_devices[@]}"; do
    device_id="${device_id//[[:space:]]/}"

    if [[ -z "${device_id}" ]]; then
      echo "ERROR: DEVICES contains an empty entry: ${devices_csv}" >&2
      exit 1
    fi

    if [[ "${device_id,,}" == "all" ]]; then
      echo "ERROR: DEVICES=all is not supported here; provide an explicit comma-separated device list." >&2
      exit 1
    fi

    DEVICE_IDS+=("${device_id}")
  done
}

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

parse_devices "${DEVICES}"

for idx in "${!DEVICE_IDS[@]}"; do
  start_replica \
    "$((idx + 1))" \
    "${DEVICE_IDS[idx]}" \
    "$((FIRST_HOST_PORT + idx))"
done

if [[ "${START_LB,,}" == "true" || "${START_LB}" == "1" || "${START_LB,,}" == "yes" ]]; then
  echo "Starting HAProxy load balancer on localhost:${LB_HOST_PORT}"
  NETWORK="${NETWORK}" \
  HOST_PORT="${LB_HOST_PORT}" \
  REPLICA_COUNT="${#DEVICE_IDS[@]}" \
  "${SCRIPT_DIR}/run_haproxy_eval_lb.sh"
fi

echo
echo "Pool startup complete."
echo "Shared HF cache:    ${SHARED_HF_CACHE}"
echo "Replica state root: ${BASE_STATE}"
echo "Devices:            ${DEVICES}"
echo "Replicas:"
for idx in "${!DEVICE_IDS[@]}"; do
  echo "  qwen-eval-$((idx + 1)) (HPU ${DEVICE_IDS[idx]}) -> http://localhost:$((FIRST_HOST_PORT + idx))/v1"
done
if [[ "${START_LB,,}" == "true" || "${START_LB}" == "1" || "${START_LB,,}" == "yes" ]]; then
  echo "Load balancer:      http://localhost:${LB_HOST_PORT}/v1"
fi
