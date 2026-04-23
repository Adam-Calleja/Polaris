#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
NETWORK=${NETWORK:-polaris_net}
BASE_STATE=${BASE_STATE:-${HOME}/.cache/polaris-gaudi}
STATE_NAME=${STATE_NAME:-llama33-70b}
STATE_DIR=${STATE_DIR:-${BASE_STATE}/${STATE_NAME}}
SHARED_HF_CACHE=${SHARED_HF_CACHE:-${BASE_STATE}/shared-hf-cache}
CONTAINER_NAME=${CONTAINER_NAME:-${STATE_NAME}}
NETWORK_ALIAS=${NETWORK_ALIAS:-${STATE_NAME}}
DEVICES=${DEVICES:-0,1,2,7}
HOST_PORT=${HOST_PORT:-8080}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-true}

: "${HF_TOKEN:?Set HF_TOKEN in your environment (export HF_TOKEN=...)}"

if [[ ! -d "${SHARED_HF_CACHE}" ]]; then
  echo "ERROR: shared HF cache directory does not exist: ${SHARED_HF_CACHE}" >&2
  echo "Create it first, then pre-populate it with a completed model download." >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"

MODEL="${MODEL}" \
DEVICES="${DEVICES}" \
CONTAINER_NAME="${CONTAINER_NAME}" \
NETWORK_ALIAS="${NETWORK_ALIAS}" \
HOST_PORT="${HOST_PORT}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
VLLM_SKIP_WARMUP="${VLLM_SKIP_WARMUP}" \
NETWORK="${NETWORK}" \
VOLUME="${STATE_DIR}" \
HF_CACHE_DIR="${SHARED_HF_CACHE}" \
VLLM_HOME_DIR="${STATE_DIR}/vllm-home" \
RECIPE_CACHE_BASE_DIR="${STATE_DIR}/recipe-cache" \
"${SCRIPT_DIR}/run_vllm_server.sh"

echo
echo "Llama 3.3 70B startup submitted."
echo "Container:          ${CONTAINER_NAME}"
echo "Alias:              ${NETWORK_ALIAS}"
echo "Devices:            ${DEVICES}"
echo "Host endpoint:      http://localhost:${HOST_PORT}/v1"
echo "Shared HF cache:    ${SHARED_HF_CACHE}"
echo "Instance state dir: ${STATE_DIR}"
