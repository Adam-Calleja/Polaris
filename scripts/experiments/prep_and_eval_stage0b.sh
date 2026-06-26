#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILES=(-f docker-compose.yaml -f docker-compose.gaudi.yaml)

HF_TOKEN="${HF_TOKEN:?export HF_TOKEN first}"

NETWORK="polaris_net"

# Generator model for row prep.
GENERATOR_MODEL="mistralai/Mistral-Large-Instruct-2407"
GENERATOR_DEVICES="${GENERATOR_DEVICES:-0,1,2,7}"
GENERATOR_CONTAINER_NAME="mistral"
GENERATOR_NETWORK_ALIAS="mistral"
GENERATOR_HOST_PORT="${GENERATOR_HOST_PORT:-8080}"

# Evaluator pool.
DEVICE_IDS=(0 1 2 5 6 7)
LB_NAME="vllm-eval-lb"
LB_HOST_PORT=18080

BASE_STATE="${HOME}/.cache/polaris-gaudi"
SHARED_HF_CACHE="${BASE_STATE}/shared-hf-cache"

MANIFEST_IN_CONTAINER="/app/experiments/protocol.final.yaml"

mkdir -p "${SHARED_HF_CACHE}"

wait_for_models() {
  local url="$1"
  local label="$2"

  until curl -fsS "${url}" >/dev/null; do
    echo "Waiting for ${label} at ${url}"
    sleep 10
  done
}

# Clear any old generator / evaluator / LB containers that may hold HPUs or ports.
for name in \
  "${GENERATOR_CONTAINER_NAME}" \
  llama33-70b \
  vllm \
  vllm-gaudi \
  "${LB_NAME}" \
  qwen-eval-1 qwen-eval-2 qwen-eval-3 qwen-eval-4 qwen-eval-5 qwen-eval-6; do
  sudo docker rm -f "${name}" >/dev/null 2>&1 || true
done

# Start Mistral generator for row preparation.
MODEL="${GENERATOR_MODEL}" \
DEVICES="${GENERATOR_DEVICES}" \
CONTAINER_NAME="${GENERATOR_CONTAINER_NAME}" \
NETWORK_ALIAS="${GENERATOR_NETWORK_ALIAS}" \
HOST_PORT="${GENERATOR_HOST_PORT}" \
NETWORK="${NETWORK}" \
VOLUME="${BASE_STATE}/mistral-prep" \
HF_CACHE_DIR="${SHARED_HF_CACHE}" \
VLLM_HOME_DIR="${BASE_STATE}/mistral-prep/vllm-home" \
RECIPE_CACHE_BASE_DIR="${BASE_STATE}/mistral-prep/recipe-cache" \
VLLM_SKIP_WARMUP=true \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=16 \
GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/gaudi/run_vllm_server.sh

wait_for_models "http://localhost:${GENERATOR_HOST_PORT}/v1/models" "Mistral generator"

# Run row preparation only after Mistral is ready.
sudo docker compose "${COMPOSE_FILES[@]}" run --no-deps --rm \
  -v "${PWD}/experiments:/app/experiments" \
  -v "${PWD}/artifacts:/app/artifacts" \
  eval sh -lc "
    python /app/scripts/experiments/run_stage.py \
      --manifest ${MANIFEST_IN_CONTAINER} \
      --stage stage0b1_docs_chunking \
      --phase prepare &&
    python /app/scripts/experiments/run_stage.py \
      --manifest ${MANIFEST_IN_CONTAINER} \
      --stage stage0b2_ticket_chunking \
      --phase prepare
  "

# Free the generator HPUs before evaluation.
sudo docker rm -f "${GENERATOR_CONTAINER_NAME}" >/dev/null 2>&1 || true

# No embed service remains after this point.
sudo docker compose "${COMPOSE_FILES[@]}" rm -sf \
  embed embed1 embed2 embed3 embed4 embed5 embed6 >/dev/null 2>&1 || true

# Start the Qwen evaluator pool and wait for it.
DEVICE_IDS_STR="${DEVICE_IDS[*]}"

MODEL="Qwen/Qwen2.5-32B-Instruct" \
NETWORK="${NETWORK}" \
BASE_STATE="${BASE_STATE}" \
SHARED_HF_CACHE="${SHARED_HF_CACHE}" \
DEVICE_IDS="${DEVICE_IDS_STR}" \
CONTAINER_PREFIX="qwen-eval" \
PORT_BASE=18080 \
LB_CONTAINER_NAME="${LB_NAME}" \
LB_HOST_PORT="${LB_HOST_PORT}" \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=4 \
GPU_MEMORY_UTILIZATION=0.90 \
VLLM_SKIP_WARMUP=true \
START_LB=true \
WAIT_UNTIL_READY=true \
READY_POLL_SECONDS=10 \
READY_TIMEOUT_SECONDS=7200 \
bash scripts/gaudi/run_qwen32_eval_pool.sh

# Run evaluation only after the Qwen pool is ready.

MANIFEST_IN_CONTAINER="/app/experiments/protocol.final.yaml"
sudo docker compose "${COMPOSE_FILES[@]}" run --no-deps --rm \
  -v "${PWD}/experiments:/app/experiments" \
  -v "${PWD}/artifacts:/app/artifacts" \
  eval sh -lc "
    python /app/scripts/experiments/run_stage.py \
      --manifest ${MANIFEST_IN_CONTAINER} \
      --stage stage0b1_docs_chunking \
      --phase evaluate &&
    python /app/scripts/experiments/run_stage.py \
      --manifest ${MANIFEST_IN_CONTAINER} \
      --stage stage0b2_ticket_chunking \
      --phase evaluate
  "
