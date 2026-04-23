#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-Qwen/Qwen2.5-72B-Instruct}
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
DEVICES=${DEVICES:-0,1,2,3}
VOLUME=${VOLUME:-${HOME}/.cache/polaris-gaudi}
NETWORK=${NETWORK:-polaris_net}

IMG=${IMG:-ghcr.io/huggingface/text-generation-inference:3.3.5-gaudi}
CONTAINER_NAME=${CONTAINER_NAME:-tgi-gaudi}

DTYPE=${DTYPE:-bfloat16}
NUM_SHARD=${NUM_SHARD:-4}
SHARDED=${SHARDED:-true}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-16}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-30000}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32000}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}
MAX_BATCH_PREFILL_TOKENS=${MAX_BATCH_PREFILL_TOKENS:-98304}
MAX_BATCH_TOTAL_TOKENS=${MAX_BATCH_TOTAL_TOKENS:-98304}
VALIDATION_WORKERS=${VALIDATION_WORKERS:-2}
WAITING_SERVED_RATIO=${WAITING_SERVED_RATIO:-1.0}
MAX_WAITING_TOKENS=${MAX_WAITING_TOKENS:-7}
REVISION=${REVISION:-}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
LIMIT_HPU_GRAPH=${LIMIT_HPU_GRAPH:-true}
JSON_OUTPUT=${JSON_OUTPUT:-false}
API_KEY=${API_KEY:-}

: "${HF_TOKEN:?Set HF_TOKEN in your environment (export HF_TOKEN=...)}"

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

count_visible_devices() {
  local devices="$1"
  if [[ -z "${devices}" ]]; then
    echo 0
    return
  fi
  if [[ "${devices}" == "all" ]]; then
    echo 0
    return
  fi
  local old_ifs="${IFS}"
  IFS=',' read -r -a parts <<< "${devices}"
  IFS="${old_ifs}"
  echo "${#parts[@]}"
}

bool_to_tgi_flag() {
  case "${1,,}" in
    true|1|yes)
      printf 'true'
      ;;
    false|0|no)
      printf 'false'
      ;;
    *)
      echo "ERROR: expected true/false value, got '${1}'" >&2
      exit 1
      ;;
  esac
}

DEVICE_COUNT="$(count_visible_devices "${DEVICES}")"

if [[ -z "${NUM_SHARD}" ]]; then
  if (( DEVICE_COUNT == 0 )); then
    echo "ERROR: NUM_SHARD must be set when DEVICES='${DEVICES}'." >&2
    exit 1
  fi
  NUM_SHARD="${DEVICE_COUNT}"
fi

case "${SHARDED,,}" in
  auto)
    if (( NUM_SHARD > 1 )); then
      SHARDED=true
    else
      SHARDED=false
    fi
    ;;
  true|1|yes)
    SHARDED=true
    ;;
  false|0|no)
    SHARDED=false
    ;;
  *)
    echo "ERROR: SHARDED must be auto, true, or false." >&2
    exit 1
    ;;
esac

if (( MAX_TOTAL_TOKENS <= MAX_INPUT_TOKENS )); then
  echo "ERROR: MAX_TOTAL_TOKENS must be greater than MAX_INPUT_TOKENS." >&2
  exit 1
fi

if [[ -z "${MAX_BATCH_PREFILL_TOKENS}" ]]; then
  MAX_BATCH_PREFILL_TOKENS=$(( MAX_BATCH_SIZE * MAX_INPUT_TOKENS ))
fi

CACHE_DIR="${VOLUME}/tgi-cache"
if ! mkdir -p "${CACHE_DIR}"; then
  echo "ERROR: Cannot create cache directory '${CACHE_DIR}'." >&2
  echo "Set VOLUME to a writable location, for example:" >&2
  echo "  VOLUME=\$HOME/.cache/polaris-gaudi ./scripts/gaudi/run_tgi_gaudi.sh" >&2
  exit 1
fi

if ! docker_cmd network inspect "${NETWORK}" >/dev/null 2>&1; then
  docker_cmd network create "${NETWORK}" >/dev/null
fi

docker_cmd rm -f "${CONTAINER_NAME}" 2>/dev/null || true

CMD=(
  docker_cmd run --rm -d
  --name "${CONTAINER_NAME}"
  --runtime=habana
  --cap-add=sys_nice
  --ipc=host
  --network "${NETWORK}"
  --network-alias vllm
  --network-alias tgi
  -p "${PORT}:${PORT}"
  -e "HF_TOKEN=${HF_TOKEN}"
  -e "HABANA_VISIBLE_DEVICES=${DEVICES}"
  -e "LIMIT_HPU_GRAPH=${LIMIT_HPU_GRAPH}"
  -e "OMPI_MCA_btl_vader_single_copy_mechanism=none"
  -v "${CACHE_DIR}:/data"
  "${IMG}"
  --model-id "${MODEL}"
  --hostname "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --max-batch-prefill-tokens "${MAX_BATCH_PREFILL_TOKENS}"
  --validation-workers "${VALIDATION_WORKERS}"
  --max-waiting-tokens "${MAX_WAITING_TOKENS}"
)

if [[ "$(bool_to_tgi_flag "${SHARDED}")" == "true" ]]; then
  CMD+=(--sharded true --num-shard "${NUM_SHARD}")
else
  CMD+=(--sharded false)
fi

if [[ -n "${MAX_BATCH_TOTAL_TOKENS}" ]]; then
  CMD+=(--max-batch-total-tokens "${MAX_BATCH_TOTAL_TOKENS}")
fi

if [[ -n "${WAITING_SERVED_RATIO}" ]]; then
  CMD+=(--waiting-served-ratio "${WAITING_SERVED_RATIO}")
fi

if [[ -n "${REVISION}" ]]; then
  CMD+=(--revision "${REVISION}")
fi

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api-key "${API_KEY}")
fi

if [[ "$(bool_to_tgi_flag "${TRUST_REMOTE_CODE}")" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ "$(bool_to_tgi_flag "${JSON_OUTPUT}")" == "true" ]]; then
  CMD+=(--json-output)
fi

"${CMD[@]}"

cat <<EOF
TGI-Gaudi is starting on ${HOST}:${PORT}
Model:        ${MODEL}
Image:        ${IMG}
Devices:      ${DEVICES}
Sharded:      ${SHARDED}
Num shard:    ${NUM_SHARD}
Max input:    ${MAX_INPUT_TOKENS}
Max total:    ${MAX_TOTAL_TOKENS}
Batch size:   ${MAX_BATCH_SIZE}
Prefill toks: ${MAX_BATCH_PREFILL_TOKENS}
Cache dir:    ${CACHE_DIR}
Script:       ${SCRIPT_DIR}/run_tgi_gaudi.sh

Logs:
  $(docker version >/dev/null 2>&1 && echo "docker" || echo "sudo docker") logs -f ${CONTAINER_NAME}

Health:
  curl -s http://localhost:${PORT}/health
  curl -s http://localhost:${PORT}/info | jq .

OpenAI-compatible chat:
  curl -s http://localhost:${PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"${MODEL}","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}'

Stop:
  $(docker version >/dev/null 2>&1 && echo "docker" || echo "sudo docker") rm -f ${CONTAINER_NAME}
EOF
