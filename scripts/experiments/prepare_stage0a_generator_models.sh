#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SETUP_SCRIPT="${SETUP_SCRIPT:-${REPO_ROOT}/scripts/gaudi/run_vllm_server.sh}"
MANIFEST_FILE="${MANIFEST_FILE:-protocol.final.yaml}"
MANIFEST_PATH="${REPO_ROOT}/experiments/${MANIFEST_FILE}"
MANIFEST_CONTAINER_PATH="/app/experiments/${MANIFEST_FILE}"
STAGE_NAME="${STAGE_NAME:-stage0a_generator_selection}"
NETWORK="${NETWORK:-polaris_net}"
PORT="${PORT:-8080}"
VLLM_CONTAINER="${VLLM_CONTAINER:-vllm-gaudi}"
MODEL_BOOT_TIMEOUT_SECONDS="${MODEL_BOOT_TIMEOUT_SECONDS:-86400}"
MODEL_BOOT_POLL_SECONDS="${MODEL_BOOT_POLL_SECONDS:-60}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"

MODEL_NAMES=(
  # "meta-llama/Llama-3.3-70B-Instruct"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  # "mistralai/Mistral-Large-Instruct-2407"
)

CONDITION_NAMES=(
  # "llama33_naive_combined"
  "mixtral8x22b_naive_combined"
  # "mistral_large_2407_naive_combined"
)

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

cleanup_model() {
  docker_cmd rm -f "${VLLM_CONTAINER}" >/dev/null 2>&1 || true
}

require_runtime_inputs() {
  if [[ ! -f "${SETUP_SCRIPT}" ]]; then
    echo "Missing setup script: ${SETUP_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Missing experiment manifest: ${MANIFEST_PATH}" >&2
    exit 1
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set." >&2
    exit 1
  fi
  if [[ -z "${JIRA_API_TOKEN:-}" ]]; then
    echo "JIRA_API_TOKEN is not set." >&2
    exit 1
  fi
  if [[ "${#MODEL_NAMES[@]}" -ne "${#CONDITION_NAMES[@]}" ]]; then
    echo "MODEL_NAMES and CONDITION_NAMES must have the same length." >&2
    exit 1
  fi
}

wait_for_model_server() {
  local expected_model="$1"
  local deadline=$((SECONDS + MODEL_BOOT_TIMEOUT_SECONDS))

  echo "Waiting for ${expected_model} on ${VLLM_CONTAINER}:${PORT} ..."
  while (( SECONDS < deadline )); do
    if ! docker_cmd inspect "${VLLM_CONTAINER}" >/dev/null 2>&1; then
      echo "Container ${VLLM_CONTAINER} is not running while waiting for ${expected_model}." >&2
      exit 1
    fi

    if docker_cmd exec \
      -e "EXPECTED_MODEL=${expected_model}" \
      -e "PORT=${PORT}" \
      "${VLLM_CONTAINER}" \
      python3.11 -c '
import json
import os
import sys
import urllib.request

expected_model = os.environ["EXPECTED_MODEL"]
port = os.environ["PORT"]

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

model_ids = {
    str(item.get("id", "")).strip()
    for item in payload.get("data", [])
    if isinstance(item, dict)
}
sys.exit(0 if expected_model in model_ids else 2)
' >/dev/null 2>&1; then
      echo "Model ready: ${expected_model}"
      return 0
    fi

    sleep "${MODEL_BOOT_POLL_SECONDS}"
  done

  echo "Timed out waiting for model readiness: ${expected_model}" >&2
  docker_cmd logs --tail 200 "${VLLM_CONTAINER}" >&2 || true
  exit 1
}

run_model_warmup() {
  local expected_model="$1"

  if (( WARMUP_REQUESTS <= 0 )); then
    echo "Skipping post-start warmup requests for ${expected_model}"
    return 0
  fi

  echo "Running ${WARMUP_REQUESTS} warmup completion request(s) for ${expected_model}"
  docker_cmd exec \
    -e "EXPECTED_MODEL=${expected_model}" \
    -e "PORT=${PORT}" \
    -e "WARMUP_REQUESTS=${WARMUP_REQUESTS}" \
    "${VLLM_CONTAINER}" \
    python3.11 -c '
import json
import os
import urllib.request

expected_model = os.environ["EXPECTED_MODEL"]
port = os.environ["PORT"]
warmup_requests = max(0, int(os.environ["WARMUP_REQUESTS"]))

context_block = (
    "Context item: A user asks how to submit and process a purchase order for HPC support. "
    "The assistant should classify the request, summarize the action, and draft a safe reply. "
    "Preserve ticket references and avoid unsupported operational claims. "
)
prompt = (
    "System: You are an HPC support assistant.\n\n"
    + "\n".join(context_block for _ in range(96))
    + "\n\nUser: Please explain how this ticket should be handled.\nAssistant:"
)

payload = json.dumps(
    {
        "model": expected_model,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "top_p": 1.0,
    }
).encode("utf-8")

for idx in range(warmup_requests):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
    print(f"warmup request {idx + 1}/{warmup_requests} complete")
'
}

run_prepare_for_condition() {
  local condition_name="$1"

  echo "Running ${STAGE_NAME} prepare for condition: ${condition_name}"
  docker_cmd compose \
    -f "${REPO_ROOT}/docker-compose.yaml" \
    -f "${REPO_ROOT}/docker-compose.gaudi.yaml" \
    run --no-deps --rm \
    -e JIRA_API_TOKEN="${JIRA_API_TOKEN}" \
    -v "${REPO_ROOT}/experiments:/app/experiments" \
    -v "${REPO_ROOT}/artifacts:/app/artifacts" \
    eval python /app/scripts/experiments/run_stage.py \
    --manifest "${MANIFEST_CONTAINER_PATH}" \
    --stage "${STAGE_NAME}" \
    --condition "${condition_name}" \
    --phase prepare
}

main() {
  require_runtime_inputs
  trap cleanup_model EXIT

  for idx in "${!MODEL_NAMES[@]}"; do
    local model_name="${MODEL_NAMES[$idx]}"
    local condition_name="${CONDITION_NAMES[$idx]}"

    echo "============================================================"
    echo "Starting model: ${model_name}"
    echo "Condition: ${condition_name}"
    echo "============================================================"

    cleanup_model
    MODEL="${model_name}" PORT="${PORT}" NETWORK="${NETWORK}" bash "${SETUP_SCRIPT}"
    wait_for_model_server "${model_name}"
    run_model_warmup "${model_name}"
    run_prepare_for_condition "${condition_name}"
    cleanup_model
  done

  echo "Stage 0A prepare complete for all generator candidates."
}

main "$@"
