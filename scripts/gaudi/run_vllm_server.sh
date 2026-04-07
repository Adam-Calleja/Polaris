#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
PORT=${PORT:-8080}
DEVICES=${DEVICES:-0,1,2,3}
VOLUME=${VOLUME:-/mnt/data/ac2650/test}
NETWORK=${NETWORK:-polaris_net}

VLLM_TAG=${VLLM_TAG:-v0.7.2+Gaudi-1.21.0}
IMG=${IMG:-vault.habana.ai/gaudi-docker/1.21.0/rhel9.4/habanalabs/pytorch-installer-2.6.0:latest}
PYTHON_BIN=${PYTHON_BIN:-python3.11}

# Optional user overrides.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-}
BLOCK_SIZE=${BLOCK_SIZE:-128}
HOST=${HOST:-0.0.0.0}
TOKENS_PER_BLOCK_BUCKET=${TOKENS_PER_BLOCK_BUCKET:-2048}
RECIPE_CACHE_DELETE=${RECIPE_CACHE_DELETE:-auto}
RECIPE_CACHE_SIZE_MB=${RECIPE_CACHE_SIZE_MB:-8192}
FORCE_REINSTALL=${FORCE_REINSTALL:-false}

ENABLE_HPU_GRAPH_DEFAULT=${ENABLE_HPU_GRAPH_DEFAULT:-false}
PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=${PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES:-0}
USE_FLASH_ATTENTION_DEFAULT=${USE_FLASH_ATTENTION_DEFAULT:-false}
FLASH_ATTENTION_RECOMPUTE_DEFAULT=${FLASH_ATTENTION_RECOMPUTE_DEFAULT:-false}
VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-true}
VLLM_EXPONENTIAL_BUCKETING=${VLLM_EXPONENTIAL_BUCKETING:-true}

: "${HF_TOKEN:?Set HF_TOKEN in your environment (export HF_TOKEN=...)}"

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

sanitize_for_path() {
  printf '%s' "$1" | tr '/:@+' '____'
}

dir_has_files() {
  local dir="$1"
  [[ -d "${dir}" ]] && find "${dir}" -mindepth 1 -print -quit | grep -q .
}

recipe_cache_delete_value() {
  case "${RECIPE_CACHE_DELETE,,}" in
    auto)
      if dir_has_files "${RECIPE_CACHE_DIR_HOST}"; then
        printf 'False'
      else
        printf 'True'
      fi
      ;;
    true|1|yes)
      printf 'True'
      ;;
    false|0|no)
      printf 'False'
      ;;
    *)
      echo "ERROR: RECIPE_CACHE_DELETE must be auto, true, or false." >&2
      exit 1
      ;;
  esac
}

case "${MODEL}" in
  meta-llama/Llama-3.3-70B-Instruct)
    [[ -n "${MAX_MODEL_LEN}" ]] || MAX_MODEL_LEN=32768
    [[ -n "${TENSOR_PARALLEL}" ]] || TENSOR_PARALLEL=4
    [[ -n "${GPU_MEMORY_UTILIZATION}" ]] || GPU_MEMORY_UTILIZATION=0.85
    [[ -n "${MAX_NUM_SEQS}" ]] || MAX_NUM_SEQS=4
    ;;
  mistralai/Mistral-Large-Instruct-2407)
    [[ -n "${MAX_MODEL_LEN}" ]] || MAX_MODEL_LEN=32768
    [[ -n "${TENSOR_PARALLEL}" ]] || TENSOR_PARALLEL=4
    [[ -n "${GPU_MEMORY_UTILIZATION}" ]] || GPU_MEMORY_UTILIZATION=0.85
    [[ -n "${MAX_NUM_SEQS}" ]] || MAX_NUM_SEQS=8
    ;;
  mistralai/Mixtral-8x22B-Instruct-v0.1)
    [[ -n "${MAX_MODEL_LEN}" ]] || MAX_MODEL_LEN=32768
    [[ -n "${TENSOR_PARALLEL}" ]] || TENSOR_PARALLEL=4
    [[ -n "${GPU_MEMORY_UTILIZATION}" ]] || GPU_MEMORY_UTILIZATION=0.85
    [[ -n "${MAX_NUM_SEQS}" ]] || MAX_NUM_SEQS=8
    ;;
  Qwen/Qwen2.5-32B-Instruct)
    [[ -n "${MAX_MODEL_LEN}" ]] || MAX_MODEL_LEN=32768
    [[ -n "${TENSOR_PARALLEL}" ]] || TENSOR_PARALLEL=4
    [[ -n "${GPU_MEMORY_UTILIZATION}" ]] || GPU_MEMORY_UTILIZATION=0.85
    [[ -n "${MAX_NUM_SEQS}" ]] || MAX_NUM_SEQS=8
    ;;
  ibm-granite/granite-3.3-8b-instruct)
    [[ -n "${MAX_MODEL_LEN}" ]] || MAX_MODEL_LEN=65536
    [[ -n "${TENSOR_PARALLEL}" ]] || TENSOR_PARALLEL=1
    [[ -n "${GPU_MEMORY_UTILIZATION}" ]] || GPU_MEMORY_UTILIZATION=0.75
    [[ -n "${MAX_NUM_SEQS}" ]] || MAX_NUM_SEQS=8
    ;;
  *)
    echo "ERROR: Unsupported MODEL=${MODEL}" >&2
    echo "Supported models:" >&2
    echo "  meta-llama/Llama-3.3-70B-Instruct" >&2
    echo "  mistralai/Mixtral-8x22B-Instruct-v0.1" >&2
    echo "  mistralai/Mistral-Large-Instruct-2407" >&2
    echo "  Qwen/Qwen2.5-32B-Instruct" >&2
    echo "  ibm-granite/granite-3.3-8b-instruct" >&2
    exit 1
    ;;
esac

if [[ "${MODEL}" == mistralai/Mistral-Large-Instruct-2407 ]]; then
  echo "WARNING: ${MODEL} is gated and under the Mistral Research License."
  echo "Make sure your HF account has accepted the license."
fi

MODEL_SAFE="$(sanitize_for_path "${MODEL}")"
TAG_SAFE="$(sanitize_for_path "${VLLM_TAG}")"

NUM_BLOCKS=$(( (MAX_NUM_SEQS * TOKENS_PER_BLOCK_BUCKET + BLOCK_SIZE - 1) / BLOCK_SIZE ))
if (( NUM_BLOCKS < 128 )); then
  NUM_BLOCKS=128
fi

HF_CACHE_DIR="${VOLUME}/hf-cache"
VLLM_HOME_DIR="${VOLUME}/vllm-home"
RECIPE_CACHE_DIR_HOST="${VOLUME}/recipe-cache/${MODEL_SAFE}/tag-${TAG_SAFE}/tp-${TENSOR_PARALLEL}/len-${MAX_MODEL_LEN}/seqs-${MAX_NUM_SEQS}/block-${BLOCK_SIZE}"
RECIPE_CACHE_DIR_CONTAINER="/data/recipe-cache"
mkdir -p "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/transformers" "${RECIPE_CACHE_DIR_HOST}" "${VLLM_HOME_DIR}"

RECIPE_CACHE_DELETE_VALUE="$(recipe_cache_delete_value)"
PT_HPU_RECIPE_CACHE_CONFIG_VALUE="${RECIPE_CACHE_DIR_CONTAINER},${RECIPE_CACHE_DELETE_VALUE},${RECIPE_CACHE_SIZE_MB}"

docker_cmd rm -f vllm-gaudi 2>/dev/null || true
docker_cmd run --rm -d \
  --name vllm-gaudi \
  --runtime=habana \
  --cap-add=sys_nice \
  --ipc=host \
  --network "${NETWORK}" \
  --network-alias vllm \
  -p "${PORT}:${PORT}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HABANA_VISIBLE_DEVICES="${DEVICES}" \
  -e PT_HPU_LAZY_MODE=1 \
  -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  -e PT_HPU_RECIPE_CACHE_CONFIG="${PT_HPU_RECIPE_CACHE_CONFIG_VALUE}" \
  -e ENABLE_HPU_GRAPH="${ENABLE_HPU_GRAPH_DEFAULT}" \
  -e VLLM_PROMPT_USE_FUSEDSDPA=true \
  -e VLLM_PROMPT_SEQ_BUCKET_MAX="${MAX_MODEL_LEN}" \
  -e VLLM_DECODE_BLOCK_BUCKET_MAX="${NUM_BLOCKS}" \
  -e VLLM_EXPONENTIAL_BUCKETING="${VLLM_EXPONENTIAL_BUCKETING}" \
  -e HF_HOME="/data/hf-cache" \
  -e HF_HUB_CACHE="/data/hf-cache" \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -e PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES="${PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES}" \
  -e USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION_DEFAULT}" \
  -e FLASH_ATTENTION_RECOMPUTE="${FLASH_ATTENTION_RECOMPUTE_DEFAULT}" \
  -e VLLM_SKIP_WARMUP="${VLLM_SKIP_WARMUP}" \
  -e FORCE_REINSTALL="${FORCE_REINSTALL}" \
  -e PYTHON_BIN="${PYTHON_BIN}" \
  -e VLLM_TAG="${VLLM_TAG}" \
  -e MODEL="${MODEL}" \
  -e PORT="${PORT}" \
  -e TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e BLOCK_SIZE="${BLOCK_SIZE}" \
  -e HOST="${HOST}" \
  -v "${HF_CACHE_DIR}:/data/hf-cache" \
  -v "${RECIPE_CACHE_DIR_HOST}:${RECIPE_CACHE_DIR_CONTAINER}" \
  -v "${VLLM_HOME_DIR}:/opt/vllm-home" \
  "${IMG}" \
  bash -lc '
    set -euo pipefail

    PY_BASE="${PYTHON_BIN}"
    TAG_SAFE="$(printf "%s" "${VLLM_TAG}" | tr "/:+" "---")"
    VENV_DIR="/opt/vllm-home/venv-${TAG_SAFE}"
    STAMP_FILE="${VENV_DIR}/.bootstrap-complete"
    REPO_DIR="/opt/vllm-home/vllm-fork"
    REPO_TAG_FILE="/opt/vllm-home/vllm-fork/.checked-out-tag"

    mkdir -p /opt/vllm-home

    NEED_BOOTSTRAP=0
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
      "${PY_BASE}" -m venv --system-site-packages "${VENV_DIR}"
      NEED_BOOTSTRAP=1
    fi

    PY="${VENV_DIR}/bin/python"

    if [ -d "${REPO_DIR}/.git" ]; then
      CURRENT_TAG="$(cat "${REPO_TAG_FILE}" 2>/dev/null || true)"
      if [ "${CURRENT_TAG}" != "${VLLM_TAG}" ]; then
        git -C "${REPO_DIR}" fetch --no-tags origin tag "${VLLM_TAG}" --force
        git -C "${REPO_DIR}" checkout -f "tags/${VLLM_TAG}"
        printf "%s\n" "${VLLM_TAG}" > "${REPO_TAG_FILE}"
        NEED_BOOTSTRAP=1
      fi
    else
      rm -rf "${REPO_DIR}"
      git clone --depth 1 --branch "${VLLM_TAG}" --single-branch \
        https://github.com/HabanaAI/vllm-fork.git "${REPO_DIR}"
      printf "%s\n" "${VLLM_TAG}" > "${REPO_TAG_FILE}"
      NEED_BOOTSTRAP=1
    fi

    EXPECTED_STAMP="${VLLM_TAG}|${PY_BASE}"
    if [ ! -f "${STAMP_FILE}" ] || [ "$(cat "${STAMP_FILE}" 2>/dev/null || true)" != "${EXPECTED_STAMP}" ]; then
      NEED_BOOTSTRAP=1
    fi

    case "${FORCE_REINSTALL,,}" in
      true|1|yes)
        NEED_BOOTSTRAP=1
        ;;
    esac

    if [ "${NEED_BOOTSTRAP}" = "1" ]; then
      "${PY}" -m pip install --upgrade pip setuptools wheel
      "${PY}" -m pip install -r "${REPO_DIR}/requirements-hpu.txt"
      "${PY}" -m pip install -e "${REPO_DIR}"
      printf "%s\n" "${EXPECTED_STAMP}" > "${STAMP_FILE}"
    fi

    "${PY}" - <<'"'"'PYCODE'"'"'
import importlib.util
import sys
print("python:", sys.executable)
print("vLLM importable:", importlib.util.find_spec("vllm") is not None)
import transformers
print("transformers:", transformers.__version__)
PYCODE

    CMD=(
      "${PY}" -m vllm.entrypoints.openai.api_server
      --model "${MODEL}"
      --tensor-parallel-size "${TENSOR_PARALLEL}"
      --max-model-len "${MAX_MODEL_LEN}"
      --download-dir /data/hf-cache
      --port "${PORT}"
      --host "${HOST}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-num-seqs "${MAX_NUM_SEQS}"
      --block-size "${BLOCK_SIZE}"
    )

    echo "launch: ${CMD[*]}"
    PYTHONUNBUFFERED=1 "${CMD[@]}"
  '

echo "vLLM is starting on ${HOST}:${PORT}"
echo "Model:              ${MODEL}"
echo "Image:              ${IMG}"
echo "TP:                 ${TENSOR_PARALLEL}"
echo "MaxLen:             ${MAX_MODEL_LEN}"
echo "Blocks:             ${NUM_BLOCKS}"
echo "GMU:                ${GPU_MEMORY_UTILIZATION}"
echo "Seqs:               ${MAX_NUM_SEQS}"
echo "vLLM home:          ${VLLM_HOME_DIR}"
echo "Recipe cache mode:  ${RECIPE_CACHE_DELETE_VALUE}"
echo "Recipe cache dir:   ${RECIPE_CACHE_DIR_HOST}"
echo "Logs:               sudo docker logs -f vllm-gaudi"
echo "Health:             curl -s http://localhost:${PORT}/v1/health | jq ."
echo "Models:             curl -s http://localhost:${PORT}/v1/models | jq ."
echo "Stop:               sudo docker rm -f vllm-gaudi"
