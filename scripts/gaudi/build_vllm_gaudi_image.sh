#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE=${BASE_IMAGE:-vault.habana.ai/gaudi-docker/1.21.0/rhel9.4/habanalabs/pytorch-installer-2.6.0:latest}
VLLM_TAG=${VLLM_TAG:-v0.7.2+Gaudi-1.21.0}
PYTHON_BIN=${PYTHON_BIN:-python3.11}
IMG_REPO=${IMG_REPO:-local/vllm-gaudi}
IMG_TAG_DEFAULT="$(printf '%s' "${VLLM_TAG}" | tr '/:+' '---')"
IMG=${IMG:-${IMG_REPO}:${IMG_TAG_DEFAULT}}

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

docker_cmd build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg VLLM_TAG="${VLLM_TAG}" \
  --build-arg PYTHON_BIN="${PYTHON_BIN}" \
  -t "${IMG}" \
  -f "${SCRIPT_DIR}/Dockerfile.vllm-gaudi" \
  "${SCRIPT_DIR}"

echo "Built image: ${IMG}"

