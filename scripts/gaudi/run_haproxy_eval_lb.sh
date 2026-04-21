#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NETWORK=${NETWORK:-polaris_net}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-eval-lb}
HOST_PORT=${HOST_PORT:-18080}
IMG=${IMG:-haproxy:lts-alpine}
CONFIG_PATH=${CONFIG_PATH:-${SCRIPT_DIR}/haproxy_eval.cfg}

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: HAProxy config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

docker_cmd rm -f "${CONTAINER_NAME}" 2>/dev/null || true
docker_cmd run --rm -d \
  --name "${CONTAINER_NAME}" \
  --network "${NETWORK}" \
  -p "${HOST_PORT}:8080" \
  -v "${CONFIG_PATH}:/usr/local/etc/haproxy/haproxy.cfg:ro" \
  "${IMG}"

echo "HAProxy eval load balancer is starting"
echo "Container:          ${CONTAINER_NAME}"
echo "Image:              ${IMG}"
echo "Network:            ${NETWORK}"
echo "Host port:          ${HOST_PORT}"
echo "Config:             ${CONFIG_PATH}"
echo "Endpoint:           http://localhost:${HOST_PORT}/v1"
echo "Backends:           qwen-eval-1..4:8080"
echo "Logs:               sudo docker logs -f ${CONTAINER_NAME}"
echo "Health:             curl -s http://localhost:${HOST_PORT}/v1/models | jq ."
echo "Stop:               sudo docker rm -f ${CONTAINER_NAME}"
