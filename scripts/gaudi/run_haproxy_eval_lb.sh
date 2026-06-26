#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NETWORK=${NETWORK:-polaris_net}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-eval-lb}
HOST_PORT=${HOST_PORT:-18080}
IMG=${IMG:-haproxy:lts-alpine}
REPLICA_COUNT=${REPLICA_COUNT:-4}
DEFAULT_CONFIG_PATH="${TMPDIR:-/tmp}/${CONTAINER_NAME}-haproxy.cfg"
CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

generate_config() {
  local config_path="$1"
  local replica_count="$2"
  local replica_num

  if (( replica_count < 1 )); then
    echo "ERROR: REPLICA_COUNT must be at least 1." >&2
    exit 1
  fi

  {
    printf '%s\n' \
      'global' \
      '    log stdout format raw local0' \
      '    maxconn 2048' \
      '' \
      'defaults' \
      '    log global' \
      '    mode http' \
      '    option httplog' \
      '    option redispatch' \
      '    retries 3' \
      '    timeout connect 5s' \
      '    timeout client 10m' \
      '    timeout server 10m' \
      '' \
      'frontend vllm_eval_frontend' \
      '    bind *:8080' \
      '    default_backend vllm_eval_backend' \
      '' \
      'backend vllm_eval_backend' \
      '    balance leastconn' \
      '    option httpchk GET /v1/models' \
      '    http-check expect status 200'

    for ((replica_num = 1; replica_num <= replica_count; replica_num++)); do
      printf '    server q%s qwen-eval-%s:8080 check\n' "${replica_num}" "${replica_num}"
    done
  } > "${config_path}"
}

if [[ "${CONFIG_PATH}" == "${DEFAULT_CONFIG_PATH}" ]]; then
  generate_config "${CONFIG_PATH}" "${REPLICA_COUNT}"
elif [[ ! -f "${CONFIG_PATH}" ]]; then
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
echo "Backends:           qwen-eval-1..${REPLICA_COUNT}:8080"
echo "Logs:               sudo docker logs -f ${CONTAINER_NAME}"
echo "Health:             curl -s http://localhost:${HOST_PORT}/v1/models | jq ."
echo "Stop:               sudo docker rm -f ${CONTAINER_NAME}"
