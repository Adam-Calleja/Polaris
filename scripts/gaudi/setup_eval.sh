#!/usr/bin/env bash
set -euo pipefail

# Required:
: "${HF_TOKEN:?Set HF_TOKEN first}"

# Core config
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
NETWORK="${NETWORK:-polaris_net}"
BASE_STATE="${BASE_STATE:-$HOME/.cache/polaris-gaudi}"
SHARED_HF_CACHE="${SHARED_HF_CACHE:-$BASE_STATE/shared-hf-cache}"

# Keep these as env vars, per your request
EVAL_DEVICE_IDS="${EVAL_DEVICE_IDS:-0 1 2 3 4 5 6 7}"   # space- or comma-separated
EVAL_REPLICA_COUNT="${EVAL_REPLICA_COUNT:-8}"

# Runtime tuning
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_SKIP_WARMUP="${VLLM_SKIP_WARMUP:-true}"

# Naming / ports
CONTAINER_PREFIX="${CONTAINER_PREFIX:-qwen-eval}"
PORT_BASE="${PORT_BASE:-18081}"
LB_CONTAINER_NAME="${LB_CONTAINER_NAME:-vllm-eval-lb}"
LB_HOST_PORT="${LB_HOST_PORT:-18080}"
HAPROXY_IMAGE="${HAPROXY_IMAGE:-haproxy:lts-alpine}"

SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)/scripts/gaudi}"
RUN_VLLM_SCRIPT="${RUN_VLLM_SCRIPT:-$SCRIPT_DIR/run_vllm_server.sh}"

docker_cmd() {
  if docker version >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

if [[ ! -x "$RUN_VLLM_SCRIPT" ]]; then
  echo "ERROR: cannot execute $RUN_VLLM_SCRIPT" >&2
  exit 1
fi

mkdir -p "$BASE_STATE"
mkdir -p "$SHARED_HF_CACHE"

# Parse device ids from either "0 1 2 3" or "0,1,2,3"
DEVICE_IDS_CLEAN="${EVAL_DEVICE_IDS//,/ }"
read -r -a DEVICE_IDS <<< "$DEVICE_IDS_CLEAN"

if (( ${#DEVICE_IDS[@]} < EVAL_REPLICA_COUNT )); then
  echo "ERROR: EVAL_REPLICA_COUNT=$EVAL_REPLICA_COUNT but only ${#DEVICE_IDS[@]} device ids were provided: $EVAL_DEVICE_IDS" >&2
  exit 1
fi

# Start replicas
for ((i=1; i<=EVAL_REPLICA_COUNT; i++)); do
  device_id="${DEVICE_IDS[$((i-1))]}"
  host_port="$((PORT_BASE + i - 1))"
  state_dir="$BASE_STATE/${CONTAINER_PREFIX}-${i}"

  mkdir -p "$state_dir"

  echo "Starting ${CONTAINER_PREFIX}-${i} on HPU ${device_id} -> localhost:${host_port}"

  MODEL="$MODEL" \
  DEVICES="$device_id" \
  CONTAINER_NAME="${CONTAINER_PREFIX}-${i}" \
  NETWORK_ALIAS="${CONTAINER_PREFIX}-${i}" \
  HOST_PORT="$host_port" \
  VLLM_SKIP_WARMUP="$VLLM_SKIP_WARMUP" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  NETWORK="$NETWORK" \
  VOLUME="$state_dir" \
  HF_CACHE_DIR="$SHARED_HF_CACHE" \
  VLLM_HOME_DIR="$state_dir/vllm-home" \
  RECIPE_CACHE_BASE_DIR="$state_dir/recipe-cache" \
  "$RUN_VLLM_SCRIPT"
done

# Generate HAProxy config for exactly these replicas
HAPROXY_CFG="$BASE_STATE/haproxy_eval.generated.cfg"
cat > "$HAPROXY_CFG" <<EOF
global
    log stdout format raw local0
    maxconn 2048

defaults
    log global
    mode http
    option httplog
    option redispatch
    retries 3
    timeout connect 5s
    timeout client 10m
    timeout server 10m

frontend vllm_eval_frontend
    bind *:8080
    default_backend vllm_eval_backend

backend vllm_eval_backend
    balance leastconn
    option httpchk GET /v1/models
    http-check expect status 200
EOF

for ((i=1; i<=EVAL_REPLICA_COUNT; i++)); do
  echo "    server r${i} ${CONTAINER_PREFIX}-${i}:8080 check" >> "$HAPROXY_CFG"
done

docker_cmd rm -f "$LB_CONTAINER_NAME" 2>/dev/null || true
docker_cmd run --rm -d \
  --name "$LB_CONTAINER_NAME" \
  --network "$NETWORK" \
  -p "${LB_HOST_PORT}:8080" \
  -v "$HAPROXY_CFG:/usr/local/etc/haproxy/haproxy.cfg:ro" \
  "$HAPROXY_IMAGE"

echo
echo "Evaluation network is up."
echo "Shared HF cache: $SHARED_HF_CACHE"
echo "Replica count:   $EVAL_REPLICA_COUNT"
echo "Device ids:      $EVAL_DEVICE_IDS"
for ((i=1; i<=EVAL_REPLICA_COUNT; i++)); do
  echo "  ${CONTAINER_PREFIX}-${i} -> http://localhost:$((PORT_BASE + i - 1))/v1"
done
echo "Load balancer -> http://localhost:${LB_HOST_PORT}/v1"
echo
echo "Verify:"
echo "  curl -s http://localhost:${LB_HOST_PORT}/v1/models"
echo
echo "Stop everything:"
echo "  sudo docker rm -f ${LB_CONTAINER_NAME} $(for ((i=1; i<=EVAL_REPLICA_COUNT; i++)); do printf '%s-%d ' "$CONTAINER_PREFIX" "$i"; done)"
