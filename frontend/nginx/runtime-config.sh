#!/bin/sh
set -eu

cat > /usr/share/nginx/html/runtime-config.js <<EOF
window.__POLARIS_RUNTIME__ = {
  apiBaseUrl: "${POLARIS_UI_API_BASE_URL:-http://localhost:8000}",
  apiEndpointPath: "${POLARIS_UI_API_ENDPOINT_PATH:-/v1/query}",
  apiTimeoutS: ${POLARIS_UI_API_TIMEOUT_S:-60},
  displayName: "${POLARIS_DISPLAY_NAME:-You}"
};
EOF
