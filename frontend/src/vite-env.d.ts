/// <reference types="vite/client" />

interface PolarisRuntimeWindowConfig {
  apiBaseUrl?: string;
  apiEndpointPath?: string;
  apiTimeoutS?: number | string;
  displayName?: string;
}

interface Window {
  __POLARIS_RUNTIME__?: PolarisRuntimeWindowConfig;
}
