import type { FrontendRuntimeConfig } from "./types";

const DEFAULT_RUNTIME: FrontendRuntimeConfig = {
  apiBaseUrl: "http://localhost:8000",
  apiEndpointPath: "/v1/query",
  apiTimeoutS: 60,
  displayName: "You",
};

function coerceString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function coerceTimeout(value: unknown, fallback: number): number {
  const parsed = Number(value);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return fallback;
}

export function readFrontendRuntimeConfig(): FrontendRuntimeConfig {
  const runtime = typeof window !== "undefined" ? window.__POLARIS_RUNTIME__ ?? {} : {};
  return {
    apiBaseUrl: coerceString(runtime.apiBaseUrl, DEFAULT_RUNTIME.apiBaseUrl),
    apiEndpointPath: coerceString(runtime.apiEndpointPath, DEFAULT_RUNTIME.apiEndpointPath),
    apiTimeoutS: coerceTimeout(runtime.apiTimeoutS, DEFAULT_RUNTIME.apiTimeoutS),
    displayName: coerceString(runtime.displayName, DEFAULT_RUNTIME.displayName),
  };
}
