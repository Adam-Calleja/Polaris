import type {
  ApiClientConfig,
  ApiProbeResult,
  FeedbackSubmissionPayload,
  NormalizedApiError,
  QueryConstraintsPayload,
  QueryResponseData,
  RetrievedContextItem,
  UiFeedbackSummary,
  UiRuntimeConfig,
} from "./types";
import { joinUrl } from "./utils";

const POLARIS_TIMEOUT_HEADER = "X-Polaris-Timeout-Ms";
const POLARIS_EVAL_POLICY_HEADER = "X-Polaris-Eval-Policy";
const POLARIS_EVAL_INCLUDE_METADATA_HEADER = "X-Polaris-Eval-Include-Metadata";

export class ApiClientError extends Error {
  error: NormalizedApiError;

  constructor(error: NormalizedApiError) {
    super(error.message);
    this.name = "ApiClientError";
    this.error = error;
  }
}

export class ApiTimeoutError extends ApiClientError {
  constructor(error: NormalizedApiError) {
    super(error);
    this.name = "ApiTimeoutError";
  }
}

function optionalInt(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function optionalFloat(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function deriveAnswerStatus(contextItems: RetrievedContextItem[]) {
  if (contextItems.length <= 0) {
    return {
      code: "no_evidence",
      detail: "No retrieved context was returned for this answer.",
    };
  }
  if (contextItems.length === 1) {
    return {
      code: "limited_evidence",
      detail: "Only one supporting context item was retrieved for this answer.",
    };
  }
  return {
    code: "grounded",
    detail: "Multiple supporting context items were retrieved for this answer.",
  };
}

function normalizeContextItems(rawContext: unknown): RetrievedContextItem[] {
  if (!Array.isArray(rawContext)) {
    return [];
  }

  return rawContext.map((item, index) => {
    if (!item || typeof item !== "object") {
      return {
        rank: index + 1,
        doc_id: "<unknown-doc-id>",
        text: String(item ?? ""),
      };
    }

    const payload = item as Record<string, unknown>;
    return {
      rank: optionalInt(payload.rank) ?? index + 1,
      doc_id: String(payload.doc_id ?? payload.id ?? payload.node_id ?? "<unknown-doc-id>"),
      text: String(payload.text ?? payload.content ?? ""),
      score: optionalFloat(payload.score),
      source: payload.source == null ? null : String(payload.source),
    };
  });
}

function normalizeQueryConstraints(rawConstraints: unknown): QueryConstraintsPayload | null {
  if (!rawConstraints || typeof rawConstraints !== "object") {
    return null;
  }

  function stringList(value: unknown): string[] {
    return Array.isArray(value)
      ? value
          .map((item) => String(item).trim())
          .filter(Boolean)
      : [];
  }

  function optionalBoolean(value: unknown): boolean | null | undefined {
    if (value == null || typeof value === "boolean") {
      return value as boolean | null | undefined;
    }
    const normalized = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "y", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "n", "off"].includes(normalized)) {
      return false;
    }
    return null;
  }

  const payload = rawConstraints as Record<string, unknown>;
  const queryType = payload.query_type == null ? null : String(payload.query_type).trim();

  return {
    query_type: queryType || undefined,
    system_names: stringList(payload.system_names),
    partition_names: stringList(payload.partition_names),
    service_names: stringList(payload.service_names),
    scope_family_names: stringList(payload.scope_family_names),
    software_names: stringList(payload.software_names),
    software_versions: stringList(payload.software_versions),
    module_names: stringList(payload.module_names),
    toolchain_names: stringList(payload.toolchain_names),
    toolchain_versions: stringList(payload.toolchain_versions),
    scope_required: optionalBoolean(payload.scope_required),
    version_sensitive_guess: optionalBoolean(payload.version_sensitive_guess),
  };
}

function normalizeTimings(rawTimings: unknown) {
  const payload = rawTimings && typeof rawTimings === "object" ? (rawTimings as Record<string, unknown>) : {};
  return {
    retrieval_elapsed_ms: optionalInt(payload.retrieval_elapsed_ms),
    generation_elapsed_ms: optionalInt(payload.generation_elapsed_ms),
  };
}

function normalizeAnswerStatus(rawStatus: unknown, contextItems: RetrievedContextItem[]) {
  if (rawStatus && typeof rawStatus === "object") {
    const payload = rawStatus as Record<string, unknown>;
    const code = payload.code == null ? "" : String(payload.code).trim();
    const detail = payload.detail == null ? "" : String(payload.detail).trim();
    if (code && detail) {
      return { code, detail };
    }
  }
  return deriveAnswerStatus(contextItems);
}

function extractAnswer(data: unknown): string {
  if (!data || typeof data !== "object") {
    return String(data ?? "");
  }

  const payload = data as Record<string, unknown>;
  for (const key of ["answer", "response", "text", "output"]) {
    if (typeof payload[key] === "string") {
      return payload[key] as string;
    }
  }

  for (const outerKey of ["result", "data"]) {
    const nested = payload[outerKey];
    if (!nested || typeof nested !== "object") {
      continue;
    }
    const nestedPayload = nested as Record<string, unknown>;
    for (const key of ["answer", "response", "text", "output"]) {
      if (typeof nestedPayload[key] === "string") {
        return nestedPayload[key] as string;
      }
    }
  }

  return String(data);
}

async function getResponseDetail(response: Response): Promise<unknown> {
  try {
    const payload = await response.json();
    if (payload && typeof payload === "object" && "detail" in payload) {
      return (payload as Record<string, unknown>).detail;
    }
    return payload;
  } catch {
    return (await response.text()).slice(0, 500);
  }
}

async function errorFromResponse(response: Response): Promise<ApiClientError> {
  const detail = await getResponseDetail(response);
  const detailObject = detail && typeof detail === "object" ? (detail as Record<string, unknown>) : null;
  const message =
    detailObject && (detailObject.error || detailObject.message)
      ? String(detailObject.error ?? detailObject.message)
      : String(detail ?? `API error ${response.status}`);
  const kind = response.status === 504 ? "timeout" : response.status >= 500 ? "server_error" : "api_error";
  const error = {
    kind,
    message,
    status_code: response.status,
    failure_class: detailObject?.failure_class == null ? null : String(detailObject.failure_class),
    detail,
  };
  return kind === "timeout" ? new ApiTimeoutError(error) : new ApiClientError(error);
}

async function fetchJson(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeoutHandle = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiTimeoutError({
        kind: "timeout",
        message: `Timed out while waiting for the API at ${url}.`,
        detail: error.message,
      });
    }
    throw new ApiClientError({
      kind: "network_error",
      message: `Failed to reach the API at ${url}.`,
      detail: error instanceof Error ? error.message : String(error),
    });
  } finally {
    window.clearTimeout(timeoutHandle);
  }
}

export async function queryBackend(
  config: ApiClientConfig,
  prompt: string,
  options: {
    queryConstraints?: QueryConstraintsPayload;
    includeEvaluationMetadata?: boolean;
    serverTimeoutMs?: number | null;
    evaluationPolicy?: string;
  } = {},
): Promise<QueryResponseData> {
  const url = joinUrl(config.baseUrl, config.endpointPath);
  const payload: Record<string, unknown> = { query: prompt };
  if (options.queryConstraints) {
    payload.query_constraints = options.queryConstraints;
  }
  if (options.includeEvaluationMetadata) {
    payload.include_evaluation_metadata = true;
  }

  const headers = new Headers({ "Content-Type": "application/json" });
  if (options.serverTimeoutMs != null) {
    headers.set(POLARIS_TIMEOUT_HEADER, String(options.serverTimeoutMs));
  }
  if (options.includeEvaluationMetadata) {
    headers.set(POLARIS_EVAL_INCLUDE_METADATA_HEADER, "true");
  }
  if (options.evaluationPolicy) {
    headers.set(POLARIS_EVAL_POLICY_HEADER, options.evaluationPolicy);
  }

  const response = await fetchJson(
    url,
    {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    },
    Math.max(1, Math.round(config.timeoutS * 1000)),
  );

  if (response.status >= 400) {
    throw await errorFromResponse(response);
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiClientError({
      kind: "invalid_response",
      message: "The API returned a non-JSON response.",
      status_code: response.status,
      detail: (await response.text()).slice(0, 500),
    });
  }

  const payloadObject = data && typeof data === "object" ? (data as Record<string, unknown>) : {};
  const context = normalizeContextItems(payloadObject.context);
  return {
    answer: extractAnswer(payloadObject),
    context,
    query_constraints: normalizeQueryConstraints(payloadObject.query_constraints),
    evaluation_metadata:
      payloadObject.evaluation_metadata && typeof payloadObject.evaluation_metadata === "object"
        ? (payloadObject.evaluation_metadata as Record<string, unknown>)
        : null,
    answer_status: normalizeAnswerStatus(payloadObject.answer_status, context),
    timings: normalizeTimings(payloadObject.timings),
  };
}

export async function probeEndpoint(config: ApiClientConfig, path: string): Promise<ApiProbeResult> {
  const url = joinUrl(config.baseUrl, path);
  try {
    const response = await fetchJson(url, { method: "GET" }, Math.max(1, Math.round(config.timeoutS * 1000)));
    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      payload = (await response.text()).slice(0, 500);
    }
    return {
      ok: response.status < 400,
      url,
      status_code: response.status,
      payload,
      message: response.status < 400 ? null : `Endpoint returned ${response.status}.`,
    };
  } catch (error) {
    if (error instanceof ApiTimeoutError) {
      return { ok: false, url, status_code: null, message: error.error.message };
    }
    if (error instanceof ApiClientError) {
      return { ok: false, url, status_code: null, message: error.error.message };
    }
    return { ok: false, url, status_code: null, message: String(error) };
  }
}

export async function getUiRuntime(config: ApiClientConfig): Promise<UiRuntimeConfig> {
  const url = joinUrl(config.baseUrl, "/v1/ui/runtime");
  const response = await fetchJson(url, { method: "GET" }, Math.max(1, Math.round(config.timeoutS * 1000)));
  if (response.status >= 400) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as UiRuntimeConfig;
}

export async function getFeedbackSummary(config: ApiClientConfig): Promise<UiFeedbackSummary> {
  const url = joinUrl(config.baseUrl, "/v1/ui/feedback/summary");
  const response = await fetchJson(url, { method: "GET" }, Math.max(1, Math.round(config.timeoutS * 1000)));
  if (response.status >= 400) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as UiFeedbackSummary;
}

export async function postFeedback(config: ApiClientConfig, payload: FeedbackSubmissionPayload): Promise<void> {
  const url = joinUrl(config.baseUrl, "/v1/ui/feedback");
  const response = await fetchJson(
    url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    Math.max(1, Math.round(config.timeoutS * 1000)),
  );
  if (response.status >= 400) {
    throw await errorFromResponse(response);
  }
}
