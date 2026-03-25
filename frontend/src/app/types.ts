export type Workspace = "Assistant" | "Evaluation" | "System";

export interface FrontendRuntimeConfig {
  apiBaseUrl: string;
  apiEndpointPath: string;
  apiTimeoutS: number;
  displayName: string;
}

export interface ApiClientConfig {
  baseUrl: string;
  endpointPath: string;
  timeoutS: number;
}

export interface ManualConstraintFields {
  queryType: string;
  scopeFamilyNames: string;
  serviceNames: string;
  softwareNames: string;
  softwareVersions: string;
}

export interface QueryConstraintsPayload {
  query_type?: string;
  system_names?: string[];
  partition_names?: string[];
  service_names?: string[];
  scope_family_names?: string[];
  software_names?: string[];
  software_versions?: string[];
  module_names?: string[];
  toolchain_names?: string[];
  toolchain_versions?: string[];
  scope_required?: boolean | null;
  version_sensitive_guess?: boolean | null;
}

export interface RetrievedContextItem {
  rank: number;
  doc_id: string;
  text: string;
  score?: number | null;
  source?: string | null;
}

export interface QueryTimings {
  retrieval_elapsed_ms?: number | null;
  generation_elapsed_ms?: number | null;
}

export interface AnswerStatus {
  code: string;
  detail: string;
}

export interface QueryResponseData {
  answer: string;
  context: RetrievedContextItem[];
  query_constraints?: QueryConstraintsPayload | null;
  evaluation_metadata?: Record<string, unknown> | null;
  answer_status: AnswerStatus;
  timings: QueryTimings;
}

export interface ApiProbeResult {
  ok: boolean;
  url: string;
  status_code?: number | null;
  payload?: unknown;
  message?: string | null;
}

export interface NormalizedApiError {
  kind: string;
  message: string;
  status_code?: number | null;
  failure_class?: string | null;
  detail?: unknown;
}

export interface AssistantUserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: string;
}

export interface AssistantReplyMessage {
  id: string;
  role: "assistant";
  content: string;
  createdAt: string;
  query: string;
  response: QueryResponseData | null;
  error: NormalizedApiError | null;
  pending: boolean;
}

export type AssistantMessage = AssistantUserMessage | AssistantReplyMessage;

export interface DemoScenario {
  scenarioId: string;
  title: string;
  description: string;
  query: string;
  queryConstraints?: QueryConstraintsPayload | null;
  serverTimeoutMs?: number | null;
  includeEvaluationMetadata?: boolean;
  focus: string;
}

export interface EvaluationScenarioResult {
  scenarioId: string;
  query: string;
  response: QueryResponseData | null;
  error: NormalizedApiError | null;
}

export interface FeedbackSummaryCountRow {
  count: number;
  scenario_id?: string;
  failure_type?: string;
}

export interface UiFeedbackSummary {
  total: number;
  helpful_yes: number;
  grounded_yes: number;
  by_scenario: FeedbackSummaryCountRow[];
  failure_types: FeedbackSummaryCountRow[];
}

export interface FeedbackSubmissionPayload {
  response_fingerprint: string;
  query: string;
  scenario_id?: string | null;
  answer_status_code: string;
  evidence_count: number;
  helpful: string;
  grounded: string;
  citation_quality: string;
  failure_type: string;
  notes: string;
}

export interface UiRuntimeConfig {
  query_endpoint_path: string;
  health_endpoint_path: string;
  ready_endpoint_path: string;
  feedback_log_path: string;
}

export interface AnswerSection {
  key: string;
  heading: string;
  body: string;
}
