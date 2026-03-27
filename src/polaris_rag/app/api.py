# polaris_rag/app/api.py
"""FastAPI application for Polaris RAG.

This module configures the public HTTP API, defines request and response models, and
exposes the query, readiness, and feedback endpoints used by the UI and evaluation
tooling.

Classes
-------
QueryConstraintsPayload
    Payload model for query Constraints.
RetrievedContextChunk
    Structured chunk for retrieved Context.
QueryAnswerStatus
    Status payload for query Answer.
QueryTimings
    Timing data for query.
QueryRequest
    Request model for query.
QueryResponse
    Response model for query.
UiRuntimeResponse
    Response model for UI Runtime.
UiFeedbackSummaryResponse
    Response model for UI Feedback Summary.
UiFeedbackSubmissionRequest
    Request model for UI Feedback Submission.
UiFeedbackSubmissionResponse
    Response model for UI Feedback Submission.

Functions
---------
parse_cors_allowed_origins
    Parse the configured UI CORS allowlist.
configure_cors
    Attach CORS middleware to the target FastAPI application.
get_feedback_log_path
    Return the configured feedback log path.
startup
    Initialize application state during FastAPI startup.
health
    Return a basic liveness response for the API service.
ready
    Return a readiness report for configured runtime dependencies.
ui_runtime
    Return runtime settings consumed by the UI.
ui_feedback_summary
    Return aggregated UI feedback statistics.
submit_ui_feedback
    Persist a UI feedback submission.
query
    Handle a Polaris RAG query request.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from polaris_rag.config import GlobalConfig
from polaris_rag.app.container import build_container
from polaris_rag.app.readiness import build_readiness_report
import logging
import traceback
from typing import Any, Mapping
from polaris_rag.common.evaluation_api import POLARIS_EVAL_INCLUDE_METADATA_HEADER
from polaris_rag.common.request_budget import (
    EVAL_POLICY_INTERACTIVE,
    FAILURE_CLASS_API_INTERNAL_ERROR,
    FAILURE_STAGE_API,
    GenerationTimeoutError,
    POLARIS_EVAL_POLICY_HEADER,
    POLARIS_TIMEOUT_HEADER,
    RequestBudget,
    RequestBudgetExceededError,
    RetrievalTimeoutError,
    build_failure_detail,
    normalize_evaluation_policy,
    resolve_evaluation_deadlines,
)
from polaris_rag.retrieval.node_utils import extract_doc_id, extract_text, serialize_source_nodes
from polaris_rag.retrieval.query_constraints import serialize_query_constraints
from polaris_rag.observability.mlflow_tracking import (
    TRACE_CHILD_RUN_HEADER,
    TRACE_PARENT_RUN_HEADER,
    TRACE_STAGE_HEADER,
    configure_mlflow_runtime,
    load_mlflow_runtime_config,
    set_span_attributes,
    set_span_outputs,
    start_span,
)
from polaris_rag.ui.feedback import FeedbackRecord, append_feedback_record, feedback_summary

app = FastAPI(title="Polaris RAG API", version="0.1.0")
logger = logging.getLogger("polaris_rag.api")

DEFAULT_UI_CORS_ALLOWED_ORIGINS = (
    "http://localhost:8500",
    "http://127.0.0.1:8500",
)
UI_CORS_ALLOWED_HEADERS = (
    "Content-Type",
    POLARIS_TIMEOUT_HEADER,
    POLARIS_EVAL_POLICY_HEADER,
    POLARIS_EVAL_INCLUDE_METADATA_HEADER,
)


class QueryConstraintsPayload(BaseModel):
    """Payload model for query Constraints.
    
    Attributes
    ----------
    query_type : str or None
        Value for query Type.
    system_names : list[str]
        Value for system Names.
    partition_names : list[str]
        Value for partition Names.
    service_names : list[str]
        Value for service Names.
    scope_family_names : list[str]
        Value for scope Family Names.
    software_names : list[str]
        Value for software Names.
    software_versions : list[str]
        Value for software Versions.
    module_names : list[str]
        Value for module Names.
    toolchain_names : list[str]
        Value for toolchain Names.
    toolchain_versions : list[str]
        Value for toolchain Versions.
    scope_required : bool or None
        Value for scope Required.
    version_sensitive_guess : bool or None
        Value for version Sensitive Guess.
    """
    query_type: str | None = None
    system_names: list[str] = Field(default_factory=list)
    partition_names: list[str] = Field(default_factory=list)
    service_names: list[str] = Field(default_factory=list)
    scope_family_names: list[str] = Field(default_factory=list)
    software_names: list[str] = Field(default_factory=list)
    software_versions: list[str] = Field(default_factory=list)
    module_names: list[str] = Field(default_factory=list)
    toolchain_names: list[str] = Field(default_factory=list)
    toolchain_versions: list[str] = Field(default_factory=list)
    scope_required: bool | None = None
    version_sensitive_guess: bool | None = None


class RetrievedContextChunk(BaseModel):
    """Structured chunk for retrieved Context.
    
    Attributes
    ----------
    rank : int
        Value for rank.
    doc_id : str
        Stable identifier for doc.
    text : str
        Text value to inspect, tokenize, or encode.
    score : float or None
        Value for score.
    source : str or None
        Source definition, source name, or source identifier to process.
    """
    rank: int
    doc_id: str
    text: str
    score: float | None = None
    source: str | None = None


class QueryAnswerStatus(BaseModel):
    """Status payload for query Answer.
    
    Attributes
    ----------
    code : str
        Value for code.
    detail : str
        Structured failure detail payload from the backend.
    """
    code: str
    detail: str


class QueryTimings(BaseModel):
    """Timing data for query.
    
    Attributes
    ----------
    retrieval_elapsed_ms : int or None
        retrieval Elapsed Ms expressed in milliseconds.
    generation_elapsed_ms : int or None
        generation Elapsed Ms expressed in milliseconds.
    """
    retrieval_elapsed_ms: int | None = None
    generation_elapsed_ms: int | None = None


class QueryRequest(BaseModel):
    """Request model for query.
    
    Attributes
    ----------
    query : str
        User query text.
    query_constraints : QueryConstraintsPayload or None
        Optional structured retrieval constraints.
    include_evaluation_metadata : bool
        Whether to include evaluation metadata in the response.
    """
    query: str
    query_constraints: QueryConstraintsPayload | None = None
    include_evaluation_metadata: bool = False


class QueryResponse(BaseModel):
    """Response model for query.
    
    Attributes
    ----------
    answer : str
        Value for answer.
    context : list[RetrievedContextChunk]
        Value for context.
    query_constraints : QueryConstraintsPayload or None
        Optional structured retrieval constraints.
    evaluation_metadata : dict[str, Any] or None
        Value for evaluation Metadata.
    answer_status : QueryAnswerStatus
        Value for answer Status.
    timings : QueryTimings
        Value for timings.
    """
    answer: str
    context: list[RetrievedContextChunk] = Field(default_factory=list)
    query_constraints: QueryConstraintsPayload | None = None
    evaluation_metadata: dict[str, Any] | None = None
    answer_status: QueryAnswerStatus
    timings: QueryTimings = Field(default_factory=QueryTimings)


class UiRuntimeResponse(BaseModel):
    """Response model for UI Runtime.
    
    Attributes
    ----------
    query_endpoint_path : str
        Filesystem path used by the operation.
    health_endpoint_path : str
        Filesystem path used by the operation.
    ready_endpoint_path : str
        Filesystem path used by the operation.
    feedback_log_path : str
        Filesystem path used by the operation.
    """
    query_endpoint_path: str = "/v1/query"
    health_endpoint_path: str = "/health"
    ready_endpoint_path: str = "/ready"
    feedback_log_path: str


class UiFeedbackSummaryResponse(BaseModel):
    """Response model for UI Feedback Summary.
    
    Attributes
    ----------
    total : int
        Value for total.
    helpful_yes : int
        Value for helpful Yes.
    grounded_yes : int
        Value for grounded Yes.
    by_scenario : list[dict[str, Any]]
        Value for by Scenario.
    failure_types : list[dict[str, Any]]
        Value for failure Types.
    """
    total: int = 0
    helpful_yes: int = 0
    grounded_yes: int = 0
    by_scenario: list[dict[str, Any]] = Field(default_factory=list)
    failure_types: list[dict[str, Any]] = Field(default_factory=list)


class UiFeedbackSubmissionRequest(BaseModel):
    """Request model for UI Feedback Submission.
    
    Attributes
    ----------
    response_fingerprint : str
        Value for response Fingerprint.
    query : str
        User query text.
    scenario_id : str or None
        Stable identifier for scenario.
    answer_status_code : str
        Value for answer Status Code.
    evidence_count : int
        Value for evidence Count.
    helpful : str
        Value for helpful.
    grounded : str
        Value for grounded.
    citation_quality : str
        Value for citation Quality.
    failure_type : str
        Value for failure Type.
    notes : str
        Value for notes.
    """
    response_fingerprint: str
    query: str
    scenario_id: str | None = None
    answer_status_code: str
    evidence_count: int = 0
    helpful: str
    grounded: str
    citation_quality: str
    failure_type: str
    notes: str = ""


class UiFeedbackSubmissionResponse(BaseModel):
    """Response model for UI Feedback Submission.
    
    Attributes
    ----------
    created_at : str
        Value for created At.
    response_fingerprint : str
        Value for response Fingerprint.
    """
    created_at: str
    response_fingerprint: str


def parse_cors_allowed_origins(raw_value: str | None) -> list[str]:
    """Parse the configured UI CORS allowlist.
    
    Parameters
    ----------
    raw_value : str or None, optional
        Raw value value to normalize.
    
    Returns
    -------
    list[str]
        Parsed CORS Allowed Origins.
    """
    if raw_value is None:
        return list(DEFAULT_UI_CORS_ALLOWED_ORIGINS)

    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        return list(DEFAULT_UI_CORS_ALLOWED_ORIGINS)
    return values


def configure_cors(target_app: FastAPI, allowed_origins: list[str] | None = None) -> None:
    """Attach CORS middleware to the target FastAPI application.
    
    Parameters
    ----------
    target_app : FastAPI
        FastAPI application instance to configure.
    allowed_origins : list[str] or None, optional
        Allowed origins that may access the API from a browser context.
    """
    origins = allowed_origins or parse_cors_allowed_origins(os.getenv("POLARIS_UI_CORS_ALLOWED_ORIGINS"))
    target_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=list(UI_CORS_ALLOWED_HEADERS),
        expose_headers=list(UI_CORS_ALLOWED_HEADERS),
    )


def get_feedback_log_path() -> str:
    """Return the configured feedback log path.
    
    Returns
    -------
    str
        Requested feedback Log Path.
    """
    return os.getenv("POLARIS_FEEDBACK_LOG_PATH", "/app/data/ui_feedback/feedback.jsonl")


configure_cors(app)


def _serialize_context(source_nodes: list[Any]) -> list[RetrievedContextChunk]:
    """Serialize context.
    
    Parameters
    ----------
    source_nodes : list[Any]
        Retrieved nodes or node wrappers to serialize.
    
    Returns
    -------
    list[RetrievedContextChunk]
        Collected results from the operation.
    """
    chunks: list[RetrievedContextChunk] = []
    for idx, source in enumerate(source_nodes, start=1):
        node = getattr(source, "node", source)
        score_raw = getattr(source, "score", None)
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None
        source_name: str | None = None

        metadata = getattr(node, "metadata", None)
        if isinstance(metadata, dict):
            source_raw = metadata.get("retrieval_source")
            if isinstance(source_raw, str) and source_raw:
                source_name = source_raw
            elif isinstance(metadata.get("retrieval_sources"), list):
                # Fallback when only list provenance exists.
                sources = [s for s in metadata.get("retrieval_sources", []) if isinstance(s, str) and s]
                if len(sources) == 1:
                    source_name = sources[0]
                elif len(sources) > 1:
                    source_name = "multi"

            chunks.append(
                RetrievedContextChunk(
                    rank=idx,
                    doc_id=extract_doc_id(node),
                    text=extract_text(node),
                    score=score,
                    source=source_name,
                )
            )
    return chunks


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """As Mapping.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _coerce_query_constraints(value: Any) -> QueryConstraintsPayload | None:
    """Coerce query Constraints.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    QueryConstraintsPayload or None
        Result of the operation.
    """
    payload = serialize_query_constraints(value)
    if payload is None:
        return None
    return QueryConstraintsPayload(**payload)


def _coerce_eval_metadata(value: Any) -> dict[str, Any] | None:
    """Coerce eval Metadata.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    dict[str, Any] or None
        Structured result of the operation.
    """
    if not isinstance(value, Mapping):
        return None
    return {str(key): value[key] for key in value.keys()}


def _coerce_optional_int(value: Any) -> int | None:
    """Coerce optional Int.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    int or None
        Result of the operation.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_timings(value: Any) -> QueryTimings:
    """Coerce timings.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    QueryTimings
        Result of the operation.
    """
    payload = _as_mapping(value)
    return QueryTimings(
        retrieval_elapsed_ms=_coerce_optional_int(payload.get("retrieval_elapsed_ms")),
        generation_elapsed_ms=_coerce_optional_int(payload.get("generation_elapsed_ms")),
    )


def _derive_answer_status(context_items: list[RetrievedContextChunk]) -> QueryAnswerStatus:
    """Derive Answer Status.
    
    Parameters
    ----------
    context_items : list[RetrievedContextChunk]
        Normalized retrieved context items associated with the answer.
    
    Returns
    -------
    QueryAnswerStatus
        Result of the operation.
    """
    context_count = len(context_items)
    if context_count <= 0:
        return QueryAnswerStatus(
            code="no_evidence",
            detail="No retrieved context was returned for this answer.",
        )
    if context_count == 1:
        return QueryAnswerStatus(
            code="limited_evidence",
            detail="Only one supporting context item was retrieved for this answer.",
        )
    return QueryAnswerStatus(
        code="grounded",
        detail="Multiple supporting context items were retrieved for this answer.",
    )


def _serialize_context_metadata(source_nodes: list[Any]) -> list[dict[str, Any]]:
    """Serialize context Metadata.
    
    Parameters
    ----------
    source_nodes : list[Any]
        Retrieved nodes or node wrappers to serialize.
    
    Returns
    -------
    list[dict[str, Any]]
        Collected results from the operation.
    """
    return serialize_source_nodes(source_nodes, include_text=False)


def _include_eval_metadata(request: Request) -> bool:
    """Include Eval Metadata.
    
    Parameters
    ----------
    request : Request
        Incoming request object or request payload.
    
    Returns
    -------
    bool
        `True` if include Eval Metadata; otherwise `False`.
    """
    raw = request.headers.get(POLARIS_EVAL_INCLUDE_METADATA_HEADER)
    if raw is None:
        return False
    normalized = str(raw).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _resolve_request_budget(request: Request) -> RequestBudget | None:
    """Resolve request Budget.
    
    Parameters
    ----------
    request : Request
        Incoming request object or request payload.
    
    Returns
    -------
    RequestBudget or None
        Result of the operation.
    """
    timeout_header = request.headers.get(POLARIS_TIMEOUT_HEADER)
    if timeout_header is None:
        return None

    try:
        requested_timeout_ms = int(timeout_header)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s header: %r", POLARIS_TIMEOUT_HEADER, timeout_header)
        return None

    policy = normalize_evaluation_policy(
        request.headers.get(POLARIS_EVAL_POLICY_HEADER),
        default=EVAL_POLICY_INTERACTIVE,
    )
    generation_cfg = _as_mapping(_as_mapping(app.state.container.config.raw).get("evaluation", {})).get("generation", {})
    deadlines = resolve_evaluation_deadlines(generation_cfg, policy=policy)
    effective_timeout_ms = min(max(1, requested_timeout_ms), deadlines.server_total_ms)
    return RequestBudget.from_timeout_ms(
        timeout_ms=effective_timeout_ms,
        policy=policy,
        retrieval_cap_ms=deadlines.retrieval_cap_ms,
        cleanup_reserve_ms=deadlines.cleanup_reserve_ms,
    )


@app.on_event("startup")
def startup():
    # Use env var so Docker can pass config location
    """Initialize application state during FastAPI startup.

    Notes
    -----
    Loads the configured `GlobalConfig`, configures MLflow runtime behavior,
    and stores the assembled dependency container on `app.state`.
    """
    cfg_path = os.environ.get("POLARIS_CONFIG", "/app/config/config.yaml")
    cfg = GlobalConfig.load(cfg_path)
    runtime_tracking_cfg = load_mlflow_runtime_config(cfg)
    configure_mlflow_runtime(runtime_tracking_cfg, strict=True)
    app.state.container = build_container(cfg)


@app.get("/health")
def health():
    """Return a basic liveness response for the API service.

    Returns
    -------
    dict[str, str]
        Static health payload indicating that the API process is running.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Return a readiness report for configured runtime dependencies.

    Returns
    -------
    JSONResponse
        Readiness payload with HTTP 200 when dependencies are ready and HTTP
        503 otherwise.
    """
    report = build_readiness_report(app.state.container.config)
    signature = json.dumps(report, sort_keys=True, default=str)
    last_logged_signature = getattr(app.state, "_last_readiness_failure_signature", None)

    if report.get("ready"):
        app.state._last_readiness_failure_signature = None
    elif signature != last_logged_signature:
        logger.warning("Readiness check failed: %s", signature)
        app.state._last_readiness_failure_signature = signature

    status_code = 200 if report.get("ready") else 503
    return JSONResponse(status_code=status_code, content=report)


@app.get("/v1/ui/runtime", response_model=UiRuntimeResponse)
def ui_runtime() -> UiRuntimeResponse:
    """Return runtime settings consumed by the UI.
    
    Returns
    -------
    UiRuntimeResponse
        Result of the operation.
    """
    return UiRuntimeResponse(feedback_log_path=get_feedback_log_path())


@app.get("/v1/ui/feedback/summary", response_model=UiFeedbackSummaryResponse)
def ui_feedback_summary() -> UiFeedbackSummaryResponse:
    """Return aggregated UI feedback statistics.
    
    Returns
    -------
    UiFeedbackSummaryResponse
        Result of the operation.
    """
    return UiFeedbackSummaryResponse(**feedback_summary(get_feedback_log_path()))


@app.post("/v1/ui/feedback", response_model=UiFeedbackSubmissionResponse)
def submit_ui_feedback(payload: UiFeedbackSubmissionRequest) -> UiFeedbackSubmissionResponse:
    """Persist a UI feedback submission.
    
    Parameters
    ----------
    payload : UiFeedbackSubmissionRequest
        Structured payload for the operation.
    
    Returns
    -------
    UiFeedbackSubmissionResponse
        Result of the operation.
    """
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record = FeedbackRecord(
        created_at=created_at,
        response_fingerprint=payload.response_fingerprint,
        query=payload.query,
        scenario_id=payload.scenario_id,
        answer_status_code=payload.answer_status_code,
        evidence_count=max(0, int(payload.evidence_count)),
        helpful=payload.helpful,
        grounded=payload.grounded,
        citation_quality=payload.citation_quality,
        failure_type=payload.failure_type,
        notes=payload.notes.strip(),
    )
    append_feedback_record(get_feedback_log_path(), record)
    return UiFeedbackSubmissionResponse(
        created_at=created_at,
        response_fingerprint=payload.response_fingerprint,
    )


@app.post("/v1/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(req: QueryRequest, request: Request):
    """Handle a Polaris RAG query request.
    
    Parameters
    ----------
    req : QueryRequest
        Value for req.
    request : Request
        Incoming FastAPI request object or request payload.
    
    Raises
    ------
    HTTPException
        If the request is invalid or the backend operation fails.
    """
    parent_run_id = request.headers.get(TRACE_PARENT_RUN_HEADER)
    child_run_id = request.headers.get(TRACE_CHILD_RUN_HEADER)
    stage_name = request.headers.get(TRACE_STAGE_HEADER)
    is_eval_request = any([parent_run_id, child_run_id, stage_name])
    trace_tags = {
        "polaris.source": "api",
        "polaris.eval_request": "true" if is_eval_request else "false",
    }
    if parent_run_id:
        trace_tags["mlflow.parent_run_id"] = str(parent_run_id)
        trace_tags["polaris.parent_run_id"] = str(parent_run_id)
    if child_run_id:
        trace_tags["polaris.child_run_id"] = str(child_run_id)
    if stage_name:
        trace_tags["polaris.stage"] = str(stage_name)
    request_budget = _resolve_request_budget(request)
    include_eval_metadata = _include_eval_metadata(request) or bool(req.include_evaluation_metadata)
    policy = request_budget.policy if request_budget is not None else normalize_evaluation_policy(None, default=EVAL_POLICY_INTERACTIVE)
    trace_tags["polaris.eval_policy"] = policy
    trace_span = None
    request_started_at = time.perf_counter()

    try:
        with start_span(
            "api.v1.query",
            inputs={"query": req.query},
            attributes={"http.route": "/v1/query"},
            tags=trace_tags,
        ) as trace_span:
            if request_budget is not None:
                set_span_attributes(trace_span, request_budget.to_attributes())
            provided_query_constraints = (
                req.query_constraints.model_dump()
                if req.query_constraints is not None
                else None
            )
            result = app.state.container.pipeline.run(
                req.query,
                request_budget=request_budget,
                query_constraints=provided_query_constraints,
            )
            answer = str(result.get("response", ""))
            context = _serialize_context(result.get("source_nodes", []))
            query_constraints = _coerce_query_constraints(result.get("query_constraints"))
            answer_status = _derive_answer_status(context)
            evaluation_metadata = None
            if include_eval_metadata:
                evaluation_metadata = _coerce_eval_metadata(
                    {
                        "reranker_profile": result.get("reranker_profile"),
                        "reranker_fingerprint": result.get("reranker_fingerprint"),
                        "retrieval_trace": result.get("retrieval_trace"),
                        "ranked_context_metadata": _serialize_context_metadata(
                            result.get("source_nodes", []) or []
                        ),
                    }
                )
            timings = _coerce_timings(result.get("timings", {}))
            response_status = "ok" if answer.strip() else "empty_response"
            context_payload = [
                c.model_dump() if hasattr(c, "model_dump") else c.dict()
                for c in context
            ]
            set_span_attributes(
                trace_span,
                {
                    "retrieval_elapsed_ms": timings.retrieval_elapsed_ms,
                    "generation_elapsed_ms": timings.generation_elapsed_ms,
                    "response_status": response_status,
                    "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                },
            )
            set_span_outputs(
                trace_span,
                {
                    "answer": answer,
                    "context": context_payload,
                    "query_constraints": (
                        query_constraints.model_dump()
                        if query_constraints is not None and hasattr(query_constraints, "model_dump")
                        else query_constraints.dict() if query_constraints is not None else None
                    ),
                    "evaluation_metadata": evaluation_metadata,
                    "answer_status": answer_status.model_dump(),
                    "timings": timings.model_dump(),
                    "policy": policy,
                },
            )
            return QueryResponse(
                answer=answer,
                context=context,
                query_constraints=query_constraints,
                evaluation_metadata=evaluation_metadata,
                answer_status=answer_status,
                timings=timings,
            )
    except (RetrievalTimeoutError, GenerationTimeoutError, RequestBudgetExceededError) as e:
        elapsed_ms = max(0, int(round((time.perf_counter() - request_started_at) * 1000.0)))
        detail = build_failure_detail(
            e,
            elapsed_ms=elapsed_ms,
            http_status=504,
            request_budget=request_budget,
        )
        if trace_span is not None:
            set_span_attributes(
                trace_span,
                {
                    "failure_class": detail["failure_class"],
                    "response_status": detail["response_status"],
                    "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                },
            )
            set_span_outputs(trace_span, detail)
        logger.warning("Timeout while handling /v1/query: %s", detail["error"])
        raise HTTPException(status_code=504, detail=detail)
    except Exception as e:
        elapsed_ms = max(0, int(round((time.perf_counter() - request_started_at) * 1000.0)))
        tb = traceback.format_exc()
        detail = build_failure_detail(
            e,
            elapsed_ms=elapsed_ms,
            http_status=500,
            request_budget=request_budget,
            response_status="error",
            traceback_text=tb,
        )
        detail.setdefault("failure_class", FAILURE_CLASS_API_INTERNAL_ERROR)
        detail.setdefault("failure_stage", FAILURE_STAGE_API)
        # Log full traceback to container logs for debugging
        logger.exception("Error while handling /v1/query")
        if trace_span is not None:
            set_span_attributes(
                trace_span,
                {
                    "failure_class": detail["failure_class"],
                    "response_status": detail["response_status"],
                    "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                },
            )
            set_span_outputs(trace_span, detail)

        raise HTTPException(
            status_code=500,
            detail=detail,
        )
