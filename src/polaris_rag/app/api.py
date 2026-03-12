# polaris_rag/app/api.py
from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from polaris_rag.config import GlobalConfig
from polaris_rag.app.container import build_container
from polaris_rag.app.readiness import build_readiness_report
import logging
import traceback
from typing import Any, Mapping
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

app = FastAPI(title="Polaris RAG API", version="0.1.0")
logger = logging.getLogger("polaris_rag.api")

class QueryRequest(BaseModel):
    query: str


class RetrievedContextChunk(BaseModel):
    rank: int
    doc_id: str
    text: str
    score: float | None = None
    source: str | None = None


class QueryResponse(BaseModel):
    answer: str
    context: list[RetrievedContextChunk] = Field(default_factory=list)


def _extract_doc_id(node: Any) -> str:
    for attr in ("id_", "node_id", "id"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            return value
    return "<unknown-doc-id>"


def _extract_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(node, "get_content"):
        try:
            content = node.get_content()
            return content if isinstance(content, str) else str(content)
        except Exception:
            return ""
    return ""


def _serialize_context(source_nodes: list[Any]) -> list[RetrievedContextChunk]:
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
                doc_id=_extract_doc_id(node),
                text=_extract_text(node),
                score=score,
                source=source_name,
            )
        )
    return chunks


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _resolve_request_budget(request: Request) -> RequestBudget | None:
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
    import os
    cfg_path = os.environ.get("POLARIS_CONFIG", "/app/config/config.yaml")
    cfg = GlobalConfig.load(cfg_path)
    runtime_tracking_cfg = load_mlflow_runtime_config(cfg)
    configure_mlflow_runtime(runtime_tracking_cfg, strict=True)
    app.state.container = build_container(cfg)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    report = build_readiness_report(app.state.container.config)
    status_code = 200 if report.get("ready") else 503
    return JSONResponse(status_code=status_code, content=report)

@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
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
            result = app.state.container.pipeline.run(req.query, request_budget=request_budget)
            answer = str(result.get("response", ""))
            context = _serialize_context(result.get("source_nodes", []))
            timings = _as_mapping(result.get("timings", {}))
            response_status = "ok" if answer.strip() else "empty_response"
            context_payload = [
                c.model_dump() if hasattr(c, "model_dump") else c.dict()
                for c in context
            ]
            set_span_attributes(
                trace_span,
                {
                    "retrieval_elapsed_ms": timings.get("retrieval_elapsed_ms"),
                    "generation_elapsed_ms": timings.get("generation_elapsed_ms"),
                    "response_status": response_status,
                    "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                },
            )
            set_span_outputs(
                trace_span,
                {
                    "answer": answer,
                    "context": context_payload,
                    "timings": timings,
                    "policy": policy,
                },
            )
            return QueryResponse(answer=answer, context=context)
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
