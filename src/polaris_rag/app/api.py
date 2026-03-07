# polaris_rag/app/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from polaris_rag.config import GlobalConfig
from polaris_rag.app.container import build_container
import logging
import traceback
from typing import Any
from polaris_rag.observability.mlflow_tracking import (
    TRACE_CHILD_RUN_HEADER,
    TRACE_PARENT_RUN_HEADER,
    TRACE_STAGE_HEADER,
    configure_mlflow_runtime,
    load_mlflow_runtime_config,
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
    trace_span = None

    try:
        with start_span(
            "api.v1.query",
            inputs={"query": req.query},
            attributes={"http.route": "/v1/query"},
            tags=trace_tags,
        ) as trace_span:
            result = app.state.container.pipeline.run(req.query)
            answer = str(result.get("response", ""))
            context = _serialize_context(result.get("source_nodes", []))
            context_payload = [
                c.model_dump() if hasattr(c, "model_dump") else c.dict()
                for c in context
            ]
            set_span_outputs(trace_span, {"answer": answer, "context": context_payload})
            return QueryResponse(answer=answer, context=context)
    except Exception as e:
        if trace_span is not None:
            set_span_outputs(
                trace_span,
                {"error": f"{type(e).__name__}: {e}"},
            )
        # Log full traceback to container logs for debugging
        logger.exception("Error while handling /v1/query")

        # Return a useful error message to clients (still a 500)
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            },
        )
