# polaris_rag/app/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from polaris_rag.config import GlobalConfig
from polaris_rag.app.container import build_container
import logging
import traceback
from typing import Any

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
    app.state.container = build_container(cfg)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = app.state.container.pipeline.run(req.query)
        answer = str(result.get("response", ""))
        context = _serialize_context(result.get("source_nodes", []))
        return QueryResponse(answer=answer, context=context)
    except Exception as e:
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
