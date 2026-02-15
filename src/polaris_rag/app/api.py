# polaris_rag/app/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from polaris_rag.config import GlobalConfig
from polaris_rag.app.container import build_container
import logging
import traceback

app = FastAPI(title="Polaris RAG API", version="0.1.0")
logger = logging.getLogger("polaris_rag.api")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

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
        # Your pipeline currently appears to return a dict with a 'response' key
        result = app.state.container.pipeline.run(req.query)["response"]
        return QueryResponse(answer=str(result))
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