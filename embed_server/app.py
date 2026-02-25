from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes"}
NORMALIZE = os.getenv("NORMALIZE_EMBEDDINGS", "1").lower() in {"1", "true", "yes"}
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
TORCH_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))
USE_MODEL_ENCODE = os.getenv("USE_MODEL_ENCODE", "0").lower() in {"1", "true", "yes"}
EMBED_MAX_BATCH_SIZE = max(1, int(os.getenv("EMBED_MAX_BATCH_SIZE", "8")))
DISABLE_MKLDNN = os.getenv("DISABLE_MKLDNN", "1").lower() in {"1", "true", "yes"}
INSTRUCTION = os.getenv("EMBED_INSTRUCTION", "").strip() or None

app = FastAPI()

_model = None
_tokenizer = None


def _patch_autocast_signature() -> None:
    """Make torch.is_autocast_enabled accept an optional device_type argument."""
    try:
        torch.is_autocast_enabled("cpu")
        return
    except TypeError:
        pass

    original = torch.is_autocast_enabled

    def _wrapped(device_type: str | None = None) -> bool:
        return original()

    torch.is_autocast_enabled = _wrapped  # type: ignore[assignment]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = None
    user: Optional[str] = None


def _load_model() -> None:
    global _model, _tokenizer

    _patch_autocast_signature()
    if DISABLE_MKLDNN and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    if TORCH_THREADS > 0:
        torch.set_num_threads(TORCH_THREADS)
        try:
            torch.set_num_interop_threads(max(1, min(TORCH_THREADS, 2)))
        except RuntimeError:
            # set_num_interop_threads can only be called once per process.
            pass

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    _model.eval()


def _mean_pool_embeddings(texts: List[str]) -> List[List[float]]:
    assert _model is not None and _tokenizer is not None

    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = _model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if NORMALIZE:
            pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu().tolist()


def _encode_embeddings(texts: List[str]) -> List[List[float]]:
    assert _model is not None and _tokenizer is not None

    if USE_MODEL_ENCODE and hasattr(_model, "encode"):
        kwargs = {"normalize_embeddings": NORMALIZE, "max_length": MAX_LENGTH}
        if INSTRUCTION:
            kwargs["instruction"] = INSTRUCTION
        embeddings = _model.encode(texts, **kwargs)
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().tolist()
        return embeddings.tolist()

    return _mean_pool_embeddings(texts)


def _encode_embeddings_batched(texts: List[str]) -> List[List[float]]:
    if len(texts) <= EMBED_MAX_BATCH_SIZE:
        return _encode_embeddings(texts)

    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_MAX_BATCH_SIZE):
        batch = texts[start:start + EMBED_MAX_BATCH_SIZE]
        vectors.extend(_encode_embeddings(batch))
    return vectors


def _count_tokens(texts: List[str]) -> int:
    assert _tokenizer is not None
    total = 0
    for text in texts:
        total += len(_tokenizer.encode(text, add_special_tokens=False))
    return total


@app.on_event("startup")
def _startup() -> None:
    _load_model()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingsRequest) -> dict:
    if request.encoding_format not in (None, "float", "base64"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported encoding_format. Use 'float' or omit the field.",
        )

    texts = request.input if isinstance(request.input, list) else [request.input]
    if not all(isinstance(text, str) for text in texts):
        raise HTTPException(status_code=400, detail="Input must be a string or list of strings.")

    start = time.time()
    vectors = _encode_embeddings_batched(texts)
    elapsed = time.time() - start

    data = [
        {"object": "embedding", "index": idx, "embedding": vec}
        for idx, vec in enumerate(vectors)
    ]
    prompt_tokens = _count_tokens(texts)

    return {
        "object": "list",
        "data": data,
        "model": request.model or MODEL_ID,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        "elapsed": elapsed,
    }
