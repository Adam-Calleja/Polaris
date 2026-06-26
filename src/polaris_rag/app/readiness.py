"""Readiness probes for Polaris RAG dependencies.

This module checks whether configured local dependencies such as the embedding service
and Qdrant are reachable and correctly provisioned for the running Polaris deployment.

Functions
---------
check_embed_readiness
    Check whether the configured embedding dependency is ready.
check_qdrant_readiness
    Check whether required Qdrant sources are ready.
build_readiness_report
    Build a combined readiness report for configured dependencies.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse

import httpx
from qdrant_client import QdrantClient

_LOCAL_EMBED_HOSTS = {
    "embed",
    "localhost",
    "127.0.0.1",
    "::1",
    "host.docker.internal",
}
_READINESS_TIMEOUT_SECONDS = 2.0


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """As Mapping.
    
    Parameters
    ----------
    obj : Any
        Value for obj.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _normalize_vector_store_kind(kind: Any) -> str:
    """Normalize vector Store Kind.
    
    Parameters
    ----------
    kind : Any
        Value for kind.
    
    Returns
    -------
    str
        Resulting string value.
    """
    if not kind:
        return "qdrant"

    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in {"qdrant", "qdrantindexstore", "qdrant_index_store"}:
        return "qdrant"
    return normalized


def _local_embed_probe_urls(api_base: str) -> tuple[str | None, list[str]]:
    """Local Embed Probe URLs.
    
    Parameters
    ----------
    api_base : str
        Base URL of the configured HTTP dependency.
    
    Returns
    -------
    tuple[str or None, list[str]]
        Collected results from the operation.
    """
    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        return None, []

    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin, [f"{origin}/health", f"{origin}/healthz"]


def _is_local_embed_dependency(api_base: str) -> bool:
    """Return whether local Embed Dependency.
    
    Parameters
    ----------
    api_base : str
        Base URL of the configured HTTP dependency.
    
    Returns
    -------
    bool
        `True` if is Local Embed Dependency; otherwise `False`.
    """
    parsed = urlparse(api_base)
    host = (parsed.hostname or "").strip().lower()
    return host in _LOCAL_EMBED_HOSTS


def _readiness_timeout() -> httpx.Timeout:
    """Readiness Timeout.
    
    Returns
    -------
    httpx.Timeout
        Result of the operation.
    """
    return httpx.Timeout(_READINESS_TIMEOUT_SECONDS, connect=1.0)


def check_embed_readiness(config: Any) -> dict[str, Any]:
    """Check whether the configured embedding dependency is ready.
    
    Parameters
    ----------
    config : Any
        Configuration object or mapping used by the operation.
    
    Returns
    -------
    dict[str, Any]
        Structured status information for the requested readiness check.
    """
    raw = _as_mapping(getattr(config, "raw", config))
    embedder_cfg = _as_mapping(raw.get("embedder", {}))
    kind = str(embedder_cfg.get("type") or embedder_cfg.get("kind") or "").strip().lower().replace("-", "_")
    api_base = str(embedder_cfg.get("api_base") or "").strip()

    if kind not in {"openai_like", "openailike", "open_ai_like", "open_ailike"}:
        return {
            "status": "skipped",
            "checked": False,
            "reason": f"embedder type {kind!r} does not use a local HTTP readiness probe",
        }

    if not api_base:
        return {
            "status": "failed",
            "checked": True,
            "reason": "embedder.api_base is missing",
        }

    if not _is_local_embed_dependency(api_base):
        return {
            "status": "skipped",
            "checked": False,
            "url": api_base,
            "reason": "embedder api_base is not a local dependency",
        }

    origin, probe_urls = _local_embed_probe_urls(api_base)
    if origin is None:
        return {
            "status": "failed",
            "checked": True,
            "url": api_base,
            "reason": "embedder api_base is not a valid absolute URL",
        }

    errors: list[str] = []
    with httpx.Client(timeout=_readiness_timeout(), follow_redirects=True) as client:
        for probe_url in probe_urls:
            try:
                response = client.get(probe_url)
            except httpx.HTTPError as exc:
                errors.append(f"{probe_url}: {exc}")
                continue

            if response.status_code < 400:
                return {
                    "status": "ok",
                    "checked": True,
                    "url": api_base,
                    "health_url": probe_url,
                }

            errors.append(f"{probe_url}: HTTP {response.status_code}")

    return {
        "status": "failed",
        "checked": True,
        "url": api_base,
        "reason": "; ".join(errors) if errors else "embed readiness probe failed",
    }


def check_qdrant_readiness(config: Any) -> dict[str, Any]:
    """Check whether required Qdrant sources are ready.
    
    Parameters
    ----------
    config : Any
        Configuration object or mapping used by the operation.
    
    Returns
    -------
    dict[str, Any]
        Structured status information for the requested readiness check.
    """
    raw = _as_mapping(getattr(config, "raw", config))
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    vector_stores = _as_mapping(raw.get("vector_stores", {}))
    raw_sources = retriever_cfg.get("sources")

    if not isinstance(raw_sources, list) or not raw_sources:
        return {
            "status": "failed",
            "checked": True,
            "reason": "retriever.sources is missing or empty",
            "sources": {},
        }

    client_cache: dict[tuple[str, int], tuple[QdrantClient | None, str | None]] = {}
    source_reports: dict[str, dict[str, Any]] = {}
    has_qdrant_source = False
    failures = 0

    for idx, raw_source in enumerate(raw_sources):
        source_cfg = _as_mapping(raw_source)
        source_name = str(source_cfg.get("name") or "").strip()
        report_key = source_name or f"source_{idx}"

        if not source_name:
            failures += 1
            source_reports[report_key] = {
                "status": "failed",
                "reason": "retriever source name is missing",
            }
            continue

        store_cfg = _as_mapping(vector_stores.get(source_name, {}))
        if not store_cfg:
            failures += 1
            source_reports[source_name] = {
                "status": "failed",
                "reason": "retriever source is not configured in vector_stores",
            }
            continue

        kind = _normalize_vector_store_kind(
            store_cfg.get("type") or store_cfg.get("kind") or store_cfg.get("provider")
        )
        if kind != "qdrant":
            source_reports[source_name] = {
                "status": "skipped",
                "reason": f"vector store type {kind!r} does not require Qdrant readiness",
            }
            continue

        has_qdrant_source = True
        host = str(store_cfg.get("host") or "localhost").strip()
        collection_name = str(store_cfg.get("collection_name") or "").strip()

        try:
            port = int(store_cfg.get("port", 6333))
        except (TypeError, ValueError):
            failures += 1
            source_reports[source_name] = {
                "status": "failed",
                "host": host,
                "port": store_cfg.get("port"),
                "collection_name": collection_name,
                "reason": f"invalid Qdrant port {store_cfg.get('port')!r}",
            }
            continue

        if not collection_name:
            failures += 1
            source_reports[source_name] = {
                "status": "failed",
                "host": host,
                "port": port,
                "reason": "collection_name is missing",
            }
            continue

        client_key = (host, port)
        if client_key not in client_cache:
            try:
                client = QdrantClient(
                    host=host,
                    port=port,
                    timeout=_READINESS_TIMEOUT_SECONDS,
                )
                client.get_collections()
                client_cache[client_key] = (client, None)
            except Exception as exc:  # pragma: no cover - specific client exceptions vary
                client_cache[client_key] = (None, str(exc))

        client, connect_error = client_cache[client_key]
        if connect_error is not None or client is None:
            failures += 1
            source_reports[source_name] = {
                "status": "failed",
                "host": host,
                "port": port,
                "collection_name": collection_name,
                "reason": f"unable to reach Qdrant: {connect_error}",
            }
            continue

        try:
            exists = bool(client.collection_exists(collection_name))
        except Exception as exc:  # pragma: no cover - specific client exceptions vary
            failures += 1
            source_reports[source_name] = {
                "status": "failed",
                "host": host,
                "port": port,
                "collection_name": collection_name,
                "reason": f"collection lookup failed: {exc}",
            }
            continue

        if not exists:
            failures += 1
            source_reports[source_name] = {
                "status": "failed",
                "host": host,
                "port": port,
                "collection_name": collection_name,
                "reason": f"collection {collection_name!r} does not exist",
            }
            continue

        source_reports[source_name] = {
            "status": "ok",
            "host": host,
            "port": port,
            "collection_name": collection_name,
        }

    if failures:
        return {
            "status": "failed",
            "checked": True,
            "reason": f"{failures} required Qdrant source(s) are not ready",
            "sources": source_reports,
        }

    if not has_qdrant_source:
        return {
            "status": "skipped",
            "checked": False,
            "reason": "no Qdrant-backed retriever sources require readiness checks",
            "sources": source_reports,
        }

    return {
        "status": "ok",
        "checked": True,
        "sources": source_reports,
    }


def build_readiness_report(config: Any) -> dict[str, Any]:
    """Build a combined readiness report for configured dependencies.
    
    Parameters
    ----------
    config : Any
        Configuration object or mapping used by the operation.
    
    Returns
    -------
    dict[str, Any]
        Constructed readiness Report.
    """
    embed = check_embed_readiness(config)
    qdrant = check_qdrant_readiness(config)
    checks = {
        "embed": embed,
        "qdrant": qdrant,
    }
    ready = all(check.get("status") in {"ok", "skipped"} for check in checks.values())
    return {
        "ready": ready,
        "status": "ok" if ready else "not_ready",
        "checks": checks,
    }


__all__ = [
    "build_readiness_report",
    "check_embed_readiness",
    "check_qdrant_readiness",
]
