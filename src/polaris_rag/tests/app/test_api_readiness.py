from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from polaris_rag.app import api, readiness
from polaris_rag.config import GlobalConfig


def _make_config(
    *,
    embed_api_base: str = "http://embed:80/v1",
    retriever_sources: list[dict[str, object]] | None = None,
    vector_stores: dict[str, dict[str, object]] | None = None,
) -> GlobalConfig:
    return GlobalConfig(
        {
            "embedder": {
                "type": "OpenAILike",
                "api_base": embed_api_base,
            },
            "vector_stores": vector_stores
            or {
                "docs": {
                    "type": "qdrant",
                    "host": "qdrant",
                    "port": 6333,
                    "collection_name": "support_docs",
                },
                "tickets": {
                    "type": "qdrant",
                    "host": "qdrant",
                    "port": 6333,
                    "collection_name": "support_tickets",
                },
            },
            "retriever": {
                "sources": retriever_sources
                or [
                    {"name": "docs"},
                    {"name": "tickets"},
                ]
            },
        }
    )


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeHTTPClient:
    def __init__(self, responses: dict[str, object]):
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str):
        response = self._responses[url]
        if isinstance(response, Exception):
            raise response
        return response


class _FakeQdrantClient:
    collections_by_host: dict[tuple[str, int], set[str]] = {}
    connection_errors: dict[tuple[str, int], str] = {}
    queried_collections: list[str] = []

    def __init__(self, *, host: str, port: int, timeout: float):
        self.host = host
        self.port = port
        self.timeout = timeout

    def get_collections(self):
        error = self.connection_errors.get((self.host, self.port))
        if error is not None:
            raise RuntimeError(error)
        return {"collections": sorted(self.collections_by_host.get((self.host, self.port), set()))}

    def collection_exists(self, collection_name: str) -> bool:
        self.queried_collections.append(collection_name)
        return collection_name in self.collections_by_host.get((self.host, self.port), set())


def _install_http_client(monkeypatch, responses: dict[str, object]) -> None:
    monkeypatch.setattr(readiness.httpx, "Client", lambda *args, **kwargs: _FakeHTTPClient(responses))


def _install_qdrant_client(monkeypatch, *, collections=None, connection_errors=None) -> None:
    _FakeQdrantClient.collections_by_host = collections or {}
    _FakeQdrantClient.connection_errors = connection_errors or {}
    _FakeQdrantClient.queried_collections = []
    monkeypatch.setattr(readiness, "QdrantClient", _FakeQdrantClient)


def test_build_readiness_report_ok(monkeypatch) -> None:
    config = _make_config()
    _install_http_client(
        monkeypatch,
        {
            "http://embed:80/health": _FakeResponse(200),
            "http://embed:80/healthz": _FakeResponse(200),
        },
    )
    _install_qdrant_client(
        monkeypatch,
        collections={("qdrant", 6333): {"support_docs", "support_tickets"}},
    )

    report = readiness.build_readiness_report(config)

    assert report["ready"] is True
    assert report["checks"]["embed"]["status"] == "ok"
    assert report["checks"]["qdrant"]["status"] == "ok"


def test_build_readiness_report_returns_503_shape_for_embed_failure(monkeypatch) -> None:
    config = _make_config()
    request = httpx.Request("GET", "http://embed:80/health")
    _install_http_client(
        monkeypatch,
        {
            "http://embed:80/health": httpx.ConnectError("connection refused", request=request),
            "http://embed:80/healthz": httpx.ConnectError("connection refused", request=request),
        },
    )
    _install_qdrant_client(
        monkeypatch,
        collections={("qdrant", 6333): {"support_docs", "support_tickets"}},
    )

    report = readiness.build_readiness_report(config)

    assert report["ready"] is False
    assert report["checks"]["embed"]["status"] == "failed"
    assert report["checks"]["qdrant"]["status"] == "ok"


def test_build_readiness_report_fails_when_qdrant_unreachable(monkeypatch) -> None:
    config = _make_config()
    _install_http_client(
        monkeypatch,
        {
            "http://embed:80/health": _FakeResponse(200),
            "http://embed:80/healthz": _FakeResponse(200),
        },
    )
    _install_qdrant_client(
        monkeypatch,
        connection_errors={("qdrant", 6333): "connection refused"},
    )

    report = readiness.build_readiness_report(config)

    assert report["ready"] is False
    assert report["checks"]["qdrant"]["status"] == "failed"
    assert "unable to reach Qdrant" in report["checks"]["qdrant"]["sources"]["docs"]["reason"]


def test_build_readiness_report_fails_when_collection_is_missing(monkeypatch) -> None:
    config = _make_config()
    _install_http_client(
        monkeypatch,
        {
            "http://embed:80/health": _FakeResponse(200),
            "http://embed:80/healthz": _FakeResponse(200),
        },
    )
    _install_qdrant_client(
        monkeypatch,
        collections={("qdrant", 6333): {"support_docs"}},
    )

    report = readiness.build_readiness_report(config)

    assert report["ready"] is False
    assert report["checks"]["qdrant"]["sources"]["tickets"]["status"] == "failed"
    assert "does not exist" in report["checks"]["qdrant"]["sources"]["tickets"]["reason"]


def test_build_readiness_report_ignores_unused_vector_stores(monkeypatch) -> None:
    config = _make_config(
        retriever_sources=[{"name": "docs"}],
        vector_stores={
            "docs": {
                "type": "qdrant",
                "host": "qdrant",
                "port": 6333,
                "collection_name": "support_docs",
            },
            "unused": {
                "type": "qdrant",
                "host": "qdrant",
                "port": 6333,
                "collection_name": "missing_collection",
            },
        },
    )
    _install_http_client(
        monkeypatch,
        {
            "http://embed:80/health": _FakeResponse(200),
            "http://embed:80/healthz": _FakeResponse(200),
        },
    )
    _install_qdrant_client(
        monkeypatch,
        collections={("qdrant", 6333): {"support_docs"}},
    )

    report = readiness.build_readiness_report(config)

    assert report["ready"] is True
    assert report["checks"]["qdrant"]["sources"].keys() == {"docs"}
    assert _FakeQdrantClient.queried_collections == ["support_docs"]


def test_build_readiness_report_skips_remote_embed(monkeypatch) -> None:
    config = _make_config(embed_api_base="https://embeddings.example.com/v1")

    def _unexpected_client(*args, **kwargs):
        raise AssertionError("remote embed should not be probed")

    monkeypatch.setattr(readiness.httpx, "Client", _unexpected_client)
    _install_qdrant_client(
        monkeypatch,
        collections={("qdrant", 6333): {"support_docs", "support_tickets"}},
    )

    report = readiness.build_readiness_report(config)

    assert report["ready"] is True
    assert report["checks"]["embed"]["status"] == "skipped"


def test_ready_endpoint_returns_200_when_report_is_ready(monkeypatch) -> None:
    api.app.state.container = SimpleNamespace(config=_make_config())
    monkeypatch.setattr(
        api,
        "build_readiness_report",
        lambda config: {"ready": True, "status": "ok", "checks": {}},
    )

    response = api.ready()

    assert response.status_code == 200
    assert json.loads(response.body) == {"ready": True, "status": "ok", "checks": {}}


def test_ready_endpoint_returns_503_when_report_is_not_ready(monkeypatch) -> None:
    api.app.state.container = SimpleNamespace(config=_make_config())
    monkeypatch.setattr(
        api,
        "build_readiness_report",
        lambda config: {"ready": False, "status": "not_ready", "checks": {"embed": {"status": "failed"}}},
    )

    response = api.ready()

    assert response.status_code == 503
    assert json.loads(response.body)["checks"]["embed"]["status"] == "failed"


def test_ready_endpoint_logs_non_ready_report_once(monkeypatch, caplog) -> None:
    api.app.state.container = SimpleNamespace(config=_make_config())
    api.app.state._last_readiness_failure_signature = None
    monkeypatch.setattr(
        api,
        "build_readiness_report",
        lambda config: {
            "ready": False,
            "status": "not_ready",
            "checks": {"qdrant": {"status": "failed", "reason": "collection missing"}},
        },
    )

    with caplog.at_level("WARNING"):
        first = api.ready()
        second = api.ready()

    assert first.status_code == 503
    assert second.status_code == 503
    assert caplog.text.count("Readiness check failed:") == 1
    assert "collection missing" in caplog.text


def test_ready_endpoint_resets_failure_log_dedup_after_success(monkeypatch, caplog) -> None:
    api.app.state.container = SimpleNamespace(config=_make_config())
    api.app.state._last_readiness_failure_signature = None
    reports = iter(
        [
            {"ready": False, "status": "not_ready", "checks": {"embed": {"status": "failed"}}},
            {"ready": True, "status": "ok", "checks": {}},
            {"ready": False, "status": "not_ready", "checks": {"embed": {"status": "failed"}}},
        ]
    )
    monkeypatch.setattr(api, "build_readiness_report", lambda config: next(reports))

    with caplog.at_level("WARNING"):
        first = api.ready()
        second = api.ready()
        third = api.ready()

    assert first.status_code == 503
    assert second.status_code == 200
    assert third.status_code == 503
    assert caplog.text.count("Readiness check failed:") == 2


def test_health_endpoint_remains_live_when_ready_fails(monkeypatch) -> None:
    api.app.state.container = SimpleNamespace(config=_make_config())
    monkeypatch.setattr(
        api,
        "build_readiness_report",
        lambda config: {"ready": False, "status": "not_ready", "checks": {}},
    )

    health_response = api.health()
    ready_response = api.ready()

    assert health_response == {"status": "ok"}
    assert ready_response.status_code == 503
