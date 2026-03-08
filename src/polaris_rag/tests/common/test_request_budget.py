from __future__ import annotations

from polaris_rag.common import request_budget


def test_request_budget_stage_timeouts_shrink_monotonically(monkeypatch) -> None:
    times = iter([100.0, 100.0, 101.0, 101.0])
    monkeypatch.setattr(request_budget.time, "monotonic", lambda: next(times))

    budget = request_budget.RequestBudget.from_timeout_ms(
        timeout_ms=5000,
        policy="official",
        retrieval_cap_ms=2000,
        cleanup_reserve_ms=500,
    )

    first = budget.stage_timeout_ms(stage="retrieval", reserve_ms=budget.cleanup_reserve_ms)
    second = budget.stage_timeout_ms(stage="generation", reserve_ms=budget.cleanup_reserve_ms)

    assert first == 2000
    assert second == 3500


def test_resolve_evaluation_deadlines_clamps_server_below_client() -> None:
    deadlines = request_budget.resolve_evaluation_deadlines(
        {
            "api_timeout": 30,
            "deadlines": {
                "official": {
                    "client_total_seconds": 30,
                    "server_total_seconds": 45,
                    "retrieval_cap_seconds": 10,
                    "cleanup_reserve_seconds": 5,
                }
            },
        },
        policy="official",
    )

    assert deadlines.client_total_seconds == 30.0
    assert deadlines.server_total_seconds < deadlines.client_total_seconds
    assert deadlines.retrieval_cap_seconds == 10.0
