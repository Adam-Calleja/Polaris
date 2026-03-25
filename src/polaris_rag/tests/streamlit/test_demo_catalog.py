from __future__ import annotations

from polaris_rag.streamlit.demo_catalog import demo_scenarios


def test_demo_catalog_exposes_six_required_scenarios() -> None:
    scenarios = demo_scenarios()

    assert len(scenarios) == 6
    assert [scenario.scenario_id for scenario in scenarios] == [
        "strong_answer",
        "ambiguous_query",
        "unsupported_query",
        "conflicting_evidence",
        "freshness_risk",
        "timeout_case",
    ]
