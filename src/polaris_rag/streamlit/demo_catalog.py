from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DemoScenario:
    scenario_id: str
    title: str
    description: str
    query: str
    query_constraints: dict[str, Any] | None = None
    server_timeout_ms: int | None = None
    include_evaluation_metadata: bool = True
    focus: str = ""


def demo_scenarios() -> list[DemoScenario]:
    return [
        DemoScenario(
            scenario_id="strong_answer",
            title="Grounded Support Answer",
            description="A direct operational question where the system should surface strong evidence and a clear action.",
            query="We need to renew RDS and transfer ownership. How should this be handled?",
            focus="Check that the answer is grounded and the evidence panel makes the rationale obvious.",
        ),
        DemoScenario(
            scenario_id="ambiguous_query",
            title="Ambiguous User Query",
            description="A realistic but underspecified query that should trigger clarification or cautious guidance.",
            query="My migrated project data no longer opens properly. What should I do?",
            focus="Verify that the answer signals uncertainty and that the interface makes weak evidence easy to spot.",
        ),
        DemoScenario(
            scenario_id="unsupported_query",
            title="Unsupported / Missing Evidence",
            description="A question that often lacks enough retrieved evidence for a confident answer.",
            query="Can you confirm the exact new path for my project data?",
            focus="Show the no-evidence or limited-evidence path and the recovery guidance.",
        ),
        DemoScenario(
            scenario_id="conflicting_evidence",
            title="Potentially Conflicting Guidance",
            description="A question framed to test whether the user can inspect source evidence rather than trust the answer blindly.",
            query="Should licence ownership changes be handled through a ticket or through the storage portal?",
            focus="Use the evidence panel to inspect the retrieved sources and compare them to the final answer.",
        ),
        DemoScenario(
            scenario_id="freshness_risk",
            title="Freshness / Version Sensitivity",
            description="A version-sensitive question where the query constraints and diagnostics matter.",
            query="What is the latest GROMACS version available on CCLake?",
            focus="Check that the UI exposes the interpreted query constraints and version-sensitivity cues.",
        ),
        DemoScenario(
            scenario_id="timeout_case",
            title="Timeout / Recovery State",
            description="A deliberately constrained request that should exercise the timeout handling path.",
            query="Summarise all relevant policies, storage rules, job submission guidance, and software-version details for HPC support.",
            server_timeout_ms=1,
            focus="Confirm that the timeout state is explicit and offers a clear recovery path.",
        ),
    ]


def scenario_map() -> dict[str, DemoScenario]:
    return {scenario.scenario_id: scenario for scenario in demo_scenarios()}
