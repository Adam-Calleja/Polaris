import type { DemoScenario } from "./types";

export const demoScenarios: DemoScenario[] = [
  {
    scenarioId: "strong_answer",
    title: "Grounded Support Answer",
    description:
      "A direct operational question where the system should surface strong evidence and a clear action.",
    query: "We need to renew RDS and transfer ownership. How should this be handled?",
    focus: "Check that the answer is grounded and the evidence panel makes the rationale obvious.",
    includeEvaluationMetadata: true,
  },
  {
    scenarioId: "ambiguous_query",
    title: "Ambiguous User Query",
    description:
      "A realistic but underspecified query that should trigger clarification or cautious guidance.",
    query: "My migrated project data no longer opens properly. What should I do?",
    focus: "Verify that the answer signals uncertainty and that the interface makes weak evidence easy to spot.",
    includeEvaluationMetadata: true,
  },
  {
    scenarioId: "unsupported_query",
    title: "Unsupported / Missing Evidence",
    description: "A question that often lacks enough retrieved evidence for a confident answer.",
    query: "Can you confirm the exact new path for my project data?",
    focus: "Show the no-evidence or limited-evidence path and the recovery guidance.",
    includeEvaluationMetadata: true,
  },
  {
    scenarioId: "conflicting_evidence",
    title: "Potentially Conflicting Guidance",
    description:
      "A question framed to test whether the user can inspect source evidence rather than trust the answer blindly.",
    query: "Should licence ownership changes be handled through a ticket or through the storage portal?",
    focus: "Use the evidence panel to inspect the retrieved sources and compare them to the final answer.",
    includeEvaluationMetadata: true,
  },
  {
    scenarioId: "freshness_risk",
    title: "Freshness / Version Sensitivity",
    description: "A version-sensitive question where the query constraints and diagnostics matter.",
    query: "What is the latest GROMACS version available on CCLake?",
    focus: "Check that the UI exposes the interpreted query constraints and version-sensitivity cues.",
    includeEvaluationMetadata: true,
  },
  {
    scenarioId: "timeout_case",
    title: "Timeout / Recovery State",
    description: "A deliberately constrained request that should exercise the timeout handling path.",
    query: "Summarise all relevant policies, storage rules, job submission guidance, and software-version details for HPC support.",
    serverTimeoutMs: 1,
    focus: "Confirm that the timeout state is explicit and offers a clear recovery path.",
    includeEvaluationMetadata: true,
  },
];
