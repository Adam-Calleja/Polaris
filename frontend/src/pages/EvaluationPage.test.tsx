import { screen, waitFor } from "@testing-library/react";
import { renderApp } from "../test/renderApp";
import { computeResponseFingerprint } from "../app/utils";

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status,
      headers: {
        "Content-Type": "application/json",
      },
    }),
  );
}

describe("EvaluationPage", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("guards against duplicate feedback for an existing response", async () => {
    const query = "We need to renew RDS and transfer ownership. How should this be handled?";
    const answer = "Grounded answer";
    const fingerprint = await computeResponseFingerprint(query, answer, {
      contextDocIds: ["doc-1"],
      scenarioId: "strong_answer",
    });

    window.sessionStorage.setItem(
      "polaris-ui-state-v1",
      JSON.stringify({
        evaluationResults: {
          strong_answer: {
            scenarioId: "strong_answer",
            query,
            response: {
              answer,
              context: [{ rank: 1, doc_id: "doc-1", text: "chunk", source: "docs" }],
              query_constraints: null,
              evaluation_metadata: null,
              answer_status: {
                code: "grounded",
                detail: "Multiple supporting context items were retrieved for this answer.",
              },
              timings: {
                retrieval_elapsed_ms: 1,
                generation_elapsed_ms: 2,
              },
            },
            error: null,
          },
        },
        feedbackSubmissionIds: [fingerprint],
      }),
    );

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/v1/ui/feedback/summary")) {
        return jsonResponse({
          total: 1,
          helpful_yes: 1,
          grounded_yes: 1,
          by_scenario: [{ scenario_id: "strong_answer", count: 1 }],
          failure_types: [{ failure_type: "none", count: 1 }],
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/evaluation");

    await waitFor(() => {
      expect(
        screen.getByText("Feedback for this exact response has already been submitted in this session."),
      ).toBeInTheDocument();
    });
  });
});
