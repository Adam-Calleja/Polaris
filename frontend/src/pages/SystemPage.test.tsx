import { screen, waitFor } from "@testing-library/react";
import { renderApp } from "../test/renderApp";

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

describe("SystemPage", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("renders health checks and runtime summary", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return jsonResponse({ status: "ok" });
      }
      if (url.endsWith("/ready")) {
        return jsonResponse({ ready: true });
      }
      if (url.endsWith("/v1/ui/runtime")) {
        return jsonResponse({
          query_endpoint_path: "/v1/query",
          health_endpoint_path: "/health",
          ready_endpoint_path: "/ready",
          feedback_log_path: "/app/data/ui_feedback/feedback.jsonl",
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/system");

    await waitFor(() => {
      expect(screen.getByText("Health check passed.")).toBeInTheDocument();
    });

    expect(screen.getByText("Readiness check passed.")).toBeInTheDocument();
    expect(screen.getByText("/app/data/ui_feedback/feedback.jsonl")).toBeInTheDocument();
    expect(screen.getByText("Architecture Flow")).toBeInTheDocument();
  });
});
