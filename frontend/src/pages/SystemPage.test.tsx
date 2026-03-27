import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

function deferredResponse() {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((resolver) => {
    resolve = resolver;
  });
  return { promise, resolve };
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

  it("hides stale probe results while a manual refresh is in flight", async () => {
    const pendingHealth = deferredResponse();
    const pendingReady = deferredResponse();
    const pendingRuntime = deferredResponse();
    let refreshStarted = false;

    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (!refreshStarted) {
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
      }

      if (url.endsWith("/health")) {
        return pendingHealth.promise;
      }
      if (url.endsWith("/ready")) {
        return pendingReady.promise;
      }
      if (url.endsWith("/v1/ui/runtime")) {
        return pendingRuntime.promise;
      }

      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/system");

    await waitFor(() => {
      expect(screen.getByText("Health check passed.")).toBeInTheDocument();
    });

    refreshStarted = true;
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Refresh Status" }));

    expect(screen.queryByText("Health check passed.")).not.toBeInTheDocument();
    expect(screen.queryByText("Readiness check passed.")).not.toBeInTheDocument();
    expect(screen.queryByText("/app/data/ui_feedback/feedback.jsonl")).not.toBeInTheDocument();
    expect(screen.getAllByText("Loading")).toHaveLength(2);
    expect(screen.getByText("Refreshing...")).toBeInTheDocument();

    pendingHealth.resolve(
      new Response(JSON.stringify({ status: "degraded" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );
    pendingReady.resolve(
      new Response(JSON.stringify({ ready: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    pendingRuntime.resolve(
      new Response(
        JSON.stringify({
          query_endpoint_path: "/v1/query",
          health_endpoint_path: "/health",
          ready_endpoint_path: "/ready",
          feedback_log_path: "/app/data/ui_feedback/feedback-refresh.jsonl",
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await waitFor(() => {
      expect(screen.getByText("Endpoint returned 503.")).toBeInTheDocument();
    });

    expect(screen.getByText("Readiness check passed.")).toBeInTheDocument();
    expect(screen.getByText("/app/data/ui_feedback/feedback-refresh.jsonl")).toBeInTheDocument();
  });
});
