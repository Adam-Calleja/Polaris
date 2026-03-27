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

describe("AssistantPage", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("submits a prompt and renders structured answer diagnostics", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/v1/query")) {
        return jsonResponse({
          answer: [
            "CLASSIFICATION",
            "Category: Storage",
            "",
            "QUICK ASSESSMENT",
            "This looks like a self-service task. [1]",
            "",
            "ACTION",
            "Use the storage portal. [1]",
            "",
            "REFERENCE KEY",
            "[1] : storage-portal-doc",
          ].join("\n"),
          context: [
            { rank: 1, doc_id: "doc-1", text: "chunk one", score: 0.91, source: "docs" },
            { rank: 2, doc_id: "doc-2", text: "chunk two", score: 0.73, source: "tickets" },
          ],
          query_constraints: {
            query_type: "local_operational",
            service_names: ["Research Data Store"],
            scope_required: true,
          },
          answer_status: {
            code: "grounded",
            detail: "Multiple supporting context items were retrieved for this answer.",
          },
          timings: {
            retrieval_elapsed_ms: 151,
            generation_elapsed_ms: 50079,
          },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/assistant");

    await userEvent.type(screen.getByRole("textbox", { name: "Prompt" }), "We need to renew RDS and transfer ownership.");
    await userEvent.click(screen.getByRole("button", { name: "➜" }));

    await waitFor(() => {
      expect(screen.getByText("Category: Storage")).toBeInTheDocument();
    });

    expect(screen.getByText("Quick Assessment")).toBeInTheDocument();
    expect(screen.getByText("Action")).toBeInTheDocument();
    expect(screen.getByText("Diagnostics")).toBeInTheDocument();
    expect(screen.getByText("Evidence Chunks")).toBeInTheDocument();
    expect(screen.getByText("Research Data Store")).toBeInTheDocument();
    expect(screen.getByText("151")).toBeInTheDocument();
    expect(screen.getByText("50079")).toBeInTheDocument();
  });

  it("clears the assistant session back to the landing state", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/v1/query")) {
        return jsonResponse({
          answer: "ACTION\nReset the workflow.",
          context: [],
          answer_status: {
            code: "grounded",
            detail: "Answer generated.",
          },
          timings: {
            retrieval_elapsed_ms: 12,
            generation_elapsed_ms: 34,
          },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/assistant");

    await userEvent.type(screen.getByRole("textbox", { name: "Prompt" }), "Reset this assistant");
    await userEvent.click(screen.getByRole("button", { name: "➜" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Clear Assistant" })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: "Clear Assistant" }));

    expect(screen.getByText("Hello Adam")).toBeInTheDocument();
    expect(screen.getByText("How can I help you today?")).toBeInTheDocument();
    expect(screen.getByText("Storage renewal")).toBeInTheDocument();
    expect(screen.queryByText("Diagnostics")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear Assistant" })).not.toBeInTheDocument();
  });
});
