import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderApp } from "./test/renderApp";

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

describe("App shell routing", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("renders the shell and navigates to evaluation", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/v1/ui/feedback/summary")) {
        return jsonResponse({
          total: 0,
          helpful_yes: 0,
          grounded_yes: 0,
          by_scenario: [],
          failure_types: [],
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/assistant");

    expect(screen.getByText("Polaris")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("link", { name: /evaluation/i }));

    await waitFor(() => {
      expect(screen.getByText("Saved Feedback")).toBeInTheDocument();
    });
  });

  it("returns to the assistant when the brand is clicked", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/v1/ui/feedback/summary")) {
        return jsonResponse({
          total: 0,
          helpful_yes: 0,
          grounded_yes: 0,
          by_scenario: [],
          failure_types: [],
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderApp("/evaluation");

    await waitFor(() => {
      expect(screen.getByText("Saved Feedback")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("link", { name: "Polaris" }));

    await waitFor(() => {
      expect(screen.getByText("How can I help you today?")).toBeInTheDocument();
    });
  });

  it("preserves the current assistant session when the brand is clicked on the assistant page", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/v1/query")) {
        return jsonResponse({
          answer: "ACTION\nOpen the relevant documentation.",
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

    await userEvent.type(screen.getByRole("textbox", { name: "Prompt" }), "Take me away from landing");
    await userEvent.click(screen.getByRole("button", { name: "➜" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Clear Assistant" })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("link", { name: "Polaris" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Clear Assistant" })).toBeInTheDocument();
    });

    expect(screen.getByText("Open the relevant documentation.")).toBeInTheDocument();
    expect(screen.queryByText("How can I help you today?")).not.toBeInTheDocument();
  });
});
