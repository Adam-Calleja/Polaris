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
      expect(screen.getByRole("heading", { name: "Evaluation" })).toBeInTheDocument();
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
      expect(screen.getByRole("heading", { name: "Evaluation" })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("link", { name: /polaris/i }));

    await waitFor(() => {
      expect(screen.getByText("How can I help you today?")).toBeInTheDocument();
    });
  });
});
