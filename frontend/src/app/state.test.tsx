import { fireEvent, screen } from "@testing-library/react";
import { AppProvider, useAppState } from "./state";
import { render } from "@testing-library/react";

const SESSION_STORAGE_KEY = "polaris-ui-state-v1";

function Probe() {
  const { state, dispatch } = useAppState();
  return (
    <div>
      <div data-testid="workspace">{state.currentWorkspace}</div>
      <div data-testid="timeout">{state.timeoutS}</div>
      <button onClick={() => dispatch({ type: "increment-timeout", delta: 5 })} type="button">
        increase
      </button>
    </div>
  );
}

describe("AppProvider", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("hydrates from session storage and persists updates", () => {
    window.sessionStorage.setItem(
      SESSION_STORAGE_KEY,
      JSON.stringify({
        currentWorkspace: "System",
        timeoutS: 77,
      }),
    );

    render(
      <AppProvider
        runtimeDefaults={{
          apiBaseUrl: "http://localhost:8000",
          apiEndpointPath: "/v1/query",
          apiTimeoutS: 60,
          displayName: "You",
        }}
      >
        <Probe />
      </AppProvider>,
    );

    expect(screen.getByTestId("workspace")).toHaveTextContent("System");
    expect(screen.getByTestId("timeout")).toHaveTextContent("77");

    fireEvent.click(screen.getByRole("button", { name: "increase" }));

    expect(JSON.parse(window.sessionStorage.getItem(SESSION_STORAGE_KEY) ?? "{}").timeoutS).toBe(82);
  });
});
