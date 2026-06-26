import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { AppProvider } from "../app/state";
import type { FrontendRuntimeConfig } from "../app/types";
import { App } from "../App";

const DEFAULT_RUNTIME: FrontendRuntimeConfig = {
  apiBaseUrl: "http://localhost:8000",
  apiEndpointPath: "/v1/query",
  apiTimeoutS: 60,
  displayName: "Adam",
};

export function renderApp(entry = "/assistant") {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AppProvider runtimeDefaults={DEFAULT_RUNTIME}>
        <App />
      </AppProvider>
    </MemoryRouter>,
  );
}

export function renderWithProvider(ui: ReactElement, runtimeDefaults: FrontendRuntimeConfig = DEFAULT_RUNTIME) {
  return render(<AppProvider runtimeDefaults={runtimeDefaults}>{ui}</AppProvider>);
}
