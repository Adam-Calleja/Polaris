import "@fontsource/manrope/400.css";
import "@fontsource/manrope/500.css";
import "@fontsource/manrope/600.css";
import "@fontsource/manrope/700.css";
import "@fontsource/manrope/800.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";

import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { AppProvider } from "./app/state";
import { readFrontendRuntimeConfig } from "./app/runtime";
import "./styles.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element was not found.");
}

ReactDOM.createRoot(root).render(
  <BrowserRouter>
    <AppProvider runtimeDefaults={readFrontendRuntimeConfig()}>
      <App />
    </AppProvider>
  </BrowserRouter>,
);
