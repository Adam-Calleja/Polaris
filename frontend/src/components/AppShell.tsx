import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect } from "react";
import type { Workspace } from "../app/types";
import { useAppState } from "../app/state";
import { Brand } from "./Brand";

const NAV_ITEMS: Array<{ workspace: Workspace; label: string; path: string; icon: string }> = [
  { workspace: "Assistant", label: "AI Assistant", path: "/assistant", icon: "✳" },
  { workspace: "Evaluation", label: "Evaluation", path: "/evaluation", icon: "◔" },
  { workspace: "System", label: "System", path: "/system", icon: "⚙" },
];

function workspaceFromPath(pathname: string): Workspace {
  if (pathname.startsWith("/evaluation")) {
    return "Evaluation";
  }
  if (pathname.startsWith("/system")) {
    return "System";
  }
  return "Assistant";
}

export function AppShell() {
  const { state, dispatch } = useAppState();
  const location = useLocation();

  useEffect(() => {
    dispatch({ type: "set-workspace", workspace: workspaceFromPath(location.pathname) });
    dispatch({ type: "set-drawer-open", open: false });
  }, [dispatch, location.pathname]);

  return (
    <div className={`app-shell ${state.drawerOpen ? "app-shell--drawer-open" : ""}`}>
      <button
        aria-hidden={!state.drawerOpen}
        className={`app-shell__backdrop ${state.drawerOpen ? "app-shell__backdrop--visible" : ""}`}
        onClick={() => dispatch({ type: "set-drawer-open", open: false })}
        tabIndex={state.drawerOpen ? 0 : -1}
        type="button"
      />
      <aside className="sidebar">
        <div className="sidebar__header-row">
          <h2 className="sidebar__heading">Menu</h2>
        </div>
        <div className="sidebar__scroll">
          <div className="sidebar__divider" />
          <nav className="sidebar__nav">
            {NAV_ITEMS.map((item) => (
              <NavLink
                className={({ isActive }) => `sidebar__nav-item ${isActive ? "sidebar__nav-item--active" : ""}`}
                key={item.path}
                onClick={() => dispatch({ type: "set-workspace", workspace: item.workspace })}
                to={item.path}
              >
                <span className="sidebar__nav-icon">{item.icon}</span>
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="sidebar__section-title">Backend</div>
          <div className="sidebar__divider" />

          <div className="sidebar__backend-fields">
            <label className="sidebar__field">
              <span className="sidebar__label">API base URL</span>
              <input
                className="sidebar__input sidebar__input--accent"
                onChange={(event) => dispatch({ type: "set-api-base-url", value: event.target.value })}
                value={state.apiBaseUrl}
              />
            </label>

            <label className="sidebar__field">
              <span className="sidebar__label">Query Endpoint</span>
              <input
                className="sidebar__input sidebar__input--accent"
                onChange={(event) => dispatch({ type: "set-api-endpoint-path", value: event.target.value })}
                value={state.apiEndpointPath}
              />
            </label>

            <div className="sidebar__field">
              <span className="sidebar__label">HTTP Timeout</span>
              <div className="timeout-control">
                <span className="timeout-control__value">{state.timeoutS}</span>
                <button
                  className="timeout-control__button"
                  onClick={() => dispatch({ type: "increment-timeout", delta: -5 })}
                  type="button"
                >
                  −
                </button>
                <button
                  className="timeout-control__button"
                  onClick={() => dispatch({ type: "increment-timeout", delta: 5 })}
                  type="button"
                >
                  +
                </button>
              </div>
            </div>
          </div>

          <button
            aria-pressed={state.debugMode}
            className={`sidebar__toggle-card ${state.debugMode ? "sidebar__toggle-card--active" : ""}`}
            onClick={() => dispatch({ type: "set-debug-mode", value: !state.debugMode })}
            type="button"
          >
            <span className="sidebar__toggle-copy">
              <span className="sidebar__toggle-title">Debug mode</span>
              <span className="sidebar__toggle-state">{state.debugMode ? "Enabled" : "Off"}</span>
            </span>
            <span className={`sidebar__switch ${state.debugMode ? "sidebar__switch--active" : ""}`} aria-hidden="true">
              <span className="sidebar__switch-thumb" />
            </span>
          </button>
          <p className="sidebar__caption">
            Debug mode requests evaluation metadata from the API and exposes raw diagnostic payloads in the
            interface.
          </p>

          <button
            className="sidebar__toggle-chip"
            onClick={() => dispatch({ type: "toggle-manual-constraints" })}
            type="button"
          >
            {state.manualConstraintsOpen ? "› Hide Manual Query Constraints" : "› Manual Query Constraints"}
          </button>

          {state.manualConstraintsOpen ? (
            <div className="sidebar__constraints">
              <label className="sidebar__field">
                <span className="sidebar__label">Query type</span>
                <select
                  className="sidebar__input"
                  onChange={(event) =>
                    dispatch({ type: "set-manual-constraint", field: "queryType", value: event.target.value })
                  }
                  value={state.manualConstraints.queryType}
                >
                  <option value="auto">auto</option>
                  <option value="local_operational">local_operational</option>
                  <option value="software_version">software_version</option>
                  <option value="general_how_to">general_how_to</option>
                </select>
              </label>

              <label className="sidebar__field">
                <span className="sidebar__label">Scope families</span>
                <input
                  className="sidebar__input"
                  onChange={(event) =>
                    dispatch({
                      type: "set-manual-constraint",
                      field: "scopeFamilyNames",
                      value: event.target.value,
                    })
                  }
                  placeholder="cclake"
                  value={state.manualConstraints.scopeFamilyNames}
                />
              </label>

              <label className="sidebar__field">
                <span className="sidebar__label">Services</span>
                <input
                  className="sidebar__input"
                  onChange={(event) =>
                    dispatch({ type: "set-manual-constraint", field: "serviceNames", value: event.target.value })
                  }
                  placeholder="Research Data Store"
                  value={state.manualConstraints.serviceNames}
                />
              </label>

              <label className="sidebar__field">
                <span className="sidebar__label">Software</span>
                <input
                  className="sidebar__input"
                  onChange={(event) =>
                    dispatch({ type: "set-manual-constraint", field: "softwareNames", value: event.target.value })
                  }
                  placeholder="GROMACS"
                  value={state.manualConstraints.softwareNames}
                />
              </label>

              <label className="sidebar__field">
                <span className="sidebar__label">Software versions</span>
                <input
                  className="sidebar__input"
                  onChange={(event) =>
                    dispatch({
                      type: "set-manual-constraint",
                      field: "softwareVersions",
                      value: event.target.value,
                    })
                  }
                  placeholder="2024.4"
                  value={state.manualConstraints.softwareVersions}
                />
              </label>
            </div>
          ) : null}
        </div>
      </aside>

      <div className="app-shell__main">
        <header className="app-shell__header">
          <button
            aria-label="Open menu"
            className="app-shell__menu-button"
            onClick={() => dispatch({ type: "toggle-drawer" })}
            type="button"
          >
            ☰
          </button>
          <Brand />
        </header>
        <main className="app-shell__content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
