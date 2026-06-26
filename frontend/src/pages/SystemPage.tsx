import { useEffect, useState } from "react";
import { getUiRuntime, probeEndpoint } from "../app/api";
import { useAppState } from "../app/state";
import type { ApiClientConfig, ApiProbeResult, UiRuntimeConfig } from "../app/types";
import { ExpandablePanel } from "../components/ExpandablePanel";
import { LoadingLabel } from "../components/LoadingLabel";

function apiConfigFromState(state: ReturnType<typeof useAppState>["state"]): ApiClientConfig {
  return {
    baseUrl: state.apiBaseUrl,
    endpointPath: state.apiEndpointPath,
    timeoutS: state.timeoutS,
  };
}

const ARCHITECTURE_STAGES = [
  ["Corpus", "Jira tickets and official documentation"],
  ["Retrieval", "Vector search plus query interpretation"],
  ["Reranking", "Validity-aware ranking and source filtering"],
  ["Prompting", "Structured ticket-style answer template"],
  ["Answer", "Grounded response with evidence inspection"],
];

const SOURCE_LEGEND = [
  ["Docs", "Official documentation and service pages."],
  ["Tickets", "Historical support tickets used as operational memory."],
  ["Multi-source", "Merged retrieval across official and experiential evidence."],
];

function ProbeCard({
  title,
  probe,
  loading,
}: {
  title: string;
  probe: ApiProbeResult | null;
  loading: boolean;
}) {
  if (!probe) {
    return (
      <div className="surface-card probe-card">
        <h3 className="probe-card__title">{title}</h3>
        <p className="probe-card__status">{loading ? <LoadingLabel label="Loading" /> : "Loading..."}</p>
      </div>
    );
  }

  return (
    <div className="surface-card probe-card">
      <h3 className="probe-card__title">{title}</h3>
      <p className={`probe-card__status ${probe.ok ? "probe-card__status--ok" : "probe-card__status--error"}`}>
        {probe.ok ? `${title} check passed.` : probe.message ?? `${title} check failed.`}
      </p>
      <p className="probe-card__url">{probe.url}</p>
      {probe.payload != null ? (
        <ExpandablePanel title={`${title} Payload`}>
          <pre className="code-block">
            {typeof probe.payload === "string" ? probe.payload : JSON.stringify(probe.payload, null, 2)}
          </pre>
        </ExpandablePanel>
      ) : null}
    </div>
  );
}

export function SystemPage() {
  const { state } = useAppState();
  const [runtime, setRuntime] = useState<UiRuntimeConfig | null>(null);
  const [healthProbe, setHealthProbe] = useState<ApiProbeResult | null>(null);
  const [readyProbe, setReadyProbe] = useState<ApiProbeResult | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [refreshing, setRefreshing] = useState(true);

  function beginRefresh() {
    setRefreshing(true);
    setHealthProbe(null);
    setReadyProbe(null);
    setRuntime(null);
    setRuntimeError(null);
  }

  useEffect(() => {
    let cancelled = false;

    async function loadSystemView() {
      beginRefresh();
      const config = apiConfigFromState(state);
      const [health, ready, runtimeConfig] = await Promise.allSettled([
        probeEndpoint(config, "/health"),
        probeEndpoint(config, "/ready"),
        getUiRuntime(config),
      ]);

      if (cancelled) {
        return;
      }

      setHealthProbe(health.status === "fulfilled" ? health.value : null);
      setReadyProbe(ready.status === "fulfilled" ? ready.value : null);
      if (runtimeConfig.status === "fulfilled") {
        setRuntime(runtimeConfig.value);
        setRuntimeError(null);
      } else {
        setRuntime(null);
        setRuntimeError(runtimeConfig.reason instanceof Error ? runtimeConfig.reason.message : String(runtimeConfig.reason));
      }
      setRefreshing(false);
    }

    void loadSystemView();
    return () => {
      cancelled = true;
    };
  }, [refreshToken, state.apiBaseUrl, state.apiEndpointPath, state.timeoutS]);

  return (
    <section className="system-page">
      <div className="page-workspace surface-card">
        <div className="page-intro">
          <div className="page-workspace__eyebrow">System overview</div>
        </div>

        <section className="card-cluster">
          <h2 className="section-heading">Architecture Flow</h2>
          <div className="stage-grid">
            {ARCHITECTURE_STAGES.map(([title, body]) => (
              <article className="surface-card stage-card" key={title}>
                <h3 className="stage-card__title">{title}</h3>
                <p className="stage-card__body">{body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="card-cluster">
          <h2 className="section-heading">Corpus Legend</h2>
          <div className="legend-grid">
            {SOURCE_LEGEND.map(([title, body]) => (
              <article className="surface-card stage-card" key={title}>
                <h3 className="stage-card__title">{title}</h3>
                <p className="stage-card__body">{body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="card-cluster">
          <div className="section-heading-row">
            <h2 className="section-heading">Live Backend Checks</h2>
            <button
              className="secondary-button secondary-button--light"
              disabled={refreshing}
              onClick={() => {
                beginRefresh();
                setRefreshToken((value) => value + 1);
              }}
              type="button"
            >
              {refreshing ? <LoadingLabel label="Refreshing status" /> : "Refresh Status"}
            </button>
          </div>
          <div className="probe-grid">
            <ProbeCard loading={refreshing} probe={healthProbe} title="Health" />
            <ProbeCard loading={refreshing} probe={readyProbe} title="Readiness" />
          </div>
        </section>

        <section className="card-cluster">
          <h2 className="section-heading">Frontend Runtime Summary</h2>
          <div className="surface-card runtime-card">
            <div className="runtime-card__row">
              <span>API base URL</span>
              <code>{state.apiBaseUrl}</code>
            </div>
            <div className="runtime-card__row">
              <span>Query endpoint</span>
              <code>{state.apiEndpointPath}</code>
            </div>
            <div className="runtime-card__row">
              <span>HTTP timeout</span>
              <code>{state.timeoutS} seconds</code>
            </div>
            <div className="runtime-card__row">
              <span>Debug mode</span>
              <code>{String(state.debugMode)}</code>
            </div>
            <div className="runtime-card__row">
              <span>Feedback log</span>
              <code>
                {runtime?.feedback_log_path ?? (runtimeError ? runtimeError : refreshing ? "Refreshing..." : "Loading...")}
              </code>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}
