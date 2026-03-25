import type { NormalizedApiError, QueryConstraintsPayload, QueryResponseData } from "../app/types";
import { formatTimingValue } from "../app/utils";
import { ExpandablePanel } from "./ExpandablePanel";

const QUERY_CONSTRAINT_LABELS: Record<string, string> = {
  query_type: "Query type",
  system_names: "Systems",
  partition_names: "Partitions",
  service_names: "Services",
  scope_family_names: "Scope families",
  software_names: "Software",
  software_versions: "Software versions",
  module_names: "Modules",
  toolchain_names: "Toolchains",
  toolchain_versions: "Toolchain versions",
  scope_required: "Scope required",
  version_sensitive_guess: "Version sensitive",
};

function ConstraintRows({ queryConstraints }: { queryConstraints?: QueryConstraintsPayload | null }) {
  if (!queryConstraints) {
    return <div className="diagnostics__empty">No explicit query constraints were returned for this answer.</div>;
  }

  const rows = Object.entries(QUERY_CONSTRAINT_LABELS)
    .map(([key, label]) => {
      const value = queryConstraints[key as keyof QueryConstraintsPayload];
      if (value == null || value === "" || (Array.isArray(value) && value.length === 0)) {
        return null;
      }
      const displayValue = Array.isArray(value) ? value.join(", ") : String(value);
      return (
        <div className="constraint-row" key={key}>
          <span className="constraint-row__label">{label}:</span>
          <span className="constraint-row__value">{displayValue}</span>
        </div>
      );
    })
    .filter(Boolean);

  if (rows.length === 0) {
    return <div className="diagnostics__empty">No explicit query constraints were returned for this answer.</div>;
  }

  return <div className="constraint-grid">{rows}</div>;
}

function DiagnosticsBody({
  response,
  error,
}: {
  response?: QueryResponseData | null;
  error?: NormalizedApiError | null;
}) {
  if (!response && !error) {
    return (
      <div className="surface-card diagnostics">
        <h3 className="diagnostics__title">Diagnostics</h3>
        <p className="diagnostics__message">
          Run a query to inspect evidence, timings, and query interpretation.
        </p>
      </div>
    );
  }

  if (response) {
    return (
      <div className="surface-card diagnostics">
        <h3 className="diagnostics__title">Diagnostics</h3>
        <div className="diagnostics__metric">
          <span className="diagnostics__metric-label">Evidence Chunks</span>
          <span className="diagnostics__metric-value">{response.context.length}</span>
        </div>
        <div className="diagnostics__metric-grid">
          <div className="diagnostics__mini-metric">
            <span className="diagnostics__mini-label">Retrieval</span>
            <span className="diagnostics__mini-value">
              {formatTimingValue(response.timings.retrieval_elapsed_ms)}
            </span>
          </div>
          <div className="diagnostics__mini-metric">
            <span className="diagnostics__mini-label">Generation</span>
            <span className="diagnostics__mini-value">
              {formatTimingValue(response.timings.generation_elapsed_ms)}
            </span>
          </div>
        </div>
        <h4 className="diagnostics__section-title">Interpreted Query Constraints</h4>
        <ConstraintRows queryConstraints={response.query_constraints} />
        {response.context.length > 0 ? (
          <ExpandablePanel title="Retrieved Evidence">
            <div className="evidence-list">
              {response.context.map((item) => (
                <article className="evidence-card" key={`${item.doc_id}-${item.rank}`}>
                  <div className="evidence-card__meta">
                    <span>#{item.rank}</span>
                    <span>{item.doc_id}</span>
                    <span>{item.source ?? "unknown"}</span>
                    <span>{typeof item.score === "number" ? item.score.toFixed(4) : "n/a"}</span>
                  </div>
                  <pre className="code-block">{item.text || "No chunk text returned."}</pre>
                </article>
              ))}
            </div>
          </ExpandablePanel>
        ) : null}
        {response.evaluation_metadata ? (
          <ExpandablePanel title="Debug Metadata">
            <pre className="code-block">{JSON.stringify(response.evaluation_metadata, null, 2)}</pre>
          </ExpandablePanel>
        ) : null}
      </div>
    );
  }

  return (
    <div className="surface-card diagnostics">
      <h3 className="diagnostics__title">Diagnostics</h3>
      <div className="diagnostics__metric">
        <span className="diagnostics__metric-label">Evidence Chunks</span>
        <span className="diagnostics__metric-value">0</span>
      </div>
      <p className="diagnostics__message">{error?.message ?? "No diagnostics available."}</p>
      {error?.detail != null ? (
        <ExpandablePanel title="Error Detail">
          <pre className="code-block">
            {typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail, null, 2)}
          </pre>
        </ExpandablePanel>
      ) : null}
    </div>
  );
}

export function DiagnosticsPanel({
  response,
  error,
}: {
  response?: QueryResponseData | null;
  error?: NormalizedApiError | null;
}) {
  return <DiagnosticsBody error={error} response={response} />;
}
