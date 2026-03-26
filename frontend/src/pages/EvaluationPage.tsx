import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useAppState } from "../app/state";
import type {
  ApiClientConfig,
  DemoScenario,
  EvaluationScenarioResult,
  FeedbackSubmissionPayload,
  NormalizedApiError,
  QueryResponseData,
  UiFeedbackSummary,
} from "../app/types";
import { demoScenarios } from "../app/scenarios";
import { ApiClientError, getFeedbackSummary, postFeedback, queryBackend } from "../app/api";
import { buildManualQueryConstraints, computeResponseFingerprint, mergeQueryConstraints } from "../app/utils";
import { AnswerCard } from "../components/AnswerCard";
import { DiagnosticsPanel } from "../components/DiagnosticsPanel";
import { ErrorCard } from "../components/ErrorCard";

function apiConfigFromState(state: ReturnType<typeof useAppState>["state"]): ApiClientConfig {
  return {
    baseUrl: state.apiBaseUrl,
    endpointPath: state.apiEndpointPath,
    timeoutS: state.timeoutS,
  };
}

function SummaryCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="surface-card metric-card">
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
    </div>
  );
}

function FeedbackForm({
  scenario,
  result,
  onSaved,
}: {
  scenario: DemoScenario;
  result: EvaluationScenarioResult;
  onSaved: () => Promise<void>;
}) {
  const { state, dispatch } = useAppState();
  const [helpful, setHelpful] = useState("yes");
  const [grounded, setGrounded] = useState("yes");
  const [citationQuality, setCitationQuality] = useState("strong");
  const [failureType, setFailureType] = useState("none");
  const [notes, setNotes] = useState("");
  const [fingerprint, setFingerprint] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadFingerprint() {
      const answer = result.response ? result.response.answer : result.error?.message ?? "";
      const contextDocIds = result.response ? result.response.context.map((item) => item.doc_id) : [];
      const value = await computeResponseFingerprint(result.query, answer, {
        contextDocIds,
        scenarioId: scenario.scenarioId,
      });
      if (!cancelled) {
        setFingerprint(value);
      }
    }

    void loadFingerprint();
    return () => {
      cancelled = true;
    };
  }, [result, scenario.scenarioId]);

  const alreadySubmitted = fingerprint ? state.feedbackSubmissionIds.includes(fingerprint) : false;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!fingerprint || alreadySubmitted || submitting) {
      return;
    }

    const payload: FeedbackSubmissionPayload = {
      response_fingerprint: fingerprint,
      query: result.query,
      scenario_id: scenario.scenarioId,
      answer_status_code: result.response ? result.response.answer_status.code : "no_evidence",
      evidence_count: result.response ? result.response.context.length : 0,
      helpful,
      grounded,
      citation_quality: citationQuality,
      failure_type: failureType,
      notes: notes.trim(),
    };

    setSubmitting(true);
    try {
      await postFeedback(apiConfigFromState(state), payload);
      dispatch({ type: "add-feedback-submission-id", fingerprint });
      await onSaved();
    } finally {
      setSubmitting(false);
    }
  }

  if (alreadySubmitted) {
    return (
      <div className="surface-card status-card">
        Feedback for this exact response has already been submitted in this session.
      </div>
    );
  }

  return (
    <form className="surface-card feedback-form" onSubmit={handleSubmit}>
      <h3 className="feedback-form__title">Record Evaluation Feedback</h3>
      <label className="feedback-form__field">
        <span>Helpful?</span>
        <select onChange={(event) => setHelpful(event.target.value)} value={helpful}>
          <option value="yes">yes</option>
          <option value="partly">partly</option>
          <option value="no">no</option>
        </select>
      </label>
      <label className="feedback-form__field">
        <span>Grounded?</span>
        <select onChange={(event) => setGrounded(event.target.value)} value={grounded}>
          <option value="yes">yes</option>
          <option value="partly">partly</option>
          <option value="no">no</option>
        </select>
      </label>
      <label className="feedback-form__field">
        <span>Citation quality</span>
        <select onChange={(event) => setCitationQuality(event.target.value)} value={citationQuality}>
          <option value="strong">strong</option>
          <option value="adequate">adequate</option>
          <option value="weak">weak</option>
        </select>
      </label>
      <label className="feedback-form__field">
        <span>Failure type</span>
        <select onChange={(event) => setFailureType(event.target.value)} value={failureType}>
          <option value="none">none</option>
          <option value="retrieval_gap">retrieval_gap</option>
          <option value="ambiguous_question">ambiguous_question</option>
          <option value="stale_or_version_risk">stale_or_version_risk</option>
          <option value="timeout">timeout</option>
          <option value="backend_error">backend_error</option>
        </select>
      </label>
      <label className="feedback-form__field">
        <span>Notes</span>
        <textarea onChange={(event) => setNotes(event.target.value)} rows={4} value={notes} />
      </label>
      <button className="feedback-form__submit" disabled={!fingerprint || submitting} type="submit">
        Save Feedback
      </button>
    </form>
  );
}

function ScenarioResult({
  result,
}: {
  result: EvaluationScenarioResult;
}) {
  return (
    <div className="evaluation-page__result-grid">
      <div>
        {result.response ? <AnswerCard response={result.response} /> : result.error ? <ErrorCard error={result.error} /> : null}
      </div>
      <div className="evaluation-page__diagnostics">
        <DiagnosticsPanel error={result.error} response={result.response} />
      </div>
    </div>
  );
}

export function EvaluationPage() {
  const { state, dispatch } = useAppState();
  const [summary, setSummary] = useState<UiFeedbackSummary | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [runningScenarioId, setRunningScenarioId] = useState<string | null>(null);

  async function loadSummary() {
    setLoadingSummary(true);
    setSummaryError(null);
    try {
      const nextSummary = await getFeedbackSummary(apiConfigFromState(state));
      setSummary(nextSummary);
    } catch (error) {
      setSummaryError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingSummary(false);
    }
  }

  useEffect(() => {
    void loadSummary();
  }, [state.apiBaseUrl, state.apiEndpointPath, state.timeoutS]);

  async function runScenario(scenario: DemoScenario) {
    setRunningScenarioId(scenario.scenarioId);
    try {
      const response = await queryBackend(apiConfigFromState(state), scenario.query, {
        queryConstraints: mergeQueryConstraints(
          buildManualQueryConstraints(state.manualConstraints),
          scenario.queryConstraints,
        ),
        includeEvaluationMetadata: state.debugMode || Boolean(scenario.includeEvaluationMetadata),
        serverTimeoutMs: scenario.serverTimeoutMs,
        evaluationPolicy: state.debugMode || scenario.includeEvaluationMetadata ? "diagnostic" : "interactive",
      });
      dispatch({
        type: "set-evaluation-result",
        result: {
          scenarioId: scenario.scenarioId,
          query: scenario.query,
          response,
          error: null,
        },
      });
    } catch (error) {
      const normalizedError: NormalizedApiError =
        error instanceof ApiClientError
          ? error.error
          : {
              kind: "unknown_error",
              message: error instanceof Error ? error.message : String(error),
            };
      dispatch({
        type: "set-evaluation-result",
        result: {
          scenarioId: scenario.scenarioId,
          query: scenario.query,
          response: null,
          error: normalizedError,
        },
      });
    } finally {
      setRunningScenarioId(null);
    }
  }

  return (
    <section className="evaluation-page">
      <div className="page-intro">
        <h1 className="page-intro__title">Evaluation</h1>
        <p className="page-intro__body">
          Curated live scenarios for your screencast, report screenshots, and lightweight usability
          evidence.
        </p>
      </div>

      <div className="metric-grid">
        <SummaryCard label="Saved Feedback" value={summary?.total ?? (loadingSummary ? "…" : 0)} />
        <SummaryCard label="Helpful = yes" value={summary?.helpful_yes ?? (loadingSummary ? "…" : 0)} />
        <SummaryCard label="Grounded = yes" value={summary?.grounded_yes ?? (loadingSummary ? "…" : 0)} />
      </div>

      {summaryError ? <div className="surface-card status-card">{summaryError}</div> : null}

      {summary && summary.total > 0 ? (
        <div className="evaluation-page__summary-grid">
          <div className="surface-card summary-table">
            <h3 className="summary-table__title">By Scenario</h3>
            {summary.by_scenario.map((row) => (
              <div className="summary-table__row" key={row.scenario_id}>
                <span>{row.scenario_id}</span>
                <span>{row.count}</span>
              </div>
            ))}
          </div>
          <div className="surface-card summary-table">
            <h3 className="summary-table__title">Failure Types</h3>
            {summary.failure_types.map((row) => (
              <div className="summary-table__row" key={row.failure_type}>
                <span>{row.failure_type}</span>
                <span>{row.count}</span>
              </div>
            ))}
          </div>
        </div>
      ) : !loadingSummary ? (
        <div className="surface-card status-card">No persistent evaluation feedback has been recorded yet.</div>
      ) : null}

      <div className="evaluation-page__scenario-list">
        {demoScenarios.map((scenario) => {
          const result = state.evaluationResults[scenario.scenarioId];

          return (
            <section className="surface-card scenario-card" key={scenario.scenarioId}>
              <div className="scenario-card__header">
                <div>
                  <h2 className="scenario-card__title">{scenario.title}</h2>
                  <p className="scenario-card__description">{scenario.description}</p>
                </div>
                <button
                  className="scenario-card__run"
                  disabled={runningScenarioId === scenario.scenarioId}
                  onClick={() => {
                    void runScenario(scenario);
                  }}
                  type="button"
                >
                  {runningScenarioId === scenario.scenarioId ? "Running..." : "Run"}
                </button>
              </div>
              <p className="scenario-card__focus">Focus: {scenario.focus}</p>
              <pre className="code-block scenario-card__query">{scenario.query}</pre>

              {result ? (
                <>
                  <ScenarioResult result={result} />
                  <FeedbackForm onSaved={loadSummary} result={result} scenario={scenario} />
                </>
              ) : null}
            </section>
          );
        })}
      </div>
    </section>
  );
}
