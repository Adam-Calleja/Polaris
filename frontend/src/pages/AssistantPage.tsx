import { useState } from "react";
import { useAppState } from "../app/state";
import type { ApiClientConfig, AssistantReplyMessage, AssistantUserMessage, NormalizedApiError } from "../app/types";
import { ApiClientError, queryBackend } from "../app/api";
import {
  assistantViewState,
  buildManualQueryConstraints,
  createMessageId,
  latestExchange,
  nowIso,
  quickPromptCards,
} from "../app/utils";
import { AnswerCard } from "../components/AnswerCard";
import { DiagnosticsPanel } from "../components/DiagnosticsPanel";
import { ErrorCard } from "../components/ErrorCard";
import { LoadingLabel } from "../components/LoadingLabel";
import { PromptComposer } from "../components/PromptComposer";

function apiConfigFromState(state: ReturnType<typeof useAppState>["state"]): ApiClientConfig {
  return {
    baseUrl: state.apiBaseUrl,
    endpointPath: state.apiEndpointPath,
    timeoutS: state.timeoutS,
  };
}

export function AssistantPage() {
  const { state, dispatch } = useAppState();
  const [submitting, setSubmitting] = useState(false);

  function clearAssistantSession() {
    dispatch({ type: "clear-assistant-session" });
    setSubmitting(false);
  }

  async function submitPrompt(prompt: string): Promise<boolean> {
    if (submitting) {
      return false;
    }

    const trimmed = prompt.trim();
    if (!trimmed) {
      return false;
    }

    const createdAt = nowIso();
    const userMessage: AssistantUserMessage = {
      id: createMessageId(),
      role: "user",
      content: trimmed,
      createdAt,
    };
    const placeholderId = createMessageId();
    const placeholder: AssistantReplyMessage = {
      id: placeholderId,
      role: "assistant",
      content: "",
      createdAt,
      query: trimmed,
      response: null,
      error: null,
      pending: true,
    };

    dispatch({ type: "append-assistant-message", message: userMessage });
    dispatch({ type: "append-assistant-message", message: placeholder });
    setSubmitting(true);

    try {
      const response = await queryBackend(apiConfigFromState(state), trimmed, {
        queryConstraints: buildManualQueryConstraints(state.manualConstraints),
        includeEvaluationMetadata: state.debugMode,
        evaluationPolicy: state.debugMode ? "diagnostic" : "interactive",
      });
      dispatch({
        type: "replace-assistant-message",
        messageId: placeholderId,
        message: {
          ...placeholder,
          content: response.answer,
          response,
          pending: false,
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
        type: "replace-assistant-message",
        messageId: placeholderId,
        message: {
          ...placeholder,
          error: normalizedError,
          pending: false,
        },
      });
    } finally {
      setSubmitting(false);
    }

    return true;
  }

  const viewState = assistantViewState(state.assistantMessages);
  const { latestUser, latestAssistant, olderMessages } = latestExchange(state.assistantMessages);

  if (viewState === "landing") {
    return (
      <section className="assistant-page assistant-page--landing">
        <div className="assistant-page__landing-spacer" />
        <div className="assistant-page__landing-inner">
          <h2 className="section-label">Quick Prompts</h2>
          <div className="quick-prompt-grid">
            {quickPromptCards().map((prompt) => (
              <button
                className="quick-prompt-card"
                key={prompt}
                onClick={() => {
                  void submitPrompt(prompt);
                }}
                type="button"
              >
                <span>{prompt}</span>
                <span className="quick-prompt-card__arrow">›</span>
              </button>
            ))}
          </div>
          <PromptComposer
            disabled={submitting}
            onSubmit={submitPrompt}
            placeholder="Ask a question..."
          />
        </div>
      </section>
    );
  }

  return (
    <section className="assistant-page">
      {latestUser ? (
        <>
          <div className="assistant-page__user-label">{state.displayName}</div>
          <div className="assistant-page__query-pill">{latestUser.content}</div>
        </>
      ) : null}

      <div className="assistant-page__grid">
        <div className="assistant-page__primary">
          <div className="assistant-page__assistant-label">Polaris</div>
          {latestAssistant?.pending ? (
            <article className="surface-card answer-card answer-card--pending">
              <h3 className="answer-card__title">
                <LoadingLabel label="Retrieving evidence and drafting an answer" />
              </h3>
            </article>
          ) : latestAssistant?.response ? (
            <AnswerCard response={latestAssistant.response} />
          ) : latestAssistant?.error ? (
            <ErrorCard error={latestAssistant.error} />
          ) : (
            <article className="surface-card answer-card">
              <div className="answer-card__empty">No assistant content available.</div>
            </article>
          )}

          {olderMessages.length > 0 ? (
            <div className="history-panel">
              <button
                className="history-panel__toggle"
                onClick={() => dispatch({ type: "toggle-assistant-history" })}
                type="button"
              >
                {state.assistantHistoryOpen ? "Hide Conversation History" : "Conversation History"}
              </button>
              {state.assistantHistoryOpen ? (
                <div className="surface-card history-panel__content">
                  {olderMessages.map((message) => (
                    <article className="history-row" key={message.id}>
                      <div className="history-row__role">{message.role === "assistant" ? "Assistant" : "User"}</div>
                      <div className="history-row__body">{message.content}</div>
                    </article>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          <div className="assistant-page__footer">
            <PromptComposer
              disabled={submitting}
              onSubmit={submitPrompt}
              placeholder="Enter a new question..."
            />
            <div className="assistant-page__actions">
              <button className="secondary-button" onClick={clearAssistantSession} type="button">
                Clear Assistant
              </button>
            </div>
          </div>
        </div>

        <div className="assistant-page__secondary">
          <DiagnosticsPanel error={latestAssistant?.error} response={latestAssistant?.response} />
        </div>
      </div>
    </section>
  );
}
