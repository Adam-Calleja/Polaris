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
  const displayName = state.displayName.trim().split(/\s+/)[0] || "there";

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
        <div className="assistant-page__landing-inner assistant-hero surface-card">
          <div className="assistant-hero__copy">
            <div className="assistant-hero__eyebrow">Assistant</div>
            <h1 className="assistant-hero__title">Hello {displayName}</h1>
            <p className="assistant-hero__subtitle">How can I help you today?</p>
          </div>

          <div className="quick-prompt-grid quick-prompt-grid--hero">
            {quickPromptCards().map((prompt) => (
              <button
                className="quick-prompt-card quick-prompt-card--hero"
                key={prompt.title}
                onClick={() => {
                  void submitPrompt(prompt.prompt);
                }}
                type="button"
              >
                <span className="quick-prompt-card__icon" aria-hidden="true">
                  {prompt.icon}
                </span>
                <span className="quick-prompt-card__title">{prompt.title}</span>
                <span className="quick-prompt-card__body">{prompt.description}</span>
              </button>
            ))}
          </div>

          <div className="assistant-hero__composer">
            <PromptComposer
              disabled={submitting}
              onSubmit={submitPrompt}
              placeholder="Ask about tickets, docs, services, or queue issues..."
            />
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="assistant-page">
      <div className="assistant-workspace surface-card">
        <div className="assistant-workspace__header">
          <div className="assistant-workspace__header-copy">
            <div className="assistant-hero__eyebrow">Assistant session</div>
            <h2 className="assistant-workspace__title">Polaris response</h2>
          </div>
          <button className="secondary-button secondary-button--light" onClick={clearAssistantSession} type="button">
            Clear Assistant
          </button>
        </div>

        {latestUser ? (
          <div className="assistant-page__conversation-intro">
            <div className="assistant-page__section-label">Your question</div>
            <div className="assistant-page__query-pill">{latestUser.content}</div>
          </div>
        ) : null}

        <div className="assistant-page__grid assistant-page__grid--workspace">
          <div className="assistant-page__primary">
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
          </div>

          <div className="assistant-page__secondary">
            <DiagnosticsPanel error={latestAssistant?.error} response={latestAssistant?.response} />
          </div>
        </div>

        <div className="assistant-page__footer assistant-page__footer--workspace">
          <PromptComposer
            disabled={submitting}
            onSubmit={submitPrompt}
            placeholder="Enter a new question..."
          />
        </div>
      </div>
    </section>
  );
}
