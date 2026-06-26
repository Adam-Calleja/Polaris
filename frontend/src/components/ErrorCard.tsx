import type { NormalizedApiError } from "../app/types";

function contentForError(error: NormalizedApiError): { title: string; body: string } {
  if (error.kind === "timeout") {
    return {
      title: "The request reached the API deadline before a full answer was returned.",
      body: "Try a shorter query, disable debug mode, or increase the request timeout.",
    };
  }
  if (error.kind === "network_error") {
    return {
      title: "The frontend could not reach the backend service.",
      body: "Check the API base URL and whether the backend container is running.",
    };
  }
  return {
    title: "The backend returned an error before the answer could be completed.",
    body: error.message,
  };
}

export function ErrorCard({ error }: { error: NormalizedApiError }) {
  const content = contentForError(error);

  return (
    <article className="surface-card answer-card answer-card--error">
      <h3 className="answer-card__title">{content.title}</h3>
      <div className="answer-card__body">{content.body}</div>
      {error.failure_class ? (
        <p className="answer-card__detail">Failure class: {error.failure_class}</p>
      ) : null}
    </article>
  );
}
