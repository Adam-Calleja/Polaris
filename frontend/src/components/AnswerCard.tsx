import type { QueryResponseData } from "../app/types";
import { parseAnswerSections } from "../app/utils";

function badgeLabel(code: string): string {
  if (code === "grounded") {
    return "Grounded";
  }
  if (code === "limited_evidence") {
    return "Limited Evidence";
  }
  return "No Evidence";
}

export function AnswerCard({ response }: { response: QueryResponseData }) {
  const sections = parseAnswerSections(response.answer);

  return (
    <article className="surface-card answer-card">
      <div className="answer-card__status-row">
        <span className={`answer-card__badge answer-card__badge--${response.answer_status.code}`}>
          {badgeLabel(response.answer_status.code)}
        </span>
        <span className="answer-card__badge answer-card__badge--neutral">
          {response.context.length} evidence {response.context.length === 1 ? "chunk" : "chunks"}
        </span>
      </div>
      <p className="answer-card__detail">{response.answer_status.detail}</p>
      {sections.length > 0 ? (
        sections.map((section) => (
          <section className="answer-card__section" key={`${section.key}-${section.heading}`}>
            <h3 className="answer-card__title">{section.heading}</h3>
            <div className="answer-card__body">{section.body}</div>
          </section>
        ))
      ) : (
        <div className="answer-card__empty">No answer text was returned.</div>
      )}
    </article>
  );
}
