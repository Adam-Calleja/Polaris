# Evaluation Methodology

This document summarises how Polaris was designed and evaluated. It distils the
project's dissertation; the dissertation itself and the raw experiment runs are
**not** included in this public repository (they were produced on a private
real-ticket corpus). The aggregate results tables referenced here live in
[`artifacts/experiments/`](artifacts/experiments/).

> All data shipped in this repo is synthetic (see `data/synthetic/`). The
> numbers below are from the original evaluation on the private real corpus.

## The system

Polaris answers HPC-support questions by retrieving over two source types —
**historical helpdesk tickets** and **official documentation** — and generating
a grounded answer with an LLM. Two design choices distinguish it from a naive
RAG baseline:

- **Source-aware authority reranking.** Retrieved chunks carry authority and
  validity metadata (which source they came from, how authoritative it is for a
  given question). Reranking favours the evidence class a question actually
  needs, rather than raw vector similarity alone.
- **Validity-aware answering.** Many support answers depend on *current*
  systems, partitions, software versions, and policy. The benchmark and the
  reranker explicitly model this validity-sensitivity.

## The benchmark

A hand-annotated benchmark of **100 queries** (70 dev / 30 test). Each item is
labelled along four axes (see `data/synthetic/eval/` for the schema):

| Label | Values | Meaning |
|---|---|---|
| `source_needed` | docs / tickets / both | minimum evidence class for a correct answer |
| `docs_scope_needed` | local_official / external_official / local_and_external / none | which documentation is authoritative |
| `validity_sensitive` | yes / no | correctness depends on current systems/versions/policy |
| `attachment_dependent` | yes / no | answer needs information in an attachment |

Benchmark composition (real corpus): **92/100** items are validity-sensitive,
**56** need local official docs, **21** are software-version queries, and **5**
are attachment-dependent. Topic mix is dominated by Schedulers/Runtime (24%),
Storage (20%), Access/Identity (19%), and Platforms (18%).

## Metrics

Answers are scored with **RAGAS**:

- `context_recall` — did retrieval surface the evidence needed?
- `faithfulness` — is the answer grounded in the retrieved context?
- `factual_correctness` — is the answer correct against the reference?

Each condition is run **3 times**; results are reported as means with **95%
percentile bootstrap confidence intervals** (50,000 resamples over held-out
queries) so differences between conditions are statistically meaningful rather
than single-run noise.

## Experiment protocol

The evaluation is staged so each design decision is isolated:

1. **Generator selection** — Llama-3.3-70B vs Mistral-Large vs Mixtral-8x22B.
2. **Chunking studies** — ticket and document chunk sizes / overlaps.
3. **Combined sanity** — docs + tickets together with the selected chunking.
4. **Source ablation** — docs-only vs tickets-only vs combined.
5. **Validity-aware reranking** — effect of the validity-sensitive reranker.
6. **External-docs ablation** — contribution of external official documentation.

## Headline results

On the held-out test set, the combined system reached approximately
**context recall 0.66, faithfulness 0.50, factual correctness 0.32** (mean of 3
runs). These are modest absolute scores, which is expected: the benchmark is
deliberately hard and validity-sensitive, and `factual_correctness` against
gold references is a strict metric. The value of the work is the **evaluation
methodology, the source-aware/validity-aware design, and the comparative
findings** across stages — not a single headline number.

Full per-stage leaderboards, condition aggregates, and bootstrap confidence
intervals are in [`artifacts/experiments/`](artifacts/experiments/).
