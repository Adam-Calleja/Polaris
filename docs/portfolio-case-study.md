# Polaris Portfolio Case Study

Polaris is an end-to-end Retrieval-Augmented Generation (RAG) assistant for
HPC service support. It was built as a final-year dissertation project to test
whether historical helpdesk tickets and official documentation could be turned
into a reusable support knowledge base without sacrificing source grounding,
validity awareness, or privacy.

## Summary

| Dimension | Detail |
| --- | --- |
| Domain | High-performance computing service support |
| Core problem | Answer operational support questions using evidence from tickets and documentation |
| Main contribution | Source-aware retrieval, validity-aware reranking, and a rigorous RAG evaluation protocol |
| Public demo strategy | Fully synthetic tickets and evaluation data, no private support content |
| Stack | Python, FastAPI, Qdrant, React, Docker Compose, MLflow, RAGAS |
| Evidence in repo | CI, backend tests, frontend tests, screenshots, methodology, experiment artifacts |

## Problem

HPC support teams accumulate large volumes of historical tickets, operational
documentation, and tacit service knowledge. That material is valuable, but it is
difficult to reuse safely:

- tickets can contain sensitive personal, operational, or credential-like data;
- documentation may be more authoritative than historical tickets for current
  policy or service state;
- many support answers are validity-sensitive because they depend on current
  systems, software versions, queues, partitions, and policies;
- generic semantic search can retrieve plausible but stale or non-authoritative
  evidence.

Polaris treats the task as a source-grounded RAG problem rather than a simple
chatbot problem. The system retrieves candidate evidence, reranks it with
source and validity metadata, constructs a grounded prompt, and returns an
answer with supporting context.

## Constraints

The project had several practical constraints that shaped the engineering
approach:

- **Private corpus:** the original ticket data could not be published.
- **Mixed evidence quality:** official docs, tickets, and external sources have
  different authority levels.
- **Validity-sensitive answers:** a correct answer often depends on whether a
  source is current for the service, software version, or policy.
- **Evaluation difficulty:** standard aggregate RAG scores hide important
  failure modes, so subgroup labels and repeated runs were needed.
- **Inspectable public release:** the repository needed to demonstrate the
  system without exposing real tickets or internal data.

## System Design

Polaris is structured as a full application rather than a notebook-only
prototype.

1. **Ingestion**
   - Jira tickets and HTML documentation are loaded and normalized.
   - Ticket and documentation text is converted into Markdown-like content.
   - Chunking is configurable so experiments can compare chunk sizes and
     overlaps.

2. **Indexing**
   - Chunks are embedded and stored in Qdrant.
   - A local document store preserves retrievable source metadata.
   - Source type, scope, and validity metadata are carried through the pipeline.

3. **Query Path**
   - A user query is embedded and matched against available evidence.
   - Retrieval can combine multiple source collections.
   - Reranking uses source-aware and validity-aware signals rather than relying
     only on semantic similarity.
   - A prompt is built with retrieved evidence and passed to an
     OpenAI-compatible LLM backend.

4. **Interfaces**
   - FastAPI exposes the query and readiness endpoints.
   - The React frontend provides assistant, evaluation, and system-inspection
     views.
   - Docker Compose wires together the API, frontend, Qdrant, embedding service,
     evaluation runner, and MLflow tracking.

## Key Engineering Decisions

### Synthetic public corpus

The public repository ships synthetic data under [`../data/synthetic/`](../data/synthetic/)
instead of anonymized real tickets. This keeps the repository inspectable while
avoiding the risk of leaking personal data, credentials, internal hostnames, or
private operational details.

### Source-aware retrieval

Historical tickets can be useful for examples, but official documentation is
often more authoritative for policy and current service state. Polaris carries
source metadata through retrieval and evaluation so the system can prefer the
right evidence class for a given query type.

### Validity-aware evaluation

The benchmark labels whether an answer is validity-sensitive. That matters in
HPC support because outdated advice can be worse than no advice. The evaluation
protocol therefore separates source ablations, validity-aware reranking, and
subgroup analysis instead of reporting one undifferentiated RAG score.

### Public tests and CI

The repository includes backend and frontend tests plus GitHub Actions CI. This
turns the public repo into inspectable software rather than just dissertation
material.

## Evaluation Approach

The original evaluation used a hand-annotated benchmark of 100 support queries
with labels for source needs, documentation scope, validity sensitivity, and
attachment dependence. Each condition was run three times, and results were
summarized with bootstrap confidence intervals.

The headline held-out scores were modest in absolute terms:

| Metric | Approximate held-out score |
| --- | --- |
| Context recall | 0.66 |
| Faithfulness | 0.50 |
| Factual correctness | 0.32 |

Those numbers are best read as evidence of a hard, validity-sensitive support
task rather than as a simple product-quality score. The main value of the work
is the evaluation protocol and the comparative design findings across
generator, chunking, source, and reranking conditions.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the detailed evaluation
methodology and [`../artifacts/experiments/`](../artifacts/experiments/) for
public aggregate artifacts.

## What the Repo Demonstrates

Polaris is intended to demonstrate several employability-relevant skills:

- turning an applied AI idea into a packaged, testable application;
- building a RAG system with realistic ingestion, retrieval, generation, and
  evaluation paths;
- handling sensitive data constraints with a privacy-aware public release;
- using Docker, FastAPI, React, Qdrant, and MLflow in one coherent system;
- designing evaluation around real failure modes rather than only benchmark
  averages;
- communicating technical tradeoffs clearly in public documentation.

## Code Paths Worth Reviewing

| Question | Start here |
| --- | --- |
| How is the API exposed? | [`../src/polaris_rag/app/api.py`](../src/polaris_rag/app/api.py) |
| How is a query processed end to end? | [`../src/polaris_rag/pipelines/rag_pipeline.py`](../src/polaris_rag/pipelines/rag_pipeline.py) |
| Where is retrieval/reranking implemented? | [`../src/polaris_rag/retrieval/`](../src/polaris_rag/retrieval/) |
| Where is generation handled? | [`../src/polaris_rag/generation/`](../src/polaris_rag/generation/) |
| Where is evaluation implemented? | [`../src/polaris_rag/evaluation/`](../src/polaris_rag/evaluation/) |
| What does the UI look like? | [`screenshots/ui-assistant.png`](screenshots/ui-assistant.png) and [`../frontend/src/`](../frontend/src/) |
| How is the stack wired together? | [`../docker-compose.yaml`](../docker-compose.yaml) |

## Tradeoffs and Limitations

- The public synthetic corpus is useful for inspection, but it is not a
  substitute for the original private corpus.
- The system is optimized for rigorous evaluation and traceability, not for
  minimal dependencies.
- Some workflows require external LLM, embedding, Jira, or MLflow configuration
  to run exactly as they did in the original project environment.
- The absolute benchmark scores show that validity-sensitive support RAG is
  difficult and requires careful human oversight.

## What I Would Improve Next

- Add a hosted public demo using only synthetic data.
- Split the Python dependency groups further so lightweight CI and local setup
  do not need the full evaluation and tracking stack.
- Add stricter linting and formatting gates once the codebase has a committed
  formatter policy.
- Expand the synthetic corpus to cover more edge cases from the benchmark
  labels.
- Add more explicit human-feedback loops for low-confidence or stale-source
  answers.

