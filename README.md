# Polaris RAG

Polaris is a Retrieval-Augmented Generation (RAG) system for HPC service support, built as a third-year project. It indexes Jira helpdesk tickets and documentation, then answers user questions using retrieval + LLM generation.

## Project Goals
- Turn historical support tickets into a searchable, reusable knowledge base.
- Improve answer quality for HPC support queries by grounding responses in source data.
- Provide a clean API and lightweight UI for experimentation and evaluation.

## Key Features
- Jira ticket ingestion with native Jira-to-Markdown conversion and configurable chunking.
- HTML documentation ingestion for user guides and knowledge base pages via HTML cleanup, Markdown normalization, and configurable chunking.
- Vector search via Qdrant with configurable retrieval strategies.
- OpenAI-compatible LLM and embedding backends (local or hosted).
- FastAPI service with a stable `/v1/query` endpoint.
- React SPA for assistant, evaluation, and system inspection workflows.
- RAGAS-based evaluation utilities and notebooks.
- MLflow experiment tracking, artifacts, and tracing.
- MLflow Prompt Registry-based runtime prompt versioning.

## Architecture Overview
- Ingestion: Jira and HTML sources are loaded, normalized to Markdown, and chunked.
- Indexing: Chunks are embedded and stored in Qdrant plus a local doc store.
- Query: User query is embedded, top-k chunks are retrieved, and a prompt is built.
- Generation: The configured LLM produces a grounded response.

## Quickstart (Docker Compose)
1. Create a `.env` file with required secrets and endpoints.
2. Build and start the stack.
3. Open the UI or call the API.

All commands below assume you are running from the repository root and using
the Docker Compose services as the execution environment.

```bash
# .env (example)
POLARIS_LLM_API_KEY=your_llm_key
GEMINI_API_KEY=your_gemini_key
JIRA_API_TOKEN=your_jira_token
```

```bash
docker compose up --build
```

To run the same stack with the Gaudi-specific overrides:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gaudi.yaml up --build
```

UI: [http://localhost:8500](http://localhost:8500)  
API: [http://localhost:8000](http://localhost:8000)
MLflow: [http://localhost:5000](http://localhost:5000)

The Dockerized frontend proxies API requests through the same origin at
`/api`, so normal browser use of the UI only needs the frontend port. Direct
API access on `8000` is still available for `curl`, notebooks, and debugging.

Check API dependency readiness with:

```bash
curl http://localhost:8000/ready
```

## Ingest Data
Jira tickets:
```bash
docker compose run --rm rag-api polaris-ingest-jira \
  -c /app/config/config.yaml \
  -s 2024-01-01 \
  -e 2025-01-01
```

HTML documentation:
```bash
docker compose run --rm rag-api polaris-ingest-html \
  -c /app/config/config.yaml \
  -p https://docs.example.org \
  --ingest-internal-links
```

`--ingest-internal-links` recursively follows linked HTML pages on the same
scheme/host within the supplied homepage path subtree. It does not discover
unlinked sections or rewrite invalid seed URLs.

The ingestion commands also accept chunking overrides for experiments:

```bash
docker compose run --rm rag-api polaris-ingest-jira \
  -c /app/config/config.yaml \
  --chunking-strategy markdown_token \
  --chunk-size-tokens 800 \
  --chunk-overlap-tokens 80
```

For repo-local development, the matching files under `scripts/` remain available
as thin compatibility wrappers around the packaged CLI entrypoints.

## Build Authority Registry
Stage 1 authority extraction is an offline artifact-generation step over the
local official docs corpus plus the official RCS services catalog. It reuses
the HTML loader, HTML preprocessing, and Markdown conversion pipeline, but does
not require Qdrant or the doc store.

```bash
python scripts/build_authority_registry.py \
  -c config/config.yaml \
  -p https://docs.hpc.cam.ac.uk/hpc/index.html \
  --ingest-internal-links
```

By default this writes:
- `data/authority/registry.local_official.v1.json`
- `data/authority/review_queue.local_official.v1.csv`

The JSON artifact contains the extracted authority entities, build metadata,
source URL list, extraction version, and summary counts. The CSV contains only
the rows that need manual audit, such as conflicting lifecycle statuses or
alias ambiguity. This stage does not yet change runtime retrieval behavior.

By default the build now combines:
- docs-derived entities from `docs.hpc.cam.ac.uk/hpc/...` with `source_scope=local_official`
- service-catalog entities from `https://www.hpc.cam.ac.uk/services` and its
  first-level service pages with `source_scope=local_official_services`

You can override the services landing page or opt out of the service-catalog
augmentation:

```bash
python scripts/build_authority_registry.py \
  -c config/config.yaml \
  -p https://docs.hpc.cam.ac.uk/hpc/index.html \
  --services-homepage https://www.hpc.cam.ac.uk/services \
  --ingest-internal-links

python scripts/build_authority_registry.py \
  -c config/config.yaml \
  -p https://docs.hpc.cam.ac.uk/hpc/index.html \
  --ingest-internal-links \
  --skip-services-catalog
```

The build metadata now records:
- `build.homepage` and `build.docs_homepage` for the docs corpus
- `build.services_homepage` for the service catalog
- `build.service_catalog_included`
- summary counts by entity type, status, and source scope

## Build External Authority Registry
Stage 6 adds a seeded external-official source register plus a dedicated build
path that preserves local and external authority scopes separately.

```bash
python scripts/build_external_authority_registry.py \
  -c config/config.yaml \
  --source-register-file data/authority/source_register.external_v1.yaml \
  --local-registry-file data/authority/registry.local_official.v1.json
```

This writes:
- `data/authority/registry.external_official.v1.json`
- `data/authority/review_queue.external_official.v1.csv`
- `data/authority/registry.official_combined.v1.json`
- `data/authority/review_queue.official_combined.v1.csv`

The combined registry is now the default runtime metadata-enrichment artifact.
If the external crawl has not yet been run successfully, the repo may carry a
local-only placeholder combined registry at the same path until the real
external build is materialized.

## Ingest External Official Docs
Registered external docs can be ingested reproducibly into their own vector
collection without activating them in the default runtime source set.

```bash
python scripts/ingest_external_docs.py \
  -c config/config.yaml \
  --source-register-file data/authority/source_register.external_v1.yaml
```

## Configuration
Configuration is split into shared and environment-specific files:
- `config/config.base.yaml`: shared retrieval, ingestion, evaluation, and MLflow settings.
- `config/config.local.yaml`: local/default model and embedder settings.
- `config/config.gaudi.yaml`: Gaudi-specific model, embedder, and eval-tuning overrides.
- `config/config.yaml`: local default alias that extends `config/config.local.yaml`.

Key sections include:
- `generator_llm` and `evaluator_llm` for model selection and parameters.
- `embedder` for embeddings endpoint configuration.
- `vector_stores` and `storage_context` for per-source persistence.
- `ingestion.conversion` for source-to-Markdown conversion engines.
- `ingestion.chunking` for shared defaults and per-source chunking strategies.
- `prompts` and `prompt_name` for prompt templates.
- `mlflow` for tracking, tracing, and prompt-registry settings.

The local Docker Compose stack uses `/app/config/config.yaml` by default. The
Gaudi override file switches runtime config to `/app/config/config.gaudi.yaml`.

Environment variables used by the stack:
- `POLARIS_LLM_API_KEY`: API key for the configured LLM or embeddings provider.
- `GEMINI_API_KEY`: API key used when `generator_llm.api_key` is configured as `${GEMINI_API_KEY}`.
- `JIRA_API_TOKEN`: Jira API token for ticket ingestion.
- `HF_TOKEN`: Hugging Face token used by the embedding service, including the Gaudi TEI image.
- `EMBED_API_BASE`: Base URL for the embeddings service.
- `POLARIS_CONFIG`: Path to the runtime config inside the API/eval containers.
- `POLARIS_FEEDBACK_LOG_PATH`: JSONL feedback log path used by the API UI endpoints.
- `POLARIS_UI_CORS_ALLOWED_ORIGINS`: Comma-separated browser origins allowed to call the API from the frontend.
- `POLARIS_UI_API_BASE_URL`: Frontend API base path injected into the frontend container. For the Dockerized frontend this should normally be `/api` so nginx proxies requests to `rag-api`.
- `POLARIS_UI_API_ENDPOINT_PATH`: Query endpoint path injected into the frontend container.
- `POLARIS_UI_API_TIMEOUT_S`: Default browser request timeout injected into the frontend container.
- `POLARIS_DISPLAY_NAME`: Optional display name shown in the frontend assistant view.
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI inside the containers. Docker
  Compose configures this as `http://mlflow:5000` for the API and eval services.

## Testing
Run tests from the repository root unless noted otherwise.

### Python Test Suite

For local Python test runs, install the package in editable mode with the
optional extras used by the API, retrieval, evaluation, and tracking paths:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[eval,llm,api,retrieval,tracking]"
pip install pytest
```

Then run the backend test suite with:

```bash
pytest -q
```

If you are running directly from the source tree without an editable install,
use the `src/` layout explicitly:

```bash
PYTHONPATH=src pytest -q
```

Some tests start temporary loopback HTTP servers on `127.0.0.1`. In sandboxed
or restricted environments that block local socket binding, those tests can
fail even when the code is otherwise correct.

### Frontend Test Suite

The React frontend has its own Vitest suite under `frontend/`:

```bash
cd frontend
npm test
```

## Evaluation
Evaluation utilities live in `src/polaris_rag/evaluation`. There are notebooks for dataset creation and analysis:
- `create_evaluation_dataset.ipynb`
- `support_ticket_analysis.ipynb`

You can run the modern RAGAS evaluation pipeline through the `eval` service:

```bash
docker compose run --rm eval
```

When the stack is already running, prefer `--no-deps` so Compose does not
recreate dependency containers unnecessarily:

```bash
docker compose run --no-deps --rm eval
```

By default, evaluation row preparation uses API mode for production-like
end-to-end evaluation. The `eval` service in Docker Compose runs this by default:

```bash
docker compose run --rm eval
```

If evaluation prep fails before the first request completes, verify the API is
fully ready before running eval:

```bash
curl http://localhost:8000/ready
```

You can still configure generation mode explicitly:

```yaml
evaluation:
  generation:
    mode: "api"
    api_url: "http://rag-api:8000/v1/query"
    retries:
      max_attempts: 1
      initial_backoff_seconds: 1.0
      max_backoff_seconds: 8.0
      jitter_seconds: 0.25
      retry_on_empty_response: true
```

You can also override this at runtime:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --generation-mode api \
  --query-api-url http://rag-api:8000/v1/query
```

Evaluation remains file-driven. The evaluator reads the dataset from
`--dataset-path` or `evaluation.dataset.input_path`, and optionally reuses
prepared rows from `--prepared-path` / `evaluation.dataset.prepared_path`.
Benchmark annotations can be supplied via `--annotations-file` or
`evaluation.dataset.annotations_path`; when present, the validated annotation
payload is joined into each row's metadata under `benchmark_annotation`.
MLflow dataset objects are logged for lineage, but eval does not resolve its
runtime input back out of MLflow.

To evaluate a specific split explicitly:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --dataset-path /app/data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --prepared-path /app/data/test/prepared_test_rows_final.json
```

To prepare rows without running RAGAS scoring:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --dataset-path /app/data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --prepared-path /app/data/test/prepared_test_rows_final.json \
  --prepare-only
```

### Reranker Switching During Evaluation

Reranker choice is config-driven. Change `retriever.rerank` in the config you
pass to `polaris-eval` rather than using a one-off CLI override. Stage 4 adds:

```yaml
retriever:
  rerank:
    type: "rrf"  # or "validity_aware"
    rrf_k: 60
    trace_enabled: true
    semantic_base:
      type: "rrf"
      rrf_k: 60
    weights_path: "config/weights/validity_reranker.dev_v3.yaml"
```

Important:
- Changing reranker settings changes retrieved context, so prepared rows must
  be regenerated for each reranker condition.
- Polaris now stores a reranker fingerprint in prepared rows and will refuse to
  reuse rows generated with a different reranker configuration.
- In `api` generation mode, start `rag-api` with the same config file used by
  `polaris-eval`; the eval client assumes the API is running the matching
  reranker configuration.

To tune the validity-aware reranker on a dev split:

```bash
python scripts/tune_validity_reranker.py \
  -c config/config.yaml \
  --dataset-path data/test/ragas_one_hop_eval_dataset_final.dev.jsonl \
  --output-path config/weights/validity_reranker.dev_v3.yaml
```

To evaluate a split while preserving benchmark subgroup labels in row metadata:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --dataset-path /app/data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --annotations-file /app/data/test/benchmark_annotations_final.csv
```

To control MLflow from CLI:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --mlflow \
  --mlflow-experiment polaris-rag-evals \
  --mlflow-run-name eval-baseline
```

To capture raw evaluator prompts and responses for RAGAS metric debugging:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --mlflow \
  --trace-evaluator-llm
```

You can also enable this in config with:

```yaml
evaluation:
  tracing:
    evaluator_llm: true
```

Evaluator traces are only emitted when MLflow tracing is enabled. They appear
under the `ragas_evaluation` stage in the MLflow trace experiment and include
the evaluator prompt, request kwargs, and raw provider response for each judged
metric call.

To enable retries during dataset preparation from CLI:

```bash
docker compose run --no-deps --rm eval \
  polaris-eval -c /app/config/config.yaml \
  --generation-max-attempts 3 \
  --generation-retry-initial-backoff 1.0 \
  --generation-retry-max-backoff 8.0 \
  --generation-retry-jitter 0.25 \
  --generation-retry-on-empty-response
```

Key outputs are written under `evaluation.output_dir` (or `--output-dir`):
- `scores.csv`
- `scores.parquet`
- `summary.json`
- `run_manifest.json`

Additional MLflow-tracked artifacts include:
- `dataset_manifest.json`
- `prepared_rows.json`
- `config_snapshot.json`
- `env_snapshot.json`

### Create Dev/Test Splits

Use `polaris-create-dev-test-sets` to create local dev/test dataset files.
These files are the executable source of truth for evaluation; the same command
can also register the splits as MLflow dataset inputs for discoverability and
lineage. The matching file under `scripts/` is a compatibility wrapper when
you want `python scripts/create_dev_test_sets.py ...` from the repo root.

Explicit test IDs:

```bash
docker compose run --rm eval \
  polaris-create-dev-test-sets \
  --dataset-file /app/data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --test-samples-file /app/data/test/eval_ticket_keys.txt
```

Stratified split from category mappings:

```bash
docker compose run --rm eval \
  polaris-create-dev-test-sets \
  --dataset-file /app/data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --categories-file /app/data/test/eval_categories.json \
  --test-size 17 \
  --random-state 42
```

Recommended multilabel split from verified benchmark annotations:

```bash
python scripts/create_dev_test_sets.py \
  --dataset-file data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --annotations-file data/test/benchmark_annotations_final.csv \
  --test-fraction 0.30 \
  --random-state 42
```

With MLflow lineage logging enabled:

```bash
docker compose run --rm eval \
  polaris-create-dev-test-sets \
  --dataset-file /app/data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --categories-file /app/data/test/eval_categories.json \
  --test-size 17 \
  --random-state 42 \
  --mlflow \
  --config-file /app/config/config.yaml
```

Notes:
- Stratified splitting uses `scikit-learn` and only stratifies categories with
  more than one sample.
- Singleton or uncategorized samples remain in the dev split.
- Annotation-driven splitting uses the verified benchmark labels to build a
  multilabel stratification profile and also writes a frozen `test_ids` file
  plus a split audit report.
- Dev and test are logged to MLflow as separate dataset objects with
  `validation` and `testing` contexts.
- Prepared rows and predictions are kept as eval run artifacts rather than
  canonical MLflow datasets.
- The `eval` service mounts `./data` to `/app/data`, so generated split files
  written under `/app/data/...` are persisted back to the host repository.

### Benchmark Annotation Workflow

Stage-3 benchmark annotation is kept separate from the canonical benchmark
JSONL. The annotation CSV is keyed by benchmark `id` and records the evidence
class, authoritative doc scope, validity sensitivity, attachment dependence,
and supporting query-type labels needed for subgroup analysis.

The compatibility wrapper scripts under `scripts/` mirror the packaged CLIs if
you want to run them directly from the repo root.

To scaffold a seeded annotation CSV from the benchmark plus the current split
files:

```bash
python scripts/benchmark_annotations.py scaffold \
  --dataset-file data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --dev-dataset-file data/test/ragas_one_hop_eval_dataset_final.dev.jsonl \
  --test-dataset-file data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --legacy-audit-file data/test/gold_standard_v1_audit_labels.csv \
  --output-file data/test/benchmark_annotations_final.csv
```

To refresh split labels in an existing verified annotation CSV after regenerating
the dev/test datasets:

```bash
python scripts/benchmark_annotations.py validate \
  --dataset-file data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --dev-dataset-file data/test/ragas_one_hop_eval_dataset_final.dev.jsonl \
  --test-dataset-file data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --annotations-file data/test/benchmark_annotations_final.csv \
  --require-verified \
  --regenerate-splits \
  --output-file data/test/benchmark_annotations_final.csv
```

After manual review, validate the file and require all rows to be marked
`review_status=verified`:

```bash
python scripts/benchmark_annotations.py validate \
  --dataset-file data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --dev-dataset-file data/test/ragas_one_hop_eval_dataset_final.dev.jsonl \
  --test-dataset-file data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --annotations-file data/test/benchmark_annotations_final.csv \
  --require-verified
```

The stage-3 label definitions and tie-break rules live in
`data/test/benchmark_annotation_codebook_v1.md`.

### Experiment 1: Benchmark Characterisation

Use `polaris-benchmark-analysis` or the matching wrapper script to generate the
experiment-1 subgroup composition artifacts:

```bash
python scripts/benchmark_analysis.py \
  --dataset-file data/test/ragas_one_hop_eval_dataset_final.jsonl \
  --dev-dataset-file data/test/ragas_one_hop_eval_dataset_final.dev.jsonl \
  --test-dataset-file data/test/ragas_one_hop_eval_dataset_final.test.jsonl \
  --annotations-file data/test/benchmark_annotations_final.csv \
  --output-dir data/test/benchmark_analysis_final
```

This writes:
- `composition_counts.csv`
- `composition_combinations.csv`
- `composition_summary.json`
- `composition_summary.md`
- `composition_figure.png`
- `composition_figure.svg`

### Manifest-Driven Experiment Automation

For the dissertation protocol, the repo now includes a thin automation layer
under `scripts/experiments/` plus a starter manifest at
`experiments/protocol.template.yaml`.

Render one generated config overlay:

```bash
python scripts/experiments/render_config.py \
  --manifest experiments/protocol.template.yaml \
  --stage stage4_source_ablation \
  --condition naive_combined
```

Run one stage from the manifest:

```bash
python scripts/experiments/run_stage.py \
  --manifest experiments/protocol.template.yaml \
  --stage stage4_source_ablation
```

Preview commands without executing them:

```bash
python scripts/experiments/run_stage.py \
  --manifest experiments/protocol.template.yaml \
  --stage stage0b_docs_chunking \
  --dry-run
```

Summarize completed runs for one stage:

```bash
python scripts/experiments/summarize_stage.py \
  --manifest experiments/protocol.template.yaml \
  --stage stage4_source_ablation
```

Generate dissertation-ready comparison sheets from one selected repeat per
condition:

```bash
python scripts/experiments/summarize_stage.py \
  --manifest experiments/protocol.template.yaml \
  --stage stage7_prompt_ablation \
  --run-comparison
```

The automation layer is intentionally thin:
- existing Polaris CLIs still do the real work for ingestion, evaluation, split creation, tuning, and benchmark analysis
- the manifest controls stage structure, config overlays, output locations, and repeats
- prompt ablation can disable the MLflow prompt registry inside the manifest so `prompt_name` changes take effect

The starter manifest is deliberately readable rather than exhaustive. Expand the
larger Stage 0B and Stage 1 grids by duplicating condition blocks so they match
your final protocol exactly.

## Prompt Registry Workflow
Runtime prompt loading is registry-first (`mlflow.prompt_registry.enabled: true`).
Register/update prompt versions from repo templates, then set alias:

```bash
docker compose run --rm rag-api \
  polaris-register-prompts-mlflow -c /app/config/config.yaml
```

Typical promotion flow:
1. Register new prompt version from local template.
2. Assign/update alias (`prod`, `staging`, etc.).
3. Restart API/eval services so runtime resolves the selected alias.

## MLflow Logging Coverage
Polaris logs the four required experiment categories:
1. Configuration parameters: flattened config + runtime/eval settings.
2. Metrics: RAGAS quality metrics and system metrics (prep/eval latency, throughput, failures, concurrency).
3. Artifacts: scores, summaries, run manifests, prepared rows, config/env snapshots.
4. Traces: `/v1/query` request trace plus pipeline spans (retrieve, prompt render, generation), linked to eval runs via request headers.

Dataset lineage is split across two layers:
- Split creation logs local dev/test files as MLflow dataset inputs for lineage.
- Each `polaris-eval` run logs the local benchmark file it consumed as an
  MLflow dataset input.

This means MLflow shows which benchmark split a run used, while the runtime
execution path still depends only on local dataset files.

## Project Structure
- `src/polaris_rag`: Core pipeline, retrieval, generation, and evaluation code.
- `embed_server`: Embedding service used by the `OpenAILike` embedder.
- `scripts`: Ingestion entrypoints for Jira and HTML.
- `config`: YAML configuration for LLMs, retrieval, and storage.
- `data`: Sample and evaluation datasets.
- `docker-compose.yaml` and `Dockerfile.*`: Deployment and service setup.

## Data Handling And Privacy
Support tickets can contain sensitive information. Treat all data as confidential, avoid committing PII, and rotate credentials if they have been exposed.

## License
Proprietary. See `pyproject.toml`.
