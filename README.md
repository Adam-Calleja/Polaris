# Polaris RAG

Polaris is a Retrieval-Augmented Generation (RAG) system for HPC service support, built as a third-year project. It indexes Jira helpdesk tickets and documentation, then answers user questions using retrieval + LLM generation.

## Project Goals
- Turn historical support tickets into a searchable, reusable knowledge base.
- Improve answer quality for HPC support queries by grounding responses in source data.
- Provide a clean API and lightweight UI for experimentation and evaluation.

## Key Features
- Jira ticket ingestion with ADF-to-text conversion and chunking.
- HTML documentation ingestion for user guides and knowledge base pages.
- Vector search via Qdrant with configurable retrieval strategies.
- OpenAI-compatible LLM and embedding backends (local or hosted).
- FastAPI service with a stable `/v1/query` endpoint.
- Streamlit chat UI for interactive demos.
- RAGAS-based evaluation utilities and notebooks.
- MLflow experiment tracking, artifacts, and tracing.
- MLflow Prompt Registry-based runtime prompt versioning.

## Architecture Overview
- Ingestion: Jira and HTML sources are loaded, cleaned, and chunked.
- Indexing: Chunks are embedded and stored in Qdrant plus a local doc store.
- Query: User query is embedded, top-k chunks are retrieved, and a prompt is built.
- Generation: The configured LLM produces a grounded response.

## Quickstart (Docker Compose)
1. Create a `.env` file with required secrets and endpoints.
2. Build and start the stack.
3. Open the UI or call the API.

```bash
# .env (example)
POLARIS_LLM_API_KEY=your_llm_key
GEMINI_API_KEY=your_gemini_key
JIRA_API_TOKEN=your_jira_token
MLFLOW_TRACKING_URI=http://localhost:5000
```

```bash
docker compose up --build
```

UI: [http://localhost:8501](http://localhost:8501)  
API: [http://localhost:8000](http://localhost:8000)
MLflow: [http://localhost:5000](http://localhost:5000)

## Local Setup
Python 3.11+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[api,ingestion,eval,tracking]"
```

## Run The API
```bash
export POLARIS_CONFIG=config/config.yaml
uvicorn polaris_rag.app.api:app --reload --host 0.0.0.0 --port 8000
```

## Run The Streamlit UI
```bash
export POLARIS_API_BASE_URL=http://localhost:8000
streamlit run src/polaris_rag/streamlit/polaris_interface.py
```

## Ingest Data
Jira tickets:
```bash
python scripts/ingest_jira_tickets.py \
  -c config/config.yaml \
  --source tickets \
  -s 2024-01-01 \
  -e 2025-01-01
```

HTML documentation:
```bash
python scripts/ingest_html_documents.py \
  -c config/config.yaml \
  --source docs \
  -p https://docs.example.org \
  --ingest-internal-links
```

## Configuration
Configuration lives in `config/config.yaml`. Key sections include:
- `generator_llm` and `evaluator_llm` for model selection and parameters.
- `embedder` for embeddings endpoint configuration.
- `vector_stores` and `storage_context` for per-source persistence.
- `prompts` and `prompt_name` for prompt templates.
- `mlflow` for tracking, tracing, and prompt-registry settings.

Environment variables used by the stack:
- `POLARIS_LLM_API_KEY`: API key for the configured LLM or embeddings provider.
- `GEMINI_API_KEY`: API key used when `generator_llm.api_key` is configured as `${GEMINI_API_KEY}`.
- `JIRA_API_TOKEN`: Jira API token for ticket ingestion.
- `EMBED_API_BASE`: Base URL for the embeddings service.
- `POLARIS_CONFIG`: Path to the runtime config used by the API.
- `POLARIS_API_BASE_URL`: API base URL used by Streamlit.
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI.

## Evaluation
Evaluation utilities live in `src/polaris_rag/evaluation`. There are notebooks for dataset creation and analysis:
- `create_evaluation_dataset.ipynb`
- `support_ticket_analysis.ipynb`

You can also run the modern RAGAS evaluation pipeline from CLI:

```bash
polaris-eval -c config/config.yaml
```

By default, evaluation row preparation uses API mode for production-like
end-to-end evaluation. The `eval` service in Docker Compose runs this by default:

```bash
docker compose run --rm eval
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
polaris-eval -c config/config.yaml --generation-mode api --query-api-url http://127.0.0.1:8000/v1/query
```

Evaluation remains file-driven. The evaluator reads the dataset from
`--dataset-path` or `evaluation.dataset.input_path`, and optionally reuses
prepared rows from `--prepared-path` / `evaluation.dataset.prepared_path`.
MLflow dataset objects are logged for lineage, but eval does not resolve its
runtime input back out of MLflow.

To evaluate a specific split explicitly:

```bash
polaris-eval -c config/config.yaml \
  --dataset-path /path/to/ragas_one_hop_eval_dataset_v1.test.jsonl \
  --prepared-path /path/to/prepared_test_rows.json
```

To control MLflow from CLI:

```bash
polaris-eval -c config/config.yaml --mlflow --mlflow-experiment polaris-rag-evals --mlflow-run-name eval-baseline
```

To enable retries during dataset preparation from CLI:

```bash
polaris-eval -c config/config.yaml \
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

Use `polaris-create-dev-test-sets` (or `scripts/create_dev_test_sets.py`) to
create local dev/test dataset files. These files are the executable source of
truth for evaluation; the same command can also register the splits as MLflow
dataset inputs for discoverability and lineage.

Explicit test IDs:

```bash
polaris-create-dev-test-sets \
  --dataset-file data/test/ragas_one_hop_eval_dataset_v1.jsonl \
  --test-samples-file data/test/eval_ticket_keys.txt
```

Stratified split from category mappings:

```bash
polaris-create-dev-test-sets \
  --dataset-file data/test/ragas_one_hop_eval_dataset_v1.jsonl \
  --categories-file data/test/eval_categories.json \
  --test-size 17 \
  --random-state 42
```

With MLflow lineage logging enabled:

```bash
polaris-create-dev-test-sets \
  --dataset-file data/test/ragas_one_hop_eval_dataset_v1.jsonl \
  --categories-file data/test/eval_categories.json \
  --test-size 17 \
  --random-state 42 \
  --mlflow \
  --config-file config/config.yaml
```

Notes:
- Stratified splitting uses `scikit-learn` and only stratifies categories with
  more than one sample.
- Singleton or uncategorized samples remain in the dev split.
- Dev and test are logged to MLflow as separate dataset objects with
  `validation` and `testing` contexts.
- Prepared rows and predictions are kept as eval run artifacts rather than
  canonical MLflow datasets.

## Prompt Registry Workflow
Runtime prompt loading is registry-first (`mlflow.prompt_registry.enabled: true`).
Register/update prompt versions from repo templates, then set alias:

```bash
polaris-register-prompts-mlflow -c config/config.yaml
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
