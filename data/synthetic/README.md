# Synthetic support-ticket corpus

Every ticket in `tickets.jsonl` is **fully synthetic** — fabricated by
`scripts/generate_synthetic_corpus.py`. No real support ticket, user, email,
or ticket identifier is represented anywhere in this corpus.

- **Identities** are invented (`*@example.org`); **keys** use a `DEMO-` prefix.
- **Scenarios** are made up. Only genuinely *public* facts (CSD3 partition
  names, public software packages) are used, to keep the tickets realistic.
- The **category mix mirrors the real benchmark's topic distribution**, so the
  corpus is representative without containing any real content.
- Tickets follow the same Jira issue schema the ingestion pipeline consumes, so
  they are drop-in ingestable (validated against `build_jira_ticket_text`).

## Evaluation set

`eval/ragas_one_hop_synthetic.jsonl` is a small, fully synthetic ragas one-hop
evaluation set over this corpus (queries + gold answers + the benchmark label
axes: `source_needed`, `docs_scope_needed`, `validity_sensitive`,
`attachment_dependent`). Its label distribution mirrors the real benchmark.

## Regenerating the demo data and stores

1. **Corpus + eval set** (no special environment needed):

   ```bash
   python scripts/generate_synthetic_corpus.py --count 120 --seed 42
   python scripts/generate_synthetic_eval.py
   ```

2. **Vector stores** (requires the runtime services). Building the persisted
   Qdrant + LlamaIndex stores needs the embedding services and Qdrant, so it
   runs in the configured environment. Ingest this corpus offline with the
   `--tickets-file` option (no Jira access required):

   ```bash
   docker compose up -d qdrant embed
   python -m polaris_rag.cli.ingest_jira_tickets \
       --tickets-file data/synthetic/tickets.jsonl --source tickets
   ```

   The load → preprocess → chunk stages are verified to run on this corpus; the
   embed → index stage runs once the services above are up.

See each directory's `MANIFEST.json` for exact counts and distributions.
