# Benchmark Annotation Codebook v1

This file defines the stage-3 benchmark labels used for subgroup characterisation and later evaluation ablations.

## Core Principles

- `source_needed` captures the minimum evidence class needed for a correct answer.
- `docs_scope_needed` captures which class of official documentation is authoritative when docs are needed.
- Labels describe the task, not the current retrieval result and not the current vector-store contents.
- `split` and `summary` are scaffolded audit columns copied from the benchmark, not research labels.
- `review_status` tracks annotation maturity; rows should be `verified` before dissertation experiments are run.

## Column Definitions

`id`
- Canonical benchmark identifier.

`split`
- Current benchmark membership: `dev` or `test`.
- This must match the maintained split files.

`summary`
- Human-readable benchmark summary copied from the benchmark row.

`source_needed`
- `docs`: the query can be answered correctly from documentary sources alone.
- `tickets`: the query requires historical ticket context or support-case memory.
- `both`: the query requires both documentary authority and ticket-specific context.

`docs_scope_needed`
- `local_official`: local CSD3 / service documentation is the authoritative doc source.
- `external_official`: external official documentation is the authoritative doc source.
- `local_and_external`: both local and external official docs are needed.
- `none`: no documentary source is required; only valid when `source_needed=tickets`.

`validity_sensitive`
- `yes`: correctness depends on current system, partition, service, workflow, or version compatibility.
- `no`: the answer is stable enough that these constraints are not central.

`attachment_dependent`
- `yes`: visible text alone is insufficient and unseen attachments, screenshots, logs, or uploaded documents materially determine the correct handling.
- `no`: the visible ticket text is sufficient for a correct first response.

`query_type`
- `local_operational`: local service behaviour, policy, access, storage, scheduling, or workflow truth.
- `software_version`: package/module/toolchain/software compatibility or version-sensitive build/runtime issue.
- `general_how_to`: stable procedural guidance that is less tied to local operational change.

`version_sensitive`
- `yes`: correctness depends on software or toolchain version.
- `no`: no material software-version dependency.

`system_scope_required`
- `yes`: correctness depends on the target system, service, partition, or platform.
- `no`: the answer does not materially depend on a specific system scope.

`review_status`
- `seeded`: scaffolded or bootstrapped but not manually reviewed.
- `verified`: manually reviewed under this codebook.

`notes`
- Optional short rationale for unusual or ambiguous cases.

## Validation Rules

- Every benchmark row must have exactly one annotation row.
- `split` must match the maintained dev/test split files.
- `summary` must match the benchmark summary.
- `docs_scope_needed=none` is only valid when `source_needed=tickets`.
- Queries with `source_needed=docs` or `source_needed=both` must not use `docs_scope_needed=none`.
- Final experiment outputs should only use rows with `review_status=verified`.
