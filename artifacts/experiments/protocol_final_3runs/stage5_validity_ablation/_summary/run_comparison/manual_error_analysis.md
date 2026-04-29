# Stage 5 Manual Error Analysis

## Scope

This note reviews the Stage 5 validity-ablation comparison generated from:

- `artifacts/experiments/protocol_final_3runs/stage5_validity_ablation/freshness_only/run_03`
- `artifacts/experiments/protocol_final_3runs/stage5_validity_ablation/naive_combined/run_03`
- `artifacts/experiments/protocol_final_3runs/stage5_validity_ablation/source_aware/run_03`
- `artifacts/experiments/protocol_final_3runs/stage5_validity_ablation/validity_aware/run_03`

The analysis uses the paired comparison artifacts in this directory:

- `combined_analysis_rows.jsonl`
- `query_review_sheet.csv`
- `manual_eval_sheet.csv`
- `manual_eval_key.csv`

The goal is to explain what each reranking condition changes on the same 30 test queries, especially whether the full validity-aware reranker improves source routing and answer behaviour relative to the weaker ablations.

## What Was Ablated

Stage 5 compares four retrieval/reranking conditions over the same docs+tickets corpus:

1. `naive_combined`
   Baseline combined retrieval with RRF reranking.

2. `freshness_only`
   Validity-aware reranker with only the freshness feature enabled.

3. `source_aware`
   Validity-aware reranker with only the authority feature enabled.

4. `validity_aware`
   Full frozen validity-aware reranker.

In practice, the source-routing behaviour was very stark:

- `naive_combined` ranked a ticket result first on all 30/30 queries.
- `freshness_only` also ranked a ticket result first on all 30/30 queries.
- `source_aware` ranked a docs result first on all 30/30 queries.
- `validity_aware` was the only mixed condition:
  - top-ranked docs on 20/30 queries
  - top-ranked tickets on 10/30 queries
  - on `source_needed=tickets`, it picked tickets first on 5/8 rows
  - on `source_needed=docs`, it still switched to tickets first on 5/14 rows

This matters because Stage 5 is not just changing ranking scores slightly. It is changing which source class the model sees as the most authoritative starting point.

## Fixed Taxonomy

The following taxonomy was used for the manual review:

1. `ticket_first_overfit`
   Ticket evidence dominates even when the answer should be governed by official docs, causing stale workflow carry-over, over-specific operational advice, or loss of policy constraints.

2. `docs_first_overcorrection`
   Authority boosting suppresses needed ticket-state context, causing the system to answer a workflow-heavy or procurement-heavy ticket as if it were a generic public-doc question.

3. `validity_routing_gain`
   The reranker moves the right source class to the top for this query, improving answer appropriateness or reducing obvious source mismatch.

4. `validity_routing_false_positive`
   The full validity-aware reranker switches source class in a way that looks plausible from query constraints but actually harms the answer by demoting the truly authoritative source.

5. `generation_bottleneck_after_good_ranking`
   The reranker improves source ordering, but the final answer still fails because the generator does not convert the better evidence into the correct concrete response.

6. `unsafe_operational_overreach`
   The answer promises or implies internal actions that should not be committed to without further verification.

## Main Findings

### 1. `freshness_only` does not behave like a meaningful improvement over `naive_combined`.

The two weakest conditions are functionally very similar at retrieval time:

- both place tickets first on all 30 queries
- they share the same top-1 retrieved item on 28/30 queries
- they share the same top-5 retrieved set ordering on 24/30 queries

Manual review reflects that. In most cases `freshness_only` reproduces the same general failure mode as `naive_combined`: a combined system that sounds locally plausible but is still dominated by ticket memory.

Representative same-query comparisons:

- `HPCSSUP-97739` (HPC password):
  both `freshness_only` and `naive_combined` tell the user to try/reset the password, whereas the reference answer is simply that there is no separate HPC password and the user should use UIS/Raven plus the CSD3 SSH TOTP.
- `HPCSSUP-92768` (checkpointing):
  both conditions hallucinate Slurm checkpoint directives rather than staying with the reference answer’s application-level checkpointing patterns.
- `HPCSSUP-96168` (two-week GPU queue delay):
  both conditions over-troubleshoot a normal service-level scheduling delay and suggest script/resource changes instead of mainly explaining queue priority.

Interpretation:

- Freshness on its own does not break the baseline’s ticket-first habit.
- In this benchmark, recency-like cues are not enough to recover authoritative or version-sensitive behaviour.

Primary error class:

- `ticket_first_overfit`

### 2. `source_aware` fixes the source-class bias, but often over-corrects.

`source_aware` is the cleanest retrieval contrast in the stage: it ranks docs first on every query. That helps on some doc-governed questions, but it also creates a new failure mode where local operational tickets get answered as if they were generic documentation questions.

Representative same-query comparisons:

- `HPCSSUP-97660` (RDS/RFS provisioning):
  this row truly depends on ticket-state workflow and Terms & Conditions acceptance. `source_aware` remains broadly on the right topic, but it adds extra group-membership and permission investigation language that is less direct than the reference workflow.
- `HPCSSUP-98292` (DAWN access assistance):
  `source_aware` imports extra public portal/application steps, including a compute-access request path, instead of staying tight to invitation acceptance and provisioning-state checking.
- `HPCSSUP-100411` (storage renewal / mistaken PO):
  `source_aware` says the user should cancel the new PO and raise a new one, while the reference answer is deliberately more cautious: do not activate anything new, confirm the target project, then verify internally whether the mistaken request can be converted or removed.

Interpretation:

- Authority-only reranking is too blunt.
- It is good at preventing stale ticket-first retrieval, but it underfits the part of the benchmark where the answer depends on internal process state rather than public rules.

Primary error classes:

- `docs_first_overcorrection`
- `unsafe_operational_overreach`

### 3. `validity_aware` is the only condition that actually adapts source class by query, but the gains are partial.

The full reranker is the only Stage 5 condition that sometimes chooses docs first and sometimes chooses tickets first. That is the retrieval behaviour the project is aiming for.

Same-query comparisons show real but limited benefits:

- `HPCSSUP-97660` (RDS/RFS provisioning):
  `source_aware` is doc-first, while `validity_aware` switches back to ticket-first. This is directionally correct because the answer depends on the provisioning workflow, not on general storage policy.
- `HPCSSUP-99428` (SNPolisher installation on SRCP):
  `validity_aware` also goes ticket-first, which is sensible for a platform-specific urgent deployment request. The answer still remains fairly generic, but the routing decision is better aligned with the task than pure authority boosting.

However, the full reranker does not reliably translate better routing into correct final answers:

- `HPCSSUP-98292` (DAWN access):
  `validity_aware` correctly switches back to ticket-first, but the answer still overpromises account setup rather than staying with invitation/provisioning-state checks.
- `HPCSSUP-100411` (storage extension):
  `validity_aware` also switches back to tickets, but still overcommits to removing the mistaken new project and reusing the PO, instead of pausing for internal verification.

Interpretation:

- The full reranker is doing something materially better than either ablation.
- But Stage 5 also shows that ranking alone is not sufficient; the generator can still operationalise the evidence unsafely.

Primary error classes:

- `validity_routing_gain`
- `generation_bottleneck_after_good_ranking`
- `unsafe_operational_overreach`

### 4. `validity_aware` also produces some false switches back to tickets on docs-governed local-operational rows.

The mixed routing is not always beneficial. There are several docs-needed queries where `validity_aware` abandons the doc-first behaviour of `source_aware` and moves tickets back to rank 1:

- `HPCSSUP-98278` (web portal login)
- `HPCSSUP-97675` (SRCP storage architecture)
- `HPCSSUP-93947` (lost Dawn SSH private key)
- `HPCSSUP-91107` (copying large file to RCS)
- `HPCSSUP-88966` (long queue time)

Representative same-query comparisons:

- `HPCSSUP-97675`:
  the reference answer is a platform-architecture clarification: SRCP storage is isolated and not interchangeable with generic RCS hardware. `validity_aware` switches to tickets first and reverts to the same generic “RCS is a good archive option” pattern seen in the ticket-first systems.
- `HPCSSUP-98278`:
  the reference answer is a targeted browser/cache/session reset for `login-web`. `validity_aware` falls back to credential/MFA troubleshooting rather than the specific redirect-cache fix.

Interpretation:

- The validity cues are sometimes too easily triggered by “local operational” wording.
- In those cases the full reranker overestimates the value of similar historic tickets and underestimates the need for public authoritative docs.

Primary error class:

- `validity_routing_false_positive`

### 5. On version-sensitive software rows, better source ordering does not automatically solve the answer.

This is where the reranker should matter most, but the manual analysis shows that Stage 5 is only a partial success.

Representative same-query comparisons:

- `HPCSSUP-98311` (newer LAMMPS on Ampere):
  ticket-first systems (`freshness_only`, `naive_combined`) surface vague “try a different compiler/module” advice from historic tickets. `source_aware` and `validity_aware` move the official LAMMPS docs to the top, which is directionally right, but they still fail to produce the specific updated supported stack from the reference answer.
- `HPCSSUP-95504` (LAMMPS + symmetrix on ukaea-amp):
  docs-first routing is better than ticket-first routing because the real issue is stack consistency, but even `validity_aware` does not reconstruct the reference answer’s clean rebuild path with the supported Ampere test stack.
- `HPCSSUP-98460` (Abaqus 2024):
  none of the conditions reach the correct answer that `abaqus/2022` is the latest centrally supported version. Ticket-first systems invent compatibility narratives from prior support conversations; docs-first systems become vague because the retrieved docs are too general and the generator does not bridge the gap.
- `HPCSSUP-98917` (Dedalus install):
  the reference answer is simple: use a personal Miniconda/Miniforge install in `hpc-work` and follow the upstream conda-forge route. All four conditions instead get dragged into legacy module/env detail.

Interpretation:

- Stage 5 improves retrieval appropriateness more than answer correctness on software/version tickets.
- The remaining bottleneck is evidence use, not just evidence ranking.

Primary error class:

- `generation_bottleneck_after_good_ranking`

## Condition-Level Summary

### `naive_combined`

Strength:

- Baseline combined coverage.
- Often picks up the existence of a relevant previous local workflow.

Weakness:

- Ticket-first on every query.
- Overfits to historic support actions even on doc-governed questions.

Overall judgement:

- Strong recall, weak arbitration.

### `freshness_only`

Strength:

- Little evidence of any useful strength beyond the baseline.

Weakness:

- Retrieval behaviour is nearly the same as `naive_combined`.
- Does not meaningfully improve source appropriateness.

Overall judgement:

- Freshness alone is not an effective control signal for this benchmark.

### `source_aware`

Strength:

- Correctly counteracts the ticket-first bias.
- Better source appropriateness on some policy and doc-governed rows.

Weakness:

- Too rigidly doc-first.
- Loses ticket-state workflow on procurement, provisioning, and internal-process questions.

Overall judgement:

- A useful correction, but too one-dimensional to be a final approach.

### `validity_aware`

Strength:

- Only condition that actually routes between docs and tickets by query.
- Best conceptual match to the benchmark’s mixed evidence requirements.

Weakness:

- Still makes some false ticket-first switches on docs-governed local-operational rows.
- Better ranking often fails to become a better final answer because the generator still overreaches or stays vague.

Overall judgement:

- The strongest Stage 5 condition, but still bottlenecked by synthesis and action safety.

## Implications For The Project

Stage 5 gives a more precise answer than Stage 4 about what needs fixing:

1. The problem is not just “use both docs and tickets”.
2. The problem is also not solved by “always trust docs first”.
3. The useful signal is query-sensitive source arbitration.
4. But once routing improves, the next bottleneck is answer-time control.

The most important follow-on requirements are:

1. Preserve the full validity-aware routing idea rather than reverting to either ticket-first or docs-first extremes.
2. Add stronger generation constraints so the model:
   - does not promise internal actions without verification,
   - does not convert historic ticket procedures into guaranteed present-day workflow,
   - does not lose concrete version/platform guidance when the docs do contain it.
3. Add explicit handling for attachment-dependent or procurement-dependent tickets so the model pauses for confirmation instead of acting as if internal checks have already happened.
4. Consider source-aware answer scaffolding:
   - if docs are dominant, frame the answer as policy/version guidance first;
   - if tickets are dominant, frame the answer as workflow-state verification first;
   - if both are needed, force the answer to separate the policy constraint from the internal next step.

## Bottom Line

Stage 5 is a useful ablation because it shows that:

- `freshness_only` adds almost nothing over the ticket-first combined baseline,
- `source_aware` fixes the wrong bias but overshoots,
- `validity_aware` is the only condition that behaves like a real query-sensitive router.

However, the manual same-query comparisons also show that retrieval is no longer the only limiting factor. Once Stage 5 gives the system a better top-ranked source, the remaining errors are often generation-side: vague answers, unsafe operational promises, and failures to convert good evidence into the specific helpdesk action actually required.

That means Stage 5 supports keeping the full validity-aware reranker, but it also shows why later stages still need stronger authority-aware and action-safe answer synthesis.
