# Stage 6 Manual Error Analysis

## Scope

This note reviews the Stage 6 external-documents ablation generated from:

- `artifacts/experiments/protocol_final_3runs/stage6_external_docs_ablation/validity_aware/run_03`
- `artifacts/experiments/protocol_final_3runs/stage6_external_docs_ablation/all_docs_validity_aware/run_03`

The analysis uses the paired comparison artifacts in this directory:

- `combined_analysis_rows.jsonl`
- `query_review_sheet.csv`
- `manual_eval_sheet.csv`
- `manual_eval_key.csv`

The goal is to explain what changes when external official documentation is added on top of the Stage 5 `validity_aware` retrieval stack, and whether that extra source class improves the same 30 held-out queries in a practically useful way.

## What Changed In Stage 6

Stage 6 is not another routing ablation in the same sense as Stage 5.

The two conditions are:

1. `validity_aware`
   Frozen validity-aware reranker over local docs plus tickets.

2. `all_docs_validity_aware`
   Frozen validity-aware reranker over local docs, external official docs, and tickets.

The key retrieval fact is that the top-ranked item did not change at all:

- the top-1 retrieved item was identical on all 30/30 queries
- external docs never ranked first
- in the external-docs condition, external docs usually appeared as secondary context, most often at ranks 3 and 6

This means Stage 6 is primarily a secondary-context ablation rather than a primary-routing ablation. The extra external documents influence answer synthesis much more than they change the main retrieval decision.

The aggregate results follow the same pattern:

- `all_docs_validity_aware` slightly improved context recall and faithfulness
- it did not improve factual correctness over `validity_aware`

So the central manual-analysis question is whether those extra external documents genuinely sharpen the answer, or whether they simply add plausible-looking but locally weaker guidance.

## Fixed Taxonomy

The following taxonomy was used for the manual review:

1. `external_clarification_gain`
   External official docs add useful generic explanation or conceptual framing that makes a locally grounded answer clearer without changing the underlying local action.

2. `partial_software_support_gain`
   External docs add relevant upstream technical context on a software/version row, but the answer still fails to recover the exact locally supported stack or support boundary.

3. `local_authority_dilution`
   External docs are topically relevant but less locally authoritative, so the answer drifts away from CSD3-specific policy, service behaviour, or workflow.

4. `upstream_troubleshooting_detour`
   External docs pull the answer into generic Slurm, package-management, or toolchain troubleshooting instead of the grounded next step from local docs or ticket state.

5. `secondary_context_no_real_fix`
   External docs change wording, citations, or explanation style, but the core operational error remains unchanged.

6. `unsafe_generation_persists`
   Even with extra official context, the generator still promises or implies internal actions that should remain verification-first.

## Main Findings

### 1. Stage 6 does not change the primary retrieval route, only the supporting context.

This is the defining difference from Stage 5.

For every query in the split:

- `validity_aware` and `all_docs_validity_aware` chose the same top-ranked item
- the response still changed, because external docs were inserted lower in the context window

Interpretation:

- The external-docs condition is not winning by finding a better primary local source.
- It is mostly changing what secondary evidence the generator sees while answering.
- That is why Stage 6 produces noticeable wording and citation differences with only weak movement in factual correctness.

Primary error class:

- `secondary_context_no_real_fix`

### 2. External docs help only when the local answer needs a small amount of generic clarification.

There are some real gains, but they are narrow and mostly explanatory rather than operational.

Representative same-query comparisons:

- `HPCSSUP-89749` (Microsoft Authenticator / CSD3 TOTP):
  `validity_aware` drifted into reset/delete guidance. `all_docs_validity_aware` was better because it stated the real distinction more clearly: the same app can be reused, but CSD3 needs a separate token and the University token will not work.
- `HPCSSUP-93947` (lost Dawn SSH private key):
  `validity_aware` added unnecessary MFA and login-command detail. `all_docs_validity_aware` was closer to the real workflow because it stayed more tightly on the key-replacement path.
- `HPCSSUP-81206` (`/tmp` full):
  the external-docs condition was slightly cleaner in explaining that the problem concerns temporary local storage rather than quota, although it still overcommitted to what support would do next.

Interpretation:

- External official docs are at their best when they provide generic conceptual clarification around an already-correct local framing.
- They are not creating new local truth here; they are only polishing the explanation.

Primary error classes:

- `external_clarification_gain`
- `secondary_context_no_real_fix`

### 3. On local operational questions, external docs often dilute the right local answer.

The most consistent Stage 6 regression is not gross irrelevance. It is locally plausible but operationally weaker guidance.

Representative same-query comparisons:

- `HPCSSUP-88966` (long queue wait on SL3 CPU):
  the reference answer is mainly reassurance plus service-level explanation and a warning not to cancel/resubmit. `all_docs_validity_aware` drifted into generic partition and queue troubleshooting instead.
- `HPCSSUP-96168` (two-week GPU queue delay):
  the correct answer is again priority/service-level explanation. The external-docs condition moved toward generic job-parameter adjustment and resubmission language.
- `HPCSSUP-98278` (`login-web` access issue):
  the reference answer is a targeted browser cache / redirect reset. The external-docs condition became more generic about credentials, email address, and login options, losing the specific local fix.
- `HPCSSUP-98537` (OnDemand Jupyter crash):
  the reference answer first treats this as a likely service or filesystem problem. `all_docs_validity_aware` instead pushed the user toward environment setup, resource changes, and interactive alternatives.

Interpretation:

- These rows already had the right local source class at rank 1.
- Adding external docs did not make the answer more authoritative.
- It made the answer broader, more generic, and less anchored to the local support action actually required.

Primary error class:

- `local_authority_dilution`

### 4. External docs are especially risky when the correct answer depends on local platform ownership, node state, or internal workflow.

This is where the external-docs condition most often pulled the answer into the wrong mode of reasoning.

Representative same-query comparisons:

- `HPCSSUP-95709` (Ampere CUDA devices unavailable):
  the reference answer is a node-health plus safe-workaround case: load the correct base stack, remove manual MPI placement overrides, and temporarily exclude the bad node. `all_docs_validity_aware` instead drifted toward generic GPU allocation and Slurm `--gres` advice.
- `HPCSSUP-99428` (urgent SRCP SNPolisher install):
  the correct first step is to prioritise escalation to the SRCP/platform team. `all_docs_validity_aware` generalised this into standard R package installation, repository checks, and package-manager procedure.
- `HPCSSUP-98608` (Dawn invalid account/partition):
  the real issue is Dawn project charging-account syntax. The external-docs condition widened the answer into generic Slurm script and partition troubleshooting, including irrelevant CSD3 partition suggestions.

Interpretation:

- On these rows, external docs are not just unnecessary.
- They actively compete with the local operational truth by introducing generic HPC advice that sounds reasonable but is not the right support action for this system.

Primary error classes:

- `upstream_troubleshooting_detour`
- `local_authority_dilution`

### 5. On version-sensitive software rows, external docs improve coverage more than they improve correctness.

Stage 6 was most plausible in principle on rows tagged `docs_scope_needed=local_and_external`, because these are the queries where upstream software documentation might actually matter.

The manual review shows only partial success.

Representative same-query comparisons:

- `HPCSSUP-98311` (newer LAMMPS on Ampere):
  `all_docs_validity_aware` was directionally better grounded because the external CUDA material reinforced that the issue was a stack/toolchain compatibility problem. However, it still failed to recover the exact updated supported module stack from the reference answer.
- `HPCSSUP-95504` (LAMMPS + symmetrix on ukaea-amp):
  the added docs changed the framing, but neither condition reached the clean supported rebuild path in the reference answer.
- `HPCSSUP-98460` (Abaqus 2024):
  the external-docs condition still did not reach the key local point that `abaqus/2022` is the latest centrally supported version.
- `HPCSSUP-98917` (Dedalus install):
  this row shows the downside clearly. The reference answer is to avoid the shared cluster conda module and use a personal Miniconda or Miniforge install in `hpc-work`. `all_docs_validity_aware` made the answer worse by drifting into generic Intel MPI, environment-variable, and parallel-run guidance.

Interpretation:

- External docs can improve the general technical framing on software rows.
- They do not reliably produce the exact local support answer.
- Without stronger answer-time control, they just broaden the technical discussion.

Primary error classes:

- `partial_software_support_gain`
- `upstream_troubleshooting_detour`

### 6. The Stage 5 generation bottleneck remains.

Stage 6 does not remove the main problem identified in Stage 5.

Representative same-query comparison:

- `HPCSSUP-98292` (DAWN access provisioning):
  adding external docs changed the wording and raised the apparent groundedness of the response, but it did not fix the core safety issue. The answer still implied account setup and key application before invitation acceptance and provisioning state had been properly verified.

Interpretation:

- The extra official evidence can make an answer look more justified.
- It does not, by itself, stop unsafe or overconfident operational reasoning.

Primary error classes:

- `secondary_context_no_real_fix`
- `unsafe_generation_persists`

## Condition-Level Summary

### `validity_aware`

Strength:

- Better preservation of local authority because the answer is restricted to local docs plus ticket evidence.
- Less likely to drift into irrelevant upstream platform or package-management guidance.

Weakness:

- Still weak on some software/version rows.
- Still affected by the answer-synthesis problems already identified in Stage 5.

Overall judgement:

- Stronger as a default support retrieval mode than the external-docs variant.

### `all_docs_validity_aware`

Strength:

- Can improve explanatory clarity on some MFA, SSH-key, and software-context rows.
- Adds relevant secondary context that sometimes improves recall or faithfulness.

Weakness:

- Does not improve the primary retrieval decision.
- Too often contaminates otherwise local answers with generic upstream advice.
- Particularly risky on local operational tickets, node-health issues, internal workflows, and cluster-specific access problems.

Overall judgement:

- Useful only as a tightly controlled supplementary source, not as the main always-on retrieval stack.

## Implications For The Project

Stage 6 does not support a simple "more official documentation is always better" conclusion.

Instead, it suggests:

1. Keep `validity_aware` as the default local retrieval stack.
2. If external docs are used, gate them much more aggressively:
   - allow them mainly on software behaviour, library syntax, upstream installation workflows, or genuinely non-local conceptual questions;
   - suppress or strongly downweight them on local operations, access provisioning, scheduling policy, SRCP/Dawn workflows, service incidents, and node-state diagnosis.
3. Do not let external docs introduce new operational actions when local docs already define the system-specific answer.
4. Add source-aware answer scaffolding so that:
   - local docs define platform policy and cluster-specific procedure,
   - tickets define local workflow or prior-case context,
   - external docs only provide secondary technical explanation where needed.

## Bottom Line

Stage 6 is not a routing success story like the intended Stage 5 `validity_aware` result.

The top-ranked retrieval choice was unchanged on all 30 queries. External docs only entered as secondary context, and their effect was therefore mainly at answer-synthesis time.

They were occasionally helpful as clarifying evidence, especially on a small number of MFA, SSH-key, and software-context rows. More often, however, they diluted local authority and pulled answers toward generic upstream troubleshooting that was less appropriate than the original `validity_aware` response.

That means Stage 6 does not justify carrying `all_docs_validity_aware` forward as the default retrieval stack. The stronger project direction is to keep `validity_aware` as the main local system and only use external official docs behind tighter query gating or stricter answer-time controls.
