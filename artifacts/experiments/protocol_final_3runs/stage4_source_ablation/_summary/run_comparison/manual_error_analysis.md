# Stage 4 Manual Error Analysis

## Scope

This note reviews the Stage 4 source-ablation comparison generated from:

- `artifacts/experiments/protocol_final_3runs/stage4_source_ablation/docs_only/run_03`
- `artifacts/experiments/protocol_final_3runs/stage4_source_ablation/naive_combined/run_03`
- `artifacts/experiments/protocol_final_3runs/stage4_source_ablation/tickets_only/run_03`

The analysis uses the standard comparison bundle under this directory:

- `combined_analysis_rows.jsonl`
- `query_review_sheet.csv`
- `manual_eval_sheet.csv`
- `manual_eval_key.csv`

This is a qualitative error analysis of representative failures across the 30-query test split. The aim is to identify stable failure modes caused by the retrieval-source ablation, not to re-score the whole stage numerically.

## Fixed Taxonomy

The following error taxonomy was used while reviewing the rows:

1. `missing_ticket_state`
   The answer relies on public documentation but misses the case-specific project state, provisioning state, incident context, or prior internal handling recorded in tickets.

2. `missing_authoritative_docs`
   The answer relies on ticket memory but lacks the authoritative policy/version/platform constraint that should have come from official docs.

3. `hybrid_synthesis_failure`
   Both source classes are present, but the model fails to combine them correctly and imports the wrong action or conclusion from one source into the final answer.

4. `unsafe_operational_overreach`
   The answer promises or recommends an internal action that should not be promised without verification, such as creating accounts, changing permissions, applying credits, or modifying procurement state.

5. `attachment_or_evidence_blindness`
   The answer proceeds as if unseen attachments or missing internal evidence had already been checked, rather than acknowledging that the next step is confirmation.

6. `incident_or_node_misdiagnosis`
   The answer treats a likely service incident or node-health problem as a user-configuration problem, or vice versa.

## Main Findings

### 1. `docs_only` fails most often when the ground truth depends on internal ticket state rather than public policy.

This condition regularly returned plausible public guidance while missing the actual service-desk workflow:

- `HPCSSUP-98575` (SL3 CPU allocation): the reference answer is "quarterly reset, no extra-credit promise". `docs_only` instead drifted into purchase/escalation language.
- `HPCSSUP-98292` (DAWN access assistance): the correct path is invitation acceptance plus provisioning-state checks. `docs_only` reverted to generic application/MFA guidance and treated the request as a standard new-access flow.
- `HPCSSUP-97675` (SRCP storage architecture): the reference explicitly says SRCP storage is isolated and not interchangeable with cheaper RCS hardware. `docs_only` answered with a generic RFS/RCS product-choice explanation.
- `HPCSSUP-100516` and `HPCSSUP-100411` (procurement/storage-extension cases with attachments): `docs_only` forced the problem back into generic portal and policy steps rather than treating it as a service-desk correction workflow.

Interpretation:

- Public documentation alone is not enough for tickets where the correct answer depends on project state, provisioning state, previous support actions, or procurement context.
- On these rows, `docs_only` often sounds polished but is operationally wrong.

Primary error classes:

- `missing_ticket_state`
- `attachment_or_evidence_blindness`
- `unsafe_operational_overreach`

### 2. `tickets_only` captures local workflow better, but often loses policy boundaries and source appropriateness.

This condition performed better on some workflow-heavy rows, but it frequently overfit to prior ticket resolutions and produced actions that were too strong, too specific, or not properly grounded:

- `HPCSSUP-97739` (HPC password): the correct answer is "use UIS/Raven password; no separate HPC password". `tickets_only` suggested that support could help set a password or that SSH-key fallback was the remedy.
- `HPCSSUP-92768` (checkpointing): the reference answer is generic application-level checkpointing guidance. `tickets_only` hallucinated Slurm `--checkpoint` directives as if checkpointing were a native scheduler feature for this case.
- `HPCSSUP-95709` (CUDA devices unavailable): the reference suggests loading the correct Ampere stack, removing manual `mpirun` placement overrides, and temporarily excluding the bad node. `tickets_only` escalated to stronger, unsupported operational steps such as `#SBATCH --reboot` and node-level diagnostics.
- `HPCSSUP-98292` (DAWN access): `tickets_only` got closer to the invitation/provisioning workflow, but still said support would create the account and apply the SSH key, which the reference explicitly avoids promising.
- `HPCSSUP-100516` (SL2 PO): `tickets_only` treated the ticket as if helpdesk could directly deposit credits once a PO existed, without first confirming which account the hours should target.

Interpretation:

- Ticket memory improves local workflow recall, but by itself it is weak on policy guardrails, version specificity, and "what support may safely promise".
- This condition often optimises for immediate actionability at the expense of authority and correctness.

Primary error classes:

- `missing_authoritative_docs`
- `unsafe_operational_overreach`
- `attachment_or_evidence_blindness`

### 3. `naive_combined` is directionally the right source mix, but it still suffers from ticket bleed into the final answer.

The combined condition usually covered more of the real task than either single-source baseline, but it still inherited bad operational habits from ticket retrieval when synthesis was weak:

- `HPCSSUP-98114` (long queue time): `naive_combined` correctly preserved the main operational message not to cancel/resubmit and that long waits can be normal. This is a good example of helpful combination.
- `HPCSSUP-95197` (VNC node issue): `naive_combined` mostly found the right shape of answer by suggesting a different node first, which matches the reference better than `docs_only`.
- `HPCSSUP-98292` (DAWN access): it mixed the correct invitation/provisioning framing with an unsafe promise to create the account and apply the SSH key.
- `HPCSSUP-98311` and `HPCSSUP-95504` (LAMMPS/GPU stack issues): instead of converging on the supported stack or clean rebuild path, it imported speculative ticket advice such as trying an alternative compiler/MPI combination.
- `HPCSSUP-98460` (Abaqus 2024): the reference says the latest supported central version is `abaqus/2022`. `naive_combined` drifted into an unsupported "cluster currently has Abaqus 2020" narrative from ticket memory.
- `HPCSSUP-100411` (storage extension): it overcommitted to removing the mistaken new project and reusing the PO, where the reference answer is deliberately more cautious and verification-first.

Interpretation:

- Combining sources is necessary for Stage 4, but naive fusion is not sufficient.
- The mixed-source model still needs stronger answer-time controls so that docs dominate on policy/version questions and ticket memory only fills in workflow context.

Primary error classes:

- `hybrid_synthesis_failure`
- `unsafe_operational_overreach`
- `attachment_or_evidence_blindness`

### 4. Incident-like or node-health issues are a consistent weak point for `docs_only`.

Several rows show that public documentation is a poor match when the real problem is a live service or node condition:

- `HPCSSUP-98537` (OnDemand Jupyter crash): the reference answer first checks for a service incident. `docs_only` pushed the user toward environment/module troubleshooting.
- `HPCSSUP-95197` (VNC on `gpu-r-4`): the reference says try a different node. `docs_only` instead moved into generic VNC session management.
- `HPCSSUP-95709` (Ampere CUDA failure): the reference treats this as a node-health plus placement issue with a safe temporary workaround. `docs_only` kept the discussion at generic submission/configuration level.
- `HPCSSUP-81206` (`/tmp` full): `docs_only` correctly recognised temporary-space context but still overgeneralised into user cleanup/quota framing instead of focusing on node identification and workaround.

Interpretation:

- When failure symptoms point to a node-specific or service-specific fault, documentation-only retrieval encourages the model to blame the user's setup.

Primary error class:

- `incident_or_node_misdiagnosis`

### 5. Attachment-dependent tickets expose an important evaluation blind spot.

The two attachment-dependent rows in this split are especially revealing:

- `HPCSSUP-100516` (PO for SL2 hours)
- `HPCSSUP-100411` (storage extension / mistaken PO)

In both cases the reference answer is intentionally cautious because the attachment contents and internal procurement state are decisive. The weaker systems instead acted as though the next administrative step were already known.

This is important because automatic relevance/recall metrics can still look reasonable if the retrieval is thematically related, even when the operational answer is unsafe.

Primary error classes:

- `attachment_or_evidence_blindness`
- `unsafe_operational_overreach`

## Condition-Level Summary

### `docs_only`

Strength:

- Best aligned with stable public process questions when the answer is fully documented and does not depend on internal case state.

Weakness:

- Brittle on workflow-heavy tickets, procurement tickets, provisioning tickets, and incident-like tickets.
- Tends to replace specific internal handling with generic public guidance.

Overall judgement:

- Too weak as a standalone retrieval mode for support work that mixes official policy with service-desk state.

### `tickets_only`

Strength:

- Best recall of local workflow and previous helpdesk handling patterns.
- Often closer than `docs_only` on tickets whose truth depends on internal provisioning or support history.

Weakness:

- Weakest source appropriateness.
- Most likely to hallucinate unsupported commands, promise admin actions, or carry stale local practice into the answer.

Overall judgement:

- Useful as supplementary context, but unsafe as the only retrieval source.

### `naive_combined`

Strength:

- Broadest coverage of the real problem space.
- Usually the best of the three when both policy and local workflow matter.

Weakness:

- Still not robust enough at arbitration between docs and ticket memory.
- Needs stronger answer-time constraints to stop ticket-derived overreach from contaminating otherwise solid answers.

Overall judgement:

- The best Stage 4 direction, but not yet the final retrieval/answering behaviour to trust without further controls.

## Implications For The Project

Stage 4 supports keeping a combined-source system rather than a single-source system, but it also shows that source combination alone is not the real fix. The remaining problem is answer synthesis.

The most valuable follow-on changes would be:

1. Give official docs priority for policy, version, account-governance, and platform-capability claims.
2. Allow ticket memory to supply workflow context, but not to authorise irreversible or internal actions on its own.
3. Add explicit prompt rules such as:
   - do not promise account creation, key installation, permission changes, credit deposits, or procurement corrections unless the evidence explicitly confirms they have been checked;
   - if attachments or internal state matter and are not visible, acknowledge the ambiguity and ask for confirmation;
   - for node/service symptoms, check the incident/node-health interpretation before blaming user configuration.
4. Consider a post-retrieval authority filter or response validator for high-risk operational actions.

## Bottom Line

The Stage 4 ablation behaved as expected:

- `docs_only` underfit the internal workflow.
- `tickets_only` overfit the internal workflow and underfit authority.
- `naive_combined` was the strongest overall condition, but it still needs authority-aware synthesis to avoid overconfident operational mistakes.

That makes Stage 4 a strong justification for the later validity-aware / authority-aware stages of the project.
