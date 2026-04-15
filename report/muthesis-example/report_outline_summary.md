# Report Outline Summary

## Project

- Working title: `Retrieval-Augmented Generation for High-Performance Computing Support`
- Core claim: heterogeneous-source, validity-aware retrieval improves answer quality for HPC support queries, especially on validity-sensitive cases.
- Target report length: about `12,500-13,000` words main text.
- Style constraints:
  - formal technical prose
  - IEEE references
  - define abbreviations on first use
  - keep Chapter 1 light on literature; deeper literature goes in Chapter 2

## Current Chapter Structure

### Chapter 1: Introduction

- `1.1 Project Context and Motivation`
- `1.2 Problem Definition and Scope`
- `1.3 Research Questions and Hypotheses`
- `1.4 Aim, Objectives, and Success Criteria`
- `1.5 Overview of Evaluation Strategy`
- `1.6 Project Plan and Milestones`
- `1.7 Report Structure`

Suggested word budget: about `1,000`

Notes:
- `1.1` is currently in good shape and has been assessed at roughly low-to-mid `80s`.
- `1.1` should establish:
  - CSD3 setting
  - fragmented support information
  - growth in support-ticket volume
  - RAG as a plausible support approach
  - relevance alone is insufficient because evidence may be unreliable or outdated
- `1.2-1.5` need to clearly derive the task, aims, and evaluation strategy from this setup.

### Chapter 2: Background and Related Work

- `2.1 HPC Support as an Information Retrieval and Question Answering Problem`
- `2.2 Retrieval-Augmented Generation`
- `2.3 Heterogeneous Support Corpora: Documentation and Tickets`
- `2.4 Authority, Freshness, and Validity in Retrieval`
- `2.5 Evaluation of RAG Systems`
- `2.6 Gap Analysis and Research Positioning`

Suggested word budget: about `2,400`

Notes:
- This chapter should be literature-led.
- It should justify why the project is not just a generic RAG system.
- The main bridge should be:
  - fragmented evidence
  - heterogeneous source roles
  - temporal validity / outdated evidence
  - need for careful evaluation

### Chapter 3: Problem Formulation and Requirements

- `3.1 Task Definition`
- `3.2 Data Sources and Knowledge Characteristics`
- `3.3 Benchmark and Query Taxonomy`
- `3.4 Requirements and Constraints`
- `3.5 Success Criteria and Requirement-to-Evaluation Mapping`
- `3.6 Ethical, Privacy, and Data-Governance Considerations`

Suggested word budget: about `1,200`

Notes:
- This chapter should turn the motivation and literature into a precise task.
- The requirement-to-evaluation mapping is important for a strong methodology narrative.

### Chapter 4: System Design and Implementation

- `4.1 System Overview`
- `4.2 Baseline Systems`
- `4.3 Corpus Ingestion and Representation`
- `4.4 Authority Registry and Metadata Extraction`
- `4.5 Validity-Aware Retrieval and Reranking Design`
- `4.6 Prompt and Generation Design`
- `4.7 API, Tooling, and Observability`
- `4.8 User Interface and Interaction Design`
- `4.9 Verification and Software Testing`
- `4.10 Design Trade-offs`

Suggested word budget: about `1,800`

Notes:
- Keep design tightly connected to evaluation.
- UI should stay proportionate unless it is explicitly evaluated as a contribution.

### Chapter 5: Experimental Methodology

- `5.1 Evaluation Goals`
- `5.2 Development and Test Split Discipline`
- `5.3 Benchmark Construction and Characterisation`
- `5.4 Frozen Backbone Decisions`
- `5.5 Metric Hierarchy and Decision Rules`
- `5.6 Development-Phase Tuning Protocol`
- `5.7 Final Test-Set Experiments`
- `5.8 Manual Evaluation Protocol`
- `5.9 Error Analysis Protocol`
- `5.10 Reproducibility and Stability Controls`

Suggested word budget: about `1,900`

Notes:
- This is one of the highest-value chapters for the mark scheme.
- The methodology should make the dev/test discipline and controlled ablation logic completely explicit.

### Chapter 6: Results

- `6.1 Development-Phase Selection Results`
- `6.2 Overview of Main Test-Set Results`
- `6.3 Controlled Ablation Results`
- `6.4 Practical Comparison Results`
- `6.5 Manual Evaluation Results`
- `6.6 Error Analysis Results`
- `6.7 Stability and Robustness Results`

Suggested word budget: about `2,200`

Notes:
- Keep causal ablations grouped together.
- Do not mix ablations, prompt comparison, and manual evaluation into one flat list.
- Present results clearly; save interpretation for Chapter 7.

### Chapter 7: Discussion and Critical Appraisal

- `7.1 Answers to the Research Questions`
- `7.2 What Improved and Why`
- `7.3 Comparison with Prior Work`
- `7.4 Where the System Still Fails`
- `7.5 Threats to Validity`
- `7.6 Review of the Project Plan`

Suggested word budget: about `1,400`

Notes:
- This chapter should do the heavy critical-analysis work.
- It should connect findings back to:
  - research questions
  - literature
  - weaknesses and limitations
  - what the results do and do not support

### Chapter 8: Conclusion and Future Work

- `8.1 Summary of Contributions`
- `8.2 Main Findings Against Objectives`
- `8.3 Limitations`
- `8.4 Future Work`

Suggested word budget: about `700`

Notes:
- Keep this concise.
- It should explicitly close the loop back to Chapter 1 aims and objectives.

## Evaluation Structure

### Main evaluation logic

- Dev set for:
  - generator selection
  - chunking tuning
  - retrieval-mode pilot
  - retrieval-budget tuning
  - validity-reranker tuning
- Test set for:
  - final ablations
  - finalist comparisons
  - manual evaluation
  - final error analysis

### Main experiments

- Source ablation:
  - `docs_only`
  - `tickets_only`
  - `naive_combined`
- Validity-signal ablation:
  - `naive_combined`
  - `source_aware`
  - `freshness_only`
  - `validity_aware`
- External-document scope ablation:
  - `validity_aware`
  - `all_docs_validity_aware`
- Prompt comparison
- Tuned-finalist comparison
- Manual evaluation
- Error analysis
- Stability checks / repeated runs

### Main metric hierarchy

- Primary metric: `factual_correctness`
- Guardrail metric: `faithfulness`
- Retrieval diagnostics:
  - `context_recall`
  - `context_precision_without_reference`
- Supporting metrics:
  - `answer_relevancy`
  - `semantic_similarity`
  - `context_entity_recall`
  - `noise_sensitivity`

## Chapter 1.1 Summary

Current function of `1.1`:

- introduce CSD3
- motivate the support-information problem
- show growth in helpdesk ticket volume
- introduce LLM/RAG support assistants
- argue that source reliability and outdated evidence matter
- hand off into formal problem definition

Current key sources used in `1.1`:

- `ResearchComputingServicesDocumentation`
- `AskHPC`
- `joslinGeneratingFrequentlyAsked2025`
- `lewisRetrievalaugmentedGenerationKnowledgeintensive2020`
- `liYouKnowWhat2024`
- `hwangRetrievalAugmentedGenerationEstimation2025`
- `CSD3UpgradeOctober`

## Mark-Scheme Priorities

- Most important report section by weight: `Technical Quality, Methodology and Evaluation` (`35%`)
- Next most important: `Background and Theory` (`25%`)
- For 80+:
  - literature should be relevant and well understood
  - aims should be derived from the setting
  - evaluation strategy should be justified, not improvised
  - the report should show honest critical analysis
  - presentation should be polished and logically structured

## Next-Step Reminder

When discussing later subsections in separate chats, keep linking comments back to:

- the exact function of that subsection in the chapter
- the report mark scheme
- how the subsection supports the overall research claim
