# Stage 7 Bootstrap Confidence Intervals

Post-protocol follow-up: prompt comparison, not part of the frozen retrieval protocol.

Primary metric: `factual_correctness`

Method:
- Per-query score = mean `factual_correctness` over the saved runs for each held-out query.
- Uncertainty = 95 percent percentile bootstrap confidence interval over held-out queries.
- Bootstrap resamples = 50,000.

Data note:
- Both prompt conditions retain all 30 held-out queries after averaging across available saved repeats.
- A small number of query-condition cells have one missing repeat score; the per-query mean uses the available saved repeats.

## Condition Means

| Condition | n queries | Mean | 95% CI |
| --- | ---: | ---: | ---: |
| `v3_ticket_prompt` | 30 | 0.278278 | [0.232444, 0.325778] |
| `v4_prompt` | 30 | 0.349833 | [0.301000, 0.402333] |

## Paired Difference

Difference is `v4_prompt - v3_ticket_prompt`.

| Comparison | n queries | Difference | 95% CI |
| --- | ---: | ---: | ---: |
| `v4_prompt - v3_ticket_prompt` | 30 | 0.071556 | [0.029778, 0.113611] |

## Interpretation

- `v4_prompt` is credibly above `v3_ticket_prompt` on held-out factual correctness in this post-protocol follow-up.
- Unlike the Stage 5 and Stage 6 retrieval comparisons, the paired interval here stays above zero, so the prompt revision shows a clearer held-out gain on the primary metric.
- Because this was collected after the frozen retrieval protocol, it should still be reported as a follow-up improvement study rather than folded into the original retrieval-ablation claims.
