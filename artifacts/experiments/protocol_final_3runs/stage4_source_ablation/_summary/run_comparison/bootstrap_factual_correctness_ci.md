# Stage 4 Bootstrap Confidence Intervals

Primary metric: `factual_correctness`

Method:
- Per-query score = mean `factual_correctness` over the three saved runs for each held-out query.
- Uncertainty = 95 percent percentile bootstrap confidence interval over held-out queries.
- Bootstrap resamples = 50,000.

## Condition Means

| Condition | n queries | Mean | 95% CI |
| --- | ---: | ---: | ---: |
| `docs_only` | 30 | 0.239722 | [0.193444, 0.286944] |
| `naive_combined` | 30 | 0.298111 | [0.251222, 0.345444] |
| `tickets_only` | 30 | 0.315500 | [0.270167, 0.363944] |

## Paired Differences

Difference is `condition_a - condition_b`.

| Comparison | n queries | Difference | 95% CI |
| --- | ---: | ---: | ---: |
| `naive_combined - docs_only` | 30 | 0.058389 | [0.014665, 0.103611] |
| `naive_combined - tickets_only` | 30 | -0.017389 | [-0.051778, 0.018111] |
| `tickets_only - docs_only` | 30 | 0.075778 | [0.028610, 0.123611] |

## Interpretation

- `naive_combined` is credibly above `docs_only` on held-out factual correctness.
- `naive_combined` is not credibly above `tickets_only`; the paired interval overlaps zero.
- The safest Stage 4 claim is that combining sources improves over `docs_only`, but not that it clearly beats both single-source baselines.
