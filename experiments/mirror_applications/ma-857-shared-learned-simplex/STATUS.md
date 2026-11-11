# MA-857 status

- Status: FAIL (quality/storage frontier and Mirror-specific attribution)
- Branch: research/ma-857-shared-learned-simplex-20261111
- Base commit: b8cde723
- Development complete: yes; LR .1 selected on seed 41
- Fresh/audit opened: yes; seeds 131, 149, 167
- Results committed: pending
- Verification committed: pending

## Decision

Worst-grid quality misses the pooled model by 5.19–6.85 percentage points across all three fresh seeds. The endpoint-plus-code payload (33,215 B) is 128 B larger than the independent three-model bank (33,087 B); it is 3.24× smaller than storing ten materialized views, but those views are not the strongest quality-matched baseline. Mirror barycentric chart and ordinary interpolation match within 5e-6 max logit error.

## Next action

Continue registry order with MA-858, Bezier-surface Mirror model chart.
