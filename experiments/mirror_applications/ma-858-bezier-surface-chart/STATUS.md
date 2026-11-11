# MA-858 status

- Status: FAIL (quality/storage frontier and Mirror-specific attribution)
- Branch: research/ma-858-bezier-surface-chart-20261111
- Base commit: 3796005d
- Development complete: yes; LR .1 selected on seed 47
- Fresh/audit opened: yes; seeds 181, 199, 211
- Results committed: pending
- Verification committed: pending

## Decision

The 9-point surface payload (97,399 B) is 2.76× smaller than 25 materialized views, but is 2.22× larger than four independent corner models (43,827 B), and the one-model pooled control is 11,933 B. Worst-grid accuracy is .733–.739 versus pooled .887–.919. Native Bezier and Mirror coordinates match exactly.

## Next action

Next eligible P0 by registry priority is MA-860; MA-859 is P1.
