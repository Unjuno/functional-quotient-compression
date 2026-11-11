# MA-860 status

- Status: FAIL (new-task quality/storage frontier and Mirror-specific attribution)
- Branch: research/ma-860-new-task-subspace-projection-20261111
- Base commit: d578bac8
- Development complete: yes; base LR .1, learned-code LR .5, random-code LR .01, full-finetune LR .01
- Fresh/audit opened: yes; seeds 223, 239, 257
- Results committed: pending
- Verification committed: pending

## Decision

Rank-3 learned-code accuracy (.375–.419) is far below full fine-tuning (.681–.786) and no better than random rank-3 (.381–.486). Base+basis+code uses 41,301 B vs 11,933 B for one full model. Native and Mirror coefficients are identical.

## Next action

Continue with MA-862 (symmetry-aligned mode surface; PA232/PA37), which tests a distinct mechanism beyond re-encoding continuous coordinates.
