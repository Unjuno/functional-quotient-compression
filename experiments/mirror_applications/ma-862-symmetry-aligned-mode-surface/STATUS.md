# MA-862 status

- Status: FAIL (symmetry alignment and Mirror-specific quality gate)
- Branch: research/ma-862-symmetry-aligned-mode-surface-20261111
- Base commit: a9b20c5d
- Development complete: yes; LR .03 selected on seed 67
- Fresh/audit opened: yes; seeds 271, 283, 307
- Results committed: pending
- Verification committed: pending

## Decision

Activation alignment improved mean grid accuracy by only .44–1.07 pp and worst-grid accuracy inconsistently (-.83 to +1.94 pp), below the 5 pp gate. Pooled model accuracy (.842–.860) remains far above interior views. Charging permutations raises endpoint storage from 43,827 B to 45,799 B. Native and Mirror chart logits match exactly.

## Next action

Continue registry priority with MA-864, shared low-loss manifold plus private off-manifold residual. MA-863 is P1.
