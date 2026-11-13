# MA-1000 status

- Status: PROMISING (synthetic conservative-derivative audit only)
- Branch: `research/ma-1000-conservative-derivative-audit-20261113`
- Base commit: `10938908f`
- Last verified commit: pending
- Development complete: no
- Fresh/audit opened: yes (100011, 100013, 100017)
- Results committed: pending
- Verification committed: pending

## Next action

Record scoped physical-validity result and continue to MA-1001 dynamics stability.

## Blockers

No trained MACE/NequIP model is available. The scoped synthetic physics audit remains executable on CPU.

## Decisions / rulings

- **FACT:** energy-derived forces pass finite-difference and rotation checks; separate force-head control fails.
- **INTERPRETATION:** the energy interface preserves physics, but Mirror is not distinct from native rank-2 coefficients and adds bytes/compute.
- **HYPOTHESIS:** long-rollout stability remains to be tested.
