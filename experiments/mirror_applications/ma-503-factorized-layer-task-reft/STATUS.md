# MA-503 status

- Status: FROZEN_SCREENING
- Branch: `research/ma-503-factorized-layer-task-reft-20261009`
- Base commit: `49e86eb0`
- Protocol frozen: yes
- Development complete: yes; selected common steps = 400 on development-only held-out NRMSE
- Fresh/audit opened: initial A0 opened; A1 runtime correction declared before rerun
- Results committed: no
- Verification committed: no

## Next action

Rerun the locked fresh worlds/seeds with end-to-end decode timing; do not tune settings or quality on fresh results.

## Blockers

None.

## Decisions / rulings

The factorization is tested on held-out layer-task pairs. The supplied basis is a controlled shared object and its bytes are charged in every serialized method.

- Amendment A1: corrected decode timing to include code/View generation; initial fresh result rows are preserved as exploratory and not used for runtime conclusions.
