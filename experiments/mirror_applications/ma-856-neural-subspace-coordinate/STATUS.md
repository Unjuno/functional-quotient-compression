# MA-856 status

- Status: FAIL (Mirror-specific attribution; all-seed quality gate also missed)
- Branch: research/ma-856-neural-subspace-coordinate-20261111
- Base commit: 7903e936
- Development complete: yes; LR .1 / β .1 frozen on seed 31
- Fresh/audit opened: yes; seeds 61, 79, 97
- Results committed: pending
- Verification committed: pending

## Decision

The learned line stores 17 alpha views in 22,665 B versus 182,743 B for 17 materialized models, but the 2-model independent bank is 22,477 B and has higher mean accuracy. The selected line misses its 2 percentage-point worst-grid quality gate on seed 61. Native alpha and Mirror m=2alpha−1 are exactly equal in logits, so there is no Mirror-specific functional gain.

## Next action

Proceed to next eligible P0, MA-857, after committing this result and cumulative registry updates.
