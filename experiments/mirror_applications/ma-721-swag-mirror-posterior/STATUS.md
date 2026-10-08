# MA-721 status

- Status: FAIL (development gate)
- Branch: `research/ma-721-swag-mirror-posterior-20261008`
- Base commit: `407ca7e2047326d1e4b753e55e05c4730f26f32b`
- Last verified commit: 147a1f8457f34838e5ee3eb483d87408bdb598e1
- Development complete: yes (60 final rows, 2 seeds)
- Fresh/audit opened: no
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Decision

Rademacher Mirror coordinates over the SWAG basis did not meet the frozen shifted NLL/ECE improvement gate over same-rank Gaussian SWAG. Independent ensemble quality was higher; rank-1 factor control improved corrupted-data calibration at a clean-NLL cost. Fresh remains unopened.

## Next action

Commit and push the checked failure, update branch-local registry/claim/status, then refresh baseline and remote branches for a new draw.

## Boundaries

Small sklearn digits MLP; post-hoc Rank-1 factor scale; no natural OOD, full Rank-1 BNN, large-model capacity or audit evidence.
