# MA-276 status

- Status: SCREENING
- Branch: `research/ma-276-boft-depth-view-20261008`
- Base commit: `19f0b5a`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Protocol is frozen at rank 4 and corrected input-side BOFT execution; commit before fresh seeds.

## Blockers

None. This is a small NumPy CPU operator screen.

## Decisions / rulings

- Random pool, draw, and remote branch check are recorded in PROTOCOL.json.
- Rank-2 per-step LoRA remains a fixed control; only the generated shared residual basis rank is selected.
