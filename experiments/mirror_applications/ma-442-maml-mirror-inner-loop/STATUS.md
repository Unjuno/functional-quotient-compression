# MA-442 status

- Status: FAIL (verified development screen)
- Branch: `research/ma-442-maml-mirror-inner-loop-20261008`
- Base commit: `ca1fa9f`
- Frozen protocol SHA-256: `f761e2f12edb72cf30a97b28d617632598317f63fe36831d8cbff6162d17bf40` (Amendment 1; initial protocol `6981a55e5a4a1dbc7c3253591526aea1024fc27dfe349d64170b7e6ab270df6a`)
- Last verified commit: pending terminal report commit
- Development complete: yes (44201, 44202)
- Fresh/audit opened: no; seeds 44211–44213 remain sealed
- Results committed: no
- Verification committed: no
- Registry row updated: FAIL

## Next action

Commit and push this verified FAIL, reconcile the live branch manifest, then refresh and select the next eligible P0.

## Blockers

None.

## Decisions / rulings

Amendment 1 preserved the initial metric artifacts and corrected wall-time intervals, training-count attribution, and training/evaluation operation accounting. No model, task, or quality-gate settings changed. Two code execution exceptions are logged and excluded.
