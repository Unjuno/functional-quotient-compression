# MA-699 status

- Status: SCREENING
- Branch: `research/ma-699-topk-lazy-attention-views-20261109`
- Base commit: `81c702ad`
- Last verified commit: pending
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Implement frozen top-k shared-cache attention and payload accounting.

## Blockers

None.

## Decisions / rulings

Charge selected view IDs. Report active attention compute as k-dependent; do not infer capacity from combination count.
