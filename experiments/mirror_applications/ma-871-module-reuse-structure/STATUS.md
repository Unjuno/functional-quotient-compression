# MA-871 status

- Status: FAIL (Mirror-specific gate); synthetic rank-2 structure screen passes its scoped quality/storage gate
- Branch: `research/ma-871-module-reuse-structure-20261111`
- Base commit: `54c00313`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze and development validation
- Results committed: pending
- Verification committed: pending

## Next action

Commit and push the checked results on this experiment branch, then integrate them into the cumulative research branch.

## Blockers

None.

## Decisions / rulings

- Native rank-2 PCA exactly matches Mirror outputs, bytes, and decode MACs; the direct Mirror-specific hypothesis failed.
- At 8 support examples all three fresh world means passed the quality gate; rank-2 structure uses 408 B vs 488 B independent, but its decode MAC proxy is 3x.

- Pre-implementation amendment 01: the payload arithmetic for six heldout task codes is 408 B for shared rank-2 structure versus 488 B for shared module bank plus six independent coefficient vectors (16.4% savings); the preregistered byte gate was set to 10% before any fresh execution.

- This is a synthetic mechanism screen derived from PA234, not a reproduction of its original experimental tasks.
- Native rank-2 PCA is an explicit functionally equivalent control; any performance result shared with that control is non-Mirror-specific.
