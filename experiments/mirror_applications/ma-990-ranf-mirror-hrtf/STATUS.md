# MA-990 status

- Status: FAIL for Mirror-specific advantage and quality gate
- Branch: `research/ma-990-ranf-mirror-hrtf-20261008`
- Base commit: `f255f0b` (`research/mirror-application-worker-ready-20261007`)
- Last verified commit: `9cf16e4`
- Development complete: yes
- Fresh/audit opened: yes, after source/protocol lock
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Record checked results and verification, push the dedicated branch, then refresh the live candidate pool and draw again.

## Blockers

RANF neural training and subjective listener testing were not run because PyTorch/GPU and listener-study resources are unavailable. This result is an acoustic signal-field screen.

## Decisions / rulings

- Development ridge selected as 0.0; model rank and sparse code size stayed fixed.
- The ordinary top-4 sparse PCA code is an exact null control and matched Mirror byte-for-byte and metric-for-metric.
- Post-fresh serializer audit added listener identity and support-index metadata; fixed conditions were regenerated and replayed, with no quality settings changed.
- A tiny external-library compatibility fix was documented; the fresh split was rerun only for deterministic replay, with no parameter changes.
