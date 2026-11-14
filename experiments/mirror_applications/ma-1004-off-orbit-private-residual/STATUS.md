# MA-1004 status

- Status: SCREENING
- Branch: `research/ma-1004-off-orbit-private-residual-20261114`
- Base commit: `28f0dceef`
- Last verified commit: pending
- Development complete: no
- Fresh/audit opened: no (uracil file downloaded; labels and coordinates unopened)
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

After protocol commit, inspect only `old_indices` and structural metadata to verify split counts; freeze selected row hashes before any energy/force evaluation. Then run CPU feasibility and PA300 sparse-control compatibility on source and development rows only.

## Blockers

The StructureCloud tensor mirror has no independent license metadata; use original ColabFit rMD17 CC0 record as authority and preserve its file hashes. Uracil is downloaded but no labels/coordinates opened. Fixed old-index windows and target selected-row manifest remain to be verified. PA300 control compatibility with this MACE-MP checkpoint must be tested.
