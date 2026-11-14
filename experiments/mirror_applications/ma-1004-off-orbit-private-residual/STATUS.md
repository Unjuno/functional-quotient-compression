# MA-1004 status

- Status: SCREENING
- Branch: `research/ma-1004-off-orbit-private-residual-20261114`
- Base commit: `28f0dceef`
- Last verified commit: pending
- Development complete: no
- Fresh/audit shard: deserialized; only old_indices metadata was accessed; fresh coordinate/label values and metrics were not examined
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

After protocol commit, inspect only `old_indices` and structural metadata to verify split counts; freeze selected row hashes before any energy/force evaluation. Then run CPU feasibility and PA300 sparse-control compatibility on source and development rows only.

## Blockers

The StructureCloud tensor mirror has no independent license metadata; use original ColabFit rMD17 CC0 record as authority and preserve its file hashes. Uracil was deserialized for row-index metadata; no coordinate or label values were indexed or inspected. Amendment 01 sets nonempty support/dev/fresh windows from old-index metadata; selected-row manifest is being regenerated before any metrics. PA300 control compatibility with this MACE-MP checkpoint must be tested.
