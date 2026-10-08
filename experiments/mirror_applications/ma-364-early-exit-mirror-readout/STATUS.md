# MA-364 status

- Status: FAIL for Mirror-specific value
- Branch: `research/ma-364-early-exit-mirror-readout-20261008`
- Base commit: `c935a90`
- Last verified commit: `b75e25d9c1c45895b8b71dcd69b646ee058c309f`
- Development complete: yes
- Fresh/audit opened: no; 36411–36413 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes on integration branch (FAIL)

## Decision summary

Mirror and direct coefficient controls had identical predictions and payload bytes on both development seeds. Shared coefficients saved bytes vs independent heads, but hard sharing was smaller with a small quality cost. No Mirror-specific benefit.

## Next action

Integrated from dedicated branch commit `b75e25d9c1c45895b8b71dcd69b646ee058c309f`; eight development rows replay exactly and three tests pass. Fresh remains sealed after exact direct-control equivalence.
