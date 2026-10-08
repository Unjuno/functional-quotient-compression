# MA-297 status

- Status: FAIL
- Branch: `research/ma-297-seta-mirror-subspace-20261008`
- Base commit: `e259f27`
- Development complete: yes
- Fresh/audit opened: yes, locked seeds only
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Commit and push checked result, registry, board, and claim-ledger updates.

## Blockers

None.

## Decisions / rulings

- The random candidate draw, pool, and index are in `PROTOCOL.json`.
- Corrected audit state reconstruction, method-specific serialization, and payload-based quality use the same fresh seeds; invalid initial audit tables are retained.
- FAIL: two fresh aligned seeds miss the 1.10 relative-quality ratio; Mirror also incurs a large fit-compute and inference-throughput penalty.
