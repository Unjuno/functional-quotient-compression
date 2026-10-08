# MA-272 status

- Status: PROMISING (3/3 fresh, Givens-aligned synthetic family; eager runtime regression)
- Branch: `research/ma-272-oftv2-mirror-input-views-20261008`
- Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
- Protocol frozen before development: yes
- Fresh worlds opened: yes, LR=0.01 frozen
- Results committed: pending
- Verification committed: pending

## Next action

Replay all rows and payloads, update status/registry/claim ledger, then commit.

## Blockers

No fused GPU kernel; measured eager CPU throughput regresses versus OFTv2.
