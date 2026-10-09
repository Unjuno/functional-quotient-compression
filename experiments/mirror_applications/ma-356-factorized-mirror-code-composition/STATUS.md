# MA-356 status

- Status: FAIL for Mirror-specific value; oracle factorized-code result
- Branch: `research/ma-356-factorized-mirror-code-composition-20261009`
- Base commit: `c935a90`
- Last verified commit: `574959cd`
- Development complete: yes
- Fresh/audit opened: no; seeds 35611–35613 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: no

## Decision summary

Factorized state represented 64 distinct held-out combinations at zero error in 2.63KB; flat/Product-Key needed 15.5–15.9KB. Direct coefficients were only 1B larger, so no Mirror-specific claim.

## Next action

Add byte/hash/metric replay tests, update registry and evidence ledger, then continue to the next P0 registry item.
