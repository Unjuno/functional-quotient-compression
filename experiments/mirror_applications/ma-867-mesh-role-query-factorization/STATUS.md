# MA-867 status

- Status: FAIL for Mirror-specific attribution; native factorization screen passes on the synthetic world
- Branch: research/ma-867-mesh-role-query-factorization-20261111
- Base commit: b9af24dc
- Development complete: fixed rank2/600-step protocol
- Fresh/audit opened: yes; seeds 601, 617, 631
- Results committed: pending
- Verification committed: pending

## Decision

Rank2 factorization generalizes the heldout low-rank utility matrix (MAE <=.0002), matches the oracle query MSE, halves policy bytes versus flat table, and uses 4x fewer messages. Native rank2 matrix factorization is identical to the Mirror decoder, so Mirror-specific value is FAIL.

## Next action

Continue to MA-869, carrier-connectivity-aware code allocation.
