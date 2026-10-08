# MA-508 status

- Status: **FAIL for Mirror-specific attribution** (rank-8 rho=0 quality/storage screen passes)
- Branch: `research/ma-508-activation-addition-mirror-basis-20261008`
- Protocol frozen before development: yes (`freeze.json`)
- Amendment 1: causal metric correction only; initial results preserved and same seeds rerun
- Development: complete (50801/50802)
- Fresh/audit opened: **no** (50811–50813 remain sealed)

## Decision

At rank 8/rho=0, heldout probe outputs are exact and actual payload is 8,324 B versus 13,194 B explicit (0.631x, passing <=.65). Native PCA exactly matches payload/output. Compute is 576 vs 64 operations/example. FAIL for Mirror attribution; no fresh access.

## Fact / interpretation / hypothesis

- Fact: every heldout behavior code changes the hidden output; 8/8 heldout views are unique.
- Fact: rho=.1 RMSE is .0629/.0799; rho=.25 rises to .155/.200.
- Interpretation: low-rank activation-addition compression works here as standard PCA, at extra inference compute.
- Hypothesis: natural behavior-vector banks may have a different storage/quality/private-state frontier.
