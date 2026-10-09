# MA-483 status

- Status: FAIL (registered <=80% fixed-depth payload gate missed)
- Branch: `research/ma-483-adaptive-rvq-depth-20261009`
- Base commit: `21c52609`
- Protocol frozen before fresh: yes
- Development complete: yes; tau=0 selected (threshold sweep tied)
- Fresh complete: yes; 3 worlds × 3 seeds
- Results/verification: yes (`529b4348`)

## Decision

H: Adaptive depth would lower bytes at equal target quality.

T: Fixed R4 versus variable length codes on heterogeneous synthetic 16D vectors, K=32; 9 fresh cases; every serialized code/depth tensor counted.

D: FAIL. Adaptive: 4,637B mean, NRMSE .05615, active depth 2.32. Fixed: 4,829B, same NRMSE, depth 4.0. Adaptive is 96.0% of fixed payload, above the preregistered 80% ceiling. Threshold sweep values all fell below the minimum .125 active amplitude and tied.

C: For N=128, per-function depth and serialization metadata erase most variable-rate savings; sparse static structure may be cheaper.

U: Native adaptive RVQ, entropy coding, larger N, and unknown residual structure.

## Next action

Run artifact tests, record FAIL in registry/claim ledger/board, integrity check, push this branch, then start the next untested P0 in registry order.
