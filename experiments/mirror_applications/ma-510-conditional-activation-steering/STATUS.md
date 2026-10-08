# MA-510 status

- Status: **FAIL**
- Branch: research/ma-510-conditional-activation-steering-20261008
- Protocol/source/tests frozen before dev: yes (freeze.json, freeze commit 36c60f0)
- Development: complete (51001/51002)
- Fresh/audit opened: **no** (51011–51013 remain sealed)
- Verification: deterministic payload+metric replay; 3 tests passed

## Decision

Rank4/rank4 misses frozen false-trigger, event-recall and behavior-accuracy gates in both dev worlds. Actual payload and compute improve over explicit CAST, but native PCA/SVD exactly matches both. FAIL; fresh stays sealed.

## Fact / interpretation / hypothesis

- Fact: at rank4/rank4 the payload is 4,334 B vs 7,532 B explicit CAST; operation proxy 640 vs 1,088/example.
- Fact: event recall .145/.071, conditional behavior accuracy .944/.830, FPR .0173/.0105 and output relative RMSE .934/.977.
- Fact: native PCA exactly aliases every Mirror rank point; replay payload hashes and metrics are identical.
- Interpretation: standard low-rank bank compression saves bytes and operations, but this router misses the frozen quality gates.
- Hypothesis: semantic condition representations in trained LMs may permit different natural condition/action frontiers; not tested here.

## Next action

After ledger/board integrity passes and this branch is pushed, proceed to MA-511. Do not open fresh seeds.
