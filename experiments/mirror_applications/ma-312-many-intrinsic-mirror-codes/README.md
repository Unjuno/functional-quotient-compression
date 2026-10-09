# MA-312 — Shared intrinsic basis + many Mirror task coordinates

Status: SCREENING. Prior art: PA33 intrinsic-dimensional fine-tuning.

## H

At larger task counts, a shared intrinsic basis plus small per-task Mirror coordinates may reduce total storage compared with independent intrinsic vectors. Generic PCA is the key control; independent task vectors measure the boundary.

## T

Synthetic regression, D=256, intrinsic d=32, 64 tasks; 64 support and 128 query examples/task. Compare direct least-squares intrinsic coefficients, a jointly optimized shared Givens vector + 64 angles (800 updates), and generic rank-2 PCA. Test aligned-orbit and independent task strata. Development worlds 31200–31201; fresh 31210–31212; three seeds each. Report full serialized payload including U and marginal adaptation payload separately.

## D

Pending development/fresh runs.

## C

Shared basis amortization can reduce task-state bytes while generic PCA matches Mirror. At 64 tasks the shared projection may still dominate full payload; Mirror optimization may remain expensive.

## U

Synthetic linear regression only; no pretrained model or natural tasks.

Development: on aligned tasks, Mirror query NRMSE is ~3.85e-6 and total payload 16,727 B vs independent coordinates 20,624 B (18.9% lower, just below the 20% gate); generic PCA is 17,018 B with similar quality. Marginal task state is 309 B Mirror vs 4,206 B direct, but shared U dominates total bytes. Mirror fit takes ~0.48 s / 800 updates versus ~3.8 ms direct. On independent tasks, Mirror NRMSE ~0.98, PCA ~0.91, while direct remains near exact. Fresh uses the locked conditions.
