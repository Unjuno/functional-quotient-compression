# MA-312 status

- Status: FAIL
- Branch: `research/ma-312-many-intrinsic-mirror-codes-20261009`
- Frozen protocol: `b8b3fb81`
- Fresh: complete, 3 worlds × 3 seeds × 2 strata; 54 rows
- Verification: complete

## H
At larger task count, a shared intrinsic basis plus many Mirror coordinates reduces total storage versus independent random-subspace task coefficients.

## T
D=256, intrinsic d=32, N=64. Compared direct least-squares intrinsic coefficients, an 800-update shared Givens code and rank-2 PCA on aligned and independent task strata. Fresh worlds 31210–31212 × seeds 0–2; full payload includes U.

## D
FAIL against the frozen total-byte gate and practical Pareto criterion. Aligned: Mirror 16,727 B / NRMSE 7.39e-6; direct 20,624 B / 3.90e-7; PCA 17,018 B / 2.89e-7. Mirror saves 18.9%, below the 20% gate, despite adaptation-state bytes 309 vs 4,206. Mirror costs ~0.463 s / 800 updates / 3.28M example presentations vs direct ~0.0036 s. Independent tasks: Mirror NRMSE ~0.971 and PCA ~0.910; direct ~3.94e-7.

## C
U dominates payload; larger N or preexisting/shared U amortization may improve the frontier. Analytic angle fitting could cut compute.

## U
No larger scale curve, pretrained model, natural tasks, or inference runtime evidence. Private intrinsic coordinates are required for unrelated tasks.
