# MA-307 status

- Status: PROMISING, narrowly scoped synthetic allocation feasibility
- Branch: `research/ma-307-packnet-mirror-before-allocation-20261009`
- Frozen protocol commit: `4b108ea5`
- Fresh: complete, 3 worlds × 3 seeds; 45 result rows
- Verification: complete

## H
Try a task View before assigning new per-task weights; allocate sparse private residuals only when a task exceeds a frozen reconstruction threshold.

## T
Synthetic D=2048 linear tasks: eight share a rank-3 basis on 128 coordinates; four add sparse novel residuals. Compared PackNet-style independent sparse weights, Mirror plus residual fallback, robust generic shared-basis fitting, ordinary PCA, and no split. Fresh worlds 30710–30712 × seeds 0–2.

## D
PROMISING only for this aligned screen. Mirror payload 1,658 B vs PackNet 9,358 B (~82% less); query NRMSE 0.000394. It split four novel tasks and stored ~62 residual values. Including shared basis and task codes, total physical values are ~482 versus 1,536 PackNet (~69% less). No-split view used 1,246 B but NRMSE 0.0996. Robust generic basis also split four tasks and was close: 1,959 B, NRMSE 0.000891, ~62 residual values.

## C
Teacher shared tasks are exactly generated from Mirror's stored basis, and their latent codes are directly supplied. A better generic factorization could close the remaining gap.

## U
No trained PackNet, optimizer retention, natural tasks, real network, or inference-latency evidence. This is representational feasibility, not a general PackNet replacement claim.
