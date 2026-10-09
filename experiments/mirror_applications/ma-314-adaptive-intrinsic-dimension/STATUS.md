# MA-314 status

- Status: PROMISING, narrowly on aligned variable-rank task functions
- Branch: `research/ma-314-adaptive-intrinsic-dimension-20261009`
- Frozen protocol: `b313de17`
- Fresh: complete, 3 worlds × 3 seeds × 2 strata; 72 rows
- Verification: complete

## H
Adaptive task dimension plus shared Mirror view may reduce storage for tasks on a common orbit; private residuals should be needed for unrelated tasks.

## T
Synthetic D=128, d=32, 24 tasks, four rank levels. Compared direct adaptive coordinates, Mirror view with/without private fallback, and adaptive PCA. Fresh worlds 31410–31412 × seeds 0–2; all payloads include U and metadata.

## D
PROMISING narrowly. Aligned: Mirror 8,557 B / NRMSE 3.78e-7 vs direct 9,110 B / 3.18e-7 and PCA 9,352 B / 5.80e-7; mean active dimension 15/32 and no private splits. Independent: 23/24 tasks split to private residuals; Mirror+private 12,978 B vs direct 9,926 B. No-private Mirror NRMSE 1.218.

## C
The teacher is generated from the same Givens orbit and uses an anchor task; total saving is modest because U costs 8,192 B.

## U
No pretrained model, learned shared basis, task distribution shift or natural-task adaptation.
