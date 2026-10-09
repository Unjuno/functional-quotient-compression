# MA-319 status

- Status: FAIL
- Branch: `research/ma-319-tucker-matrix-bank-mirror-20261009`
- Frozen protocol: `196c2649`
- Fresh: complete, 3 worlds × 3 seeds × 2 strata; 72 rows
- Verification: complete

## H
Replacing free Tucker layer coefficients with shared Givens Mirror addresses reduces total payload while preserving layer functions.

## T
K=8 bank of 32×32 matrices across 64 layers; compared full matrices, free Tucker coefficients, Mirror Givens codes and generic PCA. Fresh worlds 31910–31912 × seeds 0–2. Every payload includes fp16 bank/factors and metadata; quality measured after decode.

## D
FAIL: aligned Mirror is 16,699 B / NRMSE 5.03e-4 versus free Tucker 17,559 B / 2.93e-4 (4.9% saving, misses 10% gate) and generic PCA 16,887 B / 3.56e-4. Independent coefficients need free Tucker: NRMSE 2.92e-4 vs Mirror 1.036 and PCA 0.781. Shared bank dominates bytes.

## C
More layers or a smaller bank could increase coefficient savings; the aligned task family may also favor the Mirror transform.

## U
No natural Transformer weights, downstream language quality or fused runtime evidence.
