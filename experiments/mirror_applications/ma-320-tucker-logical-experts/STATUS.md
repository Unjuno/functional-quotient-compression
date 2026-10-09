# MA-320 status

- Status: PROMISING, narrowly for Givens-aligned logical experts
- Branch: `research/ma-320-tucker-logical-experts-20261009`
- Frozen protocol: `bf10ab7f`
- Fresh: complete, 3 worlds × 3 seeds × 2 strata; 72 rows
- Verification: complete

## H
A shared Tucker bank plus a small Mirror address can express many logical expert matrices with less storage than free per-expert coefficients.

## T
128 logical experts, K=8 shared 16×16 matrices, deterministic balanced top-1 dispatch. Compared full experts, free Tucker coefficients, Mirror Givens codes, generic PCA. Fresh worlds 32010–32012 × seeds 0–2; evaluate after fp16 serialization.

## D
PROMISING narrowly on aligned coefficients: Mirror 4,555 B / routed NRMSE 4.95e-4 vs free Tucker 6,313 B / 2.97e-4 (~28% fewer bytes); generic PCA 4,867 B / 3.52e-4. Mirror and PCA form nearby Pareto points; Mirror fit is slower (~4.0 ms vs 0.32 ms). Independent coefficients require free Tucker (NRMSE 2.93e-4); Mirror is 1.037 and PCA 0.810.

## C
The aligned teacher uses the same Givens orbit as Mirror; generic PCA is nearly as compact and more accurate.

## U
No learned router, natural experts, language model or runtime benchmark.
