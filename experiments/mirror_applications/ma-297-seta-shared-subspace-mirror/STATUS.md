# MA-297 status

- Status: FAIL for Mirror-specific value; shared-subspace compression signal observed
- Branch: `research/ma-297-seta-shared-subspace-mirror-20261009`
- Frozen protocol commit: `1eb488ce`
- Fresh: complete, 3 worlds × 3 seeds; 36 result rows
- Verification: complete

## H
After shared support discovery, compact Mirror task codes reduce storage while preserving task functions beyond shared/private sparse storage.

## T
Synthetic sparse linear vectors, D=128, eight tasks, 16 common and four private coordinates each. Compared independent sparse, SETA-style shared/private, Mirror rank-3 code, and generic PCA. Fresh worlds 29710–29712 × seeds 0–2. Actual serialized bytes include indices, values, codes, basis and metadata.

## D
FAIL for Mirror specificity. Support recovery was 100%. Mirror used 836 B and generic PCA 841 B (both query NRMSE ~3.31e-7), versus shared/private 1,055 B and independent sparse 2,004 B. The shared-code compression signal is real in this toy; Mirror is not better than PCA.

## C
The rank-3 generated task family favors generic low-rank factorization and does not represent sequentially trained SETA experts.

## U
No faithful SETA, optimization, routing, continual retention, natural tasks, or inference runtime evidence.
