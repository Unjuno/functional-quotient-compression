# MA-327 status

- Status: FAIL
- Branch: `research/ma-327-factorized-layer-expert-tucker-20261009`
- Frozen protocol: `d4b87549`
- Fresh: complete, 3 worlds × 3 seeds × 2 strata; 72 rows
- Verification: complete

## H
Factor layer and expert coefficients; Mirror expert addresses may compress the expert factor and generalize held-out combinations.

## T
16×16 layer/expert grid, K=8 orthonormal matrices, 75% observed combinations. Compared full matrices, flat Tucker coefficients, generic free factors and Mirror factors. Fresh worlds 32710–32712 × seeds 0–2.

## D
FAIL. Aligned held-out combinations: Mirror and generic free factorization both NRMSE ~2.98e-4; Mirror uses 1,552 B vs generic 1,745 B (11% smaller, misses 20% gate). Flat Tucker has held-out NRMSE 1.0. Independent factors: generic free NRMSE ~2.86e-4, Mirror ~1.073.

## C
The generic A×B factorization already provides the logical Cartesian combinations; Mirror adds no required capacity here.

## U
No real network, learned router, nonlinear task training or natural language evidence.
