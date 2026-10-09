# MA-488 status

- Status: FAIL (registered <=80% bytes for every heterogeneity level missed)
- Branch: `research/ma-488-shared-private-dictionary-20261009`
- Base: `d90dface`; development tau=.05; fresh worlds 48810-48812 × seeds 0-2

H: Selectively private residuals preserve quality below dense storage.

T: Shared rank-4 basis, 64 functions, 25/50/75% off-manifold fractions; dense/shared/all-private/adaptive controls; actual bytes.

D: Adaptive exactly restores outputs. Payload/dense ratios: 62.7%, 83.6%, 104.4% for 25/50/75%. Shared-only NRMSE .480/.605/.667. FAIL because gate requires <=80% at every fraction.

C: Full private vectors erase savings as heterogeneity grows.

U: Learned dictionaries, private low-rank codes, natural functions.
