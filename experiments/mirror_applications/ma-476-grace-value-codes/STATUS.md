# MA-476 status

- Status: FAIL (discrete VQ misses quality; continuous PCA aliases native PCA)
- Branch: `research/ma-476-grace-value-codes-20261008`
- Frozen protocol SHA-256: `19403eb095aab7702146d2ef2596e32d47b470cf89c4da94a1da9ea85e3167f6`
- Development seeds: 47601, 47602 (complete)
- Fresh/audit seeds 47611–47613: sealed, never accessed
- Metric/serialization replay: exact

VQ16/64/128 fail heldout quality; latent int8 is the stronger simple compression control. Continuous PCA is lossless on this aligned task but exactly equals native PCA.
