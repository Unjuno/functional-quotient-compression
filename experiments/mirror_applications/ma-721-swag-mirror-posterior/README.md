# MA-721 — SWAG posterior subspace with Mirror coordinates

Status: SCREENING; protocol and stratified split were frozen before development metrics.
Branch: `research/ma-721-swag-mirror-posterior-20261008`.
Base commit: `407ca7e2047326d1e4b753e55e05c4730f26f32b`.

Draw 7 selected MA-721 from 528 eligible P0 rows. Seed, index, full pool snapshot/hash and selected registry row are in `source/random_draw.json` and `source/selection_pool.csv`.

## Prior-art delta

PA181 establishes SWAG's mean plus low-rank and diagonal covariance posterior sampling. PA41 establishes rank-one factor Bayesian networks. This experiment tests whether a discrete Rademacher Mirror coordinate over the **same** SWAG covariance directions improves corruption-shift predictive calibration or quality. SWAG with matching rank and covariance is the decisive control; eight independent checkpoints are an upper reference.

The experiment uses an 80-epoch 64-128-64-10 MLP on sklearn digits. It reports clean and Gaussian-corrupted development NLL, accuracy, ECE, Brier, disagreement, real serialized payload bytes and K=8 inference cost. The fresh stratified split remains locked unless all gates in `PROTOCOL.json` pass for both seeds.
