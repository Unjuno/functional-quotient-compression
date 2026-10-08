# MA-315 — shared intrinsic Mirror view plus sparse private residuals

Status: SCREENING; frozen fresh seeds unopened. Branch: `research/ma-315-sparse-private-intrinsic-mirror-20261008`.

## H — hypothesis

A shared two-dimensional Mirror coordinate plus sparse task-private residuals should use at least 10% fewer actual bytes than a matched direct two-coefficient shared address plus the same residuals, at normalized held-out MSE <=1e-4. Sparse common/private allocation should also improve on simply storing a dense global 16D coordinate.

## T — protocol

A shared 32D linear base and 32x16 random orthonormal basis serve 128 tasks. Every task has a fixed-radius 2D orbit component and 0/2/4/8 private residual coordinates on other basis columns. Per task: 96 support, 64 validation, 128 test examples. Controls: tied, fixed dense dimensions 2/4/8/16, adaptive dense dimension, shared direct pair plus sparse residual, Mirror angle plus sparse residual, independent full weights. Sparse residual budget is selected from {0,2,4,8} using a frozen 1e-4 validation threshold. No gradient updates. Actual deterministic ZIP/NPY payloads include the base, basis, offsets, private indices and values, shared radius and all headers.

A pre-fresh strengthening refit the direct pair jointly with its selected residuals; earlier dev summaries remain under `protocol_variants/pre_direct_pair_refit/`. See `PROTOCOL_AMENDMENTS.md`. Development seeds: 31501/31502. Fresh seeds 31511–31513 are locked.

## Development observation

The stronger shared-direct control used 5,856B in both development worlds. Mirror used 6,050B (+3.3%) with normalized test MSE about 1.26e-6; the direct control had about 3.9–4.8e-7. Fixed dense d=16 used 7,790B; adaptive dense used 6,254–6,366B. Therefore sparse common/private sharing beats dense global coordinates, while Mirror has not beaten the matched direct shared/private control on development. Fresh remains sealed until this frozen result package is committed.

## C / U

**C:** the 2D teacher is deliberately generated from the same fixed-radius circular family used by Mirror; the direct control is cheaper to fit and more accurate. Actual mode/radius/header overhead may erase the per-task coefficient saving.

**U:** fresh replication, natural/noisy task functions, task-ID discovery, optimizer-driven training, nonlinear models, and capacity remain untested.
