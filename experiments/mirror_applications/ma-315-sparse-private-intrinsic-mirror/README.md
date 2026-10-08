# MA-315 — shared intrinsic Mirror view plus sparse private residuals

Status: **FAIL** under the preregistered storage gate. Dedicated branch: `research/ma-315-sparse-private-intrinsic-mirror-20261008`.

## H — hypothesis

For a mixed bank of functions sharing a fixed-radius 2D orbit with sparse private residual dimensions, replacing two common FP16 coefficients with one Mirror angle should reduce actual serialized payload by at least 10% while maintaining held-out normalized MSE <=1e-4.

## T — protocol and controls

A post-fit synthetic 32D linear task bank used one shared 32x16 random orthonormal basis and 128 tasks. Each task had a common radius-0.25 two-coordinate component plus 0/2/4/8 private residuals among the remaining coordinates. Each task had 96 support, 64 validation and 128 test examples. Development seeds were 31501/31502; frozen fresh seeds were 31511–31513. No gradient updates were used.

Controls were hard tie, fixed intrinsic dimensions 2/4/8/16, adaptive dense intrinsic codes, the matched two-coefficient shared address with identical sparse private residual allocation, Mirror angle with identical private residual mechanism, and independent full vectors. Payloads are deterministic ZIP/NPY files; all bases, task offsets/indices, codes, radius, headers and metadata are charged. The corrected direct control refits its common pair jointly with selected residuals; an earlier development variant is preserved under `protocol_variants/pre_direct_pair_refit/` and is not pooled. Frozen gates and amendment are in `PROTOCOL.json` and `PROTOCOL_AMENDMENTS.md`.

## Results

**Fact:** Fresh Mirror payloads were 6,044/6,044/6,056B; the matched direct control was 5,850/5,850/5,862B. Mirror was 3.31–3.32% larger in all three seeds, so it failed the <=0.90x promotion gate and met the frozen byte-failure rule. Both methods met the quality ceiling: Mirror test nMSE was 9.58e-7–2.43e-6; direct was 8.07e-8–1.13e-6. Mirror fit proxy was 41.9–42.2x direct and fresh throughput was 0.61–0.78x direct. See `FRESH_RESULTS.csv` and `RESULTS_CORE.csv`.

**Interpretation:** Sparse private allocation was useful against dense global coordinates in this constructed bank, but the Mirror address did not improve the storage/quality point over direct coefficients. Fixed radius was known to the direct control, so one angle did not remove enough charged state to overcome the metadata and payload layout cost.

**Hypothesis:** With the current format, task count and FP16 coding, direct common coefficients are the better implementation. Larger banks or packed formats could amortize the metadata, but require a new protocol.

## Verification

Three unit tests passed. Deterministic replay covered all 45 summary rows and 1,280 task allocation events; metric values and serialized hashes matched exactly. Fresh seeds were accessed only after commit `9bdcbdd` froze the protocol.

## C / U

**C:** the direct two-coefficient address is already a sufficient and cheaper control for this orbit; angle fitting is expensive and payload metadata erases the coordinate-size intuition.

**U:** natural task functions, nonlinear networks, optimizer-based learning, task/address discovery, other formats and larger-scale amortization remain untested. This synthetic representation screen establishes no neural-network capacity increase.
