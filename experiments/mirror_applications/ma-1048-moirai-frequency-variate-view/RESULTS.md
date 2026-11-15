# MA-1048 Results

## H / T / D / C / U

**H — Hypothesis.** A factorized frequency×variate distribution code over one frozen Moirai decoder improves probabilistic forecast quality on entirely held-out frequency/sensor combinations compared with the universal model and simpler calibration controls, at lower serialized state than independent pair tables.

**T — Trial.** Official Salesforce Moirai-1.1-R-small, revision `0c24ab99db2c1a70ea2a0fc03bf113329772ac64`, 13,827,528 FP32 parameters, CPU. Pinned fresh source PEMS-BAY (325 sensors, 52,116 five-minute rows); first 14 stored sensor columns were selected by the frozen finite-data rule. Created 5-minute and nonoverlapping 15-minute means. Fit used 16 frequency×sensor pairs ×4 origins; validation used the same pairs ×2; final audit used 12 withheld pairs ×3 origins, with 32 Moirai samples per forecast. Physical forecast horizon is one day (288 five-minute / 96 fifteen-minute steps). Controls: zero-shot, global, frequency-only, variate-only, additive, factorized View, seasonal naive. No model updates.

**D — FAIL.** The factorized View is much worse than zero-shot Moirai on both held-out frequency groups and is matched by the additive control. Development selected seasonal naive; audit settings and coefficients were not changed.

**C — Strongest counter-hypothesis.** Least-squares output calibration extrapolates poorly from a sparse checkerboard of trained frequency×sensor pairs. Factorized offsets reach magnitude 36.7 traffic-speed units, producing held-out 15-minute medians of 66–206 against targets of 9–77. This can be a code-identifiability/transfer failure rather than a limit of Moirai’s underlying capacity.

**U — Unconfirmed.** Only 14 sensors and two resampling rates from one traffic network were tested. Moirai pretraining overlap with traffic data is unknown. No GPU/H100 latency or multi-network generalization was measured. The 15-minute series are deterministic aggregates of the same 5-minute source, not a separately acquired frequency domain.

## Facts

Lower CRPS, WQL and MASE are better. Scores macro-average the six held-out sensor pairs in each frequency group. Each cell has three audit origins and 32 ensemble samples.

**Development fact.** Validation macro CRPS was 3.9846 for zero-shot Moirai, 4.2176 for additive calibration, 4.2176 for factorized View, and 2.8894 for seasonal naive. The frozen selection was seasonal naive.

### 5min held-out frequency group

| Method | CRPS | WQL | MASE | Complete payload (B) |
|---|---:|---:|---:|---:|
| zero_shot | 4.073531 | 0.062699 | 5.064673 | 55,321,242 |
| global | 4.123132 | 0.063946 | 5.163831 | 55,321,250 |
| frequency_only | 4.109959 | 0.064037 | 5.134382 | 55,321,258 |
| variate_only | 5.168267 | 0.079261 | 6.680648 | 55,321,355 |
| additive | 13.995682 | 0.235250 | 15.009147 | 55,321,363 |
| mirror_factorized | 13.994464 | 0.235229 | 15.007570 | 55,321,371 |
| seasonal_naive | 5.608989 | 0.094471 | 5.936302 | 0 |

### 15min held-out frequency group

| Method | CRPS | WQL | MASE | Complete payload (B) |
|---|---:|---:|---:|---:|
| zero_shot | 4.038798 | 0.062788 | 4.124785 | 55,321,242 |
| global | 4.049199 | 0.063147 | 4.087240 | 55,321,250 |
| frequency_only | 4.182345 | 0.064643 | 4.233056 | 55,321,258 |
| variate_only | 4.198499 | 0.066098 | 4.209786 | 55,321,355 |
| additive | 85.075931 | 1.379284 | 62.497894 | 55,321,363 |
| mirror_factorized | 85.061850 | 1.379054 | 62.487787 | 55,321,371 |
| seasonal_naive | 5.256597 | 0.087760 | 4.008909 | 0 |

## Actual serialized state

Common model payload is 55,320,882 B (55,320,200 B model.safetensors + 682 B config.json). Every method additionally pays its own float32 coefficients and actual compact method/config JSON including the selected sensor mapping.
- zero_shot: 360 B incremental; 55,321,242 B complete.
- global: 368 B incremental; 55,321,250 B complete.
- frequency_only: 376 B incremental; 55,321,258 B complete.
- variate_only: 473 B incremental; 55,321,355 B complete.
- additive: 481 B incremental; 55,321,363 B complete.
- mirror_factorized: 489 B incremental; 55,321,371 B complete.
- seasonal_naive: 360 B incremental; 55,321,242 B complete.
- Full independent 2×14 coefficient table: 585 B incremental; 55,321,467 B complete. This is a serialized identity table used only as a storage reference; no independent full-table quality result is claimed.

The candidate stores 32 float32 values (128 B), versus 30 values (120 B) for the additive control and 56 values (224 B) for a full independent table. Factorized View saves 96 coefficient bytes against that full table but costs 8 more coefficient bytes than additive. The complete model payload differs by only 96 B, about 0.00017%, while the forecast quality is much worse.

## Compute

On CPU, audit model inference p50/p95 is 0.286/0.364 seconds per univariate series forecast. Applying an affine transform to one 32×288 sample tensor measured 0.0041/0.0045 ms p50/p95. No CUDA device was available.

## Interpretation

**Interpretation.** A small factorized address does not recover useful logical forecasts on these held-out pairs. The factorized and additive methods have practically identical CRPS (within 0.02%) and both extrapolate badly, especially at 15 minutes. Native Moirai remains much better. The parameter reduction versus a full pair table is real but far too small in complete bytes to offset the quality loss.

**Hypothesis for follow-up.** Frequency×variate codes need either denser pair coverage, explicit regularization, or private residuals for unseen pair interactions. Those changes require a new experiment/amendment; this failed audit was not retuned.
