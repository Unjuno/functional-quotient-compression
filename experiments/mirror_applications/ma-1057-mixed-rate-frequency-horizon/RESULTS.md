# MA-1057 results: mixed sample-rate × horizon Moirai View

## H — hypothesis
Factorized frequency and physical-horizon calibration fitted on three cells will improve a fresh fourth cell over native and additive controls, with smaller state than independent cell calibration.

## T — protocol and execution
Official `Salesforce/moirai-1.1-R-small` at revision `0c24ab99db2c1a70ea2a0fc03bf113329772ac64`, `uni2ts==1.1.1`, CPU float32. PEMS-BAY revision `c858397fb381d82c3d55ad0a81beb51a9bb9bdaa`; selected stored sensor IDs 14–21, disjoint from MA-1048 IDs 0–13. One shared patch size 32 selected on development data. Three fit cells: 5m×1h, 5m×24h, 15m×24h; fresh audit only on 15m×1h. Matched 24-hour contexts and physical horizons. Sampling seeds 105702 and 105703, each with 24 sensor-window forecasts (768 sampled paths). These are sampling repeats on one held chronology, not independent worlds. No model updates.

## Fact
Audit metric values (CRPS / WQL / MASE):

| Method | Seed 105702 | Seed 105703 | Complete serialized payload bytes |
|---|---:|---:|---:|
| native | 1.877610 / 0.035564 / 0.618921; 1.788417 / 0.037075 / 0.603755 | 55,321,215 |
| global | 3.884595 / 0.067024 / 1.223131; 3.868359 / 0.067784 / 1.218625 | 55,321,619 |
| frequency_only | 3.452215 / 0.058175 / 1.165239; 3.414215 / 0.060274 / 1.154471 | 55,321,635 |
| horizon_only | 1.863047 / 0.035236 / 0.616796; 1.775291 / 0.036736 / 0.601148 | 55,321,633 |
| additive_log | 7.346499 / 0.140763 / 2.256485; 7.201155 / 0.147936 / 2.233561 | 55,321,641 |
| mirror_factorized | 7.346607 / 0.140765 / 2.256518; 7.201262 / 0.147938 / 2.233593 | 55,321,654 |
| independent | 3.121661 / 0.056205 / 0.961990; 3.111888 / 0.056334 / 0.961048 | 55,321,648 |
| seasonal_naive | 1.732639 / 0.035926 / 0.427422; 1.732639 / 0.035926 / 0.427422 | 388 |

Common official model weights plus config total 55,320,882 B and are charged to Moirai methods. Calibration methods additionally serialize 128 B of the frequency×sensor mean/std statistics required to reconstruct outputs. Mirror uses 32 coefficient bytes, additive uses 24 B. Seasonal naive is a separate model-free baseline and is charged only 388 B of predictor/metadata state. Native decoder p50/p95 CPU inference was 11.54/14.82 ms per window; calibration is a one-time fitted mapping and adds a scalar transform.

## D — decision
**FAIL (scoped).** The preregistered quality gates fail on both sampling seeds. Calibration makes the held-cell forecasts substantially worse. Mirror and additive controls are effectively tied; Mirror adds 13 B beyond additive and 6 B beyond the independent cell calibrator in this serialized layout. Seasonal naive materially outperforms Mirror in CRPS and MASE.

## C — strongest counter-hypothesis
Calibration fitted on only three cells does not transfer to the held frequency×horizon interaction; the shared Moirai forecasts were already strong on this cell. The 24-hour context allows seasonal persistence to be competitive. The independent-cell result is not a valid upper bound on the held cell because its cell-specific coefficients have no support data and are regularizer-driven.

## U — unresolved scope
Only one network, reused PEMS-BAY chronology, one held cell, and eight sensors were measured. Sampling seeds are not independent dataset worlds. No independent dataset-family or capacity claim.

## Interpretation
For the registered factorization, `log(s)=log(u_f)+log(v_h)` and `b=a_f+c_h`, exactly the ordinary additive-log function family. No Mirror-specific gain appears, and it costs slightly more serialized state.

## Accounting correction after audit
The first report draft omitted the train-only mean/std arrays needed by affine inference and incorrectly charged the Moirai checkpoint to the seasonal-naive comparator. The audit results were replayed from the exact serialized float32 coefficients and normalization values. Corrected actual payloads are listed above. This correction changed no fitted code, hyperparameter, data selection, or audit access; the FAIL decision is unchanged. `FRESH_FREEZE.json` preserves the protocol and pre-audit source hashes; `source/accounting_correction.py` and `source/verify.py` reproduce the correction and replay.

See `VERIFICATION.json` for hashes and replay checks.
