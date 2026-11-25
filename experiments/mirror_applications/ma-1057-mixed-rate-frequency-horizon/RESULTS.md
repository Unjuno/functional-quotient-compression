# MA-1057 results: mixed sample-rate × horizon Moirai View

## H — hypothesis
Factorized frequency and physical-horizon calibration fitted on three cells will improve a fresh fourth cell over native and additive controls, with smaller state than independent cell calibration.

## T — protocol and execution
Official `Salesforce/moirai-1.1-R-small` at revision `0c24ab99db2c1a70ea2a0fc03bf113329772ac64`, `uni2ts==1.1.1`, CPU float32. PEMS-BAY revision `c858397fb381d82c3d55ad0a81beb51a9bb9bdaa`; selected stored sensor IDs 14–21, disjoint from MA-1048 IDs 0–13. One shared patch size 32 selected on development data. Three fit cells: 5m×1h, 5m×24h, 15m×24h; fresh audit cell 15m×1h. Matched 24-hour context and horizon physical durations. Sampling seeds 105702 and 105703; each evaluates 24 sensor-window forecasts. These are sampling repeats on one held chronology, not independent worlds. No model updates.

## Fact
Audit metric values (CRPS / WQL / MASE):

| Method | Seed 105702 | Seed 105703 | Payload bytes |
|---|---:|---:|---:|
| native | 1.877610 / 0.035564 / 0.618921; 1.788417 / 0.037075 / 0.603755 | 55,321,237 |
| global | 3.884595 / 0.067024 / 1.223131; 3.868359 / 0.067784 / 1.218625 | 55,321,245 |
| frequency_only | 3.452215 / 0.058175 / 1.165239; 3.414215 / 0.060274 / 1.154471 | 55,321,261 |
| horizon_only | 1.863047 / 0.035236 / 0.616796; 1.775291 / 0.036736 / 0.601148 | 55,321,259 |
| additive_log | 7.346498 / 0.140763 / 2.256485; 7.201154 / 0.147936 / 2.233561 | 55,321,267 |
| mirror_factorized | 7.346607 / 0.140765 / 2.256518; 7.201262 / 0.147938 / 2.233593 | 55,321,280 |
| independent | 3.121661 / 0.056205 / 0.961990; 3.111888 / 0.056334 / 0.961048 | 55,321,274 |
| seasonal_naive | 1.732639 / 0.035926 / 0.427422; 1.732639 / 0.035926 / 0.427422 | 55,321,245 |

Both mirror and additive calibration worsen substantially over native; seasonal naive is best on CRPS and MASE. Mirror uses 32 coefficient bytes and 366 metadata bytes (398 B method state); additive uses 24+361=385 B. Shared official model+config is 55,320,882 B, charged once to every complete payload. Audit inference processed 24 windows per sampling seed (768 sampled paths per seed); measured Moirai CPU inference p50/p95 was 11.54/14.82 ms per window. Calibration is a one-time fitted mapping; factorized inference is just scalar transform. Forecast inference latency is common to all heads; no wall-clock advantage is claimed.

## D — decision
**FAIL (scoped).** Fails the preregistered quality criteria on both sampling seeds; mirror and additive are effectively tied and both strongly damage native forecasts. Mirror state exceeds additive by 13 B and is 6 B larger than independent-state calibration in the actual serialized layout. Seasonal naive materially outperforms Mirror.

## C — strongest counter-hypothesis
Only three training cells provide poor calibration coverage of the held frequency×horizon interaction. The native model already performs well there, while the 24-hour seasonal context makes a naive distribution competitive. The calibration family is not a meaningful capacity gain.

## U — unresolved scope
One network, reused PEMS-BAY chronology, one held cell, eight sensors; no independent dataset-family generalization or capacity claim. The limited origins also bound confidence.

## Interpretation
The registered factorized scale and offset do not create additional functional freedom over ordinary additive-log controls: `log(s)=log(u_f)+log(v_h)` and `b=a_f+c_h`. The audit confirms no Mirror-specific gain and a small serialized-state penalty.

Verification and exact frozen hashes are in `VERIFICATION.json` and `FRESH_FREEZE.json`.
