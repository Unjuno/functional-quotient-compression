# MA-1047 Results

## H / T / D / C / U

**H — Hypothesis.** A compact horizon-by-domain factorized affine output code on one frozen TimesFM decoder improves point-forecast MASE on later windows over zero-shot TimesFM and a native additive calibration, while costing less state than separate per-domain, per-horizon affine heads.

**T — Trial.** Official TimesFM 2.5 200M PyTorch checkpoint `google/timesfm-2.5-200m-pytorch`, pinned revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`; 231,289,280 FP32 parameters, no model updates. Two datasets (ETTh1, ETTm1), seven numeric channels each. Four calibration origins/channel/dataset on the first 60% of time, two validation origins on the next 20%, and three predeclared final-20% origins/channel/dataset for fresh audit. Each uses context 512 and one 720-step inference, scored at horizons 96, 192, 336, and 720. Controls: native TimesFM, global shared affine, additive domain+horizon affine, independent domain×horizon affine, seasonal naive. Code selection was frozen before the final-20% audit.

**D — FAIL.** The factorized View does not pass quality gates on either dataset and is effectively matched by the additive native low-parameter control. H100 latency is separately NOT ESTABLISHED because this environment has CPU-only PyTorch.

**C — Strongest counter-hypothesis.** The 4 fit windows per channel may be too few to estimate horizon effects robustly. The two-domain affine calibration screen is also a weak adaptation mechanism; a trained neural head or a wider set of domains may behave differently.

**U — Unconfirmed.** No learned internal TimesFM head was trained, no new domains or variates were held out, and no H100 latency/memory measurement was available. ETT data may overlap TimesFM pretraining sources; this tests later time windows from fixed series, not independent-pretraining generalization.

## Facts

Lower MASE and sMAPE are better. Audit figures macro-average 7 channels and 4 horizons within each dataset; each channel/horizon score averages the three final-period origins.

### ETTh1

| Method | MASE | sMAPE (%) | Complete payload (B) |
|---|---:|---:|---:|
| timesfm | 2.274407 | 43.2324 | 925,181,938 |
| shared | 2.252588 | 42.6299 | 925,181,957 |
| additive | 2.276271 | 43.0950 | 925,182,011 |
| independent | 2.324710 | 44.0401 | 925,182,043 |
| mirror_factorized | 2.276656 | 43.1061 | 925,182,028 |
| seasonal_naive | 2.331991 | 43.9306 | 0 |

### ETTm1

| Method | MASE | sMAPE (%) | Complete payload (B) |
|---|---:|---:|---:|
| timesfm | 4.087305 | 42.2598 | 925,181,938 |
| shared | 4.428914 | 43.2824 | 925,181,957 |
| additive | 4.594271 | 43.8617 | 925,182,011 |
| independent | 4.549058 | 43.6774 | 925,182,043 |
| mirror_factorized | 4.594772 | 43.8683 | 925,182,028 |
| seasonal_naive | 4.308486 | 41.8738 | 0 |

### Audit MASE by horizon

| Dataset | H=96 | H=192 | H=336 | H=720 |
|---|---:|---:|---:|---:|
| ETTh1 / timesfm | 2.1137 | 2.1169 | 2.2648 | 2.6023 |
| ETTh1 / additive | 2.0307 | 2.1261 | 2.3266 | 2.6216 |
| ETTh1 / mirror_factorized | 2.0297 | 2.1276 | 2.3278 | 2.6215 |
| ETTh1 / independent | 2.0202 | 2.1488 | 2.4776 | 2.6523 |
| ETTm1 / timesfm | 3.8426 | 4.0130 | 4.3811 | 4.1126 |
| ETTm1 / additive | 4.5449 | 4.4794 | 4.6862 | 4.6666 |
| ETTm1 / mirror_factorized | 4.5452 | 4.4782 | 4.6855 | 4.6702 |
| ETTm1 / independent | 4.4839 | 4.3153 | 4.8233 | 4.5738 |

## Actual serialized payload

The common checkpoint state is 925,181,579 B (925,181,104 B model.safetensors + 475 B config.json). Each method pays its saved float32 code and its compact method/config JSON. Source code is listed separately.
- timesfm: 359 B incremental; 925,181,938 B complete.
- shared: 378 B incremental; 925,181,957 B complete.
- additive: 432 B incremental; 925,182,011 B complete.
- independent: 464 B incremental; 925,182,043 B complete.
- mirror_factorized: 449 B incremental; 925,182,028 B complete.

The factorized state has 12 float32 values (48 B), compared with 16 values (64 B) for the independent affine table, but its full payload is only 15 B smaller because the checkpoint dominates. It is 17 B larger than the additive native control, which has 10 float32 values. These byte differences provide no useful compression frontier at this model size.

## Compute

Audit CPU inference p50/p95 is 0.297/0.321 seconds per 720-step forecast, measured by dividing batch wall time by batch size. Affine output-head CPU postprocess p50/p95 is 0.0031/0.0043 ms per 720-value forecast. CUDA/H100 was unavailable, so H100 latency is not measured.

## Interpretation

**Interpretation.** The factorized coordinate adds no useful logical forecasting function in this screen. On ETTh1 it is about tied to raw TimesFM; on ETTm1 it materially worsens MASE. The additive control has nearly identical quality to the factorized View at smaller incremental state. Independent affine heads are also worse than raw TimesFM. A shared two-scalar affine calibration improves ETTh1 slightly (about 0.96%) but worsens ETTm1, so it does not form a cross-domain win.

**Hypothesis for follow-up.** Horizon/domain sharing may help only when the underlying domains admit a stronger common adaptation signal than a few affine calibration terms can express. Testing a trained head would require a separate protocol and additional development data. This audit did not change the candidate after seeing fresh metrics.

## Post-audit reporting amendments

AM-01 adds the preregistered seasonal-naive reducer using saved audit origins, histories, and labels; all learned methods and decision gates are unchanged. AM-02 stores the already frozen additive table in its minimum 10-float equivalent form for exact byte accounting. Frozen factorized coefficients and all saved TimesFM forecasts remain unchanged. See `PROTOCOL.json`.
