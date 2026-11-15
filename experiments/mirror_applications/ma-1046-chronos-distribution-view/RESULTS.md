# MA-1046 Results

## H/T/D/C/U

**H — Hypothesis.** A two-value frequency-conditioned distribution View over one frozen Chronos-T5-Tiny model improves probabilistic forecast quality on fresh time-series datasets over zero-shot Chronos and native temperature calibration while adding negligible state and under 10% inference overhead.

**T — Trial.** Official `amazon/chronos-t5-tiny` revision `29d808298f1a62493e7b9a5e08529d0d930fa189`, CPU FP32; development fit on ETTh1/ETTm1 only. Fresh audit: Weather (12 preselected series, 36 windows/seed) and Australian Electricity (5 series, 15 windows/seed), 32 samples/window, seeds 104604–104606. Controls: zero-shot, native temperature, affine, normalized Hermite alias, seasonal naive. No weight updates.

**D — FAIL.** Frozen quality gates fail on both fresh datasets.

**C — Strongest counter-hypothesis.** The distribution transform is an ordinary normalized-Hermite/quadratic residual parameterization; it is not Mirror-specific. A native temperature calibration also has better normalized CRPS on both fresh datasets.

**U — Not established.** Broader model families, larger/independent series samples, GPU runtime, different horizons/frequencies, and any benefit to downstream decisions are untested. Chronos pretraining overlap with both named audit datasets means this is not a pretraining-independent audit.

## Facts

Lower is better for normalized CRPS, WQL, and MASE. Values are pooled over the three prespecified sampling seeds.

### Weather

| Method | Normalized CRPS | WQL | MASE | 90% coverage |
|---|---:|---:|---:|---:|
| chronos | 0.547306 | 0.363312 | 0.610474 | 0.5219 |
| mirror | 0.548043 | 0.369959 | 0.612292 | 0.7090 |
| hermite | 0.548043 | 0.369959 | 0.612292 | 0.7090 |
| native_temperature | 0.546610 | 0.378736 | 0.620967 | 0.7512 |
| native_affine | 0.575056 | 0.424260 | 0.650169 | 0.6744 |
| seasonal_naive | 1.000000 | 1.006417 | 0.863064 | 0.1648 |

### AustralianElectricity

| Method | Normalized CRPS | WQL | MASE | 90% coverage |
|---|---:|---:|---:|---:|
| chronos | 0.742393 | 0.050938 | 0.819200 | 0.7403 |
| mirror | 0.744450 | 0.050929 | 0.825019 | 0.7454 |
| hermite | 0.744450 | 0.050929 | 0.825019 | 0.7454 |
| native_temperature | 0.735809 | 0.048450 | 0.819433 | 0.8519 |
| native_affine | 0.746371 | 0.051074 | 0.822096 | 0.7417 |
| seasonal_naive | 1.000000 | 0.067724 | 0.801792 | 0.0000 |

Mirror normalized CRPS relative to Chronos is 1.00135 on Weather and 1.00277 on Australian Electricity (both worse). Relative to native temperature it is 1.00262 and 1.01174 (both worse). The 10,000 paired series bootstrap for Mirror/Chronos is [1.000524, 1.003946]. The Hermite result differs only at floating-point noise (normalized CRPS differences below 2e-8).

## Actual serialized inference payload

Common frozen checkpoint inference files (model weights, config, generation config) total **33,589,724 B**. Per-method payloads below include the saved float32 coefficient file and compact method metadata JSON; shared checkpoint charged once.
- mirror: incremental 237 B; complete payload 33589961 B.
- hermite: incremental 238 B; complete payload 33589962 B.
- native_temperature: incremental 249 B; complete payload 33589973 B.
- native_affine: incremental 252 B; complete payload 33589976 B.

The Mirror code itself is 8 B (two float32 values); this is not a compression result and it does not reduce the shared checkpoint bytes.

## Compute and interpretation

Decoder p95 ranged 0.742–1.496 s per Weather forecast and 0.925–1.232 s per Australian Electricity forecast across seeds. Mirror postprocessing p95 was below 0.23 ms per forecast, well below the 10% overhead gate; this negligible cost did not improve quality.

**Interpretation.** In this scoped experiment, adding two small output coordinates did not create useful additional probabilistic forecasting behavior. The operation is matched by its normalized-Hermite form, while simpler native temperature calibration is stronger on normalized CRPS. It adds bytes and a small postprocess, with no storage or quality Pareto improvement.

**Hypothesis for follow-up.** If distribution views help elsewhere, their useful degrees of freedom likely need to change more than a normalized residual shape, and should be tested against direct quantile/distribution calibration before attributing any gain to Mirror. No follow-up retuning was performed on this audit.
