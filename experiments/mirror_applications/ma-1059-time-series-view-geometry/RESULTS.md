# MA-1059 — Natural tourism-series View geometry

## H — Hypothesis

On held-out monthly tourism series, one frozen Chronos-T5-Tiny plus a shared rank-2 output LoRA basis and one support-inferred angle should recover useful forecasts with less adaptation state than independent per-series rank-2 LoRA. A support-selected scalar trend residual should be needed for at most 25% of series. A same-basis diagonal control must be beaten by at least 0.1%, and Mirror must improve by at least 2% over native and pooled LoRA.

## T — Trial

Pinned Chronos-T5-Tiny revision `29d808298f1a62493e7b9a5e08529d0d930fa189` (CPU float32) and Monash monthly Tourism revision `58aafbe2712ff481c014f562e42723f2820fd5d4`, archive SHA256 `6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c`. Forty fit series trained the shared basis for 30 updates; 16 development series fixed per-series updates and residual threshold. The 24 audit series were assigned by SHA256 ID ordering and held sealed until `AUDIT_FREEZE.json` was committed. T1–T8 were excluded after metadata preview. Context and horizon were 12 months. AM-01 fixed a pooled-hook leak and residual denominator error before audit access. Corrected selected updates are Mirror 10, diagonal 30, full two gains 30, independent 10; residual threshold 0.025.

Two fresh sampling seeds (105903, 105904) were evaluated on the same 24 held-out series. Controls: native Chronos, pooled rank-2 LoRA, same-basis one-angle Mirror, byte-near diagonal, two direct gains, independent rank-2 LoRA, Mirror plus sparse trend residual, persistence and 12-month seasonal naive. Actual inference payloads and required shared factors, coordinates, residuals and metadata were serialized. Per-method forecasts were replayed from those payload files.

## Fact — fresh audit results

Lower MASE and CRPS are better. Values below are the arithmetic mean of the two sampling seeds; each seed's metrics are retained in `runs/audit/results.json` and `RESULTS_CORE.csv`.

| Method | Macro MASE | Macro CRPS | Actual complete payload bytes | Method state bytes |
|---|---:|---:|---:|---:|
| Native Chronos | 5.564286 | 2572.600 | 33,590,178 | 454 |
| Pooled LoRA | 5.584619 | 2549.169 | 33,625,003 | 35,279 |
| Mirror | 5.587546 | 2569.544 | 33,625,090 | 35,366 |
| Diagonal gain | 5.580430 | 2568.298 | 33,625,088 | 35,364 |
| Full two gains | 5.579145 | 2568.587 | 33,625,184 | 35,460 |
| Independent per-series LoRA | 5.633677 | 2579.417 | 34,425,767 | 836,043 |
| Mirror + trend residual | 5.663872 | 2626.904 | 33,625,146 | 35,422 |
| Persistence | 6.027159 | 2884.479 | 459 | 459 |
| 12-month seasonal naive | 3.735720 | 1258.104 | 462 | 462 |

Residual selected for 3/24 audit series (12.5%); applying it increased macro MASE from 5.587546 to 5.663872. Mirror MASE was -0.42% vs native, -0.05% vs pooled LoRA, and -0.13% vs diagonal (negative means worse). Mirror falls short of the registered >=2% improvements and the >0.1% diagonal margin. Its quality is within 5% of independent LoRA. Mirror adapter state is 95.77% smaller than independent state, but complete payload is only 2.33% smaller because the common checkpoint is 33,589,724 B.

All 432 saved method/seed forecasts were recomputed from serialized payload files and matched bit-exactly. The 432 per-series MASE/CRPS rows replayed with maximum difference 0.0. Actual serialized file sizes were remeasured; see `runs/audit/verification_replay.json` and `VERIFICATION.json`.

## D — Decision

**FAIL**, for this preregistered monthly Tourism screen. Mirror fails the native, pooled-LoRA and byte-near diagonal quality gates. The seasonal-naive control is substantially better. A sparse trend residual meets the 25% frequency limit but worsens quality. The shared Mirror code is much smaller than independent adaptation state, while full-payload savings remain only about 2.3%.

## C — Strongest counter-hypothesis

Chronos-T5-Tiny with a 12-month context is poorly matched to this monthly Tourism forecast: seasonal naive lowers MASE by about one third relative to Mirror. The poor base forecast may mask useful adapter behavior; alternatively, the strong seasonal pattern makes the learned View unnecessary. Either way, this registered setup did not show a useful Mirror frontier.

## U — Unconfirmed

This is one benchmark family, two stochastic forecast draws on the same 24 audit series, and one CPU configuration. Independent worlds, other frequencies, longer contexts, near-convergence, and capacity are not tested. No broad time-series or capacity claim follows.

## Interpretation

**Fact:** Mirror is slightly worse than native Chronos, pooled LoRA and diagonal on audit MASE. Seasonal naive is much better. Mirror uses 35,366 B method state versus 836,043 B for independent LoRA, yet complete bundles differ by only 2.33%.

**Interpretation:** In this setup the small per-series angle does not improve the storage/quality frontier; a simple diagonal code slightly wins at near-identical bytes. The sparse trend residual does not recover useful quality. Adapter-state reduction is real, but it is not a practical full-checkpoint compression result.

**Hypothesis:** A seasonal or longer-context base model could change the quality picture, but this would be a new experiment and is not evidence for the frozen MA-1059 hypothesis.

## Compute and provenance

CPU float32, one Torch thread; no GPU. Across audit series, support adaptation wall time was 3.53 s Mirror, 10.19 s diagonal, 10.20 s full gains, and 3.39 s independent LoRA. The support-fold residual selector added 43.25 s. Query latency varied across the two runs; per-seed p50/p95 is in `runs/audit/results.json` (CPU shared-load/thermal variation was visible). FLOPs were not measured. Development and audit artifacts and exact serialized replay are retained. The earlier pre-AM01 FAIL and its faulty outputs remain preserved as invalidated history.
