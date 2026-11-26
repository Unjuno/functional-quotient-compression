# MA-1059 — Natural tourism-series View geometry

## H — Hypothesis

On held-out monthly tourism series, one frozen Chronos-T5-Tiny plus a shared rank-2 output LoRA basis and one support-inferred angle should recover useful forecasts with less adaptation state than independent per-series rank-2 LoRA. A support-selected scalar trend residual should be needed for at most 25% of series. A same-basis diagonal control must be beaten by at least 0.1%, and Mirror must improve by at least 2% over native and pooled LoRA.

## T — Trial

Pinned Chronos-T5-Tiny revision `29d808298f1a62493e7b9a5e08529d0d930fa189` (CPU float32) and Monash monthly Tourism benchmark revision `58aafbe2712ff481c014f562e42723f2820fd5d4`, archive SHA256 `6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c`. Forty fit and 16 development series were split by SHA256 of names before numeric parsing; the 24 audit series remain sealed. T1–T8 were excluded after archive preview exposed their rows. Context and forecast horizon are 12 months. Shared rank-2 basis: 30 updates; development-selected per-series updates: Mirror 10, diagonal 30, full two gains 30, independent LoRA 10. AM-01 corrected an active pooled-hook leak and a trend-residual denominator error before fresh audit access.

**Audit is frozen but has not yet run.** Exact source, model, data, selection, development artifact hashes, IDs and seeds are in `AUDIT_FREEZE.json`.

## Fact — corrected development results

Lower MASE is better. These are development-only screening values, not fresh audit evidence.

| Method | Macro MASE | Actual complete payload bytes | Adaptation state bytes |
|---|---:|---:|---:|
| Native Chronos | 4.607814 | 33,589,982 | 258 |
| Pooled rank-2 LoRA | 4.633805 | 33,624,807 | 35,083 |
| Mirror, 10 updates | 4.551984 | 33,624,862 | 35,138 |
| Diagonal gain, 30 updates | 4.563992 | 33,624,860 | 35,136 |
| Full two gains, 30 updates | 4.561471 | 33,624,924 | 35,200 |
| Independent rank-2 LoRA, 10 updates | 4.573154 | 34,147,043 | 557,319 |
| Persistence | 4.455252 | 283 | 283 |
| 12-month seasonal naive | 1.987944 | 284 | 284 |

The frozen development selector chose residual threshold 0.025 and selects 2/16 development series (12.5%). Mirror is 1.77% better than pooled LoRA and 0.26% better than the same-cost diagonal control, below the frozen 2% Mirror-vs-pooled gate. The seasonal-naive control is much stronger on this development subset. Actual serialized inference payload files were byte-counted; model files are included in complete payload totals.

## D — Decision

**SCREENING / PROCEED_TO_FRESH_AUDIT.** The corrected development run does not meet the Mirror quality gate, but audit execution is retained to test generalization and all registered controls. No capacity claim is made.

## C — Strongest counter-hypothesis

The 12-month-context Chronos adaptation may be weak for this monthly Tourism slice: seasonal naive substantially outperforms all learned methods. This may make adapter comparisons secondary to the base forecasting mismatch.

## U — Unconfirmed

Fresh-series quality, residual selection fraction, exact serialized forecast replay, and fresh compute/latency have not yet been measured. This is one benchmark and one deterministic development run; no broad time-series capacity conclusion follows.

## Interpretation

The angle uses far less adaptation state than independent LoRA, while a byte-matched diagonal control nearly matches its development quality. Because the checkpoint dominates full payload bytes and the seasonal baseline is stronger, this does not show a useful storage/quality frontier improvement.

## Compute and provenance

CPU float32, one Torch thread, no GPU. Shared basis training took 0.772 s for Mirror and 0.643 s for pooled LoRA. Selected per-series development fitting took 2.974/8.875/8.167/2.390 s for Mirror/diagonal/full/independent. Actual fresh audit command is frozen for the next run; tests and corrected development payload verifier pass (6 tests).
