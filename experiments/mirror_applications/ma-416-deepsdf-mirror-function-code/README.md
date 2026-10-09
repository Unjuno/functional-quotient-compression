# MA-416 — DeepSDF latent codes versus structured geometry codes

Status: **FAIL at the frozen development screen**. Prior art PA68 (DeepSDF shared decoder and per-instance latent codes).

## H — Hypothesis
Five structured geometry coordinates can reconstruct translated, rotated and scaled instances of a canonical implicit shape with lower held-out error, interpolation error, and actual decoder-plus-code bytes than five unconstrained DeepSDF latent coordinates.

> **Mirror insertion:** this experiment adds `m=(tx,ty,log_rx,log_ry,theta)` before a shared neural implicit decoder, so many transformed shape fields reuse one canonical decoder.

## T — Treatment
The task was 2D ellipse signed-distance reconstruction. The Mirror path used a 2→64→64→1 decoder trained on a canonical circle, then froze it and adapted five geometry coordinates from 256 support points per held-out shape. DeepSDF jointly trained a 7→64→64→1 decoder and unconstrained five-dimensional codes across eight training shapes, then adapted held-out codes from the same support. Eight held-out shapes used disjoint 2048-point query sets. Native affine coordinate warp used the exact same geometry map as Mirror. The analytic ellipse formula served as quality oracle. Two development seeds (41601/41602); fresh 41611–41613 remained sealed.

## D — Decision: FAIL at this fixed development screen

| Seed | Mirror query / interpolation NRMSE | DeepSDF query / interpolation NRMSE | Mirror / DeepSDF payload |
|---|---:|---:|---:|
| 41601 | 0.08474 / 0.09034 | 0.14970 / 0.16363 | 10,028 / 10,699 bytes (0.937x) |
| 41602 | 0.09310 / 0.10079 | 0.15230 / 0.16311 | 10,002 / 10,714 bytes (0.934x) |

The frozen gates required Mirror query NRMSE <=0.03, at least 2x lower than DeepSDF, interpolation NRMSE <=0.05, and payload <=0.80x DeepSDF. Both seeds missed all four limits. The analytic oracle was exact. Mirror query error was about 1.6–1.8x lower than DeepSDF, and its payload was about 6% smaller. The native affine control matched Mirror predictions exactly; payload differed only by metadata (5 bytes). Therefore the observed geometry structure is not Mirror-specific.

The neural methods remained imperfect at the frozen update budget despite an exact analytic oracle. This is a fixed-budget development FAIL and does not establish a converged capacity frontier. Fresh data were not opened.

## C — Strongest counter-hypothesis
The Mirror decoder was pretrained only on the canonical circle while DeepSDF learned a shared decoder across shapes; this setup favors structured geometry, yet the simple native affine code reproduced it exactly. The DeepSDF and Mirror decoders may both need longer training or tuning to reach near-convergence.

## U — Unknown
Near-converged quality, richer shape families, 3D reconstruction, and fresh-seed behavior remain unknown. No natural-shape result is claimed.

## Fact / Interpretation / Hypothesis
**FACT:** held-out and interpolation quality, actual bytes, and output metrics were replayed from serialized FP16 payloads. The analytic oracle was exact. Mirror and native affine were functionally identical. Fresh remained sealed.

**INTERPRETATION:** geometric coordinates improve this fixed-budget synthetic screen modestly over learned DeepSDF codes, but do not meet useful-quality or compression thresholds and have no Mirror-specific value against the direct affine control.

**HYPOTHESIS:** a richer, converged experiment could test whether geometry codes retain an efficiency advantage on non-elliptic shape families; it requires a new protocol.

## Verification correction
The first development output computed interpolation metrics from full-precision in-memory codes. Replay found small differences. `initial_pre_serializer_results.csv` and its screen are retained; canonical `RESULTS_CORE.csv` now reports interpolation metrics replayed from the serialized FP16 decoder and codes. The source verifier confirms the corrected values.
