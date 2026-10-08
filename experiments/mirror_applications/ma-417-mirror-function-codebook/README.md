# MA-417 — codebook for multiple neural functions

Status: **FAIL at the frozen development gate**. Prior art PA68 (DeepSDF).

## H — Hypothesis
For functions with clustered latent states, a small learned codebook plus one address per function can reduce actual shared-decoder-plus-code bytes while preserving held-out quality and interpolation versus full DeepSDF latent vectors.

> **Mirror insertion:** this experiment adds an eight-entry latent codebook `B` and one per-function address to a fixed shared neural coordinate decoder, so many functions select compact latent views of the same physical decoder.

## T — Treatment
One fixed 2→32→8 tanh feature decoder was shared by every method and both seeds. There were 64 training and 64 held-out functions with four private residual levels (σ=0, 0.05, 0.15, 0.30). Held-out function codes were inferred from 256 support points and scored on disjoint 2048-point query sets. Compare full DeepSDF latent vectors, eight-entry codebook/VQ, rank-4 PCA basis, an exact native nearest-centroid VQ alias, and oracle latent codes. The codebook was fit only on training-task inferred latents. Fresh 41711–41713 remained sealed.

## D — Decision: FAIL

At low residual levels, the codebook saved about 22% of total decoder-plus-code payload, but missed quality gates in both seeds.

| Seed | Private σ | Codebook NRMSE | DeepSDF NRMSE | Interpolation NRMSE | Payload ratio (codebook/DeepSDF) |
|---|---:|---:|---:|---:|---:|
| 41701 | 0.00 | 0.0538 | 0.0427 | 0.0491 | 2,323 / 2,986 = 0.778 |
| 41701 | 0.05 | 0.2337 | 0.0436 | 0.1747 | 0.778 |
| 41702 | 0.00 | 0.0660 | 0.0606 | 0.0520 | 2,323 / 2,977 = 0.780 |
| 41702 | 0.05 | 0.2005 | 0.0617 | 0.2683 | 0.780 |

Only the byte gate passed. Even at zero private residual, DeepSDF had lower query error in both seeds. A small residual caused a sharp VQ degradation, showing when private latent state becomes necessary. The rank-4 PCA baseline used nearly the DeepSDF payload and had higher low-residual query error. The native vector-quantization control produced bit-identical outputs to the Mirror codebook; therefore no Mirror-specific advantage is established.

## C — Strongest counter-hypothesis
The training and held-out latent families were generated from an eight-prototype codebook, yet nearest-centroid coding still lost too much information. The codebook therefore has a favorable constructed distribution and still misses useful quality; broader latent distributions would likely worsen the tradeoff.

## U — Unknown
A learned residual on top of the codebook might improve the frontier, but it is a separate factorization and needs a new MA/protocol. No natural decoder or language task is tested.

## Fact / Interpretation / Hypothesis
**FACT:** total inference payload fell from about 2.98KB to 2.32KB; query and interpolation gates failed; VQ exactly matched native nearest-centroid quantization; all 40 development metrics replayed from serialized payloads. Fresh remained sealed.

**INTERPRETATION:** one-byte addresses save storage but do not preserve this decoder's functions, even for clustered low-residual codes. Native VQ fully explains the representation.

**HYPOTHESIS:** adding private residuals could recover quality, at a storage cost that must be measured in a separately frozen experiment.

## Verification correction
Initial interpolation values were computed from in-memory FP32 decoder/code state and pooled all residual levels. Official `RESULTS_CORE.csv` now reports per-residual interpolation scores replayed from serialized FP16 payloads. Initial values remain in `runs/initial_pre_serializer_results.csv`.
