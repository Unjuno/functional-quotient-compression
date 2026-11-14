# MA-1007 — MoDL tied denoiser with iteration Mirror coordinate

Status: **FAIL — development gate**  
Branch: `research/ma-1007-modl-iteration-mirror-20261114`  
Prior art: PA303, MoDL (2018), paper equations at https://arxiv.org/abs/1712.02862.

## H — hypothesis

A per-iteration Givens angle over adjacent hidden-channel pairs in one shared MoDL denoiser recovers useful iteration-specific priors on a held-out synthetic coil scanner, beats the native tied denoiser and a same-size per-iteration scalar-lambda control, and preserves the measured-data forward/adjoint operators.

## T — what ran

**Fact:** An independent compact PyTorch implementation of the PA303 equations ran on CPU: three unrolled steps; one shared 2→8→8→2 residual CNN; five conjugate-gradient iterations per normal-equation solve; four analytic coil maps; 32×32 phantoms; 4× Cartesian undersampling. The official TensorFlow 1.x repository has no root license declaration and was not copied or run.

**Fact:** Scanner A pretraining used 24 examples and 24 updates. Independent and tied models reset the same initialization seed before construction; the tied model shared its CNN from initialization. Scanner B used 8 support examples and 16 development examples. Scalar-step and Mirror variants each trained 24 updates on support; independent fine-tune received 24 target updates. One seed (107001) was used.

**Fact:** The Mirror coordinate is one angle per unroll step; it rotates every adjacent pair of the eight first-layer feature channels. It sees no mask, k-space, or CG state. `A`, `A^H`, coil maps and mask are fixed across methods. The scalar control changes only per-step lambda in the same normal-equation solve.

**Fact:** The serialized development lock/checkpoint are `runs/DEV_LOCK.json` and `runs/DEV_MODELS.pt`; checkpoint SHA-256 `efa840d3117fa43f8479806f5f175408df8d243387fe08b5650f76c1bb481481`. `runs/DEV_REPLAY.json` reproduces all five methods on the 16 development cases with maximum metric difference 0. Fresh seeds 107301, 107307 and 107313 were not generated or evaluated after the development gate failed.

## Development results

| Method | NMSE ↓ | PSNR dB ↑ | Global SSIM ↑ | Measured-line residual ↓ | Payload bytes |
|---|---:|---:|---:|---:|---:|
| Independent frozen CNNs | 0.033736 | 28.8339 | 0.79262 | 0.012790 | 16,810 |
| Native tied MoDL | 0.047493 | 27.3832 | 0.74256 | 0.020961 | 6,609 |
| Tied + per-step scalar lambda | **0.047129** | **27.4171** | **0.74590** | **0.019834** | 6,609 |
| Tied + Givens Mirror feature views | 0.047501 | 27.3828 | 0.74244 | 0.020989 | 6,861 |
| Independent CNNs, target fine-tuned | 0.024894 | 30.1587 | 0.80961 | 0.013585 | 16,810 |

All byte counts are actual `torch.save` serialized full inference payloads including operator/architecture metadata. Hard tying saves 60.7% versus the independent frozen model, but worsens development NMSE by 40.8%. The scalar-step control slightly improves tied NMSE (0.77% relative). Mirror is 0.018% worse than hard tying and 0.79% worse than the scalar-step control, while costing 252 B more serialized state. Independent target fine-tuning is substantially more accurate at 2.54× the Mirror payload.

Mean CPU forward times are in `runs/DEV_LOCK.json`; observed method-to-method differences are small and noisy at this model size. CG relative normal-equation residuals are approximately `7.4e-5` for tied, Mirror, and scalar-step variants. Measured-line residual is not zero because the MoDL reconstruction solves a regularized normal equation; it is reported rather than presented as exact projection.

## D — decision

**FAIL at the development gate.** Mirror does not improve native tied MoDL, is slightly worse than the same-size scalar-step control, costs more bytes, and misses the independent fine-tune quality gate. Fresh remains sealed.

**Interpretation:** In this synthetic screen, a per-iteration orthogonal feature rotation does not recover useful function beyond hard tying. A small change to the normal-equation regularization step is a more effective control. The tied model offers a storage reduction, but the quality loss is large in this setup.

## C — strongest counter-hypothesis

The screen has one synthetic coil-map shift, one seed, and a 24-update adaptation budget. A natural image distribution or a view aligned with coil/Fourier structure could behave differently. This does not change the failed result for the locked hidden-channel Givens view.

## U — remaining unknowns

- Natural fastMRI performance, multi-seed variance, and full-scale runtime.
- Other structured view placements aligned with Fourier or coil axes.
- Whether natural scanner variation can be shared with a useful private fallback.

## Fact / interpretation / hypothesis

- **Fact:** Exact development replay matches the serialized checkpoint; no fresh data were generated. The A/A^H operator passed the adjointness test.
- **Interpretation:** Hidden-channel Givens Mirror is dominated by a scalar lambda-step control in this mechanism screen.
- **Hypothesis:** The scanner shift is better handled through regularization strength than an orthogonal reorientation of shared CNN features.
