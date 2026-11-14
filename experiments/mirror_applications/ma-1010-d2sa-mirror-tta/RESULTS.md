# MA-1010 — D2SA slice-distribution Mirror TTA

Status: **FAIL (bounded synthetic development screen)**  
Evidence lane: **mechanism / storage / compute**  
Branch: `research/ma-1010-d2sa-mirror-tta-20261114`

## H — Hypothesis

On a controlled MRI anatomy shift, a shared distribution coordinate plus compact per-slice Givens Views should stay within 1% relative NMSE of ordinary two-stage FiLM adaptation, save at least 25% of complete per-volume state bytes, and use no extra online updates.

## T — Test

PA305 is D2SA. I inspected the Apache-2.0 official repository at commit `2650d483a7cd77b13242e88719094ee837a45446`. Its source uses a SIREN prompt with patient-level shared adaptation and per-slice latent vectors, then single-slice DEBlock refinement. Its README requires pretrained checkpoints, MRI data and upstream U-Net code; none are present here. This experiment independently implements a compact D2SA-style mechanism screen. It is not an official reproduction.

A 518-parameter CNN prior was pretrained for 80 updates on 32 analytic 32×32 complex phantoms. The target distribution changes phantom shape statistics. The simulated acquisition retains 8/32 Cartesian phase-encode lines. Four target support slices train a shared distribution state in 12 updates; each of eight held-out target slices then receives 12 single-slice self-supervised updates. Adaptation uses only one 80% partition of acquired samples; the disjoint 20% partition scores consistency. Fully sampled target images are used only for post-adaptation NMSE/PSNR scoring, never in TTA losses. Fresh seeds 1010301/1010307/1010313 remain sealed.

Controls: frozen prior; native shared plus per-slice FiLM (8+8 coordinates); Givens Mirror (2+2 angles); and independent per-slice full-prior adaptation. Every alternative uses the same measurement operator and objective. Actual complete payload bytes use one common metadata schema.

## Development results

| Method | NMSE ↓ | PSNR dB ↑ | held-out k-space residual ↓ | Updates | TTA wall | Complete payload |
|---|---:|---:|---:|---:|---:|---:|
| Frozen shared prior | .680663 | 1.728 | .10251 | 0 | — | 5,141 B |
| D2SA-style FiLM | 1.204267 | -0.336 | .11704 | 108 | .274 s | 5,901 B |
| Givens Mirror | .695125 | 1.618 | .10918 | 108 | .331 s | 5,709 B |
| Independent per-slice full prior | .678976 | 1.741 | .10470 | 96 | .185 s | 34,689 B |

Mirror reduces NMSE by 42.3% relative to the unstable FiLM condition, but remains 2.1% worse than frozen sharing and 2.4% worse than independent per-slice tuning. Mirror costs 568 B more than frozen sharing. Relative to FiLM, it saves only 192 B (3.25%), below the 25% gate, and adds about 21% TTA wall time. Both adaptive-code methods perform worse than frozen sharing on hidden target image NMSE. The substantial FiLM degradation suggests an optimization/parameterization sensitivity, not a useful Mirror win.

## D — Decision

**FAIL.** The Mirror code is substantially smaller than FiLM in coordinate count, yet only reduces the complete per-volume inference/TTA package by 3.25% because shared model weights dominate. It fails the frozen byte gate and loses to hard sharing and independent adaptation in reconstruction quality. No fresh run is justified.

## C — Strongest counter-hypothesis

The tiny, undertrained synthetic prior and one fixed learning rate make FiLM optimization unstable, while the D2SA-like support split differs from the official full-volume two-stage training path. The result does not identify what happens with the released pretrained U-Net and real multi-coil data.

## U — Unconfirmed

No acquired MRI, official D2SA checkpoint, upstream U-Net checkpoint, multi-coil evaluation, real scanner shift, or clinical performance. One deterministic synthetic anatomy shift and CPU model only. This is not capacity evidence.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror NMSE .695125, FiLM 1.204267, frozen .680663, independent .678976. Complete bytes are 5,709 B Mirror, 5,901 B FiLM, 5,141 B frozen, 34,689 B independent. Fresh unopened.
- **Interpretation:** Givens coordinates are more robust than this FiLM control under this particular optimization, but do not beat hard sharing and do not materially reduce total stored state.
- **Hypothesis:** a successful D2SA Mirror must replace the large persistent per-slice prompt state while preserving the pretrained multi-coil prior's self-supervised adaptation quality; this synthetic screen does not establish that.
