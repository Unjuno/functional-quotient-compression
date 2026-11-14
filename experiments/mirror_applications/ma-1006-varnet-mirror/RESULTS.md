# MA-1006 — VarNet cascade phase-view reconstruction screen

Status: **FAIL — amended development gate**
Branch: `research/ma-1006-varnet-mirror-20261114`
Prior art: PA302; official fastMRI source commit `91f2df4711adbb6d643df1810f234e4abcf5881b` (MIT).

## H — hypothesis

A per-cascade complex Givens angle on one shared VarNet denoiser recovers scanner-shift reconstruction quality lost by hard tying, at lower serialized state than independent cascade weights, and beats an equally small scalar-gate control while leaving the VarNet data-consistency operator unchanged.

## T — what ran

**Fact:** Official `fastmri.models.varnet.VarNet` ran on CPU with three cascades and 4-channel/2-pool sensitivity and regularizer U-Nets. No CUDA device or acquired fastMRI dataset is available. The fixed mechanism dataset contains deterministic 64×64 analytic phantoms, four simulated coils, and 4× Cartesian sampling. This is synthetic evidence only, not fastMRI benchmark or clinical evidence.

**Fact:** Scanner A supplied 24 pretraining examples. Scanner B coil amplitude/phase maps supplied 8 support and 16 development examples. The independent VarNet and a separately initialized hard-tied VarNet were each trained for 24 updates on the same Scanner A examples; the hard-tied denoiser was shared from initialization. Scalar and Givens variants both began from that tied checkpoint and each received 24 view-only updates on the same Scanner B support. The independent upper control received 24 full-model target updates. One seed was used.

**Fact:** Development lock and model checkpoint are `runs/DEV_LOCK.json` and `runs/DEV_MODELS.pt`; checkpoint SHA-256 `fa184959d20de124d7b257cdff0597dd4424003716bb47bc539d8012330956de`. A serialized checkpoint replay in `runs/DEV_REPLAY.json` reproduced all five methods on all 16 development cases with maximum metric difference 0. Fresh seeds 106301, 106307 and 106313 were not generated or evaluated because the development gate failed.

**Fact:** The first pass, which averaged independently trained cascade weights to form the tied control, is preserved in `runs/DEV_LOCK_pre_amendment01.json` and `runs/DEV_MODELS_pre_amendment01.pt`. It is excluded from the decision after the initialization mismatch was found. Amendment 01 is the valid comparison below.

## Development results

| Method | Dev NMSE ↓ | PSNR dB ↑ | Acquired-line residual ↓ | Serialized full inference payload |
|---|---:|---:|---:|---:|
| Independent frozen VarNet | 0.163529 | 22.6560 | 0.31346 | 137,010 B |
| Hard-tied denoiser | 0.160539 | 22.7533 | 0.27158 | 68,678 B |
| Tied + scalar gates | **0.155246** | **22.9014** | **0.26041** | 68,678 B |
| Tied + Givens Mirror angles | 0.160418 | 22.7564 | 0.27160 | 68,678 B |
| Independent full fine-tune | 0.092474 | 25.2200 | 0.13595 | 137,010 B |

Payloads are actual `torch.save` inference objects including all method parameters and architecture metadata. Hard tying saves 49.9% bytes versus the independent frozen model and slightly improves this synthetic scanner-B metric. The three-parameter Givens view improves hard-tied NMSE by only 0.076% relative, while the same-size scalar gate improves it by 3.30%. Mirror also trails the scalar gate on PSNR and acquired-line residual. Independent target fine-tuning obtains much lower NMSE at twice the payload. Inference wall times are recorded in the run JSON; timing differences are not interpreted from this small CPU screen.

The angle is inserted after the denoiser, before the native sensitivity expansion and DC subtraction. It cannot access masks or measured k-space, and view-only methods freeze the cascade DC scalars and sensitivity estimator. The reference VarNet uses learned **soft** DC, so acquired-line residual is not expected to be zero. No exact-projection claim is made.

## D — decision

**FAIL at the amended development gate.** Givens Mirror misses the required 10% improvement over hard tying, is not within 5% of independent fine-tuning, and is dominated by a same-payload scalar gate. Fresh remains sealed.

**Interpretation:** Hard sharing itself gives a useful storage/quality point on this small synthetic case, but the complex phase coordinate adds no useful function beyond ordinary scalar modulation. The result is evidence against this Mirror placement and transform, not against every VarNet View.

## C — strongest counter-hypothesis

The screen has one scanner shift, one seed and 24 support updates. A view over intermediate denoiser channels or a real multicoil distribution shift may behave differently. This does not change the measured failure for the locked post-denoiser phase view.

## U — remaining unknowns

- Natural fastMRI performance over scanner, anatomy and acceleration splits.
- Multi-seed robustness and full-scale runtime.
- Other View placements inside denoiser feature channels.
- Whether hard tying plus an independent sensitivity estimator can retain storage savings on real acquisitions.

## Fact / interpretation / hypothesis

- **Fact:** Amended development metrics replay exactly from the serialized checkpoint; no fresh seed was opened. Actual full inference payload bytes are reported.
- **Interpretation:** The complex phase View is dominated by a scalar gate at equal bytes in this synthetic mechanism screen; hard tying alone halves state at a modest development quality change.
- **Hypothesis:** The scanner-induced residual is primarily amplitude/feature-distribution shift, not a global complex phase correction. A feature-space View requires a separate pre-registered test.
