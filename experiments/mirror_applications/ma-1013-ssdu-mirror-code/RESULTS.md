# MA-1013 — SSDU scan-level Mirror code

Status: **FAIL (bounded synthetic development gate)**  
Branch: `research/ma-1013-ssdu-mirror-code-20261115`  
Evidence: synthetic multi-coil scan-level SSDU, actual storage and CPU compute

## H — Hypothesis

A frozen multi-coil prior with one two-angle Givens code per target scan, fitted only on SSDU train samples, should improve held-out scan reconstruction, remain within 1% NMSE of same-budget affine/FiLM/private controls, and compress total scan payloads.

## T — Test

PA307's self-supervised data-splitting principle was followed. A 518-parameter CNN prior was pretrained for 64 updates on 16 analytic source phantoms. Three synthetic target scans each contain eight 32×32 slices with two-coil sensitivity maps and 8/32 Cartesian phase lines. For every slice, one complete acquired phase line is reserved from the training partition. A single scan code is fitted jointly across all eight slices for 48 updates using only zero-filled train-mask inputs, known sensitivity maps, train-mask indices and acquired train k-space. The code optimizer never reads the heldout mask or fully sampled image truth. The heldout acquired line and analytic truth are used only after adaptation for scoring.

Controls: no adaptation; a two-scalar output affine code; eight-coefficient latent FiLM; two-angle Givens Mirror; full-private prior fine-tuning. Code methods and private fine-tuning use the same 48 update steps and all eight slices in every step (1,152 slice presentations across three scans). The private control updates all 518 model parameters. Actual one-scan and three-scan `torch.save` payloads use a shared schema. Fresh scan seeds 1013301/1013307/1013313 remain sealed.

## Development results

| Method | NMSE ↓ | PSNR dB ↑ | Magnitude SSIM proxy ↑ | Heldout k-space RMSE ↓ | Updates | Adapt wall | Three-scan payload |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen prior | .64553 | 23.77 | .20093 | .06901 | 0 | — | 5,205 B |
| Native output affine (2 values) | .60978 | 24.09 | .25802 | .07081 | 144 | .577 s | 5,457 B |
| Latent FiLM (8 values/scan) | .70819 | 23.39 | .21160 | .06786 | 144 | .737 s | 5,521 B |
| Givens Mirror (2 angles/scan) | .68403 | 23.50 | .18916 | .06959 | 144 | .974 s | 5,457 B |
| Private full-prior SSDU fine-tune | .67369 | 23.56 | .21948 | .06735 | 144 | .838 s | 12,333 B |

Mirror is 5.96% worse in NMSE than the frozen prior, and 12.2% worse than the same-size, same-update two-scalar affine control. It does not improve hidden-truth quality even though its heldout k-space residual is close to the frozen model. The complete three-scan payload is exactly the same size as the affine control (5,457 B); it saves only 64 B (1.16%) versus FiLM, below the 25% requirement. Compared with three private models it saves 55.8% bytes, but loses quality to both frozen sharing and affine calibration. Mirror adaptation is 69% slower than the affine control in this CPU run.

The fixed CNN uses 516,096 forward MAC per slice. Each code-based or private method processes 1,152 slice/update pairs, about 594.5M CNN forward MAC plus backprop and 2,304 two-coil FFTs. Givens adds about 14.2M scalar rotation operations across the scan adaptations. Source pretraining took 1.54 s; optimizer/wall measurements are in `runs/development.json`.

## D — Decision

**FAIL.** The leakage-free SSDU code path is verified, but Mirror loses to a same-size ordinary affine control, fails to improve frozen reconstruction quality, and yields no actual byte saving over that affine control. Fresh remains unopened.

## C — Strongest counter-hypothesis

The known coil maps and mask are supplied to the physics operator, so they already explain the scanner shift. The tiny synthetic prior and fixed learning rate may also make learned-code adaptation unstable. The native affine result suggests a low-cost amplitude/offset correction is more appropriate than latent rotations here.

## U — Unconfirmed

No acquired fastMRI or clinical scan, real anatomy, trained VarNet, or real SSDU benchmark. The heldout measured line evaluates self-supervised consistency, not pathology preservation. No fresh replication or real scanner generalization claim.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror NMSE .68403 vs frozen .64553 and affine .60978; Mirror and affine both serialize to 5,457 B; the SSDU heldout partition is disjoint and excluded from both code and private-model training. Replay is exact; fresh unopened.
- **Interpretation:** avoiding reference-image leakage does not make the Mirror code useful; an ordinary affine code gives better synthetic reconstruction at identical bytes and updates.
- **Hypothesis:** scan-conditioned Views may help when the physical operator does not already describe the shift, but they need to beat direct native calibration on acquired, anatomy-diverse scans.
