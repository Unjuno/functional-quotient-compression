# MA-1011 — Multi-contrast MRI INR Mirror coordinates

Status: **FAIL (bounded analytic development gate)**  
Branch: `research/ma-1011-multicontrast-inr-mirror-20261114`  
Evidence lane: synthetic MRI mechanism, serialized storage and CPU compute

## H — Hypothesis

For an analytic MPnRAGE-like tissue slice with complementary undersampling over eight inversion contrasts, a shared spatial INR plus two Givens angles per contrast should stay within 1% NMSE of a native continuous-TI shared INR while saving at least 25% of actual deployment bytes versus FiLM and 50% versus independent INR fits.

## T — Test

PA306 describes an image-domain INR mapping two spatial coordinates to a vector of complex contrast values, fit jointly with contrast-specific complementary k-space masks. The paper reports healthy-volunteer MPnRAGE experiments, but no public code was located in the bounded repository search and no acquired subject scan was present in this workspace. This screen is an independent synthetic mechanism test, not a paper reproduction.

The synthetic slice is 24×24 with three analytic tissue classes and inversion-recovery curves at eight TIs (100–1,850 ms). Each contrast has six of 24 Cartesian phase-encode lines; two center lines are common and the others are complementary. Five acquired lines per contrast drive the loss; the sixth is a disjoint measured-data holdout. Fully sampled analytic truth is used only for evaluation. Compared 300 joint updates of (1) native continuous-TI shared INR, (2) shared spatial INR with eight per-contrast FiLM parameters, and (3) same shared field with two learned Givens angles per contrast. Independent native INR fits use 300 updates per contrast (2,400 total).

All deployment objects use identical outer schema and metadata. Actual complete `torch.save` payloads are reported. One CPU screen, one analytic slice, one seed; fresh seeds 1011007/1011013/1011019 remain sealed.

## Development results

| Method | NMSE ↓ | PSNR dB ↑ | global SSIM ↑ | Held-out k-space RMSE ↓ | Updates | Wall | Actual payload |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native continuous-TI shared INR | 1.1438 | 13.80 | -0.2297 | 0.1338 | 300 | 2.32 s | 24,849 B |
| Shared INR + per-contrast FiLM | 2.7256 | 10.03 | -0.1397 | 0.1438 | 300 | 2.11 s | 26,309 B |
| Shared INR + Givens Mirror | 2.6464 | 10.16 | -0.1550 | 0.1530 | 300 | 3.04 s | 26,117 B |
| Independent INR per contrast | 1.1526 | 13.77 | 0.0561 | 0.2719 | 2,400 | 3.02 s | 188,112 B |

Mirror cuts code dimension from 8 to 2 values per contrast (64 to 16 float values total), but saves only 192 B (0.73%) over FiLM because shared model weights dominate. Mirror NMSE is 131.4% worse than the native continuous-TI INR. Its held-out measurement residual is 14.3% higher than native, beyond the 1.1× gate. Mirror is 2.9% better than FiLM on NMSE, but both are much worse than native conditioning; Mirror also takes 44% longer than FiLM in this CPU run. Independent fits cost 7.6× the native payload and 8× updates while achieving similar NMSE.

## D — Decision

**FAIL.** The learned rotation family does not represent the contrast trajectory adequately in this setup, and its tiny coordinate savings do not change total storage. Native continuous-TI conditioning beats Mirror quality and costs 1,268 B less. Fresh remains unopened.

## C — Strongest counter-hypothesis

The synthetic IR tissue curves and one fixed 300-update schedule may underfit all models; the independent and native SSIM/NMSE disagree because the single global SSIM proxy and pixel-average NMSE respond differently to signed inversion signals. This is not evidence about real MPnRAGE or anatomy generalization.

## U — Unconfirmed

No real MPnRAGE/PREDICTOM data, multi-coil sensitivity maps, 3D volume or clinical metric. No fresh replication. One synthetic tissue map cannot determine whether Mirror coordinates could help a higher-capacity joint INR.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror NMSE 2.6464 vs native 1.1438; 26,117 B Mirror vs 24,849 B native and 26,309 B FiLM; holdout RMSE .1530 vs .1338 native. Replay and heldout mask separation checks pass; fresh unopened.
- **Interpretation:** learned Givens contrast views fail to capture this contrast-dependent signal more compactly than direct continuous-TI conditioning.
- **Hypothesis:** Mirror may need more expressive nonlinear trajectories, but this would add paid code or compute and must beat a stronger native contrast-conditioned INR before it merits a new experiment.
