# MA-1012 — Frozen MRI prior with site-calibrated Mirror codes

Status: **FAIL (bounded synthetic development gate)**  
Branch: `research/ma-1012-scanner-site-mirror-20261115`  
Evidence lane: synthetic multi-coil mechanism, storage and compute

## H — Hypothesis

Three held-out synthetic scanner sites with coil-map and sampling shifts can be represented by two Givens angles per site on one frozen multi-coil reconstruction prior, staying within 1% NMSE of site FiLM and independent site tuning, while saving at least 25% complete bytes versus FiLM and 50% versus independent models.

## T — Test

PA302 motivates an unrolled multi-coil reconstruction prior with explicit sensitivity maps and k-space consistency. PA305 motivates distribution-level MRI test-time adaptation. No trained VarNet/D2SA checkpoint, fastMRI split or acquired scanner data was available in the workspace, so this is an independent synthetic screen rather than a reproduction.

A 518-parameter CNN prior was pretrained for 64 updates (batch 4, 16 source phantoms) on a source scanner. Three target sites use different smooth, normalized two-coil sensitivity maps and randomized Cartesian masks. Each site has four support phantoms and eight separate development phantoms with shifted anatomy statistics. Support adaptation uses only 7/8 acquired phase-encode lines; the remaining acquired line scores held-out consistency. The fully sampled target images are used only for post-adaptation scoring. One shared prior is adapted by (i) 8-value FiLM codes per site or (ii) 2-angle Givens codes per site. An independent full-prior fine-tune is included as a private-state upper control.

All complete deployment packs use the same serialized outer schema. Each code method performs 48 updates per site, processing all four support cases per update (576 support presentations total). The full-fine-tune control performs 48 updates per site but cycles one support case per update (144 presentations); thus update counts match but support-example exposure does not. The independent control is not an equal-exposure upper bound. Fresh seeds 1012301/1012307/1012313 remain sealed.

## Development results

| Method | NMSE ↓ | PSNR dB ↑ | Global magnitude-SSIM proxy ↑ | Held-out k-space RMSE ↓ | Updates | Calibration wall | Complete 3-site payload |
|---|---:|---:|---:|---:|---:|---:|
| Frozen shared prior | .63129 | 23.63 | .20143 | .13024 | 0 | — | 5,205 B |
| Site FiLM codes | .93838 | 21.92 | .08850 | .13980 | 144 | .563 s | 5,521 B |
| Site Givens Mirror codes | .70081 | 23.15 | .17607 | .13264 | 144 | .743 s | 5,457 B |
| Independent full prior per site | .77131 | 22.74 | .13219 | .13505 | 144 | .193 s* | 12,333 B |

Mirror reduces NMSE by 25.3% relative to FiLM and by 9.1% relative to the underexposed private fine-tune. However, frozen sharing is still 11.0% better than Mirror NMSE and has the lowest held-out residual. The Mirror deployment saves only 64 B (1.16%) versus FiLM, far below the registered 25% gate, while taking 32% longer to calibrate in this CPU run. The 2-coordinate site code (6 scalars across three sites) replaces 24 FiLM scalars, but shared model weights dominate total payload. Against independent priors it saves 55.8% bytes, though it does not improve on the quality of the frozen shared model.

The CNN prior costs 516,096 forward MAC per case. Code adaptation processes 576 support presentations (about 297.3M CNN forward MAC plus two-coil FFTs/backprop); private tuning uses 144 presentations (about 74.3M forward MAC). `*` The lower private wall time is consistent with fewer support presentations, so it is not a matched compute comparison.

## D — Decision

**FAIL.** Mirror misses the byte gate versus the cheapest adaptive control and degrades quality versus frozen hard sharing. It is numerically better than FiLM in this synthetic setup, but the difference does not justify an additional site-specific code because adapting at all is worse than simply freezing the prior. Fresh remains unopened.

## C — Strongest counter-hypothesis

The explicitly supplied sensitivity maps and the fixed inverse operator already explain much of the scanner shift; the shared prior may not need calibration. The small source model and synthetic support distribution also do not reflect real sequence, coil, anatomy or reconstruction shifts. The private upper control saw fewer support examples, so its comparison is conservative for the learned codes.

## U — Unconfirmed

No acquired scanner or fastMRI data, trained VarNet, released D2SA checkpoint, real multi-coil generalization, clinical metric, or equal-example private upper control. This is not evidence that Mirror helps real site deployment.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror NMSE .70081 vs frozen .63129 and FiLM .93838; bytes 5,457 B Mirror vs 5,521 B FiLM and 12,333 B independent. Exact replay passed; fresh unopened.
- **Interpretation:** the Givens code is a better-behaved low-dimensional calibration than this FiLM parameterization, but hard sharing dominates its quality/storage tradeoff.
- **Hypothesis:** useful MRI site views may require richer shifts not already handled by measured coil maps; any future test needs released multi-coil data and matched support-example exposure.
