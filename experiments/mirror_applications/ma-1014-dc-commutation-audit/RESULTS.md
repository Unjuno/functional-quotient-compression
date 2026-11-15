# MA-1014 results

## H — Hypothesis

A nontrivial feature-space Mirror view placed after MRI data consistency (DC) will leave the acquired-data constraint set. The same view placed inside the learned prior and followed by exact DC should preserve measured samples, but may or may not improve heldout quality.

## T — Test

- Synthetic 32×32 two-coil SENSE operator, 16 source phantoms and three target sites × eight target phantoms.
- One 2-8-4-2 convolutional prior trained for 64 updates and 256 source presentations, then frozen.
- Fixed two-angle Givens Mirror `[0.37, -0.29]`; no target-truth tuning.
- Compared plain zero-filled DC; prior then exact DC; Mirror before the prior’s final projection; Mirror after DC; and Mirror after DC with final reprojection.
- Target metrics are acquired-line RMS residual, heldout k-space RMSE and image NMSE. Full target k-space is score-only. Fresh seeds 1014301, 1014307 and 1014313 remained sealed.
- Actual inference payload includes prior tensors, angles, placement/operator/format metadata. Per-example CPU inference time was recorded. The prior uses 516,096 convolution MACs per forward; Mirror adds 12,288 scalar FLOPs (2 feature-plane rotations × 1,024 pixels × 6 FLOPs).

## Results

| Placement | Acquired residual (sites 1/2/3) | Heldout k-space RMSE (sites 1/2/3) | Image NMSE (sites 1/2/3) | Bytes |
|---|---:|---:|---:|---:|
| Plain zero-filled DC | 1.84e-8 / 1.20e-8 / 1.64e-8 | .1760 / .1839 / .1716 | .7977 / .9250 / .8051 | 5,141 |
| Prior then DC | 1.88e-8 / 1.45e-8 / 1.93e-8 | .1459 / .1484 / .1413 | .5511 / .6173 / .5697 | 5,141 |
| Mirror before prior then DC | 1.89e-8 / 1.47e-8 / 1.82e-8 | .1458 / .1485 / .1419 | .5502 / .6180 / .5730 | 5,205 |
| Mirror after DC | .0778 / .0328 / .0826 | .1584 / .1703 / .1558 | .8112 / .8585 / .8806 | 5,205 |
| Mirror after DC, then reproject | 2.52e-8 / 2.19e-8 / 2.31e-8 | .1584 / .1703 / .1558 | .6816 / .8455 / .7361 | 5,205 |

The pre-prior view is effectively tied with the native prior: relative heldout error changes are −0.042%, +0.068%, +0.460% (negative means a small improvement). It fails the frozen 5% improvement gate at every site. Post-DC Mirror violates measured consistency; final reprojection restores it but quality remains worse than prior-then-DC at all sites.

The source prior costs 516,096 convolution MACs per forward. The native prior and pre-prior Mirror each use one prior forward and four complex 2D FFT calls for exact SENSE DC; Mirror adds 12,288 scalar FLOPs. Applying Mirror after DC uses two prior forwards; reprojecting afterward uses two prior forwards and eight FFT calls. The complete CPU development run took 1.543 s, including 64 source updates (1.440 s). Scoring-only metric calculation and source preprocessing are excluded from the operation proxy.

## D — Decision: FAIL

**FACT:** Exact replay matches metrics and payload sizes; checkpoint SHA-256 is in `VERIFICATION.json`. All three unit tests pass. The protocol was amended before the scored rerun after discovering the initial exploratory code was not an exact projection; that invalid attempt is excluded. Fresh remains sealed.

**INTERPRETATION:** Placement relative to DC governs hard measurement consistency. The Mirror view has no demonstrated quality/storage advantage: it adds 64 serialized bytes and 12,288 scalar FLOPs per example, while its best placement differs by <0.5% and the stronger native prior is slightly better on two target sites.

**HYPOTHESIS:** A Mirror view outside the DC-consistent subspace must be followed by a valid projection if exact measurements are required. The tested feature rotations are not a useful MRI reconstruction improvement in this screen.

**C — Strongest counter-hypothesis:** Fixed angles and short source training may be poorly aligned; a view optimized on development data or placed in a learned representation operator could behave differently.

**U — Unconfirmed:** Acquired multicoil data, realistic spatially varying sensitivities with a converged SENSE projection, VarNet/MoDL/DUNE checkpoints, learned Mirror coordinates, fresh seeds and clinical-image metrics.
