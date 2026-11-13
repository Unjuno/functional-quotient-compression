# MA-982 — Type-II site subspace with Mirror site coordinate

Status: SCREENING. Frozen synthetic mechanism screen; not a PA289 reproduction.

## H — hypothesis

A shared base subspace with a compact per-site phase coordinate can preserve useful spectral efficiency while reducing state relative to storing a separate subspace for each site. PA289 already establishes fingerprint-conditioned Type-II subspace feedback; native phase-parametric Type-II is therefore a required control.

## Mirror insertion

For site `s`, `U_s = diag(exp(i * phi_s * n)) U_0`; the persistent Mirror coordinate is `phi_s`. Per-user low-dimensional coefficients are quantized and sent as Type-II feedback. The native control uses the same phase-steering formula directly.

## T — execution boundary

Four synthetic sites each generate clustered rank-2 channels with a fixed off-subspace leakage component. Compare a shared global subspace, Mirror phase coordinate, native phase-parametric Type-II, stored per-site Type-II bases, and full quantized CSI. Count serialized persistent payload, online feedback bits and pilot probes separately.

## Limitations

This is a CPU synthetic channel screen. No public measured channel trace, PA289 implementation, pilot/coherence model, multi-user interference, RF energy or field generalization is included.
