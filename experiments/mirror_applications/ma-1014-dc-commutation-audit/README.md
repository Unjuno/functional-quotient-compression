# MA-1014 — MRI data-consistency projection commutation audit

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: worker-ready baseline `research/mirror-application-worker-ready-20261007`  
Prior art: PA303 (MoDL), PA304 (DUNE)

## Hypothesis

H: A nontrivial two-angle feature-space Mirror view applied after a learned prior generally moves an MRI reconstruction off the acquired-data consistency set, while applying the same view before an exact final data-consistency projection preserves acquired samples to numerical precision; its downstream heldout-line and image quality may still differ.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` as a two-plane Givens feature rotation at alternative interfaces around the MRI data-consistency projection so that a scan-specific logical prior can vary without duplicating the shared prior or weakening the measured-data constraint.

- Physical object: one frozen convolutional reconstruction prior.
- View placements: (A) prior feature → Mirror → prior output → DC; (B) prior feature → prior output → DC → Mirror output; (C) prior feature → Mirror → prior output → final DC.
- Simplest controls: identity view, the same view with an explicit final native DC projection, and plain DC with no view.
- Closest prior art: MoDL already ties the denoiser across unrolled iterations; DUNE preserves physics through representation-space unrolling. This screen isolates placement relative to DC, not clinical performance.

## Protocol

See `PROTOCOL.json`, amendment 01. This is a deterministic, bounded synthetic 32×32 two-coil mechanism fixture. A small prior is trained only on synthetic source phantoms; three target sites with eight fixed phantom scans each are evaluated. The Mirror codes are fixed nonzero test coordinates, not tuned against target truth. No fresh/audit seeds are opened. This experiment does not claim acquired MRI or patient-level validity. The corrected projection uses unit-norm spatially constant coil sensitivities, so the SENSE normal operator is identity; measured k-space is overwritten and mapped back with the exact adjoint. Exactness is assessed in forward acquired-line k-space.

## Gates

- **PASS (mechanism only):** all placements that promise strict DC have measured-line residual ≤1e-6, and adding the Mirror view improves heldout-line error by ≥5% over prior-then-DC in all three target scans.
- **FAIL:** a post-DC placement violates exact measured samples above 1e-4, or a projected Mirror view does not meet the heldout quality gate.
- **NOT ESTABLISHED:** deterministic data/operator or replay checks fail, or the fixture cannot distinguish the placements.

## Decision

FACT: Post-DC Mirror has acquired-line RMS residual .0328–.0826, while exact projection placements are 1.5e-8–2.5e-8. Pre-prior Mirror improves heldout k-space RMSE by 0.042% at site 1, worsens it by 0.068% at site 2 and 0.460% at site 3 versus the stronger prior-then-DC control. Reprojection after post-DC Mirror restores acquired-line consistency but increases heldout error by 8.6%, 14.7% and 10.3%, respectively. The shared prior payload is 5,141 B; adding two angles and metadata costs 64 B (5,205 B). Exact replay and three tests pass; fresh remains sealed.

INTERPRETATION: Post-DC placement violates acquired measurements. An exact final projection repairs that constraint but does not repair heldout quality. Placing the view inside the prior and projecting afterward is measurement-safe here, but does not yield the preregistered consistent quality gain and costs extra bytes and runtime.

HYPOTHESIS: This bounded fixture supports a scoped FAIL for a general useful Mirror benefit around strict DC. It supports the narrower mechanism expectation that an unconstrained post-projection view can leave the measurement-consistent set.

BOUNDARY: Synthetic operator-placement screen only; constant-sensitivity SENSE, fixed angles, 64 source updates, no acquired MRI, no clinical claim, no fresh/audit evaluation.
