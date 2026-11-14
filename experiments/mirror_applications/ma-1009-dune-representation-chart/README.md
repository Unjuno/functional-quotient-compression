# MA-1009 — DUNE latent-prior scanner Mirror chart

Status: SCREENING. Synthetic, CPU mechanism experiment.

PA304 DUNE operates in a learned representation with exact decoder-Jacobian VJP data-consistency gradients. At inspected upstream commit `8e62d1b9ca193060d09f5ac4e866dc4a6af28347`, the public repository contains only a README; the paper says code will be released upon publication. No code license or checkpoints were present. The experiment independently implements the published DUNE-z update from Eqs. 5, 6, and 8; no upstream code was copied.

## H — Hypothesis

A scanner-specific Givens coordinate inside a shared latent regularizer will improve reconstruction for a held-out synthetic scanner beyond native shared DUNE and ordinary latent FiLM, without changing exact VJP data consistency or the final acquired-sample projection.

## T — Test

A learned 4-channel latent encoder/decoder is trained with a shared latent regularizer on 24 analytic scanner-A phantoms. Scanner B changes only known two-coil sensitivity maps and is provided explicitly to the forward model. Eight target support examples update the 1-angle Mirror code, 8-coefficient FiLM code, or the full-model upper control; 16 disjoint cases choose the development decision. All methods use identical decoder VJP, coil forward/adjoint, mask, and conjugate-gradient measurement projection. Fresh seeds remain sealed unless every development gate passes.

## C / U

Strong counter-hypothesis: when scanner sensitivity maps are explicit in the operator, a scanner code in the prior is redundant. This is synthetic and cannot establish scanner or clinical performance.


Amendment 02 uses an underdetermined two-coil operator (896 complex samples for 1024 complex image values). Amendment 05 computes a stable orthogonal image-space projection through a deterministic QR of the explicit SENSE operator; it reports the re-encoded image residual. The overdetermined first run and failed CG projection attempts remain recorded as pre-metric diagnostics.
