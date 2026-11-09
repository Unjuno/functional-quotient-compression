# Invertible-flow coordinate-factorization diagnostic (2026-11-09)

## Ruling

Pause MA-773, MA-776 and MA-777 until a proposal specifies functional behavior or a deployment advantage that native conditional RealNVP, direct pair codes, or simple activation steering cannot reproduce at equal or lower serialized bytes and compute. This pause is limited to the coupling/latent-coordinate factorization lane; other invertible methods remain eligible.

## Evidence

- **MA-771:** Conditional RealNVP gives exact heldout transforms and inverse-cycle error `4.77e-7`, but its shared-plus-code payload is 701 B vs 573 B for independently serialized flows. Native conditional RealNVP is an exact payload/function alias.
- **MA-772:** Product condition × behavior factors give heldout transform nRMSE `6.08e-9` and cycle error `4.77e-7`, but the factor payload is 1,071 B vs 715 B for direct pair codes. Native factorization exactly aliases Mirror; an additive factor misses quality.

## Structural cause

Both candidates insert compact coordinates into ordinary RealNVP/conditional low-rank factors. The native conditional flow reproduces the same function, while compressed independent code arrays can cost less than the factored representation once NPZ metadata and actual values are serialized. Invertibility and low heldout error are real properties of these synthetic flows, but they do not isolate Mirror-specific value.

## Resume condition

Resume the paused rows with a test beyond reparameterizing conditional flow: a task where a Mirror view improves an explicit flow metric (likelihood, inverse robustness, code adaptation, composition or runtime) over native coupling/linear steering/equal-byte controls, with actual complete payloads and independent heldout conditions.

## Continue elsewhere

Proceed to MA-775, the i-ResNet residual-view family, which is a distinct mechanism from the tested RealNVP code factorization.
