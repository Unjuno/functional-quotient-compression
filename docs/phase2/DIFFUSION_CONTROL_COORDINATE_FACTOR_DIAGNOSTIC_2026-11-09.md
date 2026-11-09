# Diffusion-control coordinate-factor family diagnostic (2026-11-09)

## Ruling

Pause the current shared coordinate-factorization lane for MA-756, MA-757, MA-758 and MA-760 until a proposal names a functional freedom that native PCA, low-rank adapter banks, two-tower factorization, or ordinary composition cannot express.

This is an operational family pause, not a negative result for diffusion control generally. The experiments remain distinct and each result stays in its own registry row and report.

## Evidence

- **MA-752:** synthetic condition × injection-depth factor recovers its designed heldout pairs at 6,989 B vs 8,962 B independent; misses the 25% byte gate; native factorization exactly aliases Mirror.
- **MA-753:** shared rank-2 adapter basis preserves synthetic heldout controls and composition at 2,006 B vs 8,811 B full adapters, with 2× MAC proxy; ordinary PCA exactly aliases Mirror.
- **MA-755:** learned source-control × backbone two-tower predicts heldout synthetic pairs at nRMSE 4.91e-5, 1,308 B and 96 MAC vs 3,341 B / 192 MAC independent; native two-tower exactly aliases Mirror.

## Structural cause

Each designed teacher places the target function in the candidate's explicit low-description factor family. The corresponding native factorization reconstructs the same function and payload. This can establish that structured coordinates compress aligned synthetic families, but does not isolate a Mirror-specific advantage. The observed exact aliases are repeated over condition-depth, adapter-basis, and source-backbone parameterizations.

## Resume condition

Resume the paused MA rows only with a falsifiable distinction from the native shared-basis/low-rank/two-tower control, including a task family where the view coordinate changes a function that the native control at equal or lower bytes/compute cannot represent. Natural pretrained diffusion/control data and held-out composition quality would then be needed before any real-model claim.

## Continue elsewhere

The pause is limited to the diffusion-control coordinate-factorization lane. Proceed to the next eligible P0 in another family, MA-761 (goal-conditioned neural cellular automata).
