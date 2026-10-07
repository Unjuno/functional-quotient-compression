# Known Negative Results and Corrective Findings

These results constrain the current FQC / Mirror / shared-rule research program. They are not side notes; they prevent previously rejected explanations from re-entering the project under new terminology.

## N1 — Invertible transforms are not a fundamental compression source

Mirror, orthogonal rotation, whitening, or other known bijective changes of coordinates do not by themselves reduce the underlying rate-distortion burden when the distortion model is transformed consistently. Any gain must come from actual information removal/sharing, a restricted codec, decoder-known state, or other non-redundant structure.

## N2 — QK near-tie geometry is not generically low-rank

The 2026-08-27 re-audit rejected the expectation that near-tie attention boundaries alone imply a small routing tangent space. Under the Gaussian null model used there, zero-margin rank-one measurements generically span nearly the full relevant tangent space.

Therefore, if a real Transformer exhibits low-rank routing/tail geometry, treat that as evidence for additional structure such as shared query/difference subspaces, semantic clustering, GQA/MQA structure, or predictor/core reduction. Do not attribute it to the near-tie condition alone.

## N3 — A shared basis does not guarantee a shared top-r support

Even when several operators admit a common basis, different spectral orderings can make a small common support impossible. Basis alignment and support sharing must be measured separately.

## N4 — Layer/block distortion is not generally additive

Cross terms in the task metric can make independent block estimates optimistic. An additive Bellman decomposition is exact only under additional conditions.

## N5 — Low coefficient energy is not low task value

The D117 synthetic result gives an explicit counterexample: a very low-energy mode was strongly harmful to delete. Spectral or magnitude energy cannot be the sole compression admission rule.

## N6 — Synthetic search-work savings are not codec-bit savings

D70–D120 optimize experiment/query/calibration/validation work. Those results do not constitute model compression evidence unless they change actual serialized decoder state.

## N7 — More Mirror states do not imply more useful capacity

Across the MN/RF/token-period work, increasing state/view count by itself did not produce a stable quality/capacity improvement. Count, breadth, geometry, placement, and training signal are separate resources.

## N8 — Whole-core condition identity can damage sharing

In the synthetic sensor/world experiments, feeding sensor identity directly into the shared world core worsened source-only changed-law transfer. Condition-specific information should not be assumed harmless merely because parameters are technically shared.

## N9 — Always-on residual Views are not universally useful

When affine calibration already removed most sensor mismatch, residual Views worsened seen-law prediction. Views became useful only when meaningful post-calibration residual mismatch remained.

## N10 — ES gave no free continuous-parameter advantage

Independent ES samples across Mirrors reduced variance only because they increased independent sample count. At fixed total branch budget the gain disappeared, and Backpropagation substantially outperformed ES for differentiable shared parameters.

## N11 — Load balance is not routing quality

In SRM001 end-to-end routing, forcing broader address usage increased the number of used addresses but did not improve task quality. Good routing requires stable, useful functional assignments, not merely balanced occupancy.

## N12 — Compositional parameterization does not solve arbitrary credit assignment

SRM001 succeeded strongly on decomposable count targets and retained a weaker advantage on a 16-class bitmask target, but parity remained near chance for both shared/Mirror and standard variants. Factorized representation alone does not guarantee learnability of arbitrary nonlinear compositions.

## N13 — Fixed-update superiority is not a capacity proof

Standard MoE continued to improve materially when trained longer in SRM001. A fixed-update advantage must be reported as learning-efficiency evidence until near-convergence, serialized-byte-matched capacity frontiers are measured.
