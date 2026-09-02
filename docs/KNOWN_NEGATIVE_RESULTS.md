# Known Negative Results and Corrective Findings

These results constrain the current FQC research program. They are not side
notes; they prevent previously rejected explanations from re-entering the
project under new terminology.

## N1 — Invertible transforms are not a fundamental compression source

Mirror, orthogonal rotation, whitening, or other known bijective changes of
coordinates do not by themselves reduce the underlying rate-distortion burden
when the distortion model is transformed consistently. Any gain must come from
actual information removal/sharing, a restricted codec, decoder-known state, or
other non-redundant structure.

## N2 — QK near-tie geometry is not generically low-rank

The 2026-08-27 re-audit rejected the expectation that near-tie attention
boundaries alone imply a small routing tangent space. Under the Gaussian null
model used there, zero-margin rank-one measurements generically span nearly the
full relevant tangent space.

Therefore, if a real Transformer exhibits low-rank routing/tail geometry, treat
that as evidence for *additional structure* such as shared query/difference
subspaces, semantic clustering, GQA/MQA structure, or predictor/core reduction.
Do not attribute it to the near-tie condition alone.

## N3 — A shared basis does not guarantee a shared top-r support

Even when several operators admit a common basis, different spectral orderings
can make a small common support impossible. Basis alignment and support sharing
must be measured separately.

## N4 — Layer/block distortion is not generally additive

Cross terms in the task metric can make independent block estimates optimistic.
An additive Bellman decomposition is exact only under additional conditions
(e.g. cross terms vanish in expectation, the metric is diagonalized in the
chosen coordinates, or a safe interaction majorizer/tax is included).

## N5 — Low coefficient energy is not low task value

The D117 synthetic result shows an explicit counterexample: a mode with roughly
0.019% of coefficient Frobenius energy was strongly harmful to delete. Spectral
or magnitude energy cannot be used as the sole admission rule for compression.

## N6 — Synthetic search-work savings are not codec-bit savings

D70-D120 optimize experiment/query/calibration/validation work. Those results
are useful for research scheduling, but they do not constitute model
compression evidence unless they change the actual serialized decoder state.
