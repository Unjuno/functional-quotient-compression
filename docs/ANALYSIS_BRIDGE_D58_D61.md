# D58 / D61 Analysis Bridge

D58 and D61 are best treated as two different filters between architecture-correct extraction and codec admission.

## D58: structural candidate evidence

D58 asks whether compatible extracted primitives exhibit reusable structure:

- low-dimensional shared operator span;
- support concentration in a declared basis;
- commutation / joint-diagonalization evidence only where the real-symmetric theorem applies;
- RoPE operator-atom dictionary reuse;
- GQA/MQA consumer reuse;
- MLP similarity after exact architecture-specific symmetry normalization.

These are diagnostics. A small residual is not a serialized-rate theorem. Dictionary atoms, coefficients, selectors, support metadata, and private residuals are PAID unless the decoder derives them from already available state.

Important logical separations preserved by the canonical tests:

- common eigenbasis does not imply common small support;
- common support does not imply commutation;
- low shared linear span does not imply commutation;
- architectural GQA reuse is not an additional baseline-storage factor when shared K/V storage was already counted once.

## D61: functional sensitivity

D61 asks whether a declared operator perturbation can be connected to a downstream functional perturbation on a frozen input set or declared domain.

For plain bilinear attention the canonical code implements the deterministic bound

`||Delta L||_F <= |alpha| ||X||_2^2 ||Delta P||_F`.

It also implements the global rowwise softmax `1/2` Lipschitz bound and the corresponding attention residual-output bound.

At the final logits, an infinity-norm perturbation `eps` certifies unchanged top-1 whenever the reference margin is greater than `2 eps`. Cross-entropy change has the standard sufficient bounds `2||Delta z||_inf` and `sqrt(2)||Delta z||_2`.

## Admission rule

The intended flow is:

1. D57: extract an architecture-valid primitive;
2. D58: generate/score structural candidates;
3. D59/D60: compile actual paid state and reconstruction distortion;
4. D61: compute cheap functional certificates or triage scores;
5. direct real-model replay for survivors;
6. only contract-valid task evaluation or an explicitly accepted sufficient certificate can update a real compression claim.

A large D61 upper bound does **not** prove candidate failure; it can simply mean the certificate is loose. Likewise a small D58 residual does **not** prove a bit saving.

QK-normalized attention remains factor-preserving/direct-replay unless a separate valid factor-level certificate is supplied; D61 must not reintroduce the invalid fixed-P quotient forbidden by D57.
