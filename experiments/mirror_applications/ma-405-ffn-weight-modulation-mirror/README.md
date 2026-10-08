# MA-405 — conditional FFN weight modulation versus Mirror Views

Status: SCREENING; frozen protocol before development. Prior art PA65 (StyleGAN2 weight modulation/demodulation).

## H — Hypothesis
A compact non-diagonal Mirror View can recover rotation-dominated context FFNs where StyleGAN2's per-channel weight modulation is mismatched, while using fewer serialized bytes than independent context FFNs. At the diagonal-modulation endpoint, the native StyleGAN2 control should remain competitive.

> **Mirror insertion:** this experiment adds `m` as 16 pairwise Givens angles between the shared 32D feature vector and the shared FFN output projection, so eight contexts can express non-diagonal logical FFN functions without copying that projection.

## T — Setup and controls
A frozen shared 32D nonlinear feature block and one shared `W` output projection are paid once. Targets vary across eight contexts and five measured mixture levels: alpha=1 is exactly StyleGAN2 per-input-channel weight modulation/demodulation; alpha=0 is exactly `W @ R(m)`; intermediate matrices are mixtures. Controls are shared identity, StyleGAN2 modulation, Mirror Givens View, rank-one residual, and independent full affine matrices. Development seeds 40501/40502, 4096 independent train and 4096 held-out examples per context/alpha, 400 updates. Fresh seeds 40511–40513 are sealed until the full gate passes.

Serialized inference payload bytes are authoritative. NPZ payloads include shared feature and FFN weights, every task code, and metadata. All endpoints and mixture levels are reported separately.
