# MA-665 — Hyena-inspired implicit filter generator with Mirror codes

This synthetic screen tests whether a shared position-conditioned filter basis plus per-filter coordinates can compress long causal convolution filters while preserving held-out outputs. Teacher filters use eight deterministic DCT position functions and a controlled orthogonal residual. The shared DCT basis is the native implicit-filter control; learned PCA is compared with its identical native counterpart.

Measure actual serialized payloads, causal FFT output NRMSE, filter generation and FFT MAC proxies, plus CPU wall time. This is not a Hyena architecture or language-model reproduction; see `PROTOCOL.json`.
