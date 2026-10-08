# Upstream provenance and implementation boundary

- Paper: Guo et al., **CoANeRV: Coordinate-Aware Token-Space Neural Video Representation**, arXiv:2608.13938v1, submitted 2026-08-14, https://arxiv.org/abs/2608.13938.
- Public implementation: https://github.com/jialong2023/CoANeRV, audited commit `3afe1122146819d5854eca2376e5c0faabb72818` (BSD-3-Clause).
- The paper defines `T_v = E_phi(V)` and `V_hat(q) = D_psi(gamma(q), T_v)`. Its Appendix A.5/Table 7 freezes the shared decoder and optimizes only per-video tokens; this is the direct native control used here.
- The paper's main config uses a convolutional tokenizer, six-layer token former, 384 x 72 video tokens, axis-adaptive `(x,y,t)` coordinates, temperature-modulated cross-attention, and spatially tiled query processing.
- This experiment implements only a reduced coordinate-PE + temperature cross-attention + shared RGB readout core at 32 x 24 tokens/features. It does not import or claim to reproduce the full tokenizer, token former, training pipeline, GPU memory, or paper quality numbers.
- The upstream checkout and paper PDF are kept outside the project at `/tmp/coanerv-ma945`; its code is unmodified. This experiment's implementation is in `source/model.py`.
