# MA-393 — Adaptive-capacity embedding plus token Mirror view

Status: protocol frozen before development. Dedicated branch: `research/ma-393-adaptive-capacity-mirror-embedding-20261009`.

## Mirror insertion

**Mirror insertion:** add a token-specific angle after frequency-band projection to recover one functional degree of freedom without widening every rare-token embedding.

## H — Hypothesis

A per-token Mirror angle restores useful variation in low-capacity frequency bands, especially for rare tokens, at complete payload bytes no greater than a simple adaptive embedding with one extra coordinate per token.

## T — Frozen design

PA60 motivates adaptive vocabulary capacity. Use 384 tokens, three frequency bands with latent widths 16/8/4, and a fixed seeded projection into 16 dimensions. Compare full embeddings, standard adaptive widths, a byte-matched 17/9/5 adaptive control, and adaptive codes plus a token Givens angle. Train with band masses .70/.22/.08 and report frequency-weighted and per-band quality. See `PROTOCOL.json` for frozen gates and seeds.

This deliberately aligned synthetic orbit tests representation feasibility. It is not a real Zipf corpus or language-model result.
