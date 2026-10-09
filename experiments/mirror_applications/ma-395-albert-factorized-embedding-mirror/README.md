# MA-395 — ALBERT factorized embedding with latent-domain Mirror views

Status: protocol frozen before development. Dedicated branch: `research/ma-395-albert-factorized-embedding-mirror-20261009`.

## Mirror insertion

**Mirror insertion:** add four per-domain Givens angles inside the shared 8-dimensional ALBERT-style embedding bottleneck, before its common projection to 16 dimensions.

## H — Hypothesis

A low-dimensional latent Mirror view can recover held-out domain-token embedding functions at lower actual bytes than dense domain adapters, while improving over hard factorized sharing and output-side FiLM.

## T — Frozen design

PA61 ALBERT factorized embeddings provide the baseline. A fixed shared 128×8 table and 8×16 projection feed eight domain functions; a seeded 80/20 pair split tests held-out combinations. Compare independent-table oracle, hard factorized sharing, output FiLM, dense 8×8 latent transforms, and four-angle latent Mirror views. Every table, adapter, decoder and metadata object is charged. See `PROTOCOL.json` for frozen gates and seeds.

The synthetic teacher uses latent Givens rotations by construction. This isolates whether the coordinate is useful inside the bottleneck; it is not pretrained ALBERT or natural transfer evidence.
