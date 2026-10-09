# MA-389 — Hash Embedding Mirror importance codes

Status: protocol frozen before development. Dedicated branch: `research/ma-389-hash-embedding-mirror-importance-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens angle `m_i` per token to the two shared hash-component outputs so that token importance pairs can vary without storing two independent scalars for every vocabulary item.

## H — Hypothesis

A per-token angle on the shared Hash Embedding component pool preserves token embedding and decoder quality, including hash-collision cases, while reducing actual payload bytes versus a full embedding table and native two-weight Hash Embedding.

## T — Frozen design

PA58 Hash Embeddings combine shared component vectors selected by hashes with token-specific importance weights. This screen keeps the native two hash tables and fixed hash functions, and changes only the importance representation. Compare the full independent table, native per-token two-weight Hash Embedding, common tied importance, scalar gates, and Mirror angles. All tables, codes, decoder, seeds, and archive bytes are paid. See `PROTOCOL.json` for frozen seeds and development/fresh gates.

The teacher importance vectors are deliberately on a unit-circle orbit. This is an aligned feasibility test, not a natural token-embedding result.
