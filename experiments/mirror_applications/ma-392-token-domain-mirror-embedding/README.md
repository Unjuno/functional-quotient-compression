# MA-392 — Token × domain factorized embedding Mirror address

Status: protocol frozen before development. Dedicated branch: `research/ma-392-token-domain-mirror-embedding-20261009`.

## Mirror insertion

**Mirror insertion:** apply a small domain-specific Givens coordinate to a shared token embedding, producing many logical domain-token vectors without storing one vector per pair.

## H — Hypothesis

For domain-conditioned embeddings formed by orthogonal transformations of a shared token table, domain Mirror angles preserve unseen token-domain combinations while reducing complete payload bytes relative to independent tables and dense domain transforms, and improve over hard sharing and FiLM.

## T — Frozen design

PA59 motivates factorized addresses. The experiment has 128 tokens, eight domains, 16 dimensions, a fixed shared token table, and an 80/20 token-domain split in two development worlds. Compare hard sharing, FiLM, dense domain maps and eight domain-level Givens angles. A full independent oracle stores all outputs for storage/quality reference only; it is not used for held-out generalization claims. All shared tables, views, decoder and archive metadata are charged. See `PROTOCOL.json` for frozen details and gates.

Teacher domains apply disjoint 2D rotations to token vectors. This aligned synthetic screen isolates a domain-level functional coordinate; it does not test natural recommender or language data.
