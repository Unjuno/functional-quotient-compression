# MA-397 — Product-address vocabulary with collision-resolving Mirror codes

Status: protocol frozen before development. Dedicated branch: `research/ma-397-product-address-mirror-vocabulary-20261009`.

## Mirror insertion

**Mirror insertion:** add one angle per token after a deterministic Cartesian product address. The angle changes the function for two tokens that share the same pair of physical table rows.

## H — Hypothesis

A one-angle continuous View resolves two logical tokens per product address using fewer actual bytes than native direct coefficients and far fewer than an independent embedding table.

## T — Frozen design

PA43 product keys, PA58 Hash Embeddings and PA59 complementary partitions motivate the address/composition controls. There are 2,048 token IDs but only 32×32=1,024 product addresses, with exactly two token occupants per address. Compare full independent embeddings, pure table addition, native two-value token importance, and one-angle Mirror. Measure collision-pair separation as well as embedding and classifier quality. All component tables, token codes, decoder and archive overhead are paid. The fixed protocol is in `PROTOCOL.json`.

This is a deliberately collision-heavy aligned synthetic screen, not natural vocabulary evidence.
