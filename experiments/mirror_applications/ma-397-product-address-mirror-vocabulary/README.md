# MA-397 — Product-address vocabulary with collision-resolving Mirror codes

Status: protocol frozen before development. Dedicated branch: `research/ma-397-product-address-mirror-vocabulary-20261009`.

## Mirror insertion

**Mirror insertion:** add one angle per token after a deterministic Cartesian product address. The angle changes the function for two tokens that share the same pair of physical table rows.

## H — Hypothesis

A one-angle continuous View resolves two logical tokens per product address using fewer actual bytes than native direct coefficients and far fewer than an independent embedding table.

## T — Frozen design

PA43 product keys, PA58 Hash Embeddings and PA59 complementary partitions motivate the address/composition controls. There are 2,048 token IDs but only 32×32=1,024 product addresses, with exactly two token occupants per address. Compare full independent embeddings, pure table addition, native two-value token importance, and one-angle Mirror. Measure collision-pair separation as well as embedding and classifier quality. All component tables, token codes, decoder and archive overhead are paid. The fixed protocol is in `PROTOCOL.json`.

This is a deliberately collision-heavy aligned synthetic screen, not natural vocabulary evidence.

## D — Result

**Fact.** Both development worlds contained exactly 1,024 product addresses, each with two occupants. Full-table NRMSE was .00407/.00424 and native two-coefficient NRMSE .00829/.00739. Mirror NRMSE was .1311/.1356; its collision-pair separation NRMSE was .1283/.1282 versus .00838/.00735 for direct coefficients and exactly 1.0 for pure product addition (which predicted zero separation). Decoder top-1 was 98.93%/98.93% for Mirror, near the direct control's 99.90%/99.90%. Mirror payload was 13,941/13,964B: 11.25% of full and 64.9% of direct-coefficient payload. Eight payloads replayed exact bytes, hashes and metrics; four tests pass. Fresh 39711–39713 remained sealed.

**Interpretation.** The product address alone cannot distinguish its two occupants; the angle representation does create different outputs, but under the frozen update budget it does not fit the teacher's collision-resolved vectors accurately. Native two-value coefficients fit nearly exactly at a roughly 54% byte premium over Mirror. The classifier task is less sensitive than vector/pair reconstruction. This is storage compression with a strict quality failure, not an independent capacity increase.

**Hypothesis.** Random angle initialization and the periodic nonlinear fit leave a subset of token codes away from their teacher targets even after 4,000 updates; unconstrained linear coefficients avoid that optimization difficulty. No analytic-oracle angle fitting or schedule tuning was preregistered.

## H / T / D / C / U

- **H:** One angle per token resolves two tokens sharing a product address, preserving quality at lower bytes than direct importance coefficients and far fewer bytes than full embeddings.
- **T:** PA43/PA58/PA59; V=2,048, 32×32=1,024 addresses with occupancy 2, two fixed 32×16 tables, seeds 39701/39702, four controls, 4,000 Adam updates, complete compressed NPZ payloads.
- **D:** **FAIL on frozen vector and collision-separation gates.** Mirror storage was 11.25% of full and 64.9% of direct coefficients, but NRMSE was .131/.136 and pair-separation error .128/.128; fresh stayed sealed.
- **C:** The native two-coefficient control fits the same orbit nearly exactly at modest additional bytes; the fixed classifier hides much of the embedding mismatch.
- **U:** Fresh seeds, learned product-key routing, natural token frequency/address collisions, optimizer alternatives, private residual frontier and larger codebooks.
