# MA-389 — Hash Embedding Mirror importance codes

Status: **FAIL at frozen development gates; fresh stayed sealed.** Dedicated branch: `research/ma-389-hash-embedding-mirror-importance-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens angle `m_i` per token to the two shared hash-component outputs so that token importance pairs can vary without storing two independent scalars for every vocabulary item.

## H — Hypothesis

A per-token angle on the shared Hash Embedding component pool preserves token embedding and decoder quality, including hash-collision cases, while reducing actual payload bytes versus a full embedding table and native two-weight Hash Embedding.

## T — Frozen design

PA58 Hash Embeddings combine shared component vectors selected by hashes with token-specific importance weights. This screen keeps the native two hash tables and fixed hash functions, and changes only the importance representation. Compare the full independent table, native per-token two-weight Hash Embedding, common tied importance, scalar gates, and Mirror angles. All tables, codes, decoder, seeds, and archive bytes are paid. See `PROTOCOL.json` for frozen seeds and development/fresh gates.

The teacher importance vectors are deliberately on a unit-circle orbit. This is an aligned feasibility test, not a natural token-embedding result.

## D — Result

**Fact.** Two development worlds (38901, 38902), five methods each, 1,500 updates per method. Full and native Hash controls reached embedding NRMSE .0053–.0062; Mirror reached .0949/.0639. Mirror decoder top-1 was 98.44%/98.05%, against native Hash 100%/99.61%. Mirror payloads were 7,662/7,648B, 43.4% of full-table bytes (17,642/17,631B) and 88.8% of native Hash bytes (8,626/8,627B). Collision-stratified Mirror NRMSE was .0369/.0610 for collided tokens and .1038/.0646 for unique-pair tokens. All 10 stored payloads replayed exact bytes, hashes and metrics after reload; four tests pass. Fresh worlds 38911–38913 remain sealed.

**Interpretation.** Mirror compressed the deliberately aligned hash importance state relative to a full table and modestly reduced bytes against native Hash Embedding, but did not preserve embedding fidelity at the frozen .03 NRMSE gate. Decoder accuracy remained high, showing that this fixed classifier is less sensitive than vector reconstruction. Mirror beat the scalar control substantially in quality (scalar NRMSE .502/.501 at 7,968/7,932B), while costing two reconstruction ops per token and running at lower measured throughput than native Hash in these CPU timings.

**Hypothesis.** The angle coordinate is a compact orbit constraint that captures most task-relevant directions but cannot fit the unconstrained optimization endpoint exactly; the non-collision stratum dominated error in seed 38901, so hash collisions alone do not explain the miss. The native two-coefficient representation remains necessary when accurate vector reconstruction is required.

## H / T / D / C / U

- **H:** A per-token Givens angle on shared hash components preserves vector and decoder quality, including collisions, below 60% of full-table bytes.
- **T:** PA58 aligned synthetic Hash Embedding, V=256, two 32x16 shared tables, five registered controls, seeds 38901/38902, 1,500 Adam updates, actual compressed NPZ payloads including decoder and metadata.
- **D:** **FAIL.** Storage beat the full table (43.4%) but missed the NRMSE ≤.03 gate in both worlds; fresh was not opened.
- **C:** The native two unrestricted importance scalars/token fit the teacher orbit more accurately under the fixed optimizer budget; the decoder top-1 metric masks embedding distortion.
- **U:** Natural vocabulary distributions, language modeling, larger vocabularies/hash-bank scaling, quantization, and whether longer or alternative optimization recovers the angle fit.
