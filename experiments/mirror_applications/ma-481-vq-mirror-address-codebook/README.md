# MA-481 — VQ Mirror address codebook

Status: **FAIL for the registered N64 address quality gate; storage compression is real but retrieval quality misses**  
Evidence lane: address compression / collisions / storage / latency  
Branch: `research/ma-481-vq-mirror-address-20261009`  
Protocol frozen: `2f9d8213`; fresh worlds 48110–48112 × seeds 0–2.

## H — Hypothesis

For addresses on a known continuous 2D rotation orbit, shared basis plus a Mirror angle per key can preserve exact and paraphrase retrieval with less actual storage than full key vectors and discrete VQ codebooks, while collision measurements show the limit as the address count grows.

## T — Test

Each synthetic 32D memory had up to 128 addresses at near-uniform angles on a known 2D plane, with deterministic jitter of ±0.003 radians. Values were independent 4D vectors stored explicitly and identically by every method. Queries were exact addresses, four Gaussian paraphrases per address (sigma 0.01), and 512 unrelated random vectors. Retrieval used fixed cosine threshold 0.95. Controls were full explicit keys, generic Cartesian basis coefficients, continuous Mirror angles and charged uniform VQ codebooks K=8/16/32/64/128. We recorded correct-address recall, false triggers, output error, decoded-key collisions, actual serialized bytes, address decode MAC and query timing.

## D — FAIL for the registered gate

**Facts:** At N=64, Mirror used 4,505B vs 11,997B explicit keys/values (37.6%) and generic Cartesian coefficients used 4,761B. Mirror had zero decoded-key collisions and 100% exact-address recall, but mean paraphrase correct-address recall was 98.35% (range 97.27–99.22%), below the per-world 99% gate; mean paraphrase output NRMSE was 0.180 (max 0.238). The full explicit baseline had the same paraphrase recall/error, showing the failure is address crowding in this task, not a Mirror-specific collision. At N=128, Mirror used 6,297B vs explicit 21,725B and generic coefficients 6,809B; mean correct-address paraphrase recall was 98.63% (range 98.44–99.22%), still below gate in some worlds.

VQ codebooks made the collision limit clear. At N=64, K=8/16/32/64 had mean address collision fractions 0.922/0.859/0.734/0.493 and exact correct-address recall 1.6%/14.1%/26.6%/50.0%. K=128 removed collisions but used 20,441B, more than explicit storage, with only 96.9% paraphrase recall. Unrelated false triggers were zero. Query lookup timings were effectively equal after bank decode.

**Interpretation:** Continuous Mirror addresses are much smaller than explicit vectors and avoid VQ collisions on this known orbit. However, the continuous address bank itself fails the paraphrase-quality threshold as N grows, and a generic Cartesian coordinate control is within 5.4% of Mirror bytes at N=64. This is a storage result without the required usable retrieval quality; no Mirror-specific address-capacity claim is established.

## C — Strongest counter-hypothesis

The threshold failure comes from dense neighboring keys on a one-dimensional circle and paraphrase noise, which makes nearby facts ambiguous for every representation. This synthetic setup may understate natural high-dimensional address separability. Conversely, it grants Mirror exact knowledge of the key manifold.

## U — Unknown

Learned key encoders, natural retrieval embeddings, adaptive per-key radii, error-correcting addresses, and real VQ-VAE training are untested. The result should not be read as a limit on higher-dimensional or learned addresses.

## Decision

**FACT:** Mirror reduces N64 actual payload bytes to 37.6% of full explicit storage and has no quantization collisions, but its mean paraphrase recall is 98.35% and misses the per-world 99% gate. Generic coefficients are only 5.4% larger.  
**INTERPRETATION:** Low-dimensional continuous coordinates compress addresses; address crowding, rather than serialization, sets the tested quality boundary.  
**HYPOTHESIS:** Higher-dimensional learned addresses or code-aware spacing may preserve recall at the same rate.  
**BOUNDARY:** Hand-specified circular address manifold, fixed cosine retrieval and synthetic values; no VQ-VAE or natural retrieval reproduction.
