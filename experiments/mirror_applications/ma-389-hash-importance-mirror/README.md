# MA-389 — Hash Embedding Mirror importance codes

Status: FAIL — strict native-hash byte gate missed by both worlds  
Evidence lane: MECHANISM / STORAGE / RUNTIME / COLLISION SENSITIVITY  
Base commit: `f3f9240` (MA-385 result branch)  
Prior art: PA58 (Hash Embeddings)

## Hypothesis

H: When token functions are generated from a shared pair of hash-selected component tables with normalized task weights, replacing the two native per-token importance weights with one angular Mirror coordinate preserves held-out token classification quality while reducing actual payload bytes versus native Hash Embeddings; the Mirror code must also beat a direct shared-table coefficient control at the relevant byte/quality point.

## Mirror insertion

> **Mirror insertion:** this experiment adds an angular coordinate `m_i` to the importance-weight interface of a two-table Hash Embedding so that token `i` can combine the same two shared hashed vectors as `cos(m_i)e_{h_1(i)} + sin(m_i)e_{h_2(i)}` without storing two independent importance weights.

- Native method: two hash-selected component vectors plus two learned importance weights per token.
- Mirror: same physical component tables plus one angle per token.
- Cheapest controls: fixed equal weights, native two-scalar token importance, and an independent full token embedding table.
- All variants receive the same token IDs, hash functions, training examples, labels, and optimizer budget.

## Physical-to-logical claim

Two shared component banks represent a vocabulary of 128 logical embedding/readout functions. The synthetic teacher is deliberately aligned to the two-hash component family so this first experiment can falsify whether angular importance codes compress even a favorable case. Hash collisions are measured explicitly. A success would remain a synthetic alignment result, not natural language or vocabulary-scale evidence.

## Comparisons

1. Independent per-token embedding table (upper storage/quality control).
2. Native Hash Embedding with per-token two-scalar importance weights (mandatory PA58 control).
3. Same shared hash tables with fixed equal importance weights.
4. Same shared hash tables with one angle per token (Mirror).

## Gates

### PASS (development screen)

Both development worlds: independent test accuracy at least 0.80; Mirror within 2 percentage points and 0.05 NLL of native Hash Embedding; Mirror payload no more than 0.60x independent and no more than 0.90x native Hash Embedding; retrieval-free token lookup produces no extra state; collision-group accuracy is within 2 points of native; serialized-state CPU throughput is at least 0.90x native; and the direct control does not match the Mirror point at lower/equal bytes. Fresh worlds run only if all gates pass in both development worlds.

### FAIL

Either development world misses any quality, byte, collision, or runtime gate, or the native two-weight Hash Embedding control matches the Mirror result at comparable or lower bytes. Fresh remains sealed.

### NOT ESTABLISHED

Exact serialized-state replay or collision accounting fails. No fresh result may repair a development failure.

## Tuning boundary

- Development worlds: seeds 38901 and 38902.
- Fresh worlds: seeds 38911, 38912, and 38913; unopened unless both development worlds pass all gates.
- No test-time token labels or task IDs are supplied. Train, validation, and test contain disjoint draws from the same fixed token vocabulary.

## Storage and compute

Actual deterministic ZIP/NPY FP16 inference payload bytes are authoritative. Count component tables, output classifier, every token coefficient/angle, hash seeds/layout metadata, and archive headers. Hash bucket IDs are deterministic from token IDs and the charged hash parameters. Report active lookup/MAC proxy, examples and updates, training wall time, collision rate and per-collision-group quality, and CPU throughput after loading the serialized payload.

## Scope

This is a synthetic token-embedding lookup screen with a teacher aligned to a two-hash component family. It does not establish next-token language-model quality, natural lexical generalization, or large-vocabulary memory savings.

## Report

**H:** One angle per token can replace two native Hash Embedding importance weights when the token functions follow a normalized two-component orbit, reducing actual bytes while retaining quality.

**T:** Two development worlds (38901, 38902), 128 token IDs, two 16-row hash tables, 16-dimensional embeddings, 8 output classes, 512 updates and 131,072 examples per method. Compared independent token tables, fixed-weight hash tables, native two-scalar Hash Embeddings, and Mirror angles. Fresh seeds 38911–38913 were not opened.

**FACT:** Hash collisions affected 58.6%/65.6% of token IDs (84/80 unique hash pairs). Serialized payload sizes were 5,290 B independent, 2,674 B fixed-hash, 3,418 B native Hash Embedding, and 3,154 B Mirror. Mirror accuracy was 1.000/1.000 with NLL 0.0555/0.0415; native hash accuracy was 0.983/1.000 with NLL 0.0555/0.0351. Mirror collided-token and unique-signature accuracy were 1.000 in both worlds. Serialized CPU throughput was 5.24/5.94M lookups/s Mirror versus 5.41/5.58M native.

**D: FAIL (frozen strict gate).** Mirror uses 0.596x the independent-table bytes and meets quality, collision, and runtime conditions, but its payload is 0.923x the native Hash Embedding size; the protocol requires at most 0.90x. Fresh remains sealed.

**C:** The teacher is constructed directly from a normalized two-hash angular combination, which favors the Mirror family. The native two-weight control reaches nearly the same output quality with only 264 B more state, so any Mirror gain is narrow.

**U:** This is not language modeling or a natural vocabulary benchmark. No claim is established for larger token inventories, skewed frequency distributions, multilingual vocabularies, or unaligned lexical functions.

**INTERPRETATION:** A one-angle address can recover many useful token functions from two shared physical tables in this aligned synthetic case, but the observed 7.7% payload saving over native Hash Embedding misses the frozen 10% requirement. The much larger reduction against independent tables is mostly due to hash sharing itself.

**HYPOTHESIS:** A domain-specific token bank with better-than-random hash collisions may amortize the angular code more effectively, but it needs a separate preregistered experiment and natural task evidence.
