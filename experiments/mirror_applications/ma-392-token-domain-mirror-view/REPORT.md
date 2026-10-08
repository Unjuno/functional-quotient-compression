# MA-392 report

## H — Hypothesis
Shared q/r partition tables plus one learned domain Givens angle can recover useful domain-specific embeddings for withheld token-domain pairs, beyond domain-tied concatenation and simple FiLM/additive controls, without full per-domain tables.

## T — Trial
Synthetic 64-token × 4-domain task with 16D embeddings and 8 classes. The teacher applied one domain rotation to shared partition vectors. Per world, 64 of 256 token-domain combinations were held out, while every token and every domain remained represented in training. Five methods were trained for 512 Adam updates on 65,536 examples each using seeds 39201 and 39202. Methods: independent token-domain table, tied concatenation, domain FiLM, domain additive vector, and Mirror domain angle. Deterministic FP16 ZIP/NPY payloads were reloaded for evaluation. Serialized-state CPU benchmark used 100 batches of 16,384 role lookups per method and world.

## D — Decision
**FAIL.** Fresh seeds 39211–39213 remain sealed.

| Seed | Mirror seen | Mirror heldout | Tied concat heldout | FiLM heldout | Mirror bytes | Mirror / tied CPU
|---:|---:|---:|---:|---:|---:|
| 39201 | 0.9012 | 0.7911 | 0.7737 | 0.7404 | 1,906B vs 1,676B | 0.241×
| 39202 | 0.9757 | 0.7762 | 0.5572 | 0.7430 | 1,906B vs 1,676B | 0.334×

In seed 39201 Mirror was 7.39 points below FiLM on seen pairs and only 1.74 points above tied concat on held-out pairs. The payload ratio to the smallest shared control was 1.137×. Throughput was 0.241–0.334× tied concatenation. Mirror produced 256 unique FP16 role embeddings; tied concatenation produced 64.

## C — Strongest counter-hypothesis
FiLM or direct per-domain components can capture domain effects with better seen accuracy, and a native tied embedding is far faster and smaller. The held-out transfer gains over FiLM (5.07 and 3.32 points) may be real for this aligned teacher, but the frozen gates require quality, storage, and runtime together.

## U — Unconfirmed
No natural domain data, recommendation workload, fresh-world replication, near-convergence capacity, or fused accelerator runtime is established.

## Evidence classes
**Facts:** values, payload hashes, and metric replay are in `runs/`, `RESULTS_CORE.csv`, and `VERIFICATION.json`.

**Interpretation:** development FAIL: quality margin, bytes, and CPU gates missed.

**Hypothesis:** domain-coordinate recombination may aid withheld pairs when View algebra matches the task; an optimized fused kernel could alter the compute frontier, but no such result is measured here.
