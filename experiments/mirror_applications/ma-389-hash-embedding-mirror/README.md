# MA-389 — Hash Embedding Mirror importance codes

Status: **FAIL — quality and collision gates missed; fresh seeds remain sealed.**
Evidence lane: QUALITY / STORAGE / COMPUTE / COLLISION
Base commit: `81f653b`
Prior art: PA58, Hash Embeddings for Efficient Word Representations.

## H — hypothesis

For a fixed vocabulary represented through two hashed shared components, one per-token Givens angle can replace the native two-value token importance vector while retaining token-label NLL/accuracy, beating a scalar importance control, and improving full-payload storage. Collision-degree performance is an explicit boundary measurement.

## Mirror insertion

For token `t` with fixed component indexes `(i1, i2)`, native Hash Embeddings use `a_t C[i1] + b_t C[i2]`. Mirror uses `cos(theta_t) C[i1] + sin(theta_t) C[i2]`, with the same shared component pool and token-index assignments. A scalar control uses `g_t (C[i1] + C[i2])`.

## T — protocol and execution

Protocol frozen before development: vocabulary 1,024, embedding dimension 16, 8 balanced lexical classes, shared pool of 64 vectors, two fixed component indices per token, 1,200 Adam updates per method, and development seeds 38901/38902. Five controls: independent full embedding table, native two-weight Hash Embeddings, unweighted hash sum, scalar importance, and Mirror angle. All methods share labels, hash assignments and output-head initialization within each world.

The task is synthetic token-to-class prediction, not language modeling. Collision degree is the number of tokens with the same ordered hash pair. Results include sampled test and per-token collision bins. Actual FP16 ZIP/NPY inference bytes include hash indices for hashed methods, all learned weights, classifier, metadata and archive overhead. Quality is measured after reloading the payload.

## D — decision

**Fact:** In seed 38901, full/native hash/scalar/Mirror test accuracies were 1.0000/0.9643/0.7215/0.9130, with NLL 0.0019/0.1374/0.6650/0.3005. In seed 38902 the accuracies were 1.0000/0.9495/0.6191/0.9083, with NLL 0.0019/0.1480/0.7656/0.2934. Mirror payload was 7,871 B versus 9,929 B native hash (79.2%); the byte gate passed. Mirror exceeded scalar accuracy by 19.2 and 28.9 points, but fell 5.1 and 4.1 points below native hash, missing the registered 2-point quality margin. In collision degree 1, native/Mirror accuracy was 0.966/0.906 and 0.949/0.904; in degree 3 it was 1.000/0.917 and 0.952/0.857. Those populated bins miss the 5-point collision limit. Unweighted hashing scored 0.291/0.306. Full embeddings used 34,007 B and reached 1.0 accuracy. Ten serialized payloads replayed test and per-token collision metrics exactly; four tests passed. Fresh seeds 38911–38913 remain sealed.

Compute: native hash and Mirror each use two component lookups and a 32 MAC/query importance-combination proxy; scalar uses 16 MAC/query. Mirror training took 1.21/1.42s versus native 1.49/1.35s on this CPU harness. Batched test throughput is recorded per payload in `RESULTS_CORE.csv`; it is a local harness measurement, not an optimized serving kernel.

**Interpretation:** The preregistered MA-389 hypothesis FAILs because Mirror did not preserve native Hash Embedding quality or collision-bin behavior in either development seed. It did replace two importance values per token with one coordinate and reached 79.2% of the native serialized bytes. The large gain over scalar and unweighted hashing shows that a direction-changing code is more useful here than a single gain or no importance state, but it does not match the native two-coefficient freedom.

**Hypothesis:** The missing degree of freedom may be per-token magnitude: a unit-circle Mirror code constrains both coefficient norm and direction, while native Hash Embeddings learn them independently. A separately registered radius-plus-angle ablation could test this; no amendment was made after observing this result.

## C — strongest counter-hypothesis

The seeded random token classes are an artificial lexical task. Performance may overstate the value of independent native weights and may not predict real token distributions. The full table provides an attainable upper control, but no natural vocabulary or language quality is measured.

## U — unresolved

Natural language perplexity, real token-frequency skew, multilingual/OOV robustness, larger component counts, learned hash functions, efficient kernels, and near-converged fixed-byte frontiers remain untested. No general Hash Embedding or capacity claim is made.
