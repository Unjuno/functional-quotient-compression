# MA-689 — Episodic Mirror adapter to semantic consolidation

Status: **FAIL for Mirror-specific value**  
Evidence lane: MECHANISM / STORAGE / COMPUTE  
Protocol frozen: `7715d2c7`; schema-accounting amendment made before corrected development rerun.

## Hypothesis

H: For aligned episode-conditioned linear functions, transferring compact shared-basis episode codes into a persistent consolidated base and deleting the episode store preserves recall at lower complete bytes than independent full episode adapters.

## Mirror insertion

> **Mirror insertion:** this experiment adds an episode address `m` to one shared low-rank output-function basis so that many logical episode mappings can be selected without storing a full private matrix for each episode.

The physical object is a rank-4 basis in R^32. A persistent per-episode coordinate selects a reconstructed vector. The episode codes are moved into the consolidated base checkpoint before the old episode payload is deleted. The closest native method is PCA basis plus ordinary coefficients, transferred and retired identically.

## T — Treatment and controls

Synthetic 32-episode vector memory, 24 basis-fit episodes and 8 excluded episodes, rank 4, rho=0/.05/.2, seeds 68901/68902. Compared independent full vectors, shared episode codes, exact dense consolidation, and native PCA consolidation. Counted complete base+episode payload before transfer, the live duplicate peak while both copies existed, and base-only payload after episode deletion. Serialized NPZ replay was checked for every saved state. Fresh seeds 68911/68912/68913 remained sealed.

## D — Decision

**FAIL for Mirror-specific value.**

- At rho=0, post-deletion NRMSE was <7.5e-8 in both seeds. Consolidated code payloads were 1,797B, about 47% of the independent pre-transfer bank (3,782–3,798B). Lookup decode used 128 MAC/vector; independent direct retrieval used zero MACs.
- At rho=.05, post-deletion NRMSE was .0432–.0448; at rho=.2 it was .1728–.1790, both above the .02 gate. Payloads remained about 1.8KB.
- Dense consolidation retained exact vectors but used 3,557–3,581B. The temporary transfer peak reached about 7.3KB because old and new state coexisted.
- With canonical identical serialization schemas, native PCA and Mirror produced equal functions, compute, and actual bytes in every tested condition. This exact alias triggers the frozen fresh-sealing rule.

## C — Strongest counter-hypothesis

Ordinary PCA coordinates and deterministic checkpoint-state transfer fully explain the aligned-case storage result. Any byte saving comes from low-rank coding, not a Mirror-specific consolidation mechanism.

## U — Unconfirmed

This is a synthetic vector-memory test. It does not establish semantic consolidation, learned base-weight integration, natural episodic recall, language quality, adaptive forgetting, continual learning, or hardware serving latency. The compact checkpoint still stores one address per episode; it relocates the episode representation instead of deleting episode-specific information.

## Accounting amendment

The initial successful development run used longer NPZ field names for native PCA, creating a metadata-only byte difference. Its source, payloads and results are preserved under `source/attempts/initial_noncanonical_native_schema/`. Before corrected development rerun, the protocol was amended to use identical serialization field names/order for equivalent Mirror and PCA arrays. Scientific gates, task, seeds and settings did not change. See [STATUS.md](STATUS.md), [PROTOCOL.json](PROTOCOL.json), [RESULTS_CORE.csv](RESULTS_CORE.csv) and [VERIFICATION.json](VERIFICATION.json).
