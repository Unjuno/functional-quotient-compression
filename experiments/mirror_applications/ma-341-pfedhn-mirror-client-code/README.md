# MA-341 — Mirror client codes versus pFedHN

Status: SCREENING
Evidence lane: PERSONALIZATION / STORAGE / COMMUNICATION
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`

## H — falsifiable hypothesis

On held-out non-IID client partitions of the public sklearn Digits task, a shared classifier with four per-client Givens View angles can reach within two percentage points of a pFedHN-style full-model generator after the same 20 support updates, with lower actual server/client payload bytes and lower client download volume, while beating a byte-near FiLM code control.

## Draw26

MA-341 was sampled uniformly from 526 eligible P0/UNTESTED candidates after excluding current live MA branches and pre-existing MA experiment directories. The exact pool, SHA-256, seed, index and exclusions are frozen in `source/draw26_exclusions.json`.

## T — experiment

Simulate 12 clients from the public 8×8 sklearn Digits classification data with fixed Dirichlet label skew. Clients 0–9 train the shared systems; clients 10–11 remain unseen until local support adaptation. Run three fresh partition/model seeds. Compare pFedHN-style client-code-to-full-model generation, a shared MLP with Givens Mirror client codes, byte-near shared MLP + FiLM codes, a global shared MLP, and independent local models. During unseen adaptation, only the client's code is updated (the full local model for the independent control); shared weights stay frozen. The same 20 full-batch support updates are used and their optimization targets are explicit in the results.

Checkpoint selection uses development examples from training clients only. On unseen clients, adapt only client codes or the local model using support examples; query examples remain audit. Measure accuracy and cross-entropy, actual serialized server/client state, communication bytes, adaptation updates, client-model generation MACs, CPU latency and wall time.

## D — gates

**PASS for this screen** requires Mirror within two accuracy points of pFedHN after 20 support updates on both held-out clients in at least 2/3 seeds; server plus registered-client payload ≤80% of pFedHN; per-client download after the shared base is cached ≤25% of a generated full model; and the Mirror code is not dominated by byte-near FiLM.

**FAIL** if any gate fails. **NOT ESTABLISHED** if client partitioning or implementation invalidates the comparison. This simulated small-classification task does not establish production federated learning or privacy.

## C — strongest counter-hypothesis

pFedHN's client embedding already compresses personalization, and an ordinary shared base plus FiLM or low-rank code matches Mirror quality with equal or lower state and communication.

## U — boundaries

This uses simulated clients from one small public image dataset; it is not real user data, a secure aggregation protocol, or the full pFedHN reference implementation. See `PROTOCOL.json`, `RESULTS_CORE.csv`, `STATUS.md`, and `VERIFICATION.json`.
