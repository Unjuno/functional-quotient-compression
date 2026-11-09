# MA-689 — Episodic Mirror adapter to semantic consolidation

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE  
Base commit: `8c99306e`

## Hypothesis

H: When episode-specific linear functions share a low-rank semantic basis, transferring their compact episode codes into a persistent consolidated base and deleting the episode store preserves recall while reducing total serialized inference state versus independent full episode adapters.

## Mirror insertion

> **Mirror insertion:** this experiment adds an episode address `m` to one shared low-rank output-function basis so that many episodic mappings can be selected without storing a full private matrix for each episode.

- Native method: independent per-episode output vectors, addressed by episode ID.
- Physical object: one shared rank-4 basis for the output function.
- Logical functions: episode-ID-conditioned mappings reconstructed from basis coefficients.
- Persistent `m`: one code per retained episode; consolidation transfers these codes into the persistent base payload before the episodic payload is deleted.
- Cheapest native control: PCA basis plus ordinary episode coefficients, followed by the same transfer/deletion operation.
- Closest prior art: PA152's episodic low-rank adapters and optional consolidation; PA147's separation of writable long-term memory from persistent learned memory.

## Prior-art delta and comparisons

The experiment isolates the retirement event. It measures recall and complete serialized bytes immediately before transfer, after transfer with the episode store still present, and after deleting that store. Controls are independent full episode vectors, shared-basis episode codes, a dense consolidated base, and native PCA coefficients consolidated in the same way. The shared-basis and native PCA payloads are functionally matched by design; any difference is evaluated only through exact serialized bytes, replay and compute, not labeled a Mirror-specific gain.

## Scope and gates

This is a deterministic synthetic linear memory screen, not a language-model or continual-learning benchmark. Development uses seeds 68901 and 68902; 24 episodes fit the basis and 8 episodes are held out from basis fitting. Heterogeneity is swept at 0, .05 and .2. Fresh seeds remain sealed if the native PCA control exactly aliases the candidate or the development quality/storage gate fails.

PASS requires: training-episode recall NRMSE <= .02 after deleting the episode store; complete post-retirement bytes <= 75% of the independent full bank; no more than 1.25x lookup MACs versus independent retrieval; and strict storage/quality/compute improvement over native PCA consolidation.

FAIL includes loss of recall after deletion, missing the byte/compute gates, or an exact native PCA function/compute alias without strict Pareto gain. Fresh data will not be opened after an exact alias.

## H / T / D / C / U

These sections will be completed after the frozen development run. The final report distinguishes observed facts from interpretation and untested hypotheses.
