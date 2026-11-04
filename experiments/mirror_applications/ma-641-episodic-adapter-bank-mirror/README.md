# MA-641 — Episodic adapter bank Mirror compression

Status: FAIL. Evidence lane: synthetic mechanism screen.

## H

A rank-4 shared output basis with episode-specific Mirror coordinates compresses correlated episodic updates while preserving episode recall. Increasing private residual should expose the point where extra private directions become necessary.

## T

Tested a 128D key-to-answer adapter bank with 16, 64, or 128 episodes and private residual RMS 0, .05, .1, or .25. Two development seeds (64101, 64102) compare independent sparse rank-1 adapters, rank-4 Mirror codes and native rank-4 PCA. The fixed query router and dense base executor are shared and charged. Fresh seeds 64111–64113 remained sealed after Mirror/native exact function alias.

## D

At zero residual, 64 and 128 episode banks retained output quality and reduced average complete payload to 70.4% and 53.7% of the independent adapter bank. The 16 episode case reached 92.5%. The Mirror state exactly equals native PCA state and output, so there is no Mirror-specific gain. At residual RMS .05/.1/.25, nRMSE rises and cosine-threshold recall falls; rank-4 decoding uses 4x the independent operation proxy.

The preregistered text called for nearest-answer recall, while the implementation used cosine >=.99 recall. This metric deviation is explicit in `STATUS.md`; no fresh data was opened. The shared-basis/PCA alias and byte gate determine the FAIL ruling without relying on that recall metric.

## Fact / Interpretation / Hypothesis

- **Fact:** 72 development rows; actual full payload bytes and output metrics replay; all tests pass.
- **Interpretation:** Ordinary low-rank factorization explains the storage improvement; Mirror adds no distinct function or bytes advantage.
- **Hypothesis:** Real episodic banks may offer a generic low-rank storage frontier, subject to validation with realistic retrieval and edit tasks.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
