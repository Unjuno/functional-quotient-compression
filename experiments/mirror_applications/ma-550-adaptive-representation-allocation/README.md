# MA-550 — Adaptive representation allocation

Status: SCREENING  
Branch: `research/ma-550-adaptive-representation-allocation-20261009`  
Base commit: `6e1f5f7e`  
Prior art: PA96 (ReFT / LoReFT)

## H — Hypothesis

Given the same frozen output projection, a per-function selector can allocate synthetic output changes between activation coordinates and weight-space sparse/dense payloads at lower serialized bytes than either fixed allocation, while meeting a frozen reconstruction tolerance. A native shared-basis/private-code control may explain any activation-space gain, so this is not a Mirror-specific claim.

## T — Protocol

Use pinned Pythia-70M-deduped `embed_out.weight` as the shared linear map `W` (vocabulary × 512). Each seeded world contains 16 target changes: eight one-token sparse logit shifts and eight dense changes generated as `Wm`, with `m ~ N(0, .05² I)`. Eight noisy support observations per function determine the estimate; eight independent held-out observations audit it. No model training or optimizer updates are used.

Compare fixed dense weight deltas, fixed activation coordinates (least-squares `m`), sparse weight deltas where they meet the preregistered 1% relative support RMSE, and adaptive minimum-byte eligible allocation. A shared-basis/private-code control is represented by the native output basis `W` plus task codes; common `W` is already part of the paid base checkpoint. Every per-world NPZ is serialized and charged, including method IDs, function IDs, seeds, and metadata. Report per-method error, bytes, fit/decode operations and elapsed time.

Development worlds: 55001, 55002. Fresh worlds: 55011, 55012, 55013, opened only after source/protocol freeze and development decision. Selection threshold: support relative RMSE ≤ 0.01. No hyperparameter search is allowed.

## D — Decision scope

PASS only if adaptive allocation meets the support and fresh held-out gates and reduces actual payload bytes by at least 10% versus both fixed allocations, with no more than 0.01 held-out relative RMSE. FAIL if either fixed representation matches or beats adaptive bytes at the same quality, or fresh quality fails. This synthetic linear screen cannot establish natural task quality, learned Mirror capacity, or runtime Pareto improvement.

## Fact / Interpretation / Hypothesis

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: heterogeneous output functions may favor different ordinary representation families; native basis coding is the strongest counter-explanation.
