# MA-841 — AI Engram bank compressed by a shared basis

## H — hypothesis

A rank-4 basis with per-memory coefficients may preserve causal behavior of closed-form memory traces at lower actual bytes; residual memories may require private state. A matched PCA control determines whether this is ordinary subspace compression.

## T — execution

Three fresh worlds (84111–84113), each with 8 development traces estimating a rank-4 basis and 24 disjoint audit memories (16 aligned, 8 with independent residuals). Each memory binds 8 random keys to values through a closed-form ridge trace. Measured normalized sufficiency error, reactivation cosine, specificity leakage ratio, necessity effect ratio, serialized payload and a thresholded full-residual fallback.

Deviation: standalone development seed 84101 was not separately run; per-world development trace IDs were used to estimate each basis. Rank 4 stayed fixed and audit outcomes were not used for tuning.

## D — PROMISING, scoped to aligned synthetic memories

Across all 3 worlds, aligned Mirror memories averaged normalized sufficiency MSE `6.03e-4`, reactivation cosine `0.999981`, specificity leakage ratio `0.99560`, and necessity effect ratio `0.99574`. The contiguous Mirror bank used 1,552 B (64.7 B/memory) versus 6,224 B for exact traces (259.3 B/memory), or 24.9% of the exact payload. A random rank-4 basis failed badly. The same-rank PCA control matched Mirror exactly on every metric and byte, so this is not a Mirror-specific advantage.

For residual memories, code-only sufficiency error averaged `0.712`; the registered full-residual fallback restored native causal metrics and used 4,080 B, 65.6% of exact-bank bytes. The fallback activated for 8–9 of 24 memories per world.

## C — strongest counter-hypothesis

A shared PCA basis over memory traces fully explains the aligned compression result; Mirror codes are just PCA coefficients here. The residual fallback is close to storing private memory state and does not meet the aligned 50% storage ratio.

## U — unconfirmed

No natural continual-learning benchmark, language model, or learned causal retrieval system was tested. The keys/values are synthetic linear associations, and no model-level reactivation/necessity intervention beyond the stated trace metrics was run.

## Fact / Interpretation / Hypothesis

- Fact: aligned causal behavior passed with 24.9% of exact-bank bytes; PCA and Mirror were numerically identical; residual memories required paid private state.
- Interpretation: shared-basis compression works for this aligned trace family, but does not establish Mirror-specific value. The residual/private boundary is visible in both quality and bytes.
- Hypothesis: any future claim should beat PCA or demonstrate improved causal memory behavior on non-synthetic task streams.
