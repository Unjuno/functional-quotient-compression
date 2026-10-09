# MA-448 — Optimizer-state Mirror superposition

## H — Hypothesis

A shared rank-4 state basis plus per-task Mirror coordinates can reduce task-specific Adam resume state while preserving five-step continuation quality.

## T — Planned test

16D linear tasks with four latent skill factors. Store model theta plus Adam first/second moments at step 20, then resume for five updates. Compare exact independent states, shared-only, rank-4 PCA, and learned shared basis plus 4D task code. Development and fresh task IDs are disjoint. Exact serialized payloads include model state, basis/codes, counters, and indices.

## D — Pending

Protocol frozen; no numerical results.

## C — Strongest counter-hypothesis

The learned state family is already rank-4, so standard PCA or direct task arithmetic should match Mirror with fewer or equal bytes.

## U — Unknown

Whether compressed moment state preserves actual optimizer continuation and whether state bytes materially dominate task storage.
