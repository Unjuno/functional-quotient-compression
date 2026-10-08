# MA-1064 status

Status: FAIL (scoped MovieLens-100K screen)

## H

A per-rare-item Mirror angle plus rank-1 private residual improves tail AUC beyond an equal-budget ordinary rank-2 residual while preserving overall quality and beating full-table storage and DHE latency gates.

## T

MovieLens-100K chronological train/dev/audit split; three model seeds; rare IDs fixed from train-only counts. Compare full table, DHE-style hash decoder, Mirror-only, rank-2 additive residual, Mirror+rank-1 private residual and rare full residual. No audit examples are read until development checkpoints are selected.

## D

**FAIL.** Across seeds 31/47/59, DHE + Mirror angle + rank-1 residual averaged 0.44456 rare-item AUC, only +0.00078 over ordinary rank-2 residual (0.44378), below the +0.01 gate. Overall AUC was 0.49622 vs 0.49617; payload 189,240 B vs 188,896 B; P95 batch-1 latency 0.200 ms vs DHE 0.090 ms. It used 55.6% of full-table bytes but failed the latency and tail-quality gates.

The audit model family itself was weak on this split: all model logloss values were around 0.82–0.90 versus a train-prevalence constant predictor at 0.68478; the full table averaged AUC 0.506. For seed 31, DHE train AUC/logloss were 0.695/0.625 versus dev 0.537/0.799; full-table train was 0.814/0.529 versus dev 0.522/0.851, showing severe temporal generalization failure. Only 512 audit events were train-rare items. This limits conclusions to this implementation and split.

## C

Any tail gains are explained by the added ordinary residual degrees of freedom or by frequency-based allocation, while the Mirror angle adds bytes and per-request math.

## U

Generalization beyond MovieLens-100K, a better-calibrated recommender objective, optimized production DHE/TT-Rec kernels, and cold-start quality on new catalogs are unestablished.

## Fact / interpretation / hypothesis

- **Fact:** Six methods across three seeds produced 18 inference payloads. The candidate gained +0.00078 rare AUC over the equal-order rank-2 residual control, added 344 B, and ran at 2.2× DHE P95 latency. Six focused tests passed.
- **Interpretation:** The measured gains are too small and slow to justify the Mirror coordinate; ordinary residual state matches it at slightly lower storage and latency.
- **Hypothesis:** DHE plus frequency-targeted private state could still help on a stronger recommender and a larger long-tail event set; this test does not establish that.

## Amendment / execution notes

- One audit invocation stopped after evaluating the first checkpoint because the training-selection log omitted an `events_seen` field. No metrics were saved from that invocation. The field was read from the already-written development selection record and all same frozen checkpoints were evaluated; no model setting changed.
- Amendment 1 fixed non-Mirror payload metadata only. Old metrics remain in `AMENDMENT1_PRE_CORRECTION.*`; final byte values use the corrected method-specific serializer metadata.
