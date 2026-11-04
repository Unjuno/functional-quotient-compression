# MA-646 — ReLoRA restart coordinates

Status: FAIL. Evidence lane: synthetic mechanism/storage screen.

## H

Does reusing one rank-4 update basis across four restarts retain ReLoRA's rank growth? Can a small address select phase-specific directions from one shared basis without extra storage?

## T

A 32x32 rank-16 update was decomposed into four orthogonal rank-4 phase updates. Compared merged ReLoRA, static rank-4 basis reuse, a shared rank-16 dictionary plus phase address, and native block IDs. Two development seeds; full serialized inference and restart-state bytes were measured.

## D

Merged ReLoRA and shared dictionary both recover rank 16 at relative error 7.59e-8. Static basis reuse is rank 4 with relative error .837. The dictionary costs 8,780B for restart state (same as ReLoRA factors) and 8,780B for inference, above merged ReLoRA at 8,297B. Native block IDs exactly match Mirror in payload and function. Fresh remained sealed.

## Fact / Interpretation / Hypothesis

- **Fact:** 8 rows; serialization and metric replay pass.
- **Interpretation:** Static reuse caps rank; indexed subspace changes restore rank but are ordinary codebook selection.
- **Hypothesis:** Temporal composition may help where phase updates share useful subspaces, subject to matched native controls and actual-byte gains.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
