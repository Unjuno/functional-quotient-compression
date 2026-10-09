# MA-478 — View first with explicit private-value fallback

Status: SCREENING  
Branch: `research/ma-478-view-first-private-fallback-20261008`  
Base commit: `abbf9df2e005666016ee55bd272a2d29e00c7bbf`  
Prior art: PA89 (SERAC), PA90 (GRACE)

## H — Hypothesis

A shared rank-eight value view should encode aligned edits cheaply. An edit whose residual exceeds the frozen L2 threshold 0.05 receives one explicit private value. This tests whether a small private fraction can recover off-basis edits at a better complete-bank frontier than explicit values or int8.

## T — Frozen protocol

A 256-entry fixed-key memory has 64-value heldout split, including 16 private/off-basis values. Fit the shared basis only on 192 aligned training values. Compare explicit, shared-only, native PCA with the same residual fallback, int8 and no-edit. All keys/router state are paid. Fresh seeds 47811–47813 stay sealed unless all gates pass. Exact details are in `PROTOCOL.json`.

## Results

Pending frozen development runs.
