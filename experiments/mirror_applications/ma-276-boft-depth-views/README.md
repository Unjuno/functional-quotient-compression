# MA-276 — BOFT views for tied depth

## H — Hypothesis

Per-depth low-description orthogonal coordinates can turn one shared block into distinct useful depth operators with less storage than untied blocks. Independent targets identify when private capacity is needed.

## T — Conditions and provenance

Synthetic 16×16 recurrent linear block reused for four steps. Fresh initial worlds 27610–27612 × seeds 0–2 compared hard tying, rank-one residual, rank-2 LoRA, per-step Givens Mirror and untied upper; 360 rows. A preregistered generic scalar control was omitted by the initial runner. After initial fresh, an amendment added it on new worlds 27620–27622 × seeds 0–2 (A1, 432 rows). A1 is retained separately. The A1 runner wrote to the same `FRESH_RESULTS.csv` path and overwrote the original row-level CSV. Original fresh payload files remain, and the original aggregates had already been recorded in the worker transcript; original fresh rows were not regenerated. This provenance loss is explicit.

## D — Aligned-orbit result; no Mirror-specific claim

Original fresh aggregate (recorded before A1): aligned Mirror NRMSE 3.70e-8 at 1,106 B; LoRA NRMSE 3.17e-4 at 2,114 B; untied 0 at 4,165 B. Independent Mirror NRMSE 0.858, LoRA 0.620, untied 0. Per-step Mirror aligned NRMSE stayed around 3.4–3.9e-8.

A1 results on new worlds:

| Stratum | Method | Mean NRMSE | Payload B | Fit seconds |
|---|---|---:|---:|---:|
| aligned_rotations | tied | 0.137356 | 1,088 | 0.346 |
| aligned_rotations | rank1 | 0.0915526 | 1,410 | 0.399 |
| aligned_rotations | lora | 4.24343e-05 | 2,114 | 0.388 |
| aligned_rotations | mirror | 3.49998e-08 | 1,106 | 0.525 |
| aligned_rotations | generic_plane | 3.49998e-08 | 1,113 | 0.531 |
| aligned_rotations | untied | 0 | 4,165 | 0.364 |
| independent | tied | 0.873147 | 1,088 | 0.329 |
| independent | rank1 | 0.808805 | 1,410 | 0.430 |
| independent | lora | 0.615473 | 2,114 | 0.399 |
| independent | mirror | 0.862417 | 1,106 | 0.510 |
| independent | generic_plane | 0.862417 | 1,113 | 0.492 |
| independent | untied | 0 | 4,165 | 0.323 |

Fact: A1 generic scalar and Mirror outputs are identical in both strata; aligned mean NRMSE 3.50e-8, independent 0.862. Generic payload is 1,113 B vs 1,106 B because method metadata differs; same underlying code/behavior. LoRA gets aligned 4.24e-5 at 2,114 B; independent 0.615. Untied remains exact at 4,165 B.

Interpretation: one shared physical block plus one angle per step reconstructs a four-step aligned orbit with ~73% fewer bytes than untied blocks and better bytes than rank-2 LoRA. The result is a generic scalar Givens view, not Mirror-specific. Independent steps need private capacity; LoRA improves over one angle but is still far from the untied upper at this update budget.

## C — Strongest counter-hypothesis

The task teacher is exactly generated from the same one-plane rotation family, so the compression is an orbit-matched result. The generic scalar control reproduces it exactly. This does not demonstrate natural Transformer depth diversity.

## U — Limitations

Original initial-fresh row-level CSV was overwritten by the A1 runner. Original payloads and previously captured aggregates are retained, but not original per-row metrics. No natural language, nonlinear Transformer block, depth routing, or GPU runtime was tested. Registry status is FAIL for Mirror-specific value, with a scoped generic-orbit compression result preserved.
