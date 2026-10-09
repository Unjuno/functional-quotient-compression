# MA-086 — layer-group Mirror views

Status: FAIL at development gate
Evidence lane: MECHANISM
Base commit: `e0cf13997e0477e9022e4c1fcf07bd99f5075ccb`

## Hypothesis

H: Pairwise tied group bases plus learned layer Givens views recover pair-structured layer functions with at least 35% fewer serialized bytes than untied layers, while hard group tying loses quality. Larger groups may extend compression at a quality cost.

## Physical-to-logical claim

- Physical objects: one 8×8 matrix per contiguous group of two or four logical layers.
- Mirror coordinate: one learned Givens angle per logical layer.
- Multiplicity: eight logical depth maps from four or two matrices.
- Failure mode: independent layer maps need private parameters; static low-rank residuals may be competitive.

## Prior-art delta

PA01/PA06 establish tied expert/block groups and generated step modulation. MA-076/079 studied a single shared block; MA-086 measured groups of two and four against hard tying, a per-layer rank-1 residual, and untied matrices.

## Development results

Two development worlds, 600 Adam updates, LR 0.01. Fresh worlds remained sealed after the preregistered byte gate failed.

| Method | Median aligned test MSE | Serialized payload |
|---|---:|---:|
| tied groups of 2 | 0.0584 | 2,729B |
| rank-1 LoRA on groups of 2 | 0.0117 | 3,617B |
| Mirror groups of 2 | 8.35e-13 | 2,917B |
| tied groups of 4 | 0.5251 | 2,217B |
| Mirror groups of 4 | 0.4583 | 2,405B |
| untied | 1.28e-10 | 3,753B |

Mirror groups of 2 matched untied quality and beat hard tying and rank-1 LoRA, but saved only 22.3% bytes vs untied, missing the preregistered 35% reduction requirement (payload ratio 0.777 vs required ≤0.65). Groups of 4 saved 35.9% bytes but had much worse error. Independent-layer results show that both Mirror group sizes need private capacity for arbitrary maps; rank-1 LoRA was better than Mirror2 there. Mirror2's MAC proxy was 6.25% above the shared projection baseline; measured CPU wall time was about 6.1x untied in the aligned worlds.

## Decision

**FACT:** The quality gate passed on development, but the storage gate missed in both worlds, so fresh was not opened. Payload lengths are actual serialized state plus method metadata. Twenty-four result rows replayed with exact bytes and maximum MSE delta 4.40e-11; tests passed 2/2.

**INTERPRETATION:** Pairwise Givens views can recover an aligned group teacher, but their extra coordinates do not yield the preregistered byte improvement. Increasing group size improves compression and degrades quality. This candidate is FAIL under the fixed storage gate.

**HYPOTHESIS:** A useful depth-group frontier likely needs either fewer coordinates per layer or group structures with substantially more than two repeated layers.

**BOUNDARY:** Synthetic 8D linear maps, pair-aligned teacher, two development worlds, CPU only. No fresh generalization, nonlinear Transformer, language, or capacity claim.
