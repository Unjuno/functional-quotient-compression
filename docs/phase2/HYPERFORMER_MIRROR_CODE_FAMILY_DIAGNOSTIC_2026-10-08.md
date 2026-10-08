# HyperFormer/Mirror code-output family diagnostic — 2026-10-08

## Decision

Pause MA-463's unchanged factorized task × layer × position code-output formulation after MA-461 and MA-462 independently missed their frozen gates on the same central tradeoff: a compact generated coordinate can reduce some decoder state or improve withheld combinations, but does not retain the seen-task fidelity/quality needed to replace a full HyperFormer generator across worlds. Reopen this family only with a materially different insertion or a control that is not the same shared low-rank generator.

## Evidence

| MA | Mechanism | Facts | Decision |
|---|---|---|---|
| 461 | Learned task/layer embeddings → two-code generator → shared rank-two adapter basis | Payload 3,286 vs 3,854 bytes and 160 vs 352 MAC proxy/input; heldout RMSE worse than HyperFormer in both seeds, >0.05 in one. Direct native low-rank generator exactly matched output, payload and hash. | FAIL; no Mirror-specific gain and quality/storage gates miss. |
| 462 | Fixed factor embeddings → product-factor Mirror decoder over shared atoms, ranks 2/4/8 | All ranks passed heldout-vs-HyperFormer and byte gates, but no rank passed the seen-fit conjunction in both development worlds. Rank 8 passed all clauses on one world only. Fresh remained sealed. | FAIL; the compact-view heldout signal does not survive the frozen cross-world seen-quality gate. |

MA-462 evidence is on branch `research/ma-462-mirror-hyperformer-decoder-20261008`, report `experiments/mirror_applications/ma-462-mirror-hyperformer-decoder/README.md`, result commit `ac949a7bec7f29727060ef49c4cf0c3a335db3cb`. MA-461 evidence is in this branch's report and verification. Both are synthetic adapter-generation mechanism screens, not language-model capacity evidence.

## Cause and boundary

**Fact:** Both compact code families faced a seen-function fidelity requirement; MA-462 missed it across worlds, while MA-461's matched rank-two native generator made identical functions and bytes.

**Interpretation:** On this task family, low-dimensional decoder output trades seen-task fit against storage, and the functional form is already ordinary shared-basis low-rank generation.

**Hypothesis:** An unchanged factorized output-code extension at MA-463 risks repeating the same tradeoff. Its queue candidate remains UNTESTED in the registry but is operationally paused pending a distinct hypothesis.

This pause does not rule out task/layer hypernetworks generally, larger private residual frontiers, or natural PEFT deployments; those require a new non-aliased claim and matched strong controls.
