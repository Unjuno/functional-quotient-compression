# MA-403 — Token-wise generated Mirror modulation

Status: **PROTOCOL FROZEN BEFORE DEVELOPMENT**

## H — falsifiable hypothesis

An input- and context-conditioned Givens generator can produce useful per-token functional Views on a shared classifier, approaching an aligned teacher while outperforming static codes and remaining competitive with a same-input FiLM generator after charging generator state and operations.

## Prior art delta

PA63 FiLM generates feature-wise affine parameters; PA64 SPADE establishes spatially varying affine conditioning. This screen tests whether a compact orthogonal token View adds useful function under rapid per-token context changes.

## T — frozen protocol

Synthetic teacher-forced token classification. Each sequence has eight tokens and context changes per token. A shared random ReLU teacher applies one of four seeded Givens views. Compare shared no modulation, token-input/context Givens generator, same-input grouped FiLM generator, static per-context angle table, and independent per-context classifier. Development worlds 40300/40301 select LR from {0.003,0.01}; fresh 40310/40311/40312 are sealed. Actual serialized payload includes all generator/table state. Measure generator MAC per token and throughput.

The aligned synthetic teacher is a mechanism screen, not natural language evidence. All gates are fixed in `PROTOCOL.json`.
