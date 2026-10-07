# Worker queue

The queue is derived from `IDEA_REGISTRY.csv`. The registry is authoritative.

## Selection rule

Pick the first candidate satisfying all of:
1. status is UNTESTED;
2. highest available priority;
3. no other active experiment directory already claims the ID;
4. its closest prior-art controls can be implemented in the current harness.

Do not skip to a visually interesting P1/P2 idea while an executable P0 remains, unless the skipped candidate has a recorded blocker.

## Literature-derived cross-over queue

These were added after the 2026-10-07 prior-art sweep and should be considered before duplicating a simpler P0 experiment:


These have strong adjacent prior art, so the experiment must implement the cited non-Mirror method as a control.

## Current P0 sequence

### Family A — FFN / MoE / adapter
Completed this run: MA-003 -> MA-005 -> MA-009 -> MA-019 -> MA-024.

### Family B — Attention / KV
MA-063 (paused after MA-061/063 diagnostic; redesign required before more KV-view work).

### Family C — Depth / position
MA-076 and MA-079 completed PROMISING; MA-086 failed the storage gate. Independent layer functions did not benefit from shared views.
MA-116 completed FAIL at the development gate.

### Family D — Temporal
MA-121 completed PROMISING on aligned phase slots; MA-129 completed FAIL at development; three exploratory worlds were opened after gate failure and excluded from status.

### Family E — Compression / holographic
MA-156 completed PROMISING on aligned quantized weight views; strict secondary margin missed.
MA-160 completed PROMISING with its strict fresh quality gate narrowly missed: 0.647x bytes vs independent int4, 1.035x mean fresh MSE, but one seed at 1.106x crossed the 1.10 limit.
MA-171 completed PROMISING on an HRR-aligned fixed-budget teacher; 0.533x bytes and lower MSE than untied in 3/3 corrected fresh worlds, with wrong-LR exploratory access disclosed.
MA-173 completed PROMISING: learned FFT-phase views passed aligned quality/bytes in 3/3 fresh worlds (0.592x bytes, lower fixed-update MSE); active MAC proxy improved, but wall time/throughput regressed. Independent roles required private parameters.
MA-181 completed PROMISING: shared block-circulant plus charged role shifts reached 0.413x untied bytes and similar quality to independent BCA, saving 21% more bytes, but eager runtime regressed sharply; unrelated roles needed dense private state.

### Family F — Continual / optimization / distillation
MA-186 completed FAIL for registered aligned quality gate: 20B/skill vs 241B/skill rank-2 LoRA, retained prior aligned tasks 3/3, but missed relative quality in 2/3 fresh worlds; unrelated tasks needed private state. Shared hypernetwork control was degenerate and cannot establish Mirror-specific advantage. -> **MA-189 next** -> MA-199 -> MA-208

## Family handoff rule

Within one family, reuse the same parent checkpoint, serializer, data split, benchmark code and result schema whenever scientifically valid.

A worker may batch implementation work across a family, but scientific status is still assigned per MA ID.

## Stop rule

If two consecutive candidates in a family fail for the same demonstrated structural reason, stop that family and write a family diagnostic before continuing.
