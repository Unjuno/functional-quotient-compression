# MA-024 — one LoRA to many virtual LoRAs

Status: SCREENING. Branch `research/ma-024-virtual-lora-mirror-20261007`.

## H

One shared rank-2 LoRA basis with per-task rank-space Mirror rotations can represent eight adapter functions at lower serialized bytes than an independent LoRA bank. Generic shared-basis coefficients and a small task hypernetwork are the relevant controls.

## T

Synthetic task-conditioned 16D-to-12D adapter regression with eight oracle task IDs. Compare full rank-2 LoRA bank, shared LoRA, generic two-basis coefficients, task-embedding hypernetwork, and Mirror rank rotations. Aligned teacher uses the same shared rotated LoRA family; independent teacher uses unrelated rank-2 factors.

## Decision

FACT: pending development/fresh. INTERPRETATION: pending. HYPOTHESIS: pending. BOUNDARY: frozen-feature linear adapters, oracle task IDs; no language/router claim.
