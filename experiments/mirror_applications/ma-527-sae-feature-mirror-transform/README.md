# MA-527 — Mirror transform over SAE feature space

Status: **FAIL (development gate; no fresh metrics).** Protocol frozen before data extraction. Prior art: PA102 and PA96.

## H — falsifiable hypothesis

A compact Givens transform over sparse SAE coordinates will preserve held-out relation steering quality while keeping feature activity sparse, and will improve the byte/quality frontier over native SAE top-k, global sparse coding, and LoReFT.

## Insertion point

The intervention transforms a sparse feature activation code immediately before the shared tied SAE decoder maps it back to the residual stream. This isolates Mirror coordinates from neuron-space steering. The transform is parameterized by paired Givens angles; all angles, indices, coefficients and metadata are paid.

## Controls and accounting

Controls and fixed split are listed in [PROTOCOL.json](PROTOCOL.json). The pinned model and complete SAE state are shared across methods and charged once. Method-specific code is measured from actual serialized inference files. Dense residual steering is an upper reference; LoReFT rank 8 is the simple representation-space control.

## Data discipline

Fit tasks choose the shared SAE coordinate support. Development tasks used one fixed angle/scale configuration. The run had no fresh-seed causal evaluation. A harness audit found that task IDs 14–15 were nevertheless extracted and serialized during development; do not treat them as sealed audit identities.

## Evidence limits

This experiment tests four relation interventions on one model and layer. A positive result would establish only scoped feature-space steering, not arbitrary behavior capacity or routing.

## Development result

Status: **FAIL; no fresh causal metrics.** With alpha=0.5 matched for explicit and compressed codes, Mirror gold-logprob was 2.094 and 1.309 nats below explicit FV across seeds 52701/52702, missing the frozen 0.10-nat gate. The serialized Mirror code used 3,474 B versus 34,742 B for explicit FVs and retained eight nonzeros/task. Standalone Mirror deployment was 172,352,489 B after charging the 4,204,391 B SAE; explicit-FV deployment was 168,179,366 B. The dev harness inadvertently extracted and serialized task IDs 14–15 but never evaluated them; fresh seeds 52711–52713 were not run. Do not reuse those task identities for audit. Per-seed measurements and payloads are under `runs/dev/`; their actual bytes and SHA-256 values are recorded in [PAYLOAD_MANIFEST.json](PAYLOAD_MANIFEST.json). See [STATUS.md](STATUS.md) for H/T/D/C/U, controls and limitations.
