# Deterministic Real-Model Manifest Builders

This document defines the canonical bridge from a live checkpoint to the D57
Transformer extraction manifest and then to the D56 real-pilot contract.

## Design rule

Architecture interpretation is explicit; storage identity is measured.

The builder does **not** guess from tensor names that a matrix is Q, K, V, O,
that an architecture uses GQA/MQA, or that a positional operator is RoPE. An
architecture adapter must provide those facts. The builder is authoritative for
storage aliasing because it inspects the live tensor storage ranges.

## Pipeline

```text
live checkpoint tensors
  -> explicit architecture adapter
  -> storage-range inventory
  -> deterministic D57 extraction manifest
  -> validate_transformer_extraction
  -> unique baseline scalar count N
  -> deterministic D56 pilot contract
  -> floor(N/4) hard 64x bit budget for a 16-bit baseline
```

This removes a previous duplication hazard: D56's `N` is no longer manually
entered independently of the D57 tensor inventory.

## Reproducibility rules

1. Runtime storage pointers may be used to prove aliasing during extraction.
2. Runtime pointers are never public manifest identifiers.
3. Public storage-group IDs are deterministic functions of tensor membership.
4. Partial overlapping storage views are rejected unless explicit range
   accounting is implemented.
5. Canonical JSON uses sorted keys, compact separators, UTF-8, and rejects NaN.
6. Pilot artifact hashes are SHA-256 hashes of canonical JSON artifacts.
7. If the D57 model identity declares a checkpoint hash, the D56 pilot builder
   requires the same hash.

## GQA helper

`contiguous_gqa_map(q_heads, kv_heads)` is provided only as an explicit helper
for architectures that truly use contiguous grouped-query head sharing. It is
not an inference rule. Non-divisible head counts are rejected.

## Evidence boundary

A generated and validated manifest proves accounting consistency and records
architecture metadata. It does not prove that the chosen architecture adapter
is semantically correct. Adapter correctness must be checked against the model
implementation/configuration and then by replay of extracted primitives.

Likewise, a valid D56 contract does not establish compression. A real claim
still requires an actual serialized paid payload and the predeclared quality
witness.
