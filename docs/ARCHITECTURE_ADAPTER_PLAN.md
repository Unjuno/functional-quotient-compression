# Architecture Adapter Plan

The real-model extraction core intentionally does not infer tensor semantics from
checkpoint names. An architecture adapter must state which checkpoint tensor is
used for each public FQC tensor role and which model-config facts justify the
module metadata.

## Why this exists

A deterministic storage manifest is not sufficient if Q/K/V/O, GQA/MQA, RoPE,
MLP, or normalization semantics were assigned incorrectly. The adapter plan is
therefore a first-class provenance artifact.

## Adapter plan contents

- adapter ID and version;
- model identity and orientation;
- public tensor ID -> checkpoint key -> semantic role bindings;
- whether each binding participates in the baseline denominator;
- attention, MLP, and normalization module descriptions;
- external fixed state and derived primitives;
- model-config evidence used to justify the mapping.

The canonical JSON form of the adapter plan is hashed. The SHA-256 is embedded
in the generated D57 manifest together with the tensor-binding provenance.

## Separation of authority

The adapter is authoritative for architecture semantics only when supported by
the actual model implementation/configuration. The storage inventory is
authoritative for live aliasing and the unique baseline scalar count.

This separation prevents two failure modes:

1. guessing architecture semantics from names;
2. manually declaring ties/storage groups to improve the compression ratio.

## Required real-pilot check

Before a new architecture adapter is trusted, its smallest supported checkpoint
must pass:

1. D57 manifest validation;
2. projection-shape checks for the declared orientation;
3. storage-identity checks;
4. extracted-primitive replay against the original module;
5. a deterministic second extraction producing the same public manifest hash.

A valid adapter manifest is an accounting/extraction prerequisite, not evidence
of compression by itself.
