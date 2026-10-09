# MA-545 protocol amendment 2 — serialize and charge split manifest

Date: 2026-10-09 UTC

## Reason

The frozen storage list charged the support/query split manifest, but the first payload serializer kept it only as an audit sidecar. This mismatch was found during verification, before fresh-seed access.

## Change

Serialize the exact UTF-8 JSON split manifest as a uint8 array inside every inference NPZ (ICL has no added task payload), for the FV, shared-mean and rank-one controls alike. Recompute actual NPZ byte counts for existing amended-development artifacts without recomputing quality metrics. The source split, model, prompts, learning, controls, optimizer, gates and data remain unchanged. Treat the resulting serialization sizes as authoritative for the byte gate.

## Data handling

Fresh seeds remain unopened. Preserve all prior outputs. Updated amended-development NPZs and metrics are written in place; initial unseeded development outputs remain in `results/pre_amendment_1/`.
