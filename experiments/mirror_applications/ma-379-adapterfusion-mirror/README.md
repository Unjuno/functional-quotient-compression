# MA-379 — AdapterFusion bank with source-task Mirror views

Status: PROTOCOL FROZEN; no development result has been inspected. Dedicated branch: `research/ma-379-adapterfusion-mirror-20261009`.

## H — Hypothesis

A small per-source Givens coordinate inside a shared bottleneck adapter basis can reduce the stored source adapter bank while preserving input-conditioned target fusion quality, and can improve the byte/quality point over a hard-tied adapter and scalar gate.

## T — Frozen design

PA54 establishes AdapterFusion as frozen independently trained task adapters plus a learned fusion mechanism. This experiment preserves that target-side interface and tests five source-bank variants: independent rank-4 adapters, hard tying, scalar gates, unrestricted shared-basis coefficients, and Givens views. Eight source tasks feed four input-conditioned target fusion tasks. See the pre-development frozen `PROTOCOL.json` for synthetic teacher, optimizer schedule, seed split and gates.

## Exact Mirror insertion

One six-angle SO(4) code per source adapter rotates the rank-4 bottleneck activation between a shared down projection and shared up projection. The ordinary controls use the same target fusion router. All adapter, code, router and metadata bytes are charged.

## Limitations

This is a deliberately Mirror-aligned synthetic frozen-feature screen, not a pretrained AdapterFusion/Transformer reproduction. Fresh seeds stay sealed unless the development gate passes. No capacity claim follows from a fixed update budget.
