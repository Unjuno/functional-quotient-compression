# MA-272 — Input-centric OFTv2 Mirror views

## H — Hypothesis

Input-side orthogonal transforms should reproduce weight-centric OFT exactly while avoiding per-task transformed-weight materialization. A one-angle Mirror coordinate can compress task state only for aligned view families.

## T — Planned protocol

Frozen 32×32 linear map, eight tasks, aligned and independent orthogonal input views, 256 support and 1024 audit examples. Compare baseline, materialized OFT and activation-side OFT/Mirror. Fresh worlds 27210–27212 × seeds 0–2 remain locked.

## D — Pending

## C — Strongest counter-hypothesis

Input-centric OFT is algebraically equivalent to ordinary OFT; any speedup may come only from avoiding repeated materialization, while a scalar plane control may fully explain Mirror compression.

## U — Unknown

Forward equivalence, actual payload bytes, transform cost and batch latency.
