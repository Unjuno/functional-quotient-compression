# MA-286 — Cheap-LoRA Mirror column-subspace views

## H — Hypothesis

Shared column bases with compact task views may improve the quality/byte frontier over fixed Cheap-LoRA columns for aligned tasks; independent subspaces test the private-factor boundary.

## T — Planned experiment

Frozen 32×32 map, rank 4, eight synthetic tasks; compare no adapter, fixed random cLA columns, per-task Cheap-LoRA, shared generic coefficients, Mirror column addresses and full rank-4 LoRA. Fresh worlds 28610–28612 × seeds 0–2 remain locked.

## D — Pending

## C — Strongest counter-hypothesis

Generic coefficients over the same shared basis may be identical to Mirror addresses, and fixed Cheap-LoRA may already capture the task subspace.

## U — Unknown

Fresh held-out quality, actual bytes, fit time and interference.
