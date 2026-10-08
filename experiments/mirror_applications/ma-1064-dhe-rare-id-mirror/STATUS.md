# MA-1064 status

Status: SCREENING — data and protocol are frozen; numerical audit is sealed.

## H

A per-rare-item Mirror angle plus rank-1 private residual improves tail AUC beyond an equal-budget ordinary rank-2 residual while preserving overall quality and beating full-table storage and DHE latency gates.

## T

MovieLens-100K chronological train/dev/audit split; three model seeds; rare IDs fixed from train-only counts. Compare full table, DHE-style hash decoder, Mirror-only, rank-2 additive residual, Mirror+rank-1 private residual and rare full residual. No audit examples are read until development checkpoints are selected.

## D

Pending.

## C

Any tail gains are explained by the added ordinary residual degrees of freedom or by frequency-based allocation, while the Mirror angle adds bytes and per-request math.

## U

Generalization beyond MovieLens-100K and optimized production DHE/TT-Rec kernels are outside this screen.
