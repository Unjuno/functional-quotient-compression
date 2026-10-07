# MA-041 — one QKV to multiple Mirror heads

Status: SCREENING. Branch `research/ma-041-mirror-attention-heads-20261007`.

## H

One physical QKV projection plus per-head input-space Givens coordinates may recover four useful heads from a shared-view teacher, using fewer serialized bytes than ordinary MHA. Tying, gains, and low-rank residuals are controls.

## T

A synthetic 16D full-attention block (sequence length 8, four heads of dimension 4), with shared output projection. Compare independent QKV MHA, hard tying, per-head gains, rank-1 residuals, and Mirror views. A second teacher uses independent QKV projections.

## Decision

FACT: pending development/fresh. INTERPRETATION: pending. HYPOTHESIS: pending. BOUNDARY: synthetic attention block only; no language NLL claim.
