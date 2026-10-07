# MA-048 — physical-4 to logical-16 attention heads

Status: SCREENING. Branch `research/ma-048-physical4-logical16-attention-20261007`.

## H

Four physical QKV sets expanded into sixteen logical heads with per-head Mirror views can recover an aligned 16-head teacher at lower serialized payload than ordinary MHA. Hard GQA and per-head low-rank residuals are controls.

## T

Synthetic 32D full-attention block with sequence length 6, 16 logical heads of dimension 2, and four physical QKV groups. Compared full MHA, hard GQA, rank-1 generated residuals, and Mirror views. A second teacher uses sixteen independent QKV sets.

## Decision

FACT: development/fresh pending. INTERPRETATION: pending. HYPOTHESIS: pending. BOUNDARY: synthetic attention only; no language NLL claim.
