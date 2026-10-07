# MA-063 — causal Mirror-MQA

Status: SCREENING. Branch `research/ma-063-causal-mirror-mqa-20261007`.

## H

A vectorized head view over one causal MQA cache may recover multiple logical K/V roles without expanding persistent KV state. MQA/GQA quality and cache traffic are the primary controls.

## T

Synthetic causal attention, 16D, four query heads of dimension 4, sequence length 16. Compare full MHA, GQA-2, MQA, rank-1 K/V residual, and vectorized Mirror-MQA. Teacher modes use one shared causal KV pair plus head rotations or independent KV pairs.

## Decision

FACT: development/fresh pending. INTERPRETATION: pending. HYPOTHESIS: pending. BOUNDARY: synthetic causal cache only; no language NLL claim.
