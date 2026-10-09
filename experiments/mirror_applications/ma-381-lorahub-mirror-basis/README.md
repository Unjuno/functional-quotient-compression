# MA-381 — LoRAHub over a Mirror-compressed candidate basis

Status: protocol frozen before development. Dedicated branch: `research/ma-381-lorahub-mirror-basis-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens angle `m_i` inside each shared rank-2 LoRA basis so that eight candidate update functions can vary by source task without storing eight independent pairs of LoRA factors.

## H — Hypothesis

A small per-candidate Givens coordinate can preserve few-shot LoRAHub composition quality while using at most 60% of the independent candidate-bank inference payload; scalar gates should be insufficient when source updates differ by rotation.

## T — Frozen design

PA55/LoraHub freezes candidate LoRA modules and fits positive/negative scalar coefficients on few-shot target support data. Here the target-side signed coefficient fit is identical across methods; only source candidate-bank storage/function differs. Eight rank-2 source modules feed four three-way signed target compositions. See the pre-development frozen `PROTOCOL.json` for optimizer schedule, gauge audit, seeds, storage accounting and gates.

Compare independent LoRAHub candidates, hard tying, scalar source gates, unrestricted 2×2 shared-basis coefficients, and per-candidate Givens views. Fresh worlds stay sealed until every development gate passes.

## Prior-art boundary

PA55 establishes signed scalar composition as an existing non-Mirror baseline. This screen asks whether the candidate LoRA bank itself can be compressed before composition. Source updates are deliberately generated on the same SO(2) orbit, so any result is an aligned feasibility bound, not evidence about arbitrary learned LoRA banks or natural language.
