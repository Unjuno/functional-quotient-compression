# Phase II — Shared-backbone conditional models

**Active, unfinished research.**

## Current canonical state — 2026-10-07

Start with:

- [Current state](CURRENT_STATE_2026-10-07.md)
- [Current architecture — Sparse Compositional Shared-Rule Transformer](ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md)
- [Experiment registry](EXPERIMENT_REGISTRY.md)
- [SRM001 current synthetic evidence](SRM001_SHARED_RULE_MOE.md)
- [Active roadmap](../../roadmap/PHASE2_SPARSE_RULE_ROADMAP.md)

## Current architecture in one sentence

> Keep the Transformer backbone canonical and shared; put only specialist residual behavior into a sparse rule bank; select **multiple** reusable rules per token/context to construct a virtual expert.

Mirror is a candidate rule-atom parameterization, not the default transformation of the whole model.

## What changed from earlier Phase II

Earlier work explored:
- router-based global Mirror states;
- deterministic router-free views;
- token/layer/sensor coordinates;
- rotation/gain/stretch/shear families;
- reachability-guided candidate directions.

Those experiments remain important because they rejected several simpler explanations:
- more Mirror states alone did not yield a stable capacity gain;
- narrow rotation/gain families were not enough;
- a whole shared core can lose transfer when condition identity is injected too deeply;
- ES did not provide a free continuous-parameter optimization advantage;
- local reachability is not sufficient evidence of task utility.

The current SRM direction therefore localizes specialization to a sparse FFN residual instead of Mirrorizing the full computation path.

## Evidence map

| Line | Role | Status |
|---|---|---|
| MN007–MN010 | factor reuse / global Mirror exploration | historical; count-only weak, breadth mattered |
| MT / RF | routing and deterministic-view diagnostics | historical; no stable capacity result |
| Reachability | geometry analysis | analytical tool, not capacity proof |
| MS005–MS006 | ES and Backprop | ES rejected for continuous shared parameters |
| MS008–MS014 | sensor/world decomposition | supports shared-core isolation + residual specialization |
| SRM001 | sparse shared-rule / Mirror-MoE | **active strongest synthetic signal** |

See [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) for navigation.

## Historical canonical records

These are preserved for provenance and should not be mistaken for the current design:

- [Research state through MN010](RESEARCH_STATE_THROUGH_MN010.md)
- [Global Dense Mirror architecture](GLOBAL_DENSE_MIRROR_ARCHITECTURE.md)
- [Mirror Transformer proposal from 2026-10-06](MIRROR_TRANSFORMER_CURRENT_ARCHITECTURE.md)
- [Research state through 2026-10-06](RESEARCH_STATE_2026-10-06.md)
- [Reachability-adjusted Mirror analysis](REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md)
- [Semantic Mirror hypothesis](SEMANTIC_MIRROR_HYPOTHESIS.md)

## Evidence boundaries

Do not conflate:
- logical virtual-expert combinations with independent expert capacity;
- fixed-update learning speed with near-convergence capacity;
- robustness with capacity;
- parameter count with actual serialized bytes;
- routing consistency with end-task quality;
- synthetic causal-LM results with natural-language LLM performance.
