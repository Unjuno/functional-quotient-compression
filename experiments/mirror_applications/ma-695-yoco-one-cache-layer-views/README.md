# MA-695 — YOCO one-cache with layer-specific reads

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `2fc822ad`

## Hypothesis

H: A YOCO-style single global memory cache can support distinct layer reads through compact shared-basis View codes, until private residuals erase the storage advantage.

## Mirror insertion

> **Mirror insertion:** adds layer-indexed coordinates to the shared cache read transform, recovering layer-specific attention behavior without storing one K/V cache per layer.

- Native baseline: one global cache, identical read in every layer.
- Independent control: separate full K/V maps/materialized memories per layer.
- Cheapest native equivalent: shared-basis/PCA maps with direct layer coefficients, measured as an exact control.

## Prior-art delta

PA162/PA08 establish one-cache cross-layer sharing. MA-245 already showed an aligned MLKV/Mirror cache-view feasibility case (near-zero synthetic output error, half cache bytes vs two-group MLKV, but only 11.3% serialized model savings and CPU slowdown). This candidate tests four layer roles and controlled heterogeneity against hard sharing and native basis controls.

## Gates

See frozen [PROTOCOL.json](PROTOCOL.json). The task is scoped to output error, complete bytes, and read compute, not language-model quality.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic FP32 cross-layer attention only.
