# SAE feature-view family diagnostic — 2026-10-09

## Scope

This diagnostic covers the static support-derived task-function-vector (FV) representation lane tested by MA-526, MA-527 and MA-528. It does not pause conditional steering, routed feature experts, transcoder dictionaries, natural SAE behavior steering, or unrelated Mirror insertion points.

## Facts

- **MA-526:** shared activation-selected SAE pool64 with OMP8 used 3,582 B incrementally but lost 1.463/1.567 heldout gold-logprob nats vs explicit FV. Global OMP8 was smaller and was competitive; full SAE deployment exceeded explicit FV deployment by 4,172,903 B.
- **MA-527:** a 16-atom shared pool plus eight Givens angles used 3,432 B and beat same-size pairwise gains by .204/.225 nats, but lost 2.333/2.607 nats to explicit FV. Its full SAE deployment exceeded explicit FV by 4,172,997 B.
- **MA-528:** residual-selected pool64 plus OMP16 beat activation-frequency pool selection by .359/.670 nats, but lost 1.237/1.157 nats to explicit FV and .099/.090 nats to global OMP16. It used 3,990 B vs 3,840 B global OMP16; standalone deployment exceeded explicit FV by 4,173,555 B.
- All three use the same pinned SAE checkpoint and small Pythia model; all fresh worlds remained sealed. MA-527 and MA-528 have deterministic replay and retained actual payloads.

## Interpretation

The repeated bottleneck is not one atom-pool heuristic. Three representations (sparse shared-pool coefficients, orthogonal shared-code Views, and residual-selected sparse behavior codes) fail to recover these FVs' causal effect. Direct global sparse coding is at least as storage-efficient as shared-pool bookkeeping and generally better in gold likelihood. When the SAE is not already a sunk deployment cost, adding it dominates the incremental code savings.

This evidence supports pausing **unchanged static task-FV-to-SAE-bank compression variants** until the target representation or atom source changes. It does not establish that SAE features are poor steering controls generally: PA102 studies different steering targets, and these experiments use relation task FVs as steering targets.

## Family redesign requirements

A future static-bank proposal must change at least one structural premise: use a task-aligned learned/transcoder dictionary, add and byte-charge private residuals, widen the basis with an explicit frontier, or switch to direct SAE behavior targets instead of MA-516 task FVs. It must retain direct global sparse coding and full-FV controls. Simply repeating pool heuristics, angle parameterizations, or coefficient quantizers on the same pinned SAE/FV task is paused.

Conditional gates and top-k feature experts are separate insertion structures. They may proceed only with their own native controls and task-appropriate outcomes; they cannot inherit a static-bank quality or capacity claim.
