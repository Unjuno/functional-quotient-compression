# MA-379 / MA-381 adapter-composition Mirror family diagnostic

Date: 2026-10-09
Scope: shared-basis Mirror compression of small source adapter/LoRA banks before a target-side fusion or composition step.

## Facts

| Experiment | Bank / target composition | Mirror total bytes | Independent total bytes | Ratio | Relevant control |
|---|---|---:|---:|---:|---|
| MA-379 | 8 rank-4 adapters; 4 input-conditioned fusion routers | 4,634 / 4,642 B | 7,617 / 7,607 B | 60.8% / 61.0% | generic coefficients matched target quality at 4,977 / 4,970 B; scalar gates were much worse |
| MA-381 | 8 rank-2 LoRAs; 4 signed few-shot compositions | 2,396 B in both worlds | 3,935 / 3,922 B | 60.9% / 61.1% | generic coefficients matched composition quality at 2,497 / 2,500 B; scalar gates were much worse |

Both experiments used deliberately aligned synthetic shared-basis teachers. Each Mirror result missed its frozen total-payload limit of 60%, while preserving near-independent quality. Actual serialized payload included all shared factors, Mirror coordinates, target-side router/composition state, metadata, and archive overhead.

## Interpretation

The recurring failure is not inability to express the aligned source functions. It is total-payload dilution in an eight-source bank: shared bases and target-side fusion/composition state consume a substantial fixed fraction, leaving the Mirror bank at roughly 61% rather than the registered 60%. Generic coordinates are slightly larger than Mirror at near-identical quality; scalar gates are smaller but lose substantial function quality. The total result is a narrow FAIL, not evidence that larger banks fail.

The similar ratios across two very different source ranks suggest that another small eight-source adapter-bank screen would have low information value without changing bank scale or the target-side accounting structure. Fresh worlds are sealed for both IDs under their own precommitted gates.

## Family action

Pause the shared-source-adapter compression follow-ups MA-380 (fusion over a shared adapter basis) and MA-382 (Mirror-coordinate arithmetic for LoRAHub) pending a new frozen test that scales the candidate bank (for example, N=8/16/32) and reports both source-bank-only and complete target inference payloads. The complete payload remains authoritative. Keep the existing MA-379/381 outcomes unchanged; any scaling check is a new amendment or MA with fresh task IDs and the same signed/fusion controls.

Continue MA-383 as a distinct physical object and method: a retrieved prompt pool with a Mirror prompt generator. It tests prompt-vector reconstruction/retrieval rather than compressing a source adapter bank, so the fixed adapter-composition overhead finding does not predict its outcome.
