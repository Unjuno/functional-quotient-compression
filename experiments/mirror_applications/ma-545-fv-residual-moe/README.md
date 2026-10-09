# MA-545 — Few-shot-routed function-vector experts without weight experts

Status: SCREENING  
Evidence lane: LANGUAGE/REPRESENTATION_ROUTING/CAUSAL_QUALITY/ACTUAL_BYTES  
Branch: `research/ma-545-fv-residual-moe-20261009`  
Base commit: `a37cb993d0a9bef5c7a9bb39e86abca39ed30bf6`

## H — Hypothesis

On 16 held-out Pythia-70M relation tasks, a router that reads few-shot demonstrations can select support-derived residual-stream FV experts with route accuracy >=.75 and preserve oracle-routed FV quality, while using <=.75x the complete same-router rank-one MLP-output expert bank.

## Insertion and prior art

One shared frozen Pythia model receives one support-derived additive layer-3 residual direction per task. A learned affine router sees a few-shot prompt's final query representation and selects one direction. PA99 establishes activation-space function vectors; PA100 establishes contrastive additions; PA101 studies context-dependent activation steering; PA17 is a rank-one shared-weight modulation control. MA-530 failed query-only routing (.477/.516 route accuracy), so this run specifically tests demonstration-context routing.

## Fixed protocol

`PROTOCOL.json` and `freeze.json` are frozen before implementation or model evaluation. Development seeds are 54501/54502. Fresh seeds 54511–54513 stay unopened unless both development seeds pass all preregistered gates.

## T — Execution

Pending.

## Results

Pending. Data, model and intermediate artifacts remain under `results/`; do not delete negative results.

## Decision

FACT: Pending.  
INTERPRETATION: Pending.  
HYPOTHESIS: Pending.  
BOUNDARY: One Pythia-70M checkpoint and 16 relation tasks.
