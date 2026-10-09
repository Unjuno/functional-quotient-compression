# Weight-symmetry family diagnostic after MA-332 and MA-333

Date: 2026-10-09 UTC

## Trigger

The repository stop rule pauses a family after two consecutive candidate FAILs with the same demonstrated structural cause. MA-332 and MA-333 are consecutive candidates in `Weight symmetry / group actions` and both failed claims that gauge coordinates provide new logical functions.

## Fact

- MA-332 fresh audit (three worlds, fixed ReLU MLP): hidden permutation, positive rescale with inverse outgoing compensation, and compensated Givens all reproduced one output function (NRMSE < 1e-6). Their serialized coordinate payloads added bytes. Uncompensated Givens changed outputs (NRMSE .183), but task utility was not tested.
- MA-333 fresh audit (three worlds, fixed MLPs): permutation remained invariant for ReLU, GELU, tanh and LayerNorm+ReLU. Positive rescale was exact for ReLU (NRMSE 6.66e-8); sign flip was exact for tanh (0). Other activation/transform pairs changed outputs. Exact views added 83–147 bytes.
- Both are synthetic mechanism audits, not trained task quality or deployment tests.

## Interpretation

The shared cause is structural: a function-preserving symmetry changes parameter coordinates but not the realized function, so its address cannot be counted as independent functional multiplicity. Paid addresses also increase serialized payload unless they replace already-required coordinate copies. A non-symmetry transform may change the function, but a changed output alone is not evidence of useful specialization.

## Hypothesis for redesign

Any future candidate in this family must establish a concrete consumer that requires distinct parameter representatives, or add a non-gauge coordinate whose outputs improve a held-out task metric. It must compare full duplicated representatives, canonical state plus paid address, and a cheap canonical/index control at actual payload bytes and reconstruction runtime. Until such a consumer and metric are preregistered, pause MA-334..340; do not count symmetry orbits as functional capacity.

## Queue decision

Pause the Weight symmetry / group actions family after MA-332/333. Continue with the next independent P0 family candidate MA-341 (Federated personalization).
