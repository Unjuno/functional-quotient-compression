# MA-260 — BatchEnsemble rank-one Mirror ensemble

Status: **FAIL by the preregistered fresh diversity gate**. Quality, calibration and storage gates passed in aligned worlds; unrelated maps are a clear failure boundary.

## H

For four related classifier members, one shared classifier plus one Mirror angle per member should match BatchEnsemble rank-one fast weights on NLL, calibration and useful diversity without exceeding its serialized payload. Unrelated maps define the boundary.

## T

Protocol frozen before development (`PROTOCOL.json`, base `ba345ff`). Four-class, 16-input synthetic linear classifiers; aligned teachers are rotations of one shared matrix, unrelated teachers are independently sampled. Methods: independent full members, hard-tied classifier, PA17 BatchEnsemble rank-one factors, and one Mirror angle per member. Member IDs are supplied. Two development seeds and three fresh seeds were run in both regimes. All inference archives were FP16 ZIP/NPY and evaluated after reload.

## D

**FAIL by full fresh gate.** In the three fresh aligned worlds, Mirror mean member NLL was 0.720/0.678/0.657 versus BatchEnsemble 0.747/0.745/0.667; mean ECE was 0.016/0.013/0.012 versus 0.018/0.013/0.017. Mean member accuracy was 0.708/0.729/0.738 and remained within 0.02 of independent models. Mirror used 1,159 B versus 1,536 B for BatchEnsemble (24.5% smaller). Pairwise JS diversity was 0.031/0.084/0.0068; seed 26013 missed the required 0.01 diversity floor, so the registered all-fresh gate fails. Mixture NLL was 0.752/0.793/0.661 versus BatchEnsemble 0.760/0.821/0.661. Across all 40 payloads, replayed metrics matched exactly; three tests pass.

In fresh unrelated worlds, Mirror mean member accuracy was 0.458–0.474, below BatchEnsemble at 0.633–0.665 and independent models at 0.711–0.729. Mirror did not preserve useful per-member functions when teacher maps were unrelated.

CPU throughput varied by run and method (roughly 0.5–3.3M test examples/s in this harness); no stable runtime win is claimed. Mirror adds a rotation operation and was not evaluated with fused kernels.

## C

The quality and storage improvement occurs because aligned teachers were generated from the same rotation family as the Mirror view. The one fresh world below the diversity floor shows that fitting task accuracy does not guarantee distinct ensemble members. The unrelated family shows that rank-one BatchEnsemble factors represent some task-specific variation better than a single rotation orbit.

## U

This is a small linear synthetic ensemble, not a deep neural ensemble or natural-data calibration result. There is no learned member router. The diversity miss was not tuned after fresh access. Runtime needs a fused implementation and more stable benchmark before any throughput conclusion.

## Fact / interpretation / hypothesis

- **Fact:** Mirror is smaller than BatchEnsemble and has competitive aligned quality/calibration, but one fresh aligned seed fails the diversity gate; unrelated quality collapses.
- **Interpretation:** A compact orthogonal view can recover aligned member functions, but member usefulness depends on the function family and optimization retains sufficient diversity only inconsistently here.
- **Hypothesis:** A richer per-member Mirror coordinate or a diversity regularizer may improve the frontier, but requires a new experiment and must be tested against rank-one factors.


## Existing canonical MA-260 evidence

An earlier dedicated/reconciled branch (`research/ma-260-batchensemble-reconciled-20261008`, commit `8436d73948ce6561ed1cfc96ff604af2ca4c2731`) also tested four fresh rotation-aligned worlds. It reported Mirror accuracy 0.8302/ECE 0.0226 at 890 B versus BatchEnsemble 1,226 B and independent 926 B, missing its <=25% byte-reduction gate; unrelated accuracy fell to 0.6018 versus 0.8332 controls. This supports the same scoped FAIL disposition while using a different one-layer post-fit protocol. Neither protocol is a deep BatchEnsemble or natural-data result.
