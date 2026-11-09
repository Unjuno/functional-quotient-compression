# MA-762 — Shared local-rule basis with rule Views

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On 32 heldout scalar-field rules, the rank-3 basis bank has heldout update nRMSE `7.73e-8` and 12-step rollout nRMSE `2.98e-7`, with max state-norm ratio `1.0000004`. Its complete serialized payload is 1,960 B vs 3,786 B for 96 independent rules (48.2% savings). It uses 21,233,664 MACs vs 7,077,888 for independent convolution (3×) and evaluation wall time is about 113 ms vs 48 ms. Native rank-3 factorization and Mirror have identical payload hash, bytes, quality and MACs. The one-rule and scalar-gain FiLM controls have poor heldout errors.

## Interpretation

A shared low-rank local-kernel basis can encode many aligned rule dynamics with strong fidelity and materially fewer serialized bytes, while applying all three basis kernels each step triples compute. Native factorization fully explains the Mirror result. This is a useful storage/compute Pareto signal for an explicitly low-rank synthetic rule family, not Mirror-specific value.

## Hypothesis result

**FAIL** for Mirror-specific value and the strict storage/compute Pareto gate. Fresh/audit was not opened after the exact native factorization alias. The two development amendments and excluded initial outputs are preserved.

## Frozen protocol / scope

PA195 motivates repeated application of one local update rule. Here, 96 1-channel 3×3 convolution kernels are generated from a shared rank-3 basis. The basis is estimated by SVD on 64 training rules; 32 rule codes are fit from eight update-demonstration maps per heldout task. We compare independent rules, a hard-tied mean rule, per-task scalar gain, native rank-3 basis and Mirror. Rollouts use 12 recurrent tanh-convolution steps over 16×16 maps. Complete NPZ bytes, one-step and rollout error, state norm, MACs and wall time are reported.

The family is aligned to rank 3 by construction. It does not evaluate a trained Growing NCA, pattern growth or regeneration, multi-channel morphogenesis, biological tasks or hardware runtime. No capacity claim is made.

## H / T / D / C / U

- **H:** Small per-rule coordinates recover multiple stable cellular dynamics from one shared physical basis with lower complete bytes than independent local kernels.
- **T:** Seed 76201; 96 rules (64 train / 32 heldout); rank-3 3×3 scalar kernels; 8 calibration maps/task; 12-step rollouts; five controls; zero optimizer updates.
- **D:** **FAIL** for Mirror specificity. Payload falls 48.2% (1,960 B vs 3,786 B), update/rollout error is near zero, but compute is 3× and native basis exactly aliases Mirror.
- **C:** The teachers are generated from the same rank-3 basis; standard SVD factorization is sufficient.
- **U:** Learned rule discovery on NCA tasks, nonlinear multi-channel cell behavior, emergent morphology, longer/stochastic rollouts and accelerator cost.
