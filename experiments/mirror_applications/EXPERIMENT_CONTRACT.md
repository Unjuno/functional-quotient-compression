# Mirror Application Experiment Contract

Use this contract for every MA-xxx candidate unless its report explicitly preregisters a deviation.

Interpret this contract under `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md` and use `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md` for the cross-method insertion map.

## 0. Mirror parameter isolation

Every experiment must isolate the marginal contribution of the extra low-description functional parameter `m`.

Required sequence:
1. reproduce or preserve the native method `B(x; theta)`;
2. construct the minimal Mirror extension `B_M(x; theta, m)`;
3. identify the cheapest native/non-Mirror parameter that could provide similar freedom;
4. compare at relevant byte/compute/state budgets;
5. report whether `m` is replacement, complementary freedom, factorization, dynamic functional state, shared/private frontier, or no Mirror-specific value.

Generic improvement from adding parameters is not sufficient.

## A. Required baselines

At minimum:
- ordinary shared/dense baseline;
- the existing non-Mirror method the candidate is replacing;
- a byte-near simple residual/low-rank/gate control when applicable;
- an unrestricted independent-object upper control when practical.

## B. Storage

The rate is the actual serialized inference payload. Count all learned tensors, addresses, routers, indices, bases, codebooks, normalization parameters and reconstruction metadata that are not deterministically reconstructible from already-paid state.

## C. Compute

Report:
- active operations or a MAC/FLOP proxy;
- tokens/examples seen;
- optimizer updates;
- isolated wall-clock calibration;
- inference throughput when runtime is part of the hypothesis.

A fixed-update win is learning-efficiency evidence, not automatically a capacity result.

## D. Quality

Use the task's primary quality metric plus a structure-specific audit:
- expert/rule retention for MoE;
- head usefulness or ablation for attention;
- cache quality for KV;
- per-layer contribution for depth;
- joint/valid-path metrics for packet decoding;
- reconstruction distortion for quantization.

## E. Selection discipline

Development worlds/sets choose hyperparameters. Audit/fresh worlds never choose them. Negative and null results remain in the registry.

## F. Status vocabulary

UNTESTED -> SCREENING -> PROMISING / FAIL -> REPLICATED -> ADOPTED

ADOPTED requires a useful Pareto improvement over the relevant simple control, not merely a statistically or numerically detectable effect.

## G. Mirror-specific claim

A Mirror-specific claim requires the Mirror parameterization to outperform a simpler non-Mirror shared-basis or low-rank/native-parameter control at comparable storage and compute. The experiment must state the exact insertion point and marginal cost of `m`.
