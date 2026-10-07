# Mirror Application Experiment Contract

Use this contract for every MA-xxx candidate unless its report explicitly preregisters a deviation.

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

A Mirror-specific claim requires the Mirror parameterization to outperform a simpler non-Mirror shared-basis or low-rank control at comparable storage and compute.
