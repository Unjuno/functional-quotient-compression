# A13b — Post-A13 exploratory optimizer baselines

2026-11-15. Designed **after observing A13 ES vs Adam results**. DO NOT merge this with the pre-run A13 confirmatory evidence; descriptive mechanism test only. The training support set and all 36 A12 frozen parent scaffolds are unchanged. No use of IID/OOD labels for optimizer choice.

## Motivation
A13 ES improves mean held-out tent/sine errors, but task-level analysis suggests Adam has severe outlier failures; median task may already fit well. Need determine whether the advantage comes specifically from evolutionary covariance adaptation versus wider coverage/multiple initializations.

## New methods
- **4-start Adam:** four independent 240-step full-support AdamW optimizations on all eight new tasks, all shared/seen weights frozen. Restart0 uses the archived original code, restarts 1–3 use independent uniform[-4,4] codes for each unseen task. LR .01, weight decay 0. For each task pick lowest support MSE among initial and the four final codes; this produces one valid 8×2 adapted code table. Total new-task support forward examples 4×240×8×32 = 245,760 per parent (same as A13 ES/random/Adam), plus backward computations.
- **Sobol candidate search:** independently scrambled Sobol low-discrepancy sequence of 1024 2D points per task, retaining first 960, linearly map [0,1] to [-4,4]^2. Choose the lowest support-MSE point or original initial code. Exactly 960 candidate evaluations ×32 support samples ×8 tasks per parent; no derivatives or held-out labels.
- Baseline comparisons retain A13 ES, A13 random, and A13 single-start Adam results frozen before this follow-up. The 36 parent byte hashes must match A13's lock. This is *post hoc* and may be sensitive to hyperparameters.

## Evaluation
Save 2 final FP32 full checkpoints per original parent (72 new models), freeze all SHA256 and protocol hashes before independent audit, then compute old-dev/IID/OOD NRMSE and support MSE. Pair by family/world/seed/method. Compare 90th percentile support loss across 8 tasks, fraction of catastrophic new-task fits, joint quality world pass counts. Do not claim matched FLOPs or time.

## Limits
Same 3 synthetic worlds/family and reused model priors as A13, not independent confirmation. Sobol low-discrepancy sampling uses a box bound not intrinsic learned prior. MultiAdam changes initial-code distribution. Findings attribute the gradient failure partly to multistart coverage only if a multiAdam improvement appears; differences of optimizer dynamics remain possible.
