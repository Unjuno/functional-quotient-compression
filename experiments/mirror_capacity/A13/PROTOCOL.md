# A13 — Evolution strategies on frozen A12 shared scaffolds (exploratory, frozen before run)

Date: 2026-11-15. This is a *post-A12* exploratory experiment, NOT an independent confirmatory experiment or a new Mirror-specific compression claim.

## Question
Is poor held-out task-code adaptation (A12) primarily a local gradient-optimizer problem? A black-box evolution strategy should find a better 2D task code on the **same exact frozen learned shared weights** if low-code optimization is the bottleneck. Compare a random-search baseline and full-support Adam and report support/IID/OOD gaps.

## Data and frozen parents
A12 2400-step meta checkpoints from the 2026-11-15 A12 evidence ZIP, one per family {tent,sine} × world {1200,1201,1202} × seed {0,1} × architecture {mirror,shift,film} = **36 frozen shared models**. SHA-256 must match archived A12 train.lock.json. Every model has 16 seen training tasks + 8 unseen tasks and two real-valued codes per task; all learned FP32 weights including unused old codes and new codes are counted in the serialized file. The A11 generator is imported unchanged. Support is 32 examples per new task. No true teacher parameter enters the ES/Adam initialization or optimization. Test labels are not given to any search procedure.

## Competitors and same budget
For each of 36 parents, optimize all 8 unseen codes with:
1. **ES**: bounded diagonal-Gaussian (mu,lambda)-evolution strategy; per-task independent populations, P=32 and G=30 generations, 8 elites, 0.55 mean step, initial std=1.35, per-coordinate elite std with floor 0.08 and ceiling 2.50, candidate bounds [-4,4]; retain best evaluated support-loss code. The candidate in a population uses mean + z*sigma; all random streams derived from the frozen world/seed/method/task ID. Record generator means, std, best candidate and generation indices.
2. **Random**: 960 independent uniform[-4,4]^2 draws/task, with the exact same 32 labeled support examples for evaluation and the same best-support selection. Initial starting code evaluated as an extra shared common baseline; if the start wins, retain it for all methods.
3. **Adam**: adapt all 8 task codes jointly from *exactly the same initial code* using **960 full-support (8×32) optimization steps**, AdamW lr=0.01, weight_decay=0, shared weights frozen; no IID/OOD or meta-dev selection. Includes 960 full-support forward evaluations plus backwards (not FLOP-matched to ES).
ES/Random each evaluate 960 candidate code vectors * 32 support items per task (245,760 candidate-example forwards per model across 8 tasks). Adam uses 960 steps×(8×32) same number of support examples forward, plus backward and Adam math. No equal-runtime, equal-FLOP, or 960-forward-evaluation-only claims.

## Locked split and evaluation
Save 3 complete FP32 serialized models per parent, meta parent hashes, exact code policy tables, and SHA-256 ledger. All **108 final model weights** plus protocol and source hashes must be frozen before computing new-task IID/OOD and seen-task dev errors. Compare (a) new support MSE, (b) held-out IID/OOD NRMSE, (c) seen-task dev NRMSE, (d) stored bytes, (e) per-process CPU seconds and support sample forward counts, (f) code norm / clipping frequency / covariance / ES evolutionary trajectory. Keep all outcomes. No selection on test.

## Decision gates
- If ES decreases support MSE by ≥20% over Adam on ≥2/3 worlds within a family/method, call "ES support optimizer advantage" only. If same IID/OOD NRMSE advantage ≥0.03 without OOD regression on ≥2/3 worlds, call "held-out adaptation advantage". Comparison must use world-aggregated values over two seeds and eight tasks. Otherwise FAIL or UNCERTAIN.
- If no method obtains support RMSE or IID NRMSE ≤0.20 in at least two worlds, do not infer a global representational impossibility. Mark "fixed parent+2D code bottleneck under tested search domain"; a 2D bounded global search is not proof.
- Report whether ES gives a Mirror-specific benefit beyond Shift and FiLM on the same archived 1024B cap; do not derive a capacity win when none hits the joint seen/IID/OOD quality gate.

## Scope/uncertainty
Three worlds per family from the *same A11 synthetic generator*, each 2 seeds. A12 weights chosen after observing earlier experiments. Gaussian search population and hyperparameters chosen before running A13, not independently tuned. Support overfitting can worsen OOD. The method family contains no non-Mirror sine analytic prior; fitting a true sinusoid from 32 samples is a separate privileged reference, not a fair model compression baseline. CPU FP64 objective evaluator for candidate comparison, final FP32 serialization and PyTorch evaluation; numerical equivalence unit tested. Program bytes omitted from weight-file metric. No GPU/runtime generalization.
