# A10 — data-learned shared scaffold, unseen-task code adaptation

Date 2026-11-15. Fixed *before* executing the A10 main study. A10 is exploratory relative to A1–A9: the ramp family and model choices are informed by A9.

## Objective
Test whether a conditional feature scaffold can be learned from labeled tasks, rather than manually provided. Train on 16 ramp functions, then freeze all shared weights and adapt eight new ramp functions solely by fitting their task codes (multihead adapts its eight new output heads). Thresholds and end positions are never passed to a model initializer, optimizer, or model-selection rule. Teacher equations generate observations only.

## Frozen design
- Six data worlds 900–905, seeds 0 and 1; 16 meta-training tasks plus eight unseen adaptation tasks, all within one synthetic ramp-function family. The withheld task IDs' labels are not accessed during meta-training.
- Six students `mirror`, `pre`, `shift`, `film`, `basis`, `multihead`, two exact FP32 model-file caps 512 / 1024 bytes; 6×2×6×2 = 144 runs.
- Each world has train x (1,024), train-task dev x (256), for each heldout task its own 32-point adaptation support, IID audit x (512) and OOD audit x (512). Train range [-1.5,1.5], IID range [-1.5,1.5], OOD range [-2,2]. All data split streams disjoint. The support values may overlap the evaluation interval but never share identical draws with audit inputs.
- Teacher f_k(x) = relu(x - left_k) - relu(x - right_k), with left and right sampled independently of model initialization, right > left. No Mirror teacher used.
- AdamW learning rate 0.01, weight decay 0, gradient clip 5, batch 128. Meta-training 1000 updates on 16 known tasks; adaptation 300 updates drawing ONLY from 32 support pairs per new task. No early stopping or dev-based hyperparameter tuning. Fixed random index streams by world and seed, common across architectures. Deterministic CPU FP32, one thread per worker, four workers.
- All learned tensors, including training-task and new-task codes/heads, charged at 4 bytes/scalar; file adds 32-byte architecture header and 32-byte SHA-256 trailer. Decoder executable, data, training optimizer, and runtime not counted. Exact same cap, not exact bytes nor FLOPs. Record per-run active parameter counts, trainable counts during adaptation, train and adapt elapsed seconds; no matched-FLOP claim.
- Freeze meta-training weights before adaptation. At adaptation only 8×2 =16 task-code scalars may change for `mirror/pre/shift/film/basis`; `multihead` adapts 8×(hidden width+1) scalars. Fit and evaluate new codes only. Save and SHA-256-lock every final model before test score computation; reserialize and recompute audits from disk.
- Primary metric: per-task-centered NRMSE on unseen task IID/OOD (lower better) plus heldout task MSE; secondary: seen-task IID error after adaptation. Quality gate is NRMSE ≤0.20 on both unseen IID and OOD, plus seen-task IID NRMSE ≤0.20. World performance is mean of two seed-level NRMSEs; a world passes when its means satisfy all three. Descriptive paired ratios and six-world counts; no significance declaration from two seeds and one synthetic family.
- PASS for a Mirror-specific capacity advantage only if Mirror beats the validation-selected strongest non-Mirror model in ≥5/6 worlds at both caps at the same quality threshold and at least 10% lower actual stored bytes when both pass. If no model reaches the quality gate in a world, label the capacity frontier censored, not a win.

## Alternatives, limitations, uncertainty
- C: success could be general 1D ramp family structure or task ID memorization; other methods have a different number of adapter parameters, especially multihead. Fixed steps do not equal FLOPs. Training with meta tasks from the same family injects a supervised family prior, but not exact teacher thresholds.
- U: worlds share a generating distribution; N=6 worlds, 2 seeds; a new random task draws from the same family but does not establish cross-family/real-world generalization; model selection, if any, must use train-task dev only. Cross-world comparison reports descriptive range, median, and Student-t interval where meaningful.
- H/T/D/C/U and all negative evidence preserved. Never claim model-specific Shannon information creation, minimum description length optimality, or universal function capacity.

## Reproduction and audit
Use `python -m pytest -q test_a10.py` followed by `python train.py --out RUN_PATH` and `python audit.py --run RUN_PATH`. The training command only serializes models, metadata and locks: it does NOT evaluate IID/OOD. Audit independently loads and checks every locked file before computing test metrics. A10 output is a new subdirectory and never overwrites earlier evidence.
