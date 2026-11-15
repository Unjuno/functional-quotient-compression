# A11: multi-family, fine-grained byte-cap capacity screen — protocol frozen before full training

Date: 2026-11-15. **EXPLORATORY** relative to A1–A10. Previous ramp findings informed this design. No previous audit examples reused.

## Objective and research question
Assess whether task-indexed shear Mirror yields a **lower observed finite-grid stored-byte frontier** than learned direct Shift, FiLM, shared output-basis, a pre-nonlinearity-only shear control, or task-specific output heads when 16 tasks learn a common scaffold and eight unseen task IDs adapt from 32 pairs per task.

## Teacher families / no privileged teacher parameters
Three externally generated dimensionless 1D function families:
- ramp: `relu(x−a)−relu(x−b)`; b>a
- tent: `relu(x−(c−w))−2 relu(x−c)+relu(x−(c+w))`, scaled by `1/w`
- sine: `amp sin(freq*x+phase)`

Teacher-specific coefficients are used only to produce labels, never as features, initial weights, hyperparameters or architecture selectors. All models see the same meta-train and support samples within a world/family. Task ID known; unseen task **identity** not necessarily unseen function family. New independent world streams 1100 and 1101, seeds 0/1; label and input streams deterministic and disjoint.

## Fixed training protocol
- Six methods: `mirror`, `pre`, `shift`, `film`, `basis`, `multihead` inheriting A10 equation/model code (copied with attribution).
- Serialized FP32 complete-weights cap: **512, 640, 768, 1024 bytes**, including all seen/new task codes, biases, model tag 32B header and 32B SHA-256. The decoder/software is excluded.
- 2 worlds × 2 seeds × 3 teacher families × 6 methods × 4 caps = **288 frozen models**.
- Meta stage 600 AdamW steps, batch128; adapt stage 200 AdamW steps, batch128, learning rate0.01, weight decay0, gradient clip5. Each unseen task provides 32 support observations. During adaptation only the unseen task codes/heads train; all shared weights frozen.
- 1024 meta-train input positions, 256 seen-task dev positions, 256 IID held-out positions, 256 OOD held-out positions; support x drawn separately per unseen task; meta and IID x uniform [-1.5,1.5], OOD [-1.9,1.9].
- Train code **does not score dev or IID/OOD**. Write SHA-256 and complete model files, lock all 288 before audit. Audit independently reconstructs the teacher streams and scores only after verifying every recorded model and protocol/config SHA.
- Primary: three mean-seed NRMSE quantities: seen task dev, unseen IID and unseen OOD, with joint quality gate **all ≤0.20**. Secondary: raw MSE, file byte size, per-step training walltime, mean absolute error, and observed-budget pass frontier. Don't infer a minimal theoretical byte count from four caps. For models failing all caps, left/right censored (no finite frontier claim).
- For paired performance reference, choose strongest non-Mirror by *seen-task dev* (not unseen IID/OOD); do not use final audit to tune variants.
- **Exploratory PASS for a finite-grid advantage**: in a given teacher family, Mirror passes at a recorded actual byte size ≤90% of the lowest passing *non-Mirror* size in both independent worlds; other families report separately. No universal claim even if PASS: only two task worlds, shared synthetic generators.

## Controls and uncertainties
Matched **number of optimizer updates and minibatch index streams**; FLOP/MACs are not matched because width and conditional operators differ. Measure CPU wall times as descriptive only; clocks not locked. Do not convert to training-compute advantage. Report memory/software not counted separately. Six methods have different adaptation trainable scalar counts; multihead adapts more than other methods. Sample size only two independent task worlds per family and two initialization seeds: do NOT use significance testing with fabricated precision. Same teacher family is used for seen/new task IDs; no cross-family generalization measured.

## Reproduce
From `experiments/mirror_capacity/A11` with torch 2.10 CPU, numpy 2.3.5, pytest 9:
1. `python -m pytest -q test_a11.py`
2. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train.py --out RUN_DIR`
3. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit.py --run RUN_DIR`
A pre-main tiny smoke uses `python train.py --smoke --out SMOKE_DIR`, and **is excluded** from counts and comparisons.
