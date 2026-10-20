# MA-1172 — Mirror SAE encoder rank and nonlinear-control specificity

**2026-10-20 JST — completed. Mechanism rank release PASS; Mirror-specific feature separation FAIL.**

## Mirror insertion
A learned 32-scalar core, eight fixed signed-permutation pages, 64×6 learned address codes and 64 learned gains generate **64 logical 32-dimensional sparse encoder detector vectors**. Mirror applies a fixed Householder → tanh → inverse Householder weight map. The sparse code is TopK3, with three residual-refinement passes. **All methods train an independent 64×32 decoder dictionary:** neither the decoder nor the full SAE dictionary has been compressed to 32 scalars.

## Central issue
MA-1171 found an encoder numerical rank increase from 8 (linear-address) to 32 (nonlinear Mirror). MA-1172 explicitly asks whether the increase requires the Mirror/Householder structure. Strongest matched controls are **direct elementwise tanh without Householder** and a cubic nonlinearity at exactly the same 2,624 learned coefficients, plus identical-address affine and full dense one-pass SAE.

## Protocol and data
The branch's `PROTOCOL.json` was saved **before training**. Two independent synthetic worlds, three new seeds [101,103,107], five models, 30 × 500 CPU FP32 optimizer updates. One teacher has 64 independent ground-truth 32D features. The other uses an independent nonlinear orbit generator with 75% feature-pair cooccurrence in train/dev and broken cooccurrence for fresh OOD. The true factors were withheld from training. Checkpoints were selected by dev NMSE before any fresh evaluation.

## Main evidence
- **Numerical rank at 0.001 relative SVD threshold:** affine 8/32, Mirror 32/32 and **ordinary direct tanh 32/32**, consistently 6/6. FP64 recomputation from FP32-trained checkpoints confirms.
- **Energy effective rank median:** affine 4.99, Mirror 5.40, direct tanh 5.90, dense SAE 26.76. Mirror's top eight singular directions carry ~98% of energy.
- **Fresh OOD true-factor support F1 median:** affine 0.1034, Mirror 0.1141, direct tanh 0.1073, dense 0.3819. Mirror paired wins over affine 4/6, direct tanh 2/6; frozen joint +0.03 advantage gate **0/6 FAIL**.
- **CPU batch1 median:** affine 96.3 μs, direct tanh 105.0 μs, Mirror 111.7 μs, dense 29.5 μs. Host clock unlocked; no GPU test.

The rank increase is **not Mirror-specific**: a generic low-rank matrix transformed by ordinary elementwise tanh also reaches numerical rank 32, without any Householder. Numerical full rank does not imply useful functional capacity, interpretability, or low-rate compression.

## Reproduction and status
- `source/run.py`, `tests/test_sae.py`, `RESULTS_CORE.csv`, `VERIFICATION.json`, and `REPORT_JA.md` are committed in this dedicated research branch.
- `results/FP64_RECOMPUTE.csv`, `results/COUNTERFACTUAL_RANK.csv`, `results/COUNTEREXAMPLE.json`, and the complete dev/fresh/paired reports distinguish mechanism from quality.
- All 30 binary inference checkpoint packets plus source and SHA-256 manifest are available in the separately linked reproducibility ZIP.
- Five unit tests PASS, 30/30 checkpoint replay PASS. MA-1171 FAIL unchanged. No main merge without explicit review.
