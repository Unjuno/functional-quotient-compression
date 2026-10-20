# MA-1172 — Mirror SAE encoder rank and nonlinear-control specificity

**2026-10-20 JST — completed mechanism screen: partial mechanism PASS; Mirror-specific feature recovery FAIL.**

## Mirror insertion
One 32-coefficient physical core, 8 deterministic signed permutation pages, 64×6 trainable address codes and 64 gains generate 64 logical detector vectors in a sparse autoencoder. The Mirror hypothesis applies Householder–tanh–inverse-Householder **to encoder detector weights**. All detectors are trained jointly, TopK3 codes run T=3 residual-feedback passes. The 64×32 reconstruction dictionary is trained independently in every model: this is **not a full decoder compression claim**.

## What this experiment really tests
MA-1171 recorded rank 32 for Mirror and 8 for its linear counterpart. A generic elementwise nonlinear function can also break the linear rank bound. MA-1172 therefore adds **direct tanh without Householder** and **plain cubic** with *the same trainable 2,624 coefficients* and compares all against matched affine and dense SAE. A new nonlinear orbit-structured ground-truth dictionary is generated independently of student Mirror pages (no Mirror teacher shortcut). Independent seed [101, 103, 107] and world splits, sealed fresh till all 30 models train/dev-selected.

## Evidence

- `PROTOCOL.json` frozen on GitHub branch **before training**; `source/run.py` is the exact CPU experiment code.
- `RESULTS_CORE.csv` has all five models × two worlds × three seeds.
- `results/FP64_RECOMPUTE.csv` recalculates generator forward at FP64 from trained FP32 checkpoint parameters.
- `results/COUNTERFACTUAL_RANK.csv` isolates nonlinear transforms on *identical frozen Mirror parameters*, not retraining.
- `results/COUNTEREXAMPLE.json` shows a generic rank-8 matrix becomes relative numerical rank 32 under direct tanh.
- `results/DEV_RESULTS.csv` and 30 checkpoint files support replay. See `VERIFICATION.json` and `REPORT_JA.md`.

## Main result

1. Affine 8 pages have rank ≤8 algebraically. Mirror **and direct tanh** yield relative numerical rank 32 for all 6 fitted instances, both after FP64 recomputation. Cubic yields 28–32. Hence the rank result **does not establish Mirror-specific representational freedom**.
2. Mirror median energy effective rank is ~5.40, **lower** than direct tanh ~5.90, despite both having 32 numerically detectable directions. Mirror's first eight singular directions carry ~98% of detector energy.
3. Mirror OOD support micro-F1 median is ~0.114 vs affine ~0.103, direct tanh ~0.107 and dense SAE ~0.382. *Paired* Mirror gains vs affine occur in 4/6, and vs direct tanh in only 2/6. Frozen Mirror-specific quality gate: **0/6 PASS ⇒ FAIL**.
4. This does not show any real LLM interpretability improvement; the tested teacher factors are known synthetic vectors only.

**Do not merge into main without explicit review.** Past MA-1171 FAIL remains unchanged. This repository records the negative result to avoid repeated expensive searches of a generic nonlinearity alias.
