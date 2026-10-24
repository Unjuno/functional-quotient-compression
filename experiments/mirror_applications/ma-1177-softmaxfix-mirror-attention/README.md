# MA-1177: Softmax Fix × Mirror Attention — numerical and learning audit

**Question:** does Barron et al. (2026), *A Fix for Softmax and Cross-Entropy* alter outcomes for Mirror-generated virtual Attention heads when softmax saturates or a student exactly matches a teacher?

- Mathematical insertion: share an 8×8 query transformation across three rule-dependent virtual Attention experts. Mirror uses one learned 8-vector Householder reflection per rule. Matched FiLM uses one learned 8-vector gain per rule (both total 88 learned parameters); an independent 3×8×8 query projector (192 learned scalars) supplies an upper comparator.
- Data: fixed synthetic teacher attention with three rules, 12 memory keys per query. Training hard-target classification uses teacher top-1, with high-logit-scale 12.0 and low-logit-scale 2.5. Teacher world is either Householder-aligned or unrelated per-rule orthogonal.
- Numerical experiment: precision cliffs and equal-logit soft-target zero-gradient with fp32/bf16/fp64.
- Learning experiment: 2 worlds × 3 independent seeds × 3 methods × 2 logit scales × 2 hard-CE implementations = **72 models**; AdamW, 550 steps; dev monitored.
- Distillation: 2 worlds × 3 seeds × 2 architectures × 2 soft-target CE implementations × 2 eps values = **48 models**; identical teacher/student initial weights; tests numerical drift rather than representation learning.
- Fresh and OOD generated only after 120 student/teacher checkpoints hash-frozen. Compare exact same seed, minibatch order, and physical coefficients for loss variants.
- No new causal LM; no Mirror-independent benefit is claimed without beating matched FiLM on normalized rate–quality frontier.

Preregistered protocol: `PROTOCOL.json` committed **before** training. Sources: https://arxiv.org/abs/2610.31642 and https://jonbarron.info/softmaxfix/ . The prior normal-scale null result (72 model replays) remains historical evidence.

## Completed evidence

**120 trained models, 120/120 frozen checkpoint replays, 8 numeric tests PASS.**
Read `REPORT_JA.md` for exact Fresh FP64 comparisons and posthoc near-copy DEV-only notes.
Run tests: `python -m unittest discover -s tests -p 'test_*.py' -v`.
Run reproducibility checks after unpacking the complete ZIP: `python source/verify.py`.
Main train protocol: `python source/run.py train` then `python source/run.py evaluate`, only in a clean directory (overwrites are deliberately prohibited).
Results: `results/RESULTS_CORE.csv`, `results/ANALYSIS.json`, `results/NUMERIC_AUDIT.json`, `results/VERIFICATION.json`.
Previous historical CE/softmax-sensitive experiments are NOT retroactively reclassified.
