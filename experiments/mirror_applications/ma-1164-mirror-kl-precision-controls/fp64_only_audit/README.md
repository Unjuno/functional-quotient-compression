# FP64-only paired Mirror audit

This is a **post-hoc descriptive reanalysis**, not a new training experiment. All 108 rows are from the sealed MA-1164B paired fresh runs on sklearn digits (worlds 618460–618462; seeds 41,42). Every compared trained or compiled physical coefficient is FP64.

- Same-m Mirror vs plain: identical teacher NN and 64→16→10 virtual structure; the non-Mirror plain control omits nonlinear recurrence **as well as rotation**, so Mirror-specific causation is not isolated.
- Near-m standard MLP and rank-factorized controls: train FP64 from scratch; unlike Mirror, they do not require a pretrained 1,210-coefficient teacher.
- Excluded from the primary comparison: FP16/FP32 storage-only variants; these can be examined on a separate axis after controlling physical dimension and training.
- FP64 matched results: see REPORT_JA.md and results/paired_summary.csv.

To reproduce: from this directory, run python -m pytest -q tests and python source/paired_fp64_audit.py, with pandas and pytest installed.

No claim of new significance, and existing MA-1164 outcome unchanged.
