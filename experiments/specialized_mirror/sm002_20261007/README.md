# SM002: conditional virtual-expert controls

Executed in the conversation container, not GitHub Actions. This directory preserves the original core model, data generator, serializer, development/main/time runners, post-hoc diagnostic runner, immutable pre-execution plan, selected per-world counts, and verification summary. The full conversation artifact `SM002_RESULTS_2026-10-07.zip` additionally contains all models, snapshots, tests, audit/analysis scripts, frozen configs, environment, raw logs and Japanese report. Those large artifacts are not duplicated in this Git tree.

Read [the result report](../../../docs/phase2/SM002_CONDITIONAL_CONTROLS_2026-10-07.md). This is a finite-rule MLP retention experiment, NOT a Transformer, natural-language, or token-period experiment.

## Reproduction contract

PyTorch 2.10.0+cpu, NumPy, Python 3.13, FP32 eager CPU. The experiment used one thread per worker. Development/main used three workers; the actual time-stopped panel ran sequentially after they exited. Clock was not fixed. Task routing is deterministic and only one expert/view is evaluated per input.

For a new reproduction, use a fresh experiment directory retaining PLAN.md and source/. The core entrypoints are `run_sm002.py prepare-dev`, then `dev --index 0`, `dev --index 1`, `dev --index 2`, then `freeze`, then `calibrate.py`, then `run_sm002.py main --index 0/1/2`, then `run_sm002.py time`. Run `diagnose_lr.py` only after those primary results exist and keep its outcome-selected findings separate. An existing frozen protocol is deliberately not overwritten. Calibration on different hardware can generate a different time budget; the reported experiment froze 3.5 seconds before audit worlds were used.

COUNTS.json contains exact correct-input counts, explicitly ordered worlds/seeds and denominators, for main private16/32 and timed private32. It is not a confidence interval or a large-scale benchmark. Complete private4/8 results, curve hashes and saved models remain in the full artifact.

The low-rank control trains both shared base and task-specific rank-2 residuals; it is not a frozen-base LoRA reproduction. Mirror and fixed gate are seed-defined, with their method, seed and strength paid in the inference payload. Same-width models copy identical common-only parents; byte-capped different-width models cannot have byte-identical parents.
