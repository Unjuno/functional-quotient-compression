# MA-1183 — Final status, 2026-10-30

**FAIL (0/5)** on preregistered partially aligned ReLU world; no confirmatory 10-seed extension. Main 210 models, 210/210 checkpoint and Fresh/OOD replay PASS, 7/7 mechanism unit tests PASS, original training/protocol/source hashes unchanged. Separate posthoc exploratory gamma sweep 140 models and 140/140 Fresh/OOD replay PASS; this sweep is NOT preregistered confirmation and does NOT alter original FAIL.

The 5-seed engineered exactly-Mirror-aligned positive control shows Mirror beats FiLM and near-parameter independent model 5/5 at 10% threshold for ReLU and SiLU, but coupled-rank1 (strict superset) remains equivalent. Nonaligned worlds favor simpler controls. No Router, MoE, Attention, Loop, real LM, GPU, or untrained new-model generalization claim.

Main report `REPORT_JA.md`, full results and seal under `results/`, all checkpoints and exploratory results in downloadable ZIP.
