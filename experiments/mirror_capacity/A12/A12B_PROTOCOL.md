# A12b post-A12 exploratory follow-up: frozen-code dense support vs all-weight 24-task continued training

Frozen AFTER inspecting A12 preliminary outcome; this is explicitly post-discovery and is **not an independent preregistered test**. Exact parent A12 meta2400 checkpoints are reused; no alternate source checkpoints, datasets, initializations or hyperparameters may be selected using audit results.

- Families tent and sine; worlds 1200–1202, seeds 0–1, methods mirror/shift/film, stored cap 1024 B. Exactly 36 parent meta models.
- Two sibling interventions from each exact parent snapshot:
  1. `code`: shared weights and existing codes frozen, adapt only 8 new task codes using **1024 support pairs/task**, AdamW 2400 updates, batch128, lr0.01.
  2. `joint`: starting from the same parent, allow all parameters to train on old 16 tasks and new 8 tasks (new tasks each have 1024 support pairs), 2400 AdamW updates, batch128, lr0.01. Mix task IDs uniform over all 24. This is an ORACLE/DIAGNOSTIC: it changes shared weights using new tasks, so not a fair frozen-meta adaptation or capacity compression claim.
- Keep 16 seen-task training data and new-task support data disjoint from old-dev, unseen-IID and unseen-OOD. No test label used before weights/hashes locked. Compare `old_dev`, `unseen IID`, `unseen OOD` only after all 72 saved model files have been verified.
- Interpretive rule: If `code` fails and `joint` passes quality NRMSE <=0.20 on all three splits, frozen learned shared scaffold lacks useful code-only reachability under the tested optimization. If both fail, shared architecture/model capacity or optimization is still unresolved; do not assert expressibility impossibility. If code succeeds with 1024 examples while A12 n256 fails, support/code fitting was the stronger limitation. 3 worlds/family is descriptive, no valid population CI.
- Positive-control known-family oracle from A12, with 32 support examples, can identify the teacher to numerical accuracy but has privileged formula knowledge; never equate its performance with general meta-learning.
- Existing A12 main and raw parent hashes, exact FP32 byte count, method identity must match before training. Source code and evidence hashes recorded. CPU clocks not fixed, FLOPs not matched; training-time comparisons are descriptive only.
