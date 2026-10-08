# MA-932 — informational validity of Mirror stitching views

Status: SCREENING — frozen synthetic falsification protocol
Evidence lane: MECHANISM / INFORMATION / STORAGE
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Selection: Draw21, uniform over eligible P0/UNTESTED candidates; replay in `source/random_draw.json`.

## H — Hypothesis

Matching one task output across two representations does not establish that the representations retain the same information. A counterfactual task and linear information probes should expose the difference even when a shared-basis Mirror connector matches the common task.

## Mirror insertion

> **Mirror insertion:** this experiment adds a task-addressed Givens coordinate between the A/B maps of one shared low-rank representation connector, so a small address selects distinct task functions without storing one connector per task.

- Physical object: rank-2 connector basis over 8D representations.
- View coordinate: one learned rank-2 rotation per task address.
- Closest prior art: PA241, stitching validity counterexample.
- Controls: hard tying, diagonal FiLM, independent full linear connectors, unrelated structured-noise representation.

## T — Frozen task

The full encoder exposes latent features `u` and `v`; the task0-only encoder preserves `u` while replacing `v` with independent nuisance; the unrelated-noise encoder carries neither. Task 0 predicts `u`, task 1 predicts `v`, and task 2 is an independent target used only as a negative control. Seeds, updates, data splits, byte accounting, and audit rule are in `PROTOCOL.json`. The full representation is intentionally aligned with low-rank Views, so the experiment tests an information-validity diagnostic rather than natural model stitching.

## Evidence

After the frozen runs, report per-task MSE/R2, task0 cross-representation output match, linear-probe recovery of `u`/`v`, serialized bytes, and compute. A good task0 match alongside lost task1/probe performance demonstrates why output matching alone cannot establish information equivalence. Mirror success on this aligned synthetic teacher will not establish general stitching quality.

## C — Strongest counter-hypothesis

The selected observable tasks may understate semantic information, while direct probes may expose only linearly decodable content. This screen can falsify a claim of informational equivalence under its declared feature family, but cannot certify semantic equivalence.

## U — Boundaries

Only synthetic linear representation evidence is in scope. No pretrained networks, natural language, or broad stitching claim is made. Fixed-update quality is not capacity evidence. Audit data remains unopened unless the Mirror function gates pass in every fresh seed.
