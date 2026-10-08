# MA-932 — informational validity of Mirror stitching views

Status: FAIL — information diagnostic passed; Mirror compression gate failed
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

**Fact.** In all three fresh seeds the full representation with Mirror reached near-zero MSE on both tasks (largest task MSE below 1e-8). The task0-only representation retained near-zero task0 error (≤0.000056) while task1 MSE was 0.978–1.036. The full representation's linear probes recovered both features at zero measured error; task0-only recovered `u` at zero and had mean `v` probe MSE 0.998; unrelated noise had mean probe MSE 0.999 for `u` and 0.997 for `v`. Thus the output match coexisted with missing information, as predicted by PA241.

On the full representation, Mirror's payload was 2,116 B, independent two-task linear connectors 1,669 B, and FiLM 2,038 B. Mirror passed quality versus independent connectors on both tasks, but missed the ≤80% bytes gate in all three seeds and did not beat byte-matched FiLM by 2% in seed 93204. The complete Mirror function gate therefore failed and audit remained unopened. Sixty model payloads and all MSE/probe metrics replayed exactly (maximum difference 0.0). The unrelated target metric stayed near its chance MSE of 1.

**Interpretation.** The task0 match alone would have suggested equivalent representations; task1 and the probe show the information loss directly. A task-addressed Mirror view does not reconstruct a latent feature that the representation discarded. On the aligned full representation, Mirror recovered the selected functions but did not compress storage relative to independent connectors and did not reliably exceed FiLM.

**Hypothesis.** The synthetic linear task is deliberately favorable to rank-2 views and only tests linearly accessible latent information. The same diagnostic should be repeated with real stitched model representations and counterfactual semantic tasks before making any semantic-information claim.

Mean fresh training time on the full representation was 0.27 s for Mirror, 0.17 s for FiLM, and 0.11 s for independent connectors. The active multiply-add proxy was 22, 20 and 8 per example respectively. Throughput was about 2.0M, 7.4M and 10.0M examples/s.

## D — Decision

**FAIL** for the Mirror compression/superiority hypothesis. The information-validity diagnostic itself passed in all three fresh seeds.

## C — Strongest counter-hypothesis

The selected observable tasks may understate semantic information, while direct probes may expose only linearly decodable content. This screen can falsify a claim of informational equivalence under its declared feature family, but cannot certify semantic equivalence.

## U — Boundaries

Only synthetic linear representation evidence is in scope. No pretrained networks, natural language, or broad stitching claim is made. Fixed-update quality is not capacity evidence. Audit data remains unopened because Mirror function gates did not pass in every fresh seed.
