# Mirror functional capacity studies (A1–A6)

**Interpretation as of 2026-11-15:** This exploratory sequence has NOT established a Mirror-specific rate–quality advantage over carefully controlled conditional Dense, FiLM, output-basis, or low-rank models. Distinguish synthetic evidence from actual Transformer compression. Source-code implementations of the models evolve between phases, so do not pool the results as one preregistered trial.

| Study | Purpose | Result and caveat |
|---|---|---|
| A1 (2026-11-14) | Distill 8/16 functions under shear-Mirror, FiLM, independent teacher families; capacity sweep | Matching the teacher's parametric family helps; shear Mirror good on shear teachers, FiLM good on FiLM teachers. Circular teacher matching is not a general Mirror advantage. |
| A2 (2026-11-14) | External-teacher input rotations and ordinary-context MLPs; inverse-removal ablation | Pre-nonlinearity modulation sometimes better than full shear Mirror. Across generic-context MLP teachers, strong controls outperform. |
| A3 (2026-11-14) | Explicit input-rotation adapter and matched-byte controls | Simple input-coordinate adapter dominates on rotation-teacher task; full Mirror not needed. |
| A4/A4b (2026-11-14) | Continual learning of old/new tasks with code-only freeze, replay, rank-matched Mirror4 | Mirrored structure does not reliably beat FiLM / shared output-basis in old/new task Pareto quality, even after increasing code from 1 to 4. A4b is exploratory follow-up. |
| **[A5 (2026-11-15)](A5/README.md)** | Fixed 1500/3000B budget, 6 fresh worlds × 2 seeds × two **non-Mirror** teacher families, six methods | **288 models; Mirror4 loses to dev-selected best non-Mirror control on IID/OOD in 24/24 world × condition comparisons; zero >=10% capacity wins**. Independent teacher quality threshold was unmet by all methods, so some rate comparisons are censored. |

| **[A6 (2026-11-15)](A6/README.md)** | 1,797 real handwritten digit images; 16 known binary tasks, two mask/xor label families, 6 mapping/split worlds, two seeds, two FP32-byte caps, seven methods | **336 models. Mirror4 IID BCE worse than the val-selected non-Mirror control in all 24 world × family × cap comparisons.** Mask tasks favor shared MLP with task-specific output heads; XOR/ink quality target censored for all methods. No general storage advantage. |\n\n## Reproducibility and evidence boundary

- [A6 protocol](A6/PROTOCOL.md), [code](A6/a6.py), [locked training](A6/train.py), [auditing](A6/audit.py), [unit tests](A6/test_a6.py), [report](A6/REPORT_JA.md) and summary/world-level tables are tracked. Source digits are bundled with scikit-learn; complete 336 raw weight files are distributed separately from Git history.\n- [A5 protocol](A5/PROTOCOL.md), [full code](A5/a5.py), [training](A5/train.py), [auditing](A5/audit.py), [tests](A5/test_a5.py), and [results](A5/REPORT_JA.md) are part of this Git history. Small aggregated CSVs are checked into `A5/results/`.
- A1–A4 full synthetic evidence archives are retained in the earlier conversation's downloadable artifacts but are **not** silently reclassified as public Git-hosted checkpoints. The main conclusions above are from those experiment reports; the bundles can be independently introduced under explicit provenance later.
- All recorded model-file sizes exclude common decoder source code and execution frameworks. They are serialized learned weights, **not** a universal description-length lower bound or GPU runtime advantage.
- Selection and additional follow-ups based on earlier observed outcomes mean that cross-study comparisons are exploratory. Never treat increasing experiment count as increasing independent confirmation.

## Next research gate

Choose non-Mirror teacher families with *known and unknown task addresses*, compare conditional function factorization under matched stored-byte **and training/inference compute** frontiers, and require a fresh-world replication with a frozen protocol. If no low-description structured Mirror variant improves on strong controls, favor ordinary shared-basis methods rather than expanding Mirror count.
