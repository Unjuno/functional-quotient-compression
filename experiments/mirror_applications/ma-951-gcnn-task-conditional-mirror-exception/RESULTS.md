# MA-951 results — G-CNN task-conditional Mirror exception

## H — Hypothesis

A compact task phase coordinate over one exact C4 orbit representation can recover task-specific decisions for group-related functions with lower serialized payload than independent heads. A non-orbit task may need a private residual. The hypothesis is falsified at this scope if phase misses the frozen quality gate or offers no actual-byte advantage, or if a simple gate matches it.

## T — Test

**Fact.** CPU synthetic mechanism screen, 3 fresh seeds (95111, 95113, 95117), 512 train and 256 held-out examples per seed, 500 Adam updates, batch 64, lr .003. One shared 2-convolution CNN evaluated at four rotations yields the exact C4 feature orbit. Tasks 0/1 are exact transformed copies; task 2 is a separate central-patch class label. Controls: independent heads, fixed phase [0,1,0], phase plus task-2 private residual, and learned per-task group-slot gate. Training saw orientations 0–2; test includes all four. Serialized payload includes model state, phase/gate/private state, and metadata. This is a synthetic mechanism screen, not natural-image evidence.

## Results

**Fact.** Means over the three seeds:

| Condition | Task 0 accuracy | Task 1 accuracy | Task 2 accuracy | Payload bytes | Train seconds | Inference examples/s | MAC proxy/example |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent heads | 1.000 | 0.827 | 1.000 | 6,425 | 9.96 | 16,501 | 663,648 |
| Fixed Mirror phase | 0.915 | 0.901 | 0.569 | 6,432 | 8.41 | 17,473 | 663,648 |
| Phase + private residual | 1.000 | 1.000 | 1.000 | 6,851 | 7.97 | 17,463 | 663,680 |
| Ordinary learned gate | 0.751 | 0.746 | 1.000 | 6,445 | 7.86 | 15,622 | 663,744 |

Per-seed rows are in `RESULTS_CORE.csv`. Phase-only missed the preregistered ≥95% native task-0/task-1 retention gate on two seeds and was 7 bytes larger than independent heads. It did not solve task 2 (mean 56.9%). The ordinary gate also missed task 0/1 quality. Adding a private residual restored all tasks to 100% on all seeds, but cost 426 bytes (6.6%) over independent heads. All conditions had exact C4 feature equivariance error 0.0 and exact serialized replay error 0.0. Group-feature MAC proxy dominated: the private variant added 32 MACs/example, gate 96.

**Interpretation.** In this construction, the chosen phase code did not yield useful task specialization at a byte saving. The non-orbit task required private state for quality, and that state erased any payload advantage. The measured compute proxy was almost unchanged; wall-clock values are short CPU runs and do not establish a stable serving-speed advantage.

## D — Decision

**FAIL (scoped synthetic screen).** Frozen quality and actual-byte gates were not met. The exact C4 representation is validated mechanically, but no useful Mirror compression or independent-capacity gain is established. Exact transformed copies are not counted as additional independent logical functions.

## C — Strongest counter-hypothesis

The failure may reflect an unfavorable fixed readout/phase parameterization and short optimization budget rather than a general limit of task-conditioned group Views. However, the learned ordinary gate also failed the group-related tasks here, while the full independent heads solve them, and no phase byte benefit appeared.

## U — Unconfirmed

No natural-image benchmark, trained G-CNN checkpoint, alternate equivariant feature types, near-convergence capacity frontier, or deployment-kernel measurement was tested. Task 0/1 are deliberately related copies, not independent tasks. This result does not generalize to G-CNNs broadly.

## Verification

The run script was syntax-checked and executed on CPU. State was serialized and reloaded for each condition; maximum output replay difference was 0.0. Exact C4 feature equivariance maximum absolute error was 0.0. Full row-level metrics and run metadata are stored in `RESULTS_CORE.csv` and `source/results.json`.
