# MA-1018 — quantum gate-group factorized Mirror coordinates

Status: SCREENING; draw and protocol are frozen before any fidelity score.
Branch: `research/ma-1018-quantum-gate-mirror-20261008`.
Baseline: `feb56df43d45068ad957bc028a7837cbd61d006b`.
Draw 8 selected MA-1018 from 560 eligible P0 rows. Seed/index/pool SHA and exact registry row are in `source/random_draw.json`; the ordered pool is `source/selection_pool.csv`.

## Prior-art delta

PA308 already establishes data re-uploading; it is not claimed as Mirror. PA309 TensorHyper-VQC is a direct compact parameter-generation control. This screen compares a task × layer × qubit × axis CP Mirror coordinate with shared angle low-rank and TT-SVD controls, plus independent full angle vectors.

## Frozen mechanism screen

Exact CPU statevector simulation of 4-qubit, 6-layer circuits with ordered RX/RY/RZ rotations and nearest-neighbor CNOT chains. Each task has 72 rotation angles; all task methods use the same 90-gate skeleton and logical depth 36. Synthetic task tensors vary across three declared heterogeneity levels, six basis-training task identities and two held-out development tasks. The held-out Mirror code is fit from known target angles, so this is an **oracle representability/compression screen**, not task-data learning. Process fidelity, complete serialized library bytes, private residual bytes, simulator operations and wall time are recorded.

No QPU, shots, hardware compilation or real quantum task performance is claimed. See `PROTOCOL.json` for gates and the fresh-opening rule.
