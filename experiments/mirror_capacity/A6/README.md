# A6: handwritten-digit multi-function capacity benchmark (2026-11-15)

**Evidence lane: small real-image dataset, known-task IDs, CPU FP32; no Mirror-specific capacity benefit established.** This is the follow-up to A1–A5, not an LLM compression or a general benchmark result.

- `PROTOCOL.md`: frozen before full training; distinguishes discovery, controls, and quality thresholds.
- `a6.py`: 7 conditional student architectures, deterministic stratified data/task generation, exact FP32 serializer.
- `train.py`: training and weight freeze; never reads IID/OOD labels for model selection.
- `audit.py`: locks/checks weights and code hashes before computing all held-out scores.
- `replay.py`: independently retrains 14 cases and checks byte identity.
- `test_a6.py`: unit tests for shear equivalence, data independence, bytes, checksums, and task-conditioned behavior.
- `results/`: repository-safe aggregate reports, model-level train manifest, and world-level CSV; full numeric model weights are in a separate conversation ZIP rather than ordinary Git history.
- `REPORT_JA.md`: Japanese technical report with math, interpretation, negative outcomes and reproduction command.

## Run

Dependencies: Python 3.13.5, numpy 2.3.5, scipy, torch 2.10.0+cpu, sklearn 1.8.0, pandas, pytest. Linux x86-64 CPU, no GPU required. Different library/CPU configurations need not reproduce exact FP32 bytes.

```bash
cd experiments/mirror_capacity/A6
python -m pytest -q test_a6.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --out /tmp/a6-fresh --worlds 600 601 602 603 604 605 --families mask xorink --seeds 0 1 --caps 6000 12000 --steps 700 --jobs 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python audit.py --out /tmp/a6-fresh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python replay.py --out /tmp/a6-fresh
```

The output directory must not already exist. `train.py` locks the model files before `audit.py` can read any test target. Auditing is immutable (refuses overwrite). `replay.py` re-trains 14 fixed cases and requires byte-identical weights.

## Current headline

Mirror4 lost to the validation-selected strongest non-Mirror model in IID BCE in all six mapping/split worlds for each of the two task families and both byte caps (0/6 wins in four cells). The class-membership tasks were best solved by a shared MLP with per-task output heads. The XOR/ink tasks were censored at the preregistered quality threshold for all methods. Evidence does **not** demonstrate 10% stored-byte savings at equal quality. All mapping worlds reuse the same underlying digit image dataset.

See the full report for confidence intervals, raw world means, limits and comparisons.
