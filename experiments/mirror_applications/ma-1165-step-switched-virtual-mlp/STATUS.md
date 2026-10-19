# MA-1165 status

- Branch: `research/ma-1165-step-switched-mirror-20261019`
- Date: 2026-10-19
- Stage: **SCREENING / FAIL (Mirror-specific matched-control gate)**.
- Parent: MA-1163/1164. Their negative decisions remain unchanged.
- Frozen plan committed **before training** as `PROTOCOL.json`, no retuning after fresh results.
- Experiment: 72/72 CPU FP32 synthetic runs, 3 seeds, loops 2/4, one 192-vector trainable shared physical object, nonlinear step-varying virtual-weight Mirror.
- Unit tests: 8/8 PASS.
- Checkpoint integrity: 72/72 SHA256 match; fresh replay: 72/72 matches.
- **Primary scientific gate:** independent teacher with T=4, relative 5% gain versus strongest simple shared control in >=2/3 paired seeds; observed 1/3 (vs gain): FAIL.
- Aligned teacher shows some switched-vs-same gains; all-task superiority is NOT demonstrated.
- **CPU-only evidence:** serialized inference state and batch-1 latency; no GPU improvement or capacity law established.
- Authoritative human-readable report: `REPORT_JA.md`. Seed-level table: `results/RESULTS_CORE.csv`; aggregation: `results/SUMMARY.csv`; runtime: `results/BENCHMARK.csv`.
- Code/test/source hashes and verification are under `source/`, `tests/`, `SOURCE_SHA256.json`, `VERIFICATION.json`.
- Large checkpoints / detailed per-seed JSON remain in conversation experiment archive rather than Git history; `source/run_experiment.py` recreates them.
- Branch scoped only, not merged to main.

## Reproduction

Requires Python3, PyTorch 2.10+ CPU, pytest; no data downloads. In this directory:

```bash
python -m pytest -q tests/test_model.py
OMP_NUM_THREADS=1 python source/run_experiment.py --out .
OMP_NUM_THREADS=1 python source/verify.py
OMP_NUM_THREADS=1 python source/benchmark.py
```

**Note:** Existing `results/RESULTS_CORE.csv` is a slim seed-level *published* table.
For a fresh full rerun, start with an **empty copy of this experiment directory's `results/` and `checkpoints/` subdirectories** (keep `PROTOCOL.json` and source files). The script writes full per-run metrics and checkpoint hashes, then the verifier checks them. If only the slim published CSV is present, the script will consider its labels complete and **will not** recreate missing checkpoints automatically; remove that CSV first. Raw full tables and checkpoints are also distributed in the chat experiment zip.
