# A8/A8b — Rank-certified conditional function-capacity experiment

**Status (2026-11-15): completed synthetic CPU exploratory study; not a Mirror-specific capacity win.**

A8 isolates a mathematical limitation of task-independent shared representations with linear task-specific heads. For 16 shifted-hinge / interval-ramp functions, the centered output matrix has rank 16 on 17 carefully selected input points. Therefore a shared embedding width below 16 followed by task-affine heads cannot represent all these sampled functions exactly. Full mathematical proof, variable table, dimensional checks: [`THEORY_JA.md`](THEORY_JA.md).

**Crucial negative result:** A standard shear-Mirror can encode either family exactly in 220/244 B when teacher thresholds are given directly, but equally simple non-Mirror conditional shifts can do so. Random-initialized Mirror in 600-update A8 did **not** achieve a general saved-byte advantage over FiLM, shifts, input affine adapters, no-inverse shear or shared output heads. In four new-world exploratory A8b runs at 2400 updates, ramp Mirror remained below the quality gate 0/4 worlds while the selected controls passed 4/4. Expressivity and optimization must be separated.

## Reproduce

Requirements: Python 3.11+, PyTorch (tested 2.10.0+cpu), NumPy, pandas, SciPy, pytest. CPU only; run with one BLAS numerical thread per process. Run in the current folder:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q test_a8.py
python train.py --out /tmp/a8-run --jobs 4
python audit.py --out /tmp/a8-run
python replay.py --out /tmp/a8-run
python train.py --out /tmp/a8b-run --worlds 816 817 818 819 --families ramp --seeds 0 1 --caps 512 --kinds mirror pre film shift --steps 2400 --jobs 4
python audit.py --out /tmp/a8b-run
```

`train.py` refuses to reuse an existing output directory. `audit.py` refuses to evaluate before the training manifest is locked and refuses to overwrite an existing audit. `a8.py` builds the deterministic non-Mirror teachers, model families and exact learned-FP32-weight serializer. No full expert matrix per token is materialized by the Mirror forward. Checkpoints and complete per-model score tables are stored separately in the downloadable experiment ZIP, rather than bloating Git history.

## Published small evidence

- [`PROTOCOL.md`](PROTOCOL.md) — main freeze before study, H/T/D/C/U.
- [`A8B_PROTOCOL.md`](A8B_PROTOCOL.md) — after-seeing-results, explicitly exploratory new-world follow-up.
- [`THEORY_JA.md`](THEORY_JA.md) — full linear-head rank-bound and constructive Mirror/shift equivalence.
- [`REPORT_JA.md`](REPORT_JA.md) — world-level results, limitations, uncertainty and model selection details.
- [`results/summary.json`](results/summary.json), [`results/contrasts.csv`](results/contrasts.csv), [`results/frontier.csv`](results/frontier.csv), [`results/rank.csv`](results/rank.csv), [`results/strict_controls.csv`](results/strict_controls.csv), [`results/exact_witness.csv`](results/exact_witness.csv) — primary measured evidence.
- [`results_extended/summary.json`](results_extended/summary.json), [`results_extended/contrasts.csv`](results_extended/contrasts.csv) — exploratory longer-training evidence.
- [`EVIDENCE.json`](EVIDENCE.json) — SHA-256 provenance ledger and separate-archive hashes.

## Non-claims

No quality-preserving 64× transformer compression, general Mirror-native capacity multiplier, new Shannon information, unknown task-ID generalization, real-world digit performance, GPU acceleration or matched training/inference FLOP advantage follows from A8. Hinge/ramp teacher formulas are deliberately simple shift-controlled functions. Program-code description bytes are not included in model weight sizes. The exact oracle construction is not learned from data.
