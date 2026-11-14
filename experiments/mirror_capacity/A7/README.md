# A7 — Spatial AND/XOR conditional-capacity experiment

**Status: measured, exploratory, negative for Mirror-specific rate-quality advantage.**

A7 tests whether task-dependent hidden nonlinearities outperform shared feature readout when each of 16 known tasks is a two-feature image predicate, without Mirror teachers. It reuses the **exact A6 model/serializer** from ../A6/a6.py with a new train-calibrated digit-task generator.

## Files

- `PROTOCOL.md` — frozen before full training, SHA256 f593518140d6f634a83bf7b15a8dba7920151c9b99a4fd6f8f2110296607bb76.
- `a7.py` — spatial projection task and train/dev/test split generator.
- `train.py` — 336-model deterministic training and whole-file lock, without reading audit metrics.
- `audit.py` — verifies locked weights, scores val/IID/OOD and selects controls from val only.
- `test_a7.py` — six tests.
- `replay.py` — independent-process byte-identical replay of selected checkpoints.
- `REPORT_JA.md` — full Japanese research report, numerical tables, limitations, error checks.
- `results/world_ratios.csv` — 24 world×family×budget validation-selected contrasts; full 1,008-score audit and checkpoint hashes are in the separately archived A7 evidence ZIP.

## Reproduce

From the parent `experiments/mirror_capacity` directory, with Python3.13+, PyTorch2.10 CPU, NumPy, SciPy, pandas, scikit-learn and pytest:

```bash
python -m pytest A7/test_a7.py -q
python A7/train.py --out A7/reproduction
python A7/audit.py --out A7/reproduction
python A7/replay.py --out A7/reproduction
```

Use an **absent output directory**. Original result files must not be overwritten. Replay costs additional training; data and code are sufficient without the archived weights.

## Result

- 336 models (6 worlds × 2 families × 2 seeds × 2 byte caps × 7 architectures); 1,008 independently scored splits.
- Mirror4 loses IID and OOD BCE vs validation-selected best non-Mirror controls in **all six worlds** at all four family/cap comparisons.
- A7 rate-quality gate fails: 0/6 practical wins; XOR is partly right-censored (multihead passes in 2/6, Mirror in 0/6).
- All six mapping worlds use a **single** underlying digits dataset, not six independent source domains.
- Results do not establish general capacity impossibility, a deployment-level byte advantage, or an LLM/GPU extrapolation.
