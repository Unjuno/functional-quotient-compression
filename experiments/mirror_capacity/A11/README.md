# A11 — cross-family observed-byte capacity frontier (2026-11-15)

Evidence type: **CPU synthetic exploratory research; no general Mirror compression claim.** This experiment follows A10's failure to learn a shared ramp scaffold and tests whether the outcome changes across three externally defined function families and four storage caps. Compared models are the six A10 implementations, copied into a versioned A11 module.

## Executive result

- **288** frozen trained models / **864** independently scored seen-task dev, unseen IID, unseen OOD splits.
- Families: `ramp`, `tent`, `sine` (24 tasks/family/world: 16 meta-training IDs + 8 withheld code-adaptation IDs), two independent world parameter draws (1100, 1101), two training seeds (0,1).
- Byte caps 512/640/768/1024, 600 AdamW meta updates and 200 code-only adaptation updates per model; batch128 and 32 labeled support points per new task.
- Ramp: ordinary **direct Shift passes in both worlds at every budget**. Full Mirror passes in one world at only the 768-B cap, requiring **764 B**, while Shift passed in that world at **500 B**. The other ramp world had no passing Mirror across all four caps. No 10% stored-byte Mirror win.
- Tent and sine: **zero passing method/budget/world cases** at the fixed joint NRMSE ≤0.20 threshold. A minimum required byte count cannot be inferred; these frontiers are **censored**, not proof of impossible representation.
- Independent replay of 12 source-locked model training cases was byte-identical 12/12; 10 A11 unit tests passed. All 288 saved FP32 files passed SHA256/size/deserialize–reserialize verification before score computation.

## Reproduce

From `experiments/mirror_capacity/A11` using Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, pytest 9.0.2:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q test_a11.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train.py --out /tmp/a11-main
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit.py --run /tmp/a11-main
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python replay.py --run /tmp/a11-main --limit 12
python analyze.py --run /tmp/a11-main
```

All output destinations must be fresh; `audit.py` refuses existing audits, `train.py` refuses overwriting run directories. `train.py` creates the model-file lock **without** opening dev/IID/OOD results; `audit.py` checks every model's SHA and round-trip first. Large individual model files are distributed as a separate evidence ZIP and intentionally **not** committed to Git history.

## Files

- [`PROTOCOL.md`](PROTOCOL.md): fixed design and threshold; read before results
- [`a11.py`](a11.py): models, synthetic data generator, exact weight codec; derived from versioned A10 source
- [`train.py`](train.py): training only; no held-out access
- [`audit.py`](audit.py): locked-weight first, then evaluation
- [`replay.py`](replay.py): independent training replay on selected frozen runs
- [`analyze.py`](analyze.py): censor-aware observed-grid frontier
- [`test_a11.py`](test_a11.py): ten unit/integration tests
- [`results/`](results): per-world summary and explicit censoring, not independently generated checkpoint files
- [`REPORT_JA.md`](REPORT_JA.md): methods, variable table, interpretation, limitations, error check

## Interpretation boundary

No observation here establishes that shear Mirror is universally worse than other conditional models; the tested shear structure and its optimization are specific. Three teacher families do **not** constitute three real-world domains. Two world draws per family are too few for inferential population claims. The 4-point budget search and its observed passing bytes are not true minimum description lengths. Fixed update counts do not mean equal FLOPs; source decoder and execution environment are not charged in the weight files. This line does not establish a generalizable Mirror capacity multiplier.
