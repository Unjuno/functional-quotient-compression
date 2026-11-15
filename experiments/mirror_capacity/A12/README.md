# A12/A12b — diagnosis of two failures after A11

**CPU synthetic, exploratory; no Mirror-specific capacity advantage claimed.**

Question: Is the A11 tent/sine failure due to too few support pairs, too short code optimization, or a shared scaffold that cannot be adapted to unseen tasks?

- A12: 2 families × 3 worlds × 2 seeds × 3 models × 2 meta stages (600/2400) × 2 support counts (32/256) × 2 adaptation stages (200/800) = **288 final checkpoints** plus 72 parent meta checkpoints.
- A12b, designed after A12 results: 36 meta2400 parents, 1024 support pairs/task, 2400 updates, compare code-only adaptation vs all-weight joint training = **72 final checkpoints**.
- A11 model and exact FP32 serialization are inherited from sibling `../A11/a11.py`. Teacher parameters generate labels only; no oracle teacher coefficients enter the learned models.
- A12B's all-weight joint condition is not matched few-shot deployment; use only as a mechanism diagnostic.

## Main empirical conclusion

Tent: 32→256 support pairs improves Mirror unseen IID error 0.727→0.582 at fixed meta2400/adapt800, but correct-family regression from 32 examples attains ~1e-6 NRMSE. Therefore this is not a fundamental lack of sample information; learned code-fitting/scaffold geometry is limiting. Tent FiLM all-weight joint training passes 3/3 worlds; frozen-code adaptation passes 1/3.

Sine: 32→256 barely changes Mirror unseen IID error (0.475→0.463), and 2400 meta updates reduce old-task error without reliably changing unseen-task error. All-weight joint training improves IID but no model passes OOD; extrapolation remains a separate limitation.

## Reproduce

Prerequisite: `../A11/a11.py` present, Python 3.13, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0, pytest. Run all commands from this A12 directory. Never reuse existing output directories.

```sh
python -m pytest -q test_a12.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py --out /tmp/a12-work/main --workers 4
python -c 'from a12 import audit_run; audit_run("/tmp/a12-work/main")'
python oracle.py --out /tmp/a12-work/main/oracle.csv
python analyze.py --root /tmp/a12-work
python replay.py --run /tmp/a12-work/main --receipt /tmp/a12-work/main/replay.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python a12b.py --parent /tmp/a12-work/main --out /tmp/a12-work/a12b --workers 4
python a12b.py --parent /tmp/a12-work/main --out /tmp/a12-work/a12b --audit
```

**Note:** `analyze.py --root` requires a directory containing the `main/` experiment folder. The example consistently uses `/tmp/a12-work/main`.

Sources: [A12 frozen protocol](PROTOCOL.md), [A12b post-discovery protocol](A12B_PROTOCOL.md), [full Japanese report](REPORT_JA.md), [evidence manifest](EVIDENCE.json). Raw checkpoints are in the separate attached ZIP to avoid duplicating binary weights in Git history.
