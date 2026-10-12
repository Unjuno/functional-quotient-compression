# Reproduce MA701-LP01

CPU Python3.13, torch2.10.0+cpu, numpy2.3.5, scipy1.17.0, scikit-learn1.8.0, pytest. Run inside this execution directory, not repository root. No GPU or API key. The dataset is packaged with sklearn.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
python -m source.runner calibration --family aligned
python -m source.runner calibration --family digits8
python -m source.runner select
for s in 17101 17102 17103; do
  for f in aligned partial unrelated digits8 digits4; do
    python -m source.runner dev --family "$f" --seed "$s"
  done
done
python -m source.audit --benchmark
python -m source.diagnostic calibration
for s in 27101 27102 27103; do python -m source.diagnostic dev --seed "$s"; done
mkdir -p diagnostic
# A link avoids copying checkpoints and keeps diagnostic audit outputs separate.
[ -e diagnostic/runs ] || ln -s ../runs_adjust diagnostic/runs
python -m source.audit --root diagnostic/runs --benchmark
python verify_retrain.py
```

The runner skips completed directories marked DONE. To run from scratch, use a new execution directory; do not erase the evidence. The primary source and protocol must match PRE_RUN_LOCK.json; the GitHub wrapper format and the local flat format are both supported. Original fresh seeds17111–17115 have not been evaluated; do not use them for tuning.

The full evidence ZIP contains all356 primary/diagnostic checkpoint files plus2 training replay checkpoints and raw per-run metrics. The GitHub source checkout can regenerate these files; it does not itself include all binary checkpoints. CPU timing is stochastic and not expected to match bitwise. Results refer to the common FP32 deployment path; compiler/folding adds model-state memory.
