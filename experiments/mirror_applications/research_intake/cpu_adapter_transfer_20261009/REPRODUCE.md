# CPU reproduced experiments MAT01 / MAT02

Use Python 3.13 with `torch==2.10.0+cpu`, `numpy==2.3.5`, `scipy==1.17.0` and `scikit-learn==1.8.0`; `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and PyTorch CPU threads=1. Both experiments use the public scikit-learn digits dataset loaded locally, no network downloads.

From this directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/test_mat01.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/test_mat02.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run_mat01.py --phase all --out /tmp/mat01_verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run_mat02.py --phase all --out /tmp/mat02_verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python replay_check.py
python check_mat.py
```

The original preregistrations are preserved separately and can be inspected in `MAT01_FROZEN_PROTOCOL.json` and `MAT02_FROZEN_PROTOCOL.json`. Primary fits are 320 base-model, 145 source-adapter, 175 target-adapter steps per role. MAT01 dev [11,12,13], fresh [101..105]; MAT02 dev [21..23], fresh [201..205]. Source and target task labels come from only train/support IDs; audit image IDs and scaler/normalization are held out. No hyperparameter sweep or fresh tuning. For each seed, all controls see the same support original images.

**Exact replay:** same-seed replay for the first fresh MAT01 and MAT02 world verifies all original task NLL, accuracy and serialized byte sizes with absolute difference zero. Wall-clock P50/P95 varies with unlocked virtualized CPU clocks and is NOT byte-exact.

**Important:** Runtime in these files uses ONE CANONICAL UNTRANSFORMED input sample for a collection of heads; each quality task is evaluated on its OWN transformed image, so that common-input runtime is NOT an end-to-end cost for simultaneously handling four distinct image corruptions. Do not infer single-forward specialist speedup from it.

**Scientific outcome:** Mirror-only loses to native same-byte linear codes and rank4 LoRA. Adding paid rank2 private residual closes much of the quality gap but still loses the strict joint quality/storage/latency gate. All statuses in the authoritative repository remain unchanged.
