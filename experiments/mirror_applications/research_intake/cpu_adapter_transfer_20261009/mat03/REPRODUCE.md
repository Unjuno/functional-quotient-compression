# MAT03 / MAT03R exact reproducibility instructions

Python 3.13, PyTorch 2.10+CPU, NumPy 2.3.5, sklearn 1.8.0. Environment `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, torch intraop threads=1. No network data downloads. Dataset `sklearn.datasets.load_digits()` baked into sklearn.

From this `mat03` directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/test_mat03.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run_mat03.py --phase all --out /tmp/mat03_repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run_mat03_replica.py --out /tmp/mat03_replica_repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python replay_mat03.py
python check_mat03.py
```

Initial dev `[31,32,33]`, initial fresh `[301..305]`, replication fresh `[401..405]` from unchanged source frozen before the latter. Initial source was NOT committed before initial fresh; its protocol was committed. Replication code and hypothesis were frozen before replication fresh. Output quality/NLL and NPZ bytes replay deterministically; CPU P95 microbenchmarks do not due unlocked clock/jitter.

Five outputs are digit parity, digit >=5, train-median thresholded horizontal stroke asymmetry, vertical stroke asymmetry and center density. All use exact same original 8x8 image pixel inputs and train-derived normalization/thresholds. All methods have 64→128→32 shared FFN trunk and one actual trunk invocation. Native **independent one-pass five-head** is a mandatory control (NOT 5 full forward passes). Other controls include native IA3, ordinary Hadamard 4D affine gating, original-coordinate 4D, learned linear dictionary and analogous private rank1.

Physical storage is serialized NPZ including shared trunk, whole code bank, generator seed and preprocessing. Numerical outputs are checked on heldout original image IDs. The five seed splits are strongly correlated because all use the same 1,797 real images, and no LLM or GPU inference claim is allowed.
