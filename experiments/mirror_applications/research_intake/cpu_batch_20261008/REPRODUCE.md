# Reproduction (CPU only)

Run from a complete checkout of this isolated research branch; Python 3.13, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu, scikit-learn 1.8.0. No internet/model download or CUDA necessary for this mechanism suite.

```sh
cd experiments/mirror_applications/research_intake/cpu_batch_20261008
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python source/test_ma1171.py
python source/test_ma1183.py
python source/test_ma1183_real.py
python source/ma1171_joint_codec.py --phase dev --output /tmp/ma1171_dev
python source/ma1171_joint_codec.py --phase fresh --output /tmp/ma1171_fresh
python source/ma1183_tabm_style.py --phase dev --output /tmp/ma1183_dev
python source/ma1183_tabm_style.py --phase fresh --output /tmp/ma1183_fresh
python source/ma1183_real_tabular.py --output /tmp/ma1183_real
python check_batch.py
```

**Cautions:** Data seeds/labels and learned outputs are deterministic under the declared one-thread runtime, but CPU P50/P95 and cold Python decoder clocks differ between runs. Compare deterministic columns to checked-in `results/*RAW.csv`, not raw clock values byte for byte. Set output directories outside the repository to avoid modifying checked-in results. All tests and model loads run on CPU; no real GPU transfer. These implementations are standalone mechanism tests inspired by native code, not reproductions of official TabM or other source-paper benchmarks.

**Frozen branch provenance:** initial dual experiment protocol commit `5002b45a76d27c8b0e1632e70abb5c3c9d989122`; source freeze before fresh `48a29306ad2c8b38c1cd8778aaa4c130b2d34ecb`; independent real tabular protocol `a87010db95e6b90a23648596d8c06d6210b6bcc7`; real source freeze `da65b039faf8af0551f871e5f9795ca4035ecf0e`.
