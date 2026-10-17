# MA-1162 multihead Mirror folding experiment (2026-10-18)

**Mirror insertion:** one shared physical Q/K/V projection triple generates multiple causal logical attention heads using compact per-head Q and post-attention output View coordinates, without duplicated resident KV histories.

This MA-1162 extension tests the user-proposed **multi-head sharded Mirror execution**, not a single scalar network. Parallel H heads share physical K/V, head-specific relative Q rotations change attention distributions, and nonlinear query/output Mirror sandwiches can add non-foldable functions. Recurrent vertical depth is outside this first L0; do not call heads additional depth.

Controlled positive teachers are generated from known Mirror coordinates. Negative off-orbit teachers have unrelated dense per-head orthogonal maps. Compare equal-code scalar Gain and no-view null, plus native static projection folding, and count cache/program/model bytes.

Sources and tests in `source/`, `tests/`. The training benchmark is an aligned synthetic mechanism only, **not natural-language quality or general compression evidence**. Registered against current historical MA-1162 and MA-691/MA-253, without editing previous status.

Reproduction (CPU single thread):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_multihead.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/train.py --phase fresh --out results/fresh_teacher.csv --updates 160 --batch 32 --lr .06 --eval-n 256
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/benchmark.py --out results/decode_benchmark.csv --repeats 25 --warmups 6
```

All results are written only to new files, never overwriting an existing CSV. Frozen protocol and verification must be preserved along with negative results.
