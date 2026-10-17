# MA-1164 — matrix-free Mirror CPU fusion speed pilot (2026-10-18)

**Status: runtime engineering pilot / positive speed improvement; not Mirror-specific task-quality adoption.**

Research scope: speed up the existing MA-1164 online Mirror operator and the separate, previously trained MA-1164 digit-weight compiler. No prior FAIL/PROMISING status, fresh split, claim ledger or `main` is changed.

**Mirror insertion:** apply a shared physical coefficient vector through multiple deterministic sign/Hadamard Mirror addresses and optional nonlinear sandwiches across K directions and R recurrent steps, to execute distinct virtual operators without storing full weight matrices. The existing learned-digit Householder/Givens/tanh decoder is separately recompiled into one CPU execution kernel while preserving its outputs.

## Two intentionally different speed probes

1. `source/fused_online.py`: **strict matrix-free** implicit Hadamard operator; NO d×d virtual matrix allocated. Provides both original `tanh(mean(W_j x))` recurrence and nonlinear Mirror sandwich `Q^-1 tanh(diag(u) Qx)`.
2. `source/fused_digits.py`: **existing learned image model** weight-compiler optimization. Uses a single transient 1,210-element scratch vector and directly consumes it to calculate logits; unlike (1), **not** virtual-weight-free. It does not improve measured model quality.

Baseline options: original `inference_microaudit/source/online_operator.py` (identical function), cached fully materialized d×d dense matrices (equal function), and original trained `ma-1164/.../source/core.py` decoder. Native cached MLP is a fast non-Mirror control.

## Run (repository checkout)

Install `torch`, `numpy`, `numba`, `pytest` (and for the optional digits replay, `scikit-learn`, `scipy`). Execute from this folder:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/benchmark.py --out results/benchmark_replay.csv
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/benchmark_digits.py \
  --old-checkpoint /path/to/previous/ma1163/results/native_11.pt \
  --out results/learned_digits_replay.csv
```

`benchmark_digits.py` requires an existing prior teacher checkpoint; no checkpoint is shipped in Git history. The earlier conversation's `MA1163_pretrained_baseline_mirror_compilation.zip` includes it. Freshness: that trained-digit split had been previously evaluated, so this is a **posthoc runtime/equality replay**, never a new sealed fresh quality study.

### Outcome and boundaries

- 18 new tests PASS, plus 7 original microaudit tests independently PASS.
- Structure family d=64/128/256/512 × batch 1/64 × (K,R)=(1,1)/(4,4): all 16 conditions faster than old PyTorch eager, 12/16 faster than preexpanded dense matmul at **equal 1 CPU thread**.
- d=256,B=64,K=R=4: fused 1-thread 3.003ms vs old 8.658ms vs cached dense 3.505ms; fused 4-thread .780ms (not a same-thread comparison).
- Trained digit m=768 K=8 R=4, batch64: old 3.458ms, fused .932ms, cached native .025ms. Mirror cannot claim a net native speed advantage.
- Numba JIT startup excluded from steady-state, measured separately. No GPU, no Numba differentiable backward kernel, no FP16/FP8000 high-precision speed claims.

See [Japanese report](REPORT_JA.md), [protocol](PROTOCOL_DIAGNOSTIC.json), [results](results/benchmark.csv), [verification](VERIFICATION.json).
