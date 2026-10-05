# Reachability-adjusted Mirror: finite verification

This directory verifies the algebra in [the analysis note](../../../docs/phase2/REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md).
It does **not** run Transformer training, checkpoint analysis, or a capacity benchmark.

## Run

From this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_geometry.py
```

Dependencies: NumPy and SciPy, plus Python's standard library. The recorded run used
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, CPU float64. These are recorded conditions,
not a claim that other versions were tested. This is a standalone experiment;
no runtime dependency or Transformer source in the main package is changed.

The command runs 15 tests and rewrites `verification.json` with test counts,
finite-fixture measurements, environment information, and SHA-256 values of
both Python source files. Exit status is nonzero on failure. Do not interpret
the unittest runtime as a model speed benchmark.

## Scope

- Exact linearized minimum-metric-norm reachability, with explicit feasibility.
- Baseline/Mirror cost ratio in the correct orientation.
- Coordinate changes with the corresponding transformed metric.
- Fixed-view folding, nonlinear generator derivatives, fixed-preconditioner dynamics.
- Simplex moment identities and counterexamples to token-cycle cancellation.
- Volume preservation versus numerical conditioning.

A large efficiency eigenvalue is **not** evidence of useful task progress.
The rank tolerance is numerical and finite-data dependent. The dense routines
are small reference implementations, not scalable JVP/VJP implementations.
Metric/candidate selection, task-utility validation, full joint token/layer
curvature, and finite-step Transformer replay remain to be executed.

`provenance.json` identifies the previous local report without promoting its
checkpoint-derived K suggestions into verified optima. Review was an author
self-review, not an independent third-party review.
