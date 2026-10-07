# Compute-reuse research intake — 2026-10-07

Research support for existing MA-003 / MA-672 / MA-691..700; not a new MA training result. **No trained models, no throughput benchmark, no capacity claim.**

Read `REPORT.md` for proofs, assumptions, variable definitions and source interpretation. `SUBEXPERIMENTS.csv` specifies 16 follow-up subtests. `SOURCES.json` identifies 12 primary papers; prior-art performance claims were not reproduced here.

## Immediate worker use

- MA-003: share common up/down projections; test sign-View fusion against the exactly equivalent gate/bypass model; test antipodal cancellation before claiming View diversity.
- MA-694/699: keep original key-width temperature after latent absorption; distinguish shared KV from shared attention probabilities; fuse value-only Views only when attention weights agree.
- MA-697/698/700: preserve prefix provenance and source-token history; test the missing-information counterexample; assess quantization error after each intended View.

A differing key/query View usually still needs a separate softmax map. Different upstream nonlinear hidden states cannot generally be repaired from a canonical cache alone.

## Run locally inside the worker container

Requires Python 3.10+ and PyTorch. The checked environment is recorded in `VERIFICATION.json`.

```bash
python -m unittest test_algebra -v
python measure.py
```

The measurement writes `IDENTITY_METRICS.csv` and `MEASUREMENT_SUMMARY.json`. No external data or network is needed. Diagnostics use fixed random algebra fixtures, not fresh task worlds.

## Coordination

This intake is append-only. Do not overwrite the 700-row global registry or reset existing statuses, runs, seeds or parent checkpoints. CR IDs are local subtests, not newly reserved global MA IDs. At the inspected snapshots, the dedicated MA-251 branch was ahead of the shared status board. Check live dedicated branches before claiming work; do not rerun completed MA-248..251 because of a stale board. Full provenance is in `REPORT.md`.
