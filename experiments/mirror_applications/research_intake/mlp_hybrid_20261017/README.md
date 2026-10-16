# MLP-HYBRID-001 — Dual-input Mirror SwiGLU + private output capacity

**Complete exploratory CPU research 2026-10-17.** See [full Japanese report](REPORT_JA.md) and [frozen protocol](PROTOCOL.json). This is a standalone research-intake experiment, not a new canonical MA ID or Transformer result.

**Mirror insertion:** Add compact gate/up Givens codes to one shared SwiGLU and compare no output correction vs pair bias, shared rank-2 output adjustment and full pair-specific output adjustment. Native gain with the exact same extra parameters, inexpensive separate readout, and independent MLP are mandatory controls.

- Primary: 5 held-out new data split worlds × 2 seeds × 2 digit-data roles × 11 methods = **220 CPU models**, 550 updates, validation-chosen checkpoints.
- Separately prelocked, postprimary long follow-up: 3 NEW worlds × 2 seeds × 2 roles × 5 methods = **60 models**, 1,400 updates.
- All 560 final+best saved inference payloads audited by hash, bytes and independent NumPy FP64 computations. Eleven tests pass; three complete model retrainings match state hashes and all checkpoints.
- Primary gate: **FAIL** for Mirror+rank2 vs same-count gain+rank2 on both classification and image reconstruction. Native task-specific readout has lower bytes and equal/better quality than the tested Mirror hybrids.
- Follow-up does not establish a Mirror-specific quality/byte frontier; CPU eager Mirror forward is slower than native readout.

Reproduce with installed requirements `python3.13`, `torch2.10.0+cpu`, `numpy2.3.5`, `scipy`, `sklearn1.8`, `pytest`:

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH=. pytest -q tests
python runner.py --phase smoke
python runner.py --phase fresh --workers 3
python audit.py
python analyze.py
python followup.py --workers 3
python audit_long.py
python -c "import analyze; analyze.summary('long')"
python retrain.py
python benchmark.py
```

**Keep original evidence immutable when reproducing; use a fresh copy.** Freeze locks refer to code/test files at study start, not every post-study analysis file. Serialized raw payloads and logs are in the conversation's full evidence archive. Source and aggregate summaries are published on a dedicated research branch; main and canonical MA decisions are unchanged.