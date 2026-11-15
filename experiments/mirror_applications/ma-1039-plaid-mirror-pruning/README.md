# MA-1039 — PLAID centroid pruning with a query-role task View

**Decision: FAIL.** A two-angle centroid-bank View improves pre-pruning candidate-union recall but misses final MRR/Recall gates and exactly matches the ordinary inverse query-side Givens control.

- [Protocol](PROTOCOL.json): frozen conditions, controls, gates, bounded-data scope and CPU codebook-sample amendment.
- [Results](RESULTS.md): H/T/D/C/U, scoped audit metrics, actual bytes, compute and fact/interpretation/hypothesis separation.
- [Verification](VERIFICATION.json): tests, exact metric replay, exact candidate map and actual payload replay.
- `runs/summary.json`, `runs/audit_rankings.json`, `runs/DEV_FREEZE.json`, `runs/verification_replay.json`: numeric results, per-query replay records and artifact hashes. The checkpoint, raw corpus and PLAID index remain external; their provenance/hashes are recorded.

## Reproduce

From the repository root in the pinned environment, run:

```bash
PYTHONPATH=/tmp/ma1039deps:/tmp/ma1038/ColBERT python experiments/mirror_applications/ma-1039-plaid-mirror-pruning/source/verify_results.py
pytest -q experiments/mirror_applications/ma-1039-plaid-mirror-pruning/tests
```

The experiment driver is `source/run_experiment.py`. It requires the official BEIR TREC-COVID archive, pinned ColBERTv2 checkpoint, CPU FAISS and the official ColBERT source revision listed in `PROTOCOL.json` and `RESULTS.md`.
