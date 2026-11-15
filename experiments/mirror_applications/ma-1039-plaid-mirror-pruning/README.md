# MA-1039 — PLAID centroid pruning with a query-role task View

This experiment adds two development-selected Givens coordinates to the shared centroid scoring bank used by official ColBERTv2 PLAID. It asks whether query-length roles can improve held-out retrieval while reusing one residual-coded token index. The required native query-side inverse-Givens control tests whether the insertion point provides any function beyond ordinary score-preserving parameter movement.

The run uses the pinned ColBERT source and checkpoint with CPU PLAID traversal, a bounded positive-qrel-plus-background TREC-COVID collection, and a qid-disjoint development/audit split. Read `PROTOCOL.json` before running. Audit data must not be opened until the source and protocol freeze commit is recorded.

Run from the repository root with the pinned environment paths available:

```bash
python experiments/mirror_applications/ma-1039-plaid-mirror-pruning/source/run_experiment.py
```

The driver writes development freeze and audit summaries under `runs/`; large external model, index, and dataset files remain outside Git. The result is scoped to the bounded collection and CPU execution path, not a full-corpus or production-latency claim.
