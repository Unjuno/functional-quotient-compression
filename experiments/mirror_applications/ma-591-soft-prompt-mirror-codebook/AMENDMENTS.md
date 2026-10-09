# MA-591 amendments

## Amendment 1 — inference timing instrumentation

- Discovered after initial development runs, before any fresh/audit access: implementation saved prompt-training and reconstruction time but omitted the preregistered inference-time metric.
- Correction: time the four held-out windows per task for each frozen prompt representation and report per-task and total evaluation time.
- No data, task selection, model, optimizer, update count, quality gate, storage method, or hyperparameter changed. Initial dev artifacts are retained externally but superseded; registered dev seeds are rerun from the amended frozen source.
