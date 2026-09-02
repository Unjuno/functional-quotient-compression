# E006 — Exact conditional Pareto cache

Status: **exact mechanism reconstructed; archived cache sizes not yet reproduced**.

The durable result is that local signatures can be grouped and Pareto-pruned
while keeping bit costs exact. This preserves hard-budget answers within each
signature class and avoids unsafe bit bucketing near the budget boundary.

`src/fqc/cache.py` implements exact conditional Pareto construction. The
archived m=2/4/6/8 entry counts are retained as provenance claims until the
original 16-leaf generator is recovered or independently reconstructed.
