# MA-399 — MatFormer nested speculative drafter with Mirror view

Status: protocol frozen before development. Dedicated branch: `research/ma-399-matformer-speculative-drafter-mirror-20261009`.

## Mirror insertion

**Mirror insertion:** four Givens angles rotate the narrow width-8 draft activations before projection, while the width-16 verifier remains unchanged.

## H — Hypothesis

A tiny role-specific Mirror code improves the acceptance overlap of a nested MatFormer drafter with the full verifier enough to raise exact speculative throughput, using fewer actual bytes than a separate drafter/verifier pair.

## T — Frozen design

PA53 establishes nested MatFormer subnetworks; PA15 motivates routed slim-verifier comparisons. A frozen width-16 MLP supplies width-8 and width-12 nested draft roles. Compare hard width-8, width-12 reference, width-8 plus Mirror, width-8 plus FiLM, and an independent width-8 model. Measure held-out draft/full overlap, exact residual-corrected output, actual bytes and draft-plus-verifier latency. The PA15 hierarchical rejection router itself is outside this registered screen. Full details and gates are in `PROTOCOL.json`.

This is a one-step synthetic next-token distribution benchmark, not language-model generation evidence.
