# MA-699 — Top-k Mirror attention Views over one physical cache

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `81c702ad`

## Hypothesis

H: Multiple active logical attention Views can share one physical cache and retain materialized-view outputs, while their compute still scales with top-k.

## Mirror insertion

> **Mirror insertion:** applies selected per-view key/query and value/output transforms on reads from one canonical K/V cache; per-query selected indices are paid.

- Native baseline: independent per-view materialized caches.
- Single-view baseline: one attention pass.
- Cheapest equivalent: direct plane-phase coordinates, included as native control.

## Prior-art delta

PA156 and PA161 motivate lazy cache reads and latent K/V projections. MA-699 tests several simultaneous logical attention roles and explicitly measures active attention cost as k grows. View-combination counts are not treated as capacity.

## Results discipline

Frozen seeds, top-k values, and gates are in [PROTOCOL.json](PROTOCOL.json). All routing indices and view codes are serialized. Compare actual outputs and MAC proxies to materialized independent views.

## Results and decision

**FACT:** Across three locked audit seeds, lazy top-k output matched materialized-view output with maximum relative error 1.86e-7, 1.40e-7, 1.14e-7, and 8.95e-8 for k=1,2,4,8. At k=8 the shared payload was 16.06% of independent cache payload. Active compute was 1.031k, 2.063k, 4.125k, and 8.25k times a single attention pass; compared to independent top-k, lazy query/output transforms add about 3.125%. Mean pairwise view-output distance was .654. Native direct phase coordinates matched exactly.

**INTERPRETATION:** One cache can support multiple simultaneous logical attention roles at much lower cache storage. The actual compute cost scales with k and view outputs are diverse in this synthetic fixture. Direct phase coordinates exactly reproduce the Mirror representation, so there is no Mirror-specific result and output diversity does not establish independent capacity.

**HYPOTHESIS:** The cache-sharing algebra survives top-k composition, with a useful storage frontier and compute proportional to active k; this is ordinary structured attention algebra.

**BOUNDARY:** Synthetic FP32 attention only; no trained model, language task, capacity claim, GPU throughput, or production serving result. Router indices are charged and logical combinations are not treated as expert count.
