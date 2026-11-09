# MA-695 — YOCO one-cache with layer-specific reads

Status: FAIL  
Evidence lane: MECHANISM  
Base commit: `2fc822ad`

## Hypothesis

H: A YOCO-style single global memory cache can support distinct layer reads through compact shared-basis View codes, until private residuals erase the storage advantage.

## Mirror insertion

> **Mirror insertion:** adds layer-indexed coordinates to the shared cache read transform, recovering layer-specific attention behavior without storing one K/V cache per layer.

- Native baseline: one global cache, identical read in every layer.
- Independent control: separate full K/V maps/materialized memories per layer.
- Cheapest native equivalent: shared-basis/PCA maps with direct layer coefficients, measured as an exact control.

## Prior-art delta

PA162/PA08 establish one-cache cross-layer sharing. MA-245 already showed an aligned MLKV/Mirror cache-view feasibility case (near-zero synthetic output error, half cache bytes vs two-group MLKV, but only 11.3% serialized model savings and CPU slowdown). This candidate tests four layer roles and controlled heterogeneity against hard sharing and native basis controls.

## Gates

See frozen [PROTOCOL.json](PROTOCOL.json). The task is scoped to output error, complete bytes, and read compute, not language-model quality.

## Results and decision

**FACT:** Across three corrected audit seeds, aligned rho=0/q=0 layer reads matched independent caches within 2.82e-7, at 4,410 B versus 49,926 B (8.83%) and 0.293x the full-width attention MAC proxy. At rho=.1, q<=1 errors were .141–.151; q=2 .131–.147. q=4 restored output error below 2.88e-7 at 6,538 B (13.1% independent) and 0.293x compute. rho=.25 similarly required q=4 for exact output. The native shared-basis control exactly matched Mirror functions and payload hashes. Hard identical reads lose layer-specific behavior.

**INTERPRETATION:** A shared latent read path provides a strong synthetic storage/compute frontier, including exact private rank-4 recovery at low bytes relative to materialized caches. The stricter frozen gate requiring rho=.1 with q<=1 fails, and all savings/functions are exactly the ordinary native shared-basis latent factorization; this is not Mirror-specific.

**HYPOTHESIS:** YOCO-style one-cache layer diversity is feasible with latent-space read transforms, but the tested mechanism reduces to native basis coefficients and needs private rank for heterogeneous layers.

**BOUNDARY:** Synthetic FP32 attention operators, four layers, T=96, D=16, latent R=4; no trained YOCO model, language NLL, GPU kernel, or production throughput claim. Initial bad-scaling attempt is preserved; corrected math replayed all locked seeds without tuning.
