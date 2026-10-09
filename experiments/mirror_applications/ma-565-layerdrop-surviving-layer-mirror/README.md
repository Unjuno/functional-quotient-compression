# MA-565 — LayerDrop surviving-layer Mirror roles

Status: **FAIL**  
Branch: `research/ma-565-layerdrop-surviving-layer-mirror-20261009`  
Base commit: `0ff3df2f`  
Prior art: PA110 LayerDrop

## H

Depth-specific Mirror role codes on surviving residual blocks recover useful behavior of shallow LayerDrop subnetworks without restoring omitted layers, at lower bytes than independent shallow subnetworks and beyond ordinary direct gates.

## Frozen mechanism screen

Four residual linear blocks map 8D to 8D. Depth-2/3/4 masks are evaluated on fixed Gaussian probes. The full-depth model is the target teacher. Compare plain LayerDrop mask, scalar Mirror role on each surviving block, direct per-block scalar gates, and independent shallow operators. Oracle least-squares fits gates on training probes and evaluates held-out probes, two seeds. Count actual deterministic NPZ bytes for each inference package and report MAC/wall proxies. This is not LayerDrop training or a Transformer result.

PASS requires shallow depth output nMSE ≤1e-3, ≥20% fewer bytes than independent shallow models, and ≥10% fewer than direct gates. FAIL on direct gate alias or quality miss. Fresh sealed on specificity gate.

## H / T / D / C / U

- **H:** Small role coordinates on surviving layers compensate for omitted depth.
- **T:** Depth-2/3/4 synthetic residual stack, two seeds, 24 rows; least-squares gates, no updates.
- **D:** FAIL. Direct gates only marginally reduce nMSE (.174→.169 at depth 2; .081→.073 at depth 3); independent shallow operator is 747B with zero error. Mirror and direct gate are identical.
- **C:** Direct learned residual gates and independent shallow models.
- **U:** LayerDrop pretraining, attention/FFN Transformer blocks, LM quality and GPU latency.
