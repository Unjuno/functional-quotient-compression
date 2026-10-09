# MA-366 — Depth × expert factorized Mirror routing

Status: **FAIL for Mirror-specific value** (development only; fresh split invalid)  
Branch: `research/ma-366-depth-expert-factorized-routing-20261009`
Base: `c935a90`  
Prior art: PA02 shared/path-constrained MoE routing; PA49 Mixture-of-Depths.

## H — Hypothesis

Factoring logical compute paths into depth and expert coordinates can reproduce held-out depth×expert combinations with fewer actual payload bytes and routing operations than a flat path table, while preserving task quality.

## Mirror insertion

> **Mirror insertion:** this experiment adds a factorized address `(m_depth,m_expert)` to a shared block/expert basis so a pair selects a logical computation path without storing every path independently.

Compare flat path table, PA02-style factorized routing, factorized Mirror path coordinates, and direct pair coefficients over identical basis maps. Routing decisions are fixed identically and path entropy is measured; combinatorial path count is not capacity.

## T — Frozen protocol

Synthetic 4-depth × 4-expert bank of 16D→8D linear operators. Generate targets exactly from `W0 + alpha_depth*D + beta_expert*E + alpha_depth*beta_expert*C`. Hold out 4 of 16 pairs as unseen compositions. For each of two dev seeds, evaluate on 512 Gaussian inputs per path. Fresh IDs 36611/36612/36613 were inadvertently generated before the development gate was adjudicated; the rows are preserved under `protocol_variants/accidentally_exposed_fresh/` and excluded from all claims. Fresh integrity is invalid. Controls serialize 16 independent path maps, shared depth/expert factor maps plus coefficients, and direct pair coefficients. Report nMSE, distinct operators, address collisions/path entropy, actual serialized bytes, route lookup MACs and approximate active operator MACs. No training updates; oracle factor bank is a mechanism/storage screen.

## Gates

PASS both dev seeds if held-out nMSE <=1e-6, all 16 paths distinct, >=10% smaller than PA02 factorized control and flat path table, and route MAC <= factorized baseline. Direct coefficients must be >=10% larger for Mirror-specific result. FAIL if direct control is within 10%, quality fails, or compute rises >10%; fresh sealed.

## Boundaries

Oracle path bank only; no learned router or Transformer, no capacity claim, no path-count-as-expert-count. All bases and coordinates are charged.

## Results

### Fact

Both development seeds recovered all 16 paths with zero held-out nMSE and no address collisions. The flat operator table serialized to 7,941–7,967 bytes; PA02-style shared factorization to 3,353–3,359 bytes; the factorized Mirror path representation to 3,713–3,719 bytes. Direct pair coefficients also used 3,713–3,719 bytes and produced identical decoded outputs and routing MAC counts. Mirror/direct routing used 2 proxy MACs versus 16 for flat and PA02. Every condition used the same 1,048,576 operator-MAC proxy per seed. Fact: the runner was accidentally invoked without `--dev-only`, so it generated rows for fresh seed IDs 36611–36613 before the failure gate was evaluated. This protocol deviation is disclosed in `artifacts/results.csv`; those rows are excluded from analysis and the fresh split is now considered exposed/invalid.

### Interpretation

The synthetic factor bank supports compositional path recovery and reduces payload versus a flat path table. It does not beat the PA02-style factorized control on storage, and the direct coefficient control matches Mirror exactly. The preregistered Mirror-specific gate therefore fails. The accidental fresh-run invocation breaks the split integrity; the fresh rows are disclosed but not analyzed and cannot support any claim. The routing proxy omits learned routing, dispatch, memory traffic, and end-to-end model execution. `RESULTS_CORE.csv` contains only the eight development rows and separates route lookup MACs from the operator MAC proxy; the accidentally generated fresh rows are provenance-only.

### H / T / D / C / U

- **H:** Depth and expert addresses can recover held-out path combinations with fewer bytes and route operations than flat paths and PA02 factorization. The held-out reconstruction sub-hypothesis passed on development; the Mirror-specific superiority gate failed.
- **T:** Synthetic 4×4, 16D→8D oracle linear bank, four held-out pairs, 512 probes/path, development seeds 36601 and 36602; flat table, PA02 factorization, Mirror path and direct coefficient controls. No optimizer updates. Runner was accidentally run on seeds 36611–36613 before the gate; these rows are retained but excluded and fresh integrity is invalid.
- **D:** FAIL for Mirror-specific value.
- **C:** Explicit coefficients and categorical Mirror coordinates are equivalent encodings here; the compact result follows from a shared bilinear basis rather than Mirror.
- **U:** Learned routers, actual MoD+MoE execution, natural language quality, dispatch/runtime cost, near-convergence and fixed-byte quality frontier remain untested. An independent fresh split requires a new MA/amendment because the registered fresh IDs were exposed.
