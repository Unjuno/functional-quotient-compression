# MA-715 — RegMean statistics to Mirror merge coefficients

Status: SCREENING  
Evidence lane: MECHANISM / QUALITY / STORAGE  
Base commit: `d65f326e`  
Prior art: PA177 RegMean; PA26 Task Arithmetic.

## Hypothesis

H: For aligned synthetic layer deltas, solving a low-dimensional shared-basis coefficient from RegMean input-Gram sufficient statistics can preserve the full RegMean objective within 5% while reducing the *complete* stored merge state (including those statistics) by at least 25% against full RegMean and the nearest native low-rank coefficient solve.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, the compact coefficient vector over a shared delta basis, to the RegMean layer-weight solve so that multiple merged logical layers can be recovered without storing a full merged weight matrix per request.

- Native method: RegMean solves a layer's least-squares merge from layer input Gram matrices and cross-statistics.
- Physical object: one shared layer-delta basis and fixed source deltas.
- Logical multiplicity: multiple synthetic merged outputs.
- Cheapest ordinary control: direct ridge/least-squares solve in the same basis (native projected RegMean); also full RegMean and Task Arithmetic.
- Closest prior art: RegMean sufficient-statistic merge (PA177); task-vector composition (PA26).

## Frozen mechanism screen

Use D=64 inputs, output width O=32, 8 synthetic source tasks, rank-4 shared basis, 12 merge requests each combining 2/4/6 tasks. For each source task, generate a positive-semidefinite input Gram `G_t=X_t^T X_t` and cross-statistic `C_t=X_t^T Y_t`; target maps are generated from a shared rank-4 delta basis plus controlled orthogonal private residual fractions 0 and 0.1. Compare (i) full RegMean solving `(sum G + λI)W=sum C`, (ii) Task Arithmetic weight averaging, (iii) Mirror basis coefficient solve under summed Gram objective, and (iv) native projected RegMean solve with same basis and coefficients. Development seed 71501 only. No fresh/audit data are opened because this is a minimal mechanism screen; any configuration change requires a new experiment/amendment.

### Gates

PASS requires every request's mean RegMean objective <=1.05x full RegMean, at least 25% lower complete source-plus-output serialized bytes than both full RegMean and native low-rank control, and no >1.25x active solve-work proxy. Any exact function/payload alias to native projected RegMean is FAIL for Mirror-specific value. Failure of quality/bytes/compute gates is FAIL for this scoped screen; implementation or metric non-replayability is NOT ESTABLISHED.

## Storage and compute

Actual single-container serialized payload bytes are authoritative. Include shared basis, source deltas or their coefficients, all input Gram and cross-statistics needed at merge/inference time, request task IDs, outputs and metadata. Full RegMean and low-rank variants are serialized as complete inference/reconstruction bundles. Report examples/statistics, optimizer updates (zero), solve work proxy, and wall clock. This synthetic CPU result cannot establish model-level or language quality.

## Results — development seed 71501

**FACT:** On 12 requests (2/4/6 source tasks repeated), the complete serialized bundles were full RegMean 610,012 B, shared-basis Mirror 673,471 B, native projected RegMean 673,471 B, and Task Arithmetic 314,456 B. Mean weight NRMSE to full RegMean was 0.000, 0.514, 0.514, and 0.515 respectively. Mirror and native projected payloads are byte-identical (SHA-256 `7dc52fe60f8ec3ded55b0a569b5486ee2894987a938e03969f21d9c8f664c0f1`). Projected active solve proxy is 8,256 versus 393,216 for full solve; no optimizer updates. The 48-row output reports one identical complete bundle per method across each request (not 12 independently serialized bundles).

**INTERPRETATION:** The stored Gram/cross-statistics dominate the compact code savings: Mirror is 10.4% larger than full RegMean and misses the 25% byte gate; its restricted solution also has substantial functional error. The cheaper native projected method exactly explains it. Task Arithmetic is smaller but has similar error and is not a quality-matched replacement.

**HYPOTHESIS:** A learned basis may help only if its statistic-to-basis alignment reduces error substantially and the statistics can be amortized/shared across many outputs. This run does not support that claim.

**DECISION (D): FAIL** for the frozen synthetic mechanism gate; fresh/audit unopened.

**COUNTER-HYPOTHESIS (C):** The synthetic basis was not trained on RegMean statistics, and charging per-task Gram/cross-statistics may be inappropriate if an application already stores or can deterministically reconstruct them; either could change the frontier.

**UNCONFIRMED (U):** Learned bases, multi-layer neural models, empirical activation statistics, amortization across substantially larger output banks, actual merge-time wall-clock scaling, and natural-task quality.

## H / T / D / C / U

- **H:** Compact Mirror coefficients solved from RegMean statistics preserve its objective and reduce complete deployment bytes.
- **T:** One synthetic linear layer, D=64, O=32, eight source tasks, rank-4 shared basis, 12 merge requests, development seed 71501; compared full RegMean, Task Arithmetic, Mirror coefficient solve and native projected RegMean. No optimizer updates; 48 result rows cover 12 requests × 4 controls.
- **D:** FAIL for the frozen development screen. Full statistical-state bundle and quality gates both fail; native control exactly aliases.
- **C:** A learned basis and amortized/reusable sufficient statistics could alter this frontier.
- **U:** Fresh seeds, learned basis, natural task models, statistic acquisition cost, and actual runtime on production implementations.

Facts, interpretations and hypotheses are separated in the Results section above.
