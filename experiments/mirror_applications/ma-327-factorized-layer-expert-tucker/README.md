# MA-327 — Factorized layer x expert Tucker address

Status: PROTOCOL FROZEN BEFORE DEVELOPMENT
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE
Base commit: <sha>
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## H — hypothesis

For a planted low-rank Tucker matrix bank of logical layer-expert transforms, factorized layer and expert codes will recover held-out combinations at test nMSE <= 1e-5 while saving at least 20% actual bytes over flat pair codes and at least 10% over the nearest ordinary coefficient-product control.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` to <exact object/interface> so that <claimed logical variation> can be expressed without <targeted physical duplication/cost>.

- Native method before adding `m`: PA35-style shared Tucker bank with coefficients per layer-expert pair.
- Exact insertion point for `m`: coefficient tensor is represented as a product of layer and expert coordinates.
- Persistent or dynamic `m`: persistent task coordinates.
- Mirror resources: fixed rank-2 coefficient product; no fresh tuning of rank.
- Cheapest ordinary parameter that might provide the same freedom: direct rank-2 factorized coefficient product.

## Physical-to-logical claim

- Physical object being shared:
- Mirror/View coordinate:
- Claimed logical multiplicity:
- Why this could save storage:
- Why it might fail:

## Prior-art delta

Read the registry row and referenced PA items first.

- Closest prior art:
- What prior art already establishes:
- Exact Mirror-specific delta tested here:
- Cheapest simpler control that could explain the result:

## Comparisons

Primary:
1. ordinary baseline;
2. existing non-Mirror method being replaced;
3. byte-near low-rank/gate/shared-basis control;
4. Mirror candidate;
5. unrestricted independent-object upper control when practical.

## Gates

### PASS
All three fresh seeds pass held-out nMSE <=1e-5, Mirror bytes <=0.80x flat pair control, and Mirror is at least 10% smaller than ordinary product coefficients at comparable quality.

### FAIL
Any two fresh seeds miss quality or 20% flat-control byte reduction, ordinary product control comes within 10% bytes, or fit compute exceeds 10x flat coefficients.

### NOT ESTABLISHED
Task/control implementation failure, serialization replay mismatch, or fresh data access before protocol/source freeze.

## Tuning boundary

Development:
- worlds/seeds: 32701, 32702.
- hyperparameters allowed to change: training and numerical settings only if report amendment is committed before fresh access.

Fresh/audit:
- worlds/seeds:
- seeds 32711, 32712, 32713; never used for tuning.

## Storage contract

List every paid inference object. Actual serialized payload is authoritative.

## Compute contract

Record tokens/examples, optimizer updates, active-compute proxy, isolated wall-clock, and inference throughput if relevant.

## Results

Do not write conclusions until RESULTS_CORE.csv and VERIFICATION.json exist.

## Decision

FACT: pending.
INTERPRETATION: pending.
HYPOTHESIS: pending.
BOUNDARY: synthetic linear Tucker screen only.
