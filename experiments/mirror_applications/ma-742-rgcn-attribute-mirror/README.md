# MA-742 — Relation-attribute factorized R-GCN Mirror views

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`

## Hypothesis

**H:** A shared relation-transform basis plus a multiplicative Mirror coordinate composed from two semantic relation attributes will predict held-out attribute combinations with lower serialized inference bytes than per-relation R-GCN basis coefficients, while remaining within 10% of their seen-relation MSE.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m(a,b)` to the coefficient interface that selects shared R-GCN relation transforms, so unseen attribute combinations can express logical relation transforms without storing a free coefficient vector for every relation ID.

- Native method: R-GCN relation-basis decomposition (PA188).
- Shared object: one bank of learned relation-transform basis matrices.
- Coordinate: `m(a,b) = tanh(U[a]) ⊙ tanh(V[b])`, decoded into basis coefficients.
- Logical objects: relation transforms for pairs of relation attributes, including held-out pairs.
- Cheapest control: additive attribute factorization of the basis coefficients; native free per-relation coefficients remain a seen-relation control.
- Prior art: PA188; relation identity is already a compact address, so the test is whether compositional m adds held-out-combination generalization over ordinary basis coefficients.

## Comparisons

1. Independent full relation matrices (upper control, seen relations only).
2. Native R-GCN basis decomposition with free coefficients per relation (seen-relation control).
3. Additive relation-attribute coefficient factorization (simple compositional control).
4. Mirror multiplicative relation-attribute coordinate.

## Gates

### PASS
On all three fresh worlds, Mirror held-out-combination MSE is at least 20% below additive control, within 10% of free per-relation basis control on seen combinations, and serialized marginal relation-coordinate payload (excluding the shared basis common to all methods) is at most 60% of the native free coefficient-table payload. Full inference payload bytes are also reported and must be lower than the free-coefficient model. Payload v1 charges its full eight-byte header and every FP32 tensor byte; reconstruction is checked exactly. Runtime is reported as a separate axis.

Pre-fresh amendments (1) clarified marginal relation-code storage alongside complete payload bytes, (2) permitted an equal-budget 600-update development check after the initial 150-update screen exposed a possible optimization confound, and (3) replaced generic pickle serialization with a compact, versioned inference format after pickle metadata dominated the small payload comparison. All methods use the same format and its complete header is charged. No fresh data were accessed; seeds, model, data, quality gates and the selected equal training budget are unchanged.

### FAIL
Mirror does not improve on additive factors on held-out combinations, exceeds the seen-relation quality tolerance, or fails to reduce actual serialized bytes against free per-relation coefficients. If additive factors match Mirror, the result is not Mirror-specific.

### NOT ESTABLISHED
Synthetic mechanism only; no real-KG or link-prediction claim.

## Tuning boundary

- Development worlds: seeds 7421 and 7422. Ranks `(2, 4, 8)` and equal optimizer updates `(150, 600)` are dev-only choices; choose the lowest-compute setting meeting both predeclared seen-quality and held-out-combination gates.
- Fresh worlds: seeds 74201, 74202 and 74203. Rank 2 and 600 updates freeze before evaluation; settings and hashes are in `source/frozen_config.json`.

## Random selection provenance

Draw 2 selected MA-742 from 533 P0 UNTESTED IDs without a remote `research/ma-*` branch. Seed, index, pool hash and exact eligible list are in `source/random_draw.json` and `source/selection_pool.csv`.

## Decision

### Development facts (fresh still unopened)

Across seeds 7421/7422, rank-2 Mirror mean held-out MSE was 0.00022764816 versus 0.22263122 for additive factors; seen MSE was 0.00022378165 versus 0.0002248915 for native basis coefficients. The serialized marginal coordinate was 104 B versus 264 B; full payload was 1128 B versus 1288 B.

FACT: fresh pending.

INTERPRETATION: developmental evidence meets both quality selection gates; fresh confirmation is required.

HYPOTHESIS: compositional multiplicative relation coordinates may improve unseen attribute-combination behavior with a smaller marginal code state.

BOUNDARY: development-only synthetic relation-conditioned message operator; no natural-graph evidence.
