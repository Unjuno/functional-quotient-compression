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
On all three fresh worlds, Mirror held-out-combination MSE is at least 20% below additive control, within 10% of free per-relation basis control on seen combinations, and serialized marginal relation-coordinate payload (excluding the shared basis common to all methods) is at most 60% of the native free coefficient-table payload. Full inference payload bytes are also reported and must be lower than the free-coefficient model. Runtime is reported as a separate axis.

Pre-fresh amendments clarify the storage denominator and permit a 600-update equal-budget development option after the initial 150-update screen exposed a potential optimization confound. No fresh data were accessed. The storage amendment states: the shared basis is common to all models, so the compression gate is on the separately serialized marginal relation-coordinate payload. No model, data, optimizer or quality threshold changed.

### FAIL
Mirror does not improve on additive factors on held-out combinations, exceeds the seen-relation quality tolerance, or fails to reduce actual serialized bytes against free per-relation coefficients. If additive factors match Mirror, the result is not Mirror-specific.

### NOT ESTABLISHED
Synthetic mechanism only; no real-KG or link-prediction claim.

## Tuning boundary

- Development worlds: seeds 7421 and 7422. Ranks `(2, 4, 8)` and equal optimizer updates `(150, 600)` are dev-only choices; choose the lowest-compute setting meeting both predeclared seen-quality and held-out-combination gates.
- Fresh worlds: seeds 74201, 74202 and 74203. Rank and update count freeze before evaluation.

## Random selection provenance

Draw 2 selected MA-742 from 533 P0 UNTESTED IDs without a remote `research/ma-*` branch. Seed, index, pool hash and exact eligible list are in `source/random_draw.json` and `source/selection_pool.csv`.

## Decision

FACT: pending.

INTERPRETATION: pending.

HYPOTHESIS: pending.

BOUNDARY: synthetic relation-conditioned message operator only; no natural-graph evidence.
