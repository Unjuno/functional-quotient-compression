# MA-921 — Factorized endpoint Views for reusable flow maps

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Prior art: PA260, Flow Map Matching.

## H — hypothesis

Factorized Mirror endpoint coordinates over one shared two-time flow-map basis can recover held-out interval maps and preserve semigroup composition with lower actual inference bytes than native Flow Map Matching time conditioning, while outperforming a byte-matched additive endpoint factor control.

> **Mirror insertion:** this experiment adds `m(s,t)` to the coefficient interface of one shared flow-map basis, so a distinct interval transport can be expressed without storing a separate map or free code for every interval pair.

## Prior-art delta and controls

PA260 learns a two-time transport map and frames consistency-model formulations. This experiment tests the marginal representation of the endpoint pair. Native FMM conditioning is the strongest registered control; additive endpoint factors test whether multiplicative Mirror composition adds anything beyond simpler factorization.

1. Native FMM-conditioned map `F(x,s,t)` (two-time endpoint input).
2. Shared flow-map basis with additive `U[s]+V[t]` coefficient control.
3. Shared basis with multiplicative Mirror `U[s]⊙V[t]` code and learned coefficient decoder.
4. Independent full 2×2 matrix per interval, as a seen-pair upper control.

## Protocol

- Synthetic non-autonomous ODE: `dx/dt = (A0 + sin(2πt) A1)x`, with new noncommuting 2×2 matrices per world. Reference maps use fixed RK4 integration.
- Time grid: 21 endpoints in `[0,1]`, spacing `0.05`; 198 training pairs; 12 held-out exact interval pairs and their listed long-range compositions.
- Development seeds: 9211, 9212. Fresh seeds: 92101, 92102, 92103. Ranks `(2, 4, 8)` and equal update budgets `(200, 500)` are dev-only choices.
- Held-out metrics: mean map MSE, normalized error for a two-segment composition against the true flow, and model semigroup discrepancy. Direct map NFE is 1; two-segment NFE is 2.
- Storage: actual serialized inference checkpoints, including every basis, endpoint code, decoder and metadata byte.

## Gates

Development selects the lowest update count then rank that meets on both dev worlds: Mirror heldout MSE ≤1.10× native FMM; ≤0.80× additive control; normalized semigroup discrepancy ≤0.01; and complete serialized payload ≤70% of native FMM. Fresh settings are frozen before access. All three fresh worlds must pass.

A failed dev gate is recorded and fresh data remain unopened. A failed fresh world makes the candidate FAIL; settings are not changed afterward.

## Random selection provenance

Draw 3 selected MA-921 from 532 registered P0/UNTESTED IDs with no remote `research/ma-*` branch. Seed, pool hash, index and eligible list are in `source/random_draw.json` and `source/selection_pool.csv`.

## Decision

FACT: pending.

INTERPRETATION: pending.

HYPOTHESIS: pending.

BOUNDARY: synthetic mechanism screen only; no image-generation, diffusion-quality or arbitrary continuous-time claim.
