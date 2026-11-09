# MA-732 — factorized PDE coefficient × boundary-condition Views

Status: SCREENING

## Hypothesis

H: Factorized Mirror coordinates over PDE coefficient and boundary-condition factors recover held-out combinations from one shared operator with fewer complete bytes than storing one operator per combination, while beating ordinary DeepONet parameter conditioning.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m=(1/a,b0,b1)`, a factorized coordinate over diffusion coefficient and boundary values, to one shared Green operator so held-out coefficient×boundary combinations do not require separate physical operators.

Prior art: FNO PA186; DeepONet PA187. Controls: independent complete operator bank, shared Green operator plus ordinary scalar parameter conditioning, factorized Mirror coordinate, analytic Poisson solution as known-law upper control.

## Frozen protocol

1D Poisson equation `-a u''=f` on [0,1], Dirichlet boundaries `(b0,b1)`. Coefficients `{.5,1,2,4}` and boundary pairs `{00,10,01,11}` produce 16 combinations. Hold out four combinations while every coefficient and boundary category remains represented in 12 training combinations. For each grid, estimate one shared Green matrix and boundary basis from 128 observed random forcing/solution pairs on each of the 12 declared training combinations. Evaluate the four held-out combinations on disjoint random forcing functions. Compare independent complete operator bank, shared ordinary DeepONet-like parameter conditioning, factorized Mirror code, and analytic exact solve. Charge actual serialized inference bundles. Development seed 73201. Amendment 01 corrects an initial implementation that bypassed training data; those exploratory results are preserved in RESULTS_INITIAL_ANALYTIC.csv.

### Gates

PASS requires held-out solution relative L2 error <=.02 at both resolutions, >=25% complete bytes saved versus independent bank, and a strictly better quality/byte frontier than ordinary shared scalar conditioning. Exact ordinary parameter-conditioning alias is FAIL for Mirror-specific value. No audit after development failure.

## Scope

Known linear elliptic PDE only; no trained FNO/DeepONet or unseen physical law claim.

## Results — amendment 01, development seed 73201

**FACT:** After fitting the shared Green matrix and boundary basis from the 12 training combinations, the held-out four combinations achieve relative L2 errors ~4.1e-15 at grid32 and ~1.05e-14 at grid64. Independent bank payloads are 5,291 B and 21,281 B; native shared parameter-conditioning bundles are 2,380 B and 7,734 B, respectively (55.0% and 63.7% savings). Mirror factor-code payload hashes and outputs exactly match native parameter conditioning. The analytic exact operator is 2,168 B and 7,529 B. Held-out forcing fields are disjoint from training observations. The initial direct-analytic implementation is preserved separately and is not used as generalization evidence.

**INTERPRETATION:** Fitting one Green operator from training combinations generalizes to held-out coefficient×boundary combinations in this linear known-law family and saves substantial storage over independent operators. Ordinary scalar parameter conditioning explains the entire benefit; Mirror adds no distinct function or state.

**HYPOTHESIS:** Factorized codes may be useful for unknown nonlinear operators if they beat standard DeepONet/FNO conditioning; this screen does not test that situation.

**D:** FAIL for Mirror-specific attribution; native control matches exactly. The shared operator storage mechanism itself passes this synthetic feasibility screen. Audit unopened.

**C:** The exact linear PDE is separable in `1/a`, `b0`, and `b1`, so it strongly favors factorization and ordinary physical-parameter conditioning.

**U:** Learned neural operator, nonlinear PDEs, fresh parameter families, noisy observations, extrapolation, production runtime and whether structured Views beat native conditioning.

### H / T / D / C / U

- **H:** Factorized PDE coefficient/boundary Views recover unseen combinations from one shared operator with less state than independent operators and more value than ordinary conditioning.
- **T:** 1D Poisson, 12 train combos to estimate shared Green/boundary bases, 4 held-out combos, 128 disjoint forcing fields/combination, grids32/64, dev seed73201; independent/native/Mirror/analytic controls.
- **D:** FAIL for Mirror attribution; native conditioning exactly aliases Mirror. Shared fit saves 55–64% vs independent bank at near-zero error.
- **C:** Known linear separable law and direct scalar conditioning.
- **U:** Unknown nonlinear operators, trained neural models and fresh domains.

The first non-training implementation is retained in `RESULTS_INITIAL_ANALYTIC.csv` as exploratory-only evidence and is excluded from this decision.
