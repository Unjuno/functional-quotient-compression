# MA-450 status

**FAIL for the meta-learned controller; the simple validation rule establishes a useful View/private allocation frontier.**

## H / T

Tested mixed related and unrelated 16D tasks. The development rule selected threshold 0.01; a development-only logistic controller predicted whether full private fitting improved query loss. Fresh worlds were 45010–45012, with 120 tasks per world.

## D — Fact

Mean query NRMSE / exact bytes per task: always Mirror 0.482 / 340.3B; always private 3.05e-07 / 404.3B; simple threshold 2.35e-07 / 372.3B; learned controller 2.35e-07 / 380.2B; oracle 2.35e-07 / 372.3B. Both adaptive rules allocated private vectors to 50% of tasks and matched oracle quality. Learned controller and simple threshold had equal quality, but the controller costs 7.9B/task more.

## Interpretation

The validation rule shows when private capacity is needed: related tasks stay in a 2D View; out-of-subspace tasks receive a 16D private vector. This reduces bytes by about 8% versus always-private at essentially the same quality. The learned controller adds no benefit over the simple threshold.

## C — Strongest counter-hypothesis

A single validation residual threshold is sufficient; meta-learning the decision adds policy bytes and compute without improving allocation.

## U — Unknown

No natural tasks or nonlinear networks; only two deliberately separated linear task types. Timing is a tiny CPU mechanism measurement, not deployment evidence.
