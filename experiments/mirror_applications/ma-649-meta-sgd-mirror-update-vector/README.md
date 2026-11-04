# MA-649 — Meta-SGD Mirror update vector

Status: FAIL. Evidence lane: synthetic few-shot adaptation/storage.

## H

Can per-family Meta-SGD rules for correlated task families be represented by a shared rank-4 basis and small Mirror codes with lower complete payload and retained query quality?

## T

32D diagonal quadratic tasks with eight families, 256 support and 512 query examples per family, and heterogeneity 0/.25/.5. Two development seeds. Compared global Meta-SGD, independent family vectors, rank-4 Mirror/PCA and random-subspace codes. Actual serialized payload bytes and curvature-weighted query loss were measured. Fresh seeds remained sealed.

## D

At zero heterogeneity, rank4 Mirror/PCA query loss is nearly equal to independent family Meta-SGD, but payload is only about 4% smaller (1,834–1,835B vs 1,910–1,913B), missing the 75% storage gate. At heterogeneity .25/.5, Mirror loss rises to 1.56–1.72x / 6.37–9.08x. Native PCA exactly matches Mirror. FAIL.

An initial implementation omitted curvature weights in its analytic denominator. The corrected run uses the frozen objective, and the initial artifacts remain under `source/attempts/initial_buggy_run/` with the reason recorded in `source/attempt_log.json`.

## Fact / Interpretation / Hypothesis

- **Fact:** 30 corrected development rows; tests, byte accounting and metric replay pass.
- **Interpretation:** Small Meta-SGD vectors do not amortize to the required payload frontier at this bank size; high task diversity exceeds rank4 capacity.
- **Hypothesis:** Larger banks may amortize shared bases while retaining quality if their update rules remain correlated.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
