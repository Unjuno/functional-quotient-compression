# Mirror confirmation design boundaries from T199–T207

Evidence lane: SYNTHETIC / discovery-anchored planning. These results do **not** add new real-model compression claims.

## 1. Pairing benefit must be evaluated in the predeclared contrast

A single scalar cross-width correlation is insufficient. For the predeclared small-vs-large contrast `[-0.5,-0.5,+0.5,+0.5]`, the same nominal correlation can either help or hurt depending on covariance topology.

Representative synthetic result at `rho=0.7`, six independent training replicas per scale:

- independent widths: power ~0.656;
- global common-mode correlation: ~0.986;
- correlation across opposite-sign contrast pairs: ~0.987;
- AR(1)-like adjacent correlation: ~0.805;
- correlation confined within the small and large groups: ~0.449.

Therefore pilot pairing diagnostics should estimate variance reduction in the exact predeclared contrast, not average pairwise correlation.

## 2. Pilot correlation is planning evidence only

With only 4–8 independent model replicas, sample cross-width correlation is very noisy. Even when the underlying common-mode variance fraction is substantial, the realized pilot correlation can span a wide interval.

Do not use a rule such as “if pilot correlation exceeds X, stop at six replicas” as the sole confirmatory stopping criterion. Retain unknown-variance t-based confirmation or separately calibrated sequential boundaries.

## 3. Do not precommit to a 4→6 replica confirmation

The existing real-model discovery ratios for WQ shared/local are `[1.769, 1.276, 0.927, 0.909]`, giving a small-vs-large log-ratio contrast of about `-0.4927`. This is exploratory and selected, so independent confirmation should expect shrinkage.

Representative required independent replicas for one-sided alpha 0.025:

- 25% discovery-effect retention, paired contrast SD 0.10: about 8 replicas for 80% power, 10 for 90%;
- same retention, SD 0.20: about 23 / 30;
- same retention, SD 0.30: about 49 / 65;
- 10% retention, SD 0.10: about 35 / 46.

Thus confirmation sample size must be recomputed from empirical independent-replica contrast variance plus a conservative shrinkage target. The earlier 4→6 design is justified only if pilot data actually show unusually low paired contrast variance.

## 4. Incomplete 28M paired families

If the confirmatory estimand is explicitly a small-vs-large step and 28M training failure is MCAR-like, incomplete families can retain information by using 8M alone as the large-side estimate when 28M is missing.

In a representative synthetic eight-family design with 50% 28M completion:

- complete-case power ~0.322;
- partial-family power ~0.665;
- an 8M-only paired fallback ~0.586;
- type-I errors remained near nominal under MCAR.

This estimator is not generally valid for arbitrary linear or late-only alternatives.

## 5. Outcome-dependent missingness is a hard boundary

When 28M completion probability depends on the mirror metric, both partial-family and complete-case analyses become anti-conservative. In the tested MNAR stress family, nominal 2.5% false-positive rates rose to about 3.9% at the strongest tested dependence.

Therefore:

1. training completion and failure reasons must be logged before decoding mirror outcomes;
2. incomplete-family estimators require an audited MCAR-like assumption or an explicit missingness model;
3. if outcome-dependent failure is plausible, rerun failed models or use a narrower predeclared fallback whose estimand does not depend on the missing 28M outcome.

## Updated confirmatory boundary

- Existing 1M/3M/8M/28M checkpoints remain discovery only.
- Confirmation uses newly trained independent model replicas.
- Replica pairing and the confirmatory contrast are fixed before training.
- Pairing quality is assessed through contrast-aligned covariance/variance reduction.
- Pilot correlation and pilot SD are planning inputs, not endpoints.
- Required replica count is recomputed after pilot independent replicas using conservative effect shrinkage.
- Missingness is audited before any incomplete-family analysis.
