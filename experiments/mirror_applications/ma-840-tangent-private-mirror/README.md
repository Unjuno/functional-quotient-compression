# MA-840 — Tangent Mirror basis + nonlinear private residual

## H — hypothesis

A shared rank-2 tangent basis with task Mirror codes should cover small, approximately linearizable adaptations cheaply; high residual tasks should benefit from a private rank-1 nonlinear residual without paying for a full update.

## T — execution

Three fresh synthetic worlds (84011–84013), each with 4 development task IDs used to estimate the rank-2 tangent basis and 8 fresh task IDs (4 low-residual, 4 high-residual). Each fresh task used 32 support and 256 query examples. Controls were tangent-only task codes, tangent plus private rank-1 nonlinear residual, and independent full weight updates. Inference payloads were serialized with safetensors, including shared base weights and basis where required.

## D — FAIL

The preregistered low-residual tangent quality gate passed in all 3 seeds: mean query MSE was `1.78e-6`, with 2,256 B shared tangent payload. In the high-residual stratum, tangent-only mean MSE was `1.415e-2`; adding the private rank-1 residual produced `1.415e-2` as well, failing the registered requirement for at least 50% improvement. Full per-task updates reached mean MSE about `1.67e-6` at 5,232 B. The hybrid payload was 4,024 B but did not recover quality.

## C — strongest counter-hypothesis

The chosen rank-1 residual parameterization is ineffective under the frozen 32-example support budget; this does not show that all private residuals fail. Full updates show that nonlinear capacity is useful for the hard stratum, while this particular residual implementation did not learn it.

## U — unconfirmed

No natural tasks or pretrained network were tested. The tangent basis was estimated from synthetic task deltas that share a rank-2 subspace. Whether a better private residual, selective allocation rule or more support data fixes the hard stratum remains unknown.

## Fact / Interpretation / Hypothesis

- Fact: tangent-only fits the low-residual group; hybrid and tangent-only errors are effectively equal on the high-residual group; full updates improve high-residual quality by roughly four orders of magnitude.
- Interpretation: shared tangent coordinates compress this aligned low-residual task family, but the tested private residual does not bridge the nonlinear boundary.
- Hypothesis: private state needs a more learnable factorization or residual selection rule for nonlinear tasks.
