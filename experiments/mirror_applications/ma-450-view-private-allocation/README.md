# MA-450 — View vs private residual allocation

## H — Hypothesis

A learned controller will allocate private residuals only when a compact View fails, preserving always-private quality with fewer serialized bytes and outperforming a simple validation rule.

## T — Test

Synthetic 16D linear tasks, half in the shared rank-2 subspace and half with orthogonal residuals. Three fresh worlds × three seeds × 40 tasks; development-only threshold and logistic controller selection. Compare always Mirror, always private, simple threshold, learned controller, and oracle allocation. Full payload serializers include codes, private weights, flags, controller and metadata.

## D — FAIL (controller), useful allocation evidence

Adaptive quality was essentially exact (2.35e-07 NRMSE) with 50% private allocation. Simple threshold used 372.3B/task vs 404.3B always-private, an 8% reduction. Learned controller matched quality but used 380.2B/task, 7.9B more than the simple rule.

## C — Strongest counter-hypothesis

The validation threshold is enough; the meta-controller only adds bytes.

## U — Unknown

Natural task complexity and nonlinear models remain untested.
