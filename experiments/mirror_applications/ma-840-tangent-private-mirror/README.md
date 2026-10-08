# MA-840 — Tangent Mirror basis + nonlinear private residual

## H — hypothesis

A shared rank-2 tangent basis may represent small task adaptations compactly; large nonlinear deviations may require private rank-1 residual parameters.

## T — planned test

Synthetic vector regression around one tanh base network, stratified by teacher residual amplitude. Compare tangent-only task codes, tangent plus private rank-1 nonlinear residual, and independent full updates. Fresh seeds and gates are frozen in `PROTOCOL.json`. PA226 motivates tangent linearization; PA18 provides the low-rank adapter control.

## D — status

SCREENING.

## C — strongest counter-hypothesis

The teacher is constructed in a shared tangent subspace, so low-residual gains may reflect alignment with the known basis; rank-1 private control may also simply reproduce the teacher's hard-task construction.

## U — unconfirmed

No result yet; no natural task or pretrained network is included.

## Fact / Interpretation / Hypothesis

- Fact: PA226 studies when fine-tuning is explained by first-order linearization; PA18 is a low-rank adapter method.
- Interpretation: a useful screen must separate the tangent regime from the nonlinear residual regime.
- Hypothesis: a small shared code plus selective private state may give a better storage/quality frontier across both strata.
