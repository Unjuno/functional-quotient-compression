# MA-771 — RealNVP coupling Mirror activation views

## H — hypothesis

A shared invertible RealNVP-style activation chart plus a small task code may represent many nonlinear task-specific invertible maps with lower bytes than private flow weights and better held-out fit than linear activation steering.

## T — planned test

A synthetic 2D invertible teacher `F_m(x)=g^{-1}(g(x)+m)`; fit shared chart on development task IDs, then adapt held-out codes from 8 support activations and score 256 query activations. Compare additive steering, affine, 2-16-2 residual MLP and independent flow. Fresh seeds and gates are locked in `PROTOCOL.json`. PA199 establishes RealNVP coupling; PA97 covers representation steering.

## D — status

SCREENING. Protocol frozen before fresh audit.

## C — strongest counter-hypothesis

A small affine or MLP activation intervention may recover the task transformations at comparable serialized bytes, making the invertible chart unnecessary.

## U — unconfirmed

No results yet. The synthetic teacher is structurally aligned to the Mirror family, so even a positive result would establish only a mechanism case.

## Fact / Interpretation / Hypothesis

- Fact: RealNVP coupling layers provide tractable exact inverse and log-determinant operations (PA199).
- Interpretation: invertibility makes a strong structural diagnostic, but it does not establish useful specialization.
- Hypothesis: a shared chart with task-local latent translations may trade a small code for substantial nonlinear task variation.
