# MA-732 — Factorized PDE-coefficient × boundary-condition operator views

## H — hypothesis

A shared low-dimensional operator basis plus factorized physics and boundary coordinates may recover held-out solution combinations at fewer bytes than independent solution maps. It must also improve on a direct conditioned DeepONet-style operator for Mirror-specific evidence.

## T — execution

Synthetic equation `-kappa*u''=1` on `[0,1]`, with Dirichlet values L,R. Exact solution is `(1-x)L + xR + x(1-x)/(2*kappa)`. One development seed 73201 fixed the representation; fresh worlds were 73211–73213. Each had 4 coefficient values × 4 boundary pairs, with 12 training and 4 heldout diagonal cross-combinations, evaluated on a 64-point grid. Compared factorized Mirror, direct conditioned operator with the same function basis and codes, and independent sampled solution maps. No gradient updates; 768 grid examples/world. Actual canonical NPZ inference payload bytes were measured.

## D — FAIL (Mirror-specific gate)

On all 12 heldout fresh combinations, Mirror solution MSE averaged `1.26e-16` (max `5.85e-16`); analytic PDE residual was zero to floating-point precision. Factorized Mirror payload was 1,510 B versus 4,400 B for independent solution maps (34.3%). But the direct conditioned operator had the exact same shared basis and task coordinates, same 1,510 B payload, and equivalent heldout errors. Thus storage compression against independent tables is established for this exact separable equation, while the preregistered Mirror-specific gate fails because a native direct conditioner matches it.

## C — strongest counter-hypothesis

The closed-form Poisson solution is already separable. The result is ordinary operator conditioning using physics and boundary values; it is not evidence of learned FNO/DeepONet generalization or a new Mirror benefit.

## U — unconfirmed

No trained neural operator, noisy observations, nonlinear PDE, discretization transfer, or natural coefficient/boundary distribution was tested. Heldout combinations test algebraic composition only.

## Fact / Interpretation / Hypothesis

- **Fact:** factorized and direct conditioned methods use 1,510 B and nearly identical heldout solution quality; independent sampled maps use 4,400 B.
- **Interpretation:** factorized coordinates compactly represent this exact family but add no function or storage advantage over direct operator conditioning.
- **Hypothesis:** learned operators may benefit when a low-dimensional task code generalizes to unseen physical combinations beyond what native conditioning provides; not tested here.
