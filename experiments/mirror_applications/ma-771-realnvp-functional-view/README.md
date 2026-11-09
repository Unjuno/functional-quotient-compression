# MA-771 — RealNVP coupling as a nonlinear functional view

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On four heldout task codes, the conditional RealNVP map has nRMSE `0`, maximum inverse-cycle error `4.77e-7`, mean logdet `-3.75e-4`, and minimum sampled Jacobian determinant `.210`. Its serialized shared-flow-plus-code payload is 701 B, larger than the compressed independent flow bank at 573 B. Linear steering and the equally sized four-parameter residual MLP have heldout nRMSE `.2891` and `.2918`; MLP minimum sampled determinant is `1.042`, but no exact inverse is provided. The native conditional RealNVP and Mirror payload hashes, bytes, output, logdet and MACs are identical.

## Interpretation

The coupling map gives nonlinear input-dependent behavior that these same-size linear and residual-MLP controls do not reproduce on the heldout codes, while retaining exact invertibility and tractable logdet. But actual compressed serialization favors the independent task-flow bank in this small parameterization, and ordinary conditional RealNVP is exactly the same function as the Mirror view. This is a mechanistic feasibility result, not a Mirror-specific or density-model gain.

## Hypothesis result

**FAIL** for Mirror-specific value and the byte gate. The transform preserves invertibility, but native RealNVP aliases exactly and the complete view payload is larger than the independent bank. Fresh/audit was not opened.

## Frozen protocol / scope

PA199/PA97 motivate an invertible nonlinear coordinate chart and representation steering. A 2D affine coupling map uses one scalar task code to modulate scale and translation. Twelve of 16 task codes fit a four-scalar residual MLP; four are held out. Compare independent per-task flows, linear activation steering, equal-size MLP, native conditional RealNVP and Mirror. Measure actual NPZ bytes, heldout transform error, analytic logdet, inverse-cycle error, sampled Jacobian determinant, MAC/exp operations and training/evaluation wall time.

The teacher is generated from the same conditional coupling family. This does not evaluate a trained density estimator, likelihood, downstream behavior, higher-dimensional flow or accelerator runtime.

## H / T / D / C / U

- **H:** An invertible coupling view expresses nonlinear task-specific activation transforms beyond linear steering and an equal-size MLP while remaining compact.
- **T:** Seed 77101; dimension 2; 16 task codes; 12 train/4 heldout; 256 train and 512 test examples/task; 2,000 MLP updates; five controls.
- **D:** **FAIL.** Flow inversion works (max error `4.77e-7`), but shared state is 701 B vs 573 B independent; native conditional RealNVP exactly aliases Mirror.
- **C:** The useful nonlinear behavior is standard RealNVP conditioning; serialization compression rewards the nearly redundant independent parameter bank.
- **U:** Natural data, likelihood quality, learned flow conditioning, higher dimensions, stochastic views and hardware cost.
