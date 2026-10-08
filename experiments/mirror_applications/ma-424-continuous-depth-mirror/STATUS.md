# MA-424 status: FAIL (quality works; compression gate fails)

## H — falsifiable hypothesis

Four rotated mode dynamics could be recovered by one shared vector field plus one angle per mode at <=60% of independent payload bytes, with near-equal trajectory quality and no worse stiffness.

## T — executed

Four 2D stable linear ODE modes generated as rotations of one shared symmetric matrix with eigenvalues -0.3 and -1.4. Fit vector fields from derivative samples; integrate fresh initial states to t=1 with RK4 at 8 and 16 steps. Compared shared, angle Mirror, scalar mode gate, rank-one residual, and independent matrices. 300 AdamW updates × batch 256; 2 development worlds × 3 seeds selected LR 0.01; 3 fresh worlds × 3 seeds. Actual serialized inference tensors retained.

## D — FAIL

Fresh mean normalized derivative RMSE / RK4-16 endpoint RMSE / bytes: shared 0.33738 / 0.27774 / 1,577B; Mirror 0.00217 / 0.00193 / 1,829B; scalar 0.33328 / 0.27887 / 1,829B; rank-one 0.02004 / 0.02654 / 2,081B; independent 0.01623 / 0.00831 / 1,641B. Mirror quality is strong and NFE is identical (64 at RK4-16), but the actual payload is 11.5% larger than independent, not <=60%. Maximum stiffness ratio was 4.693 Mirror vs 4.687 independent and teacher 4.667, so the strict no-higher-stiffness gate also misses.

## C — strongest counter-hypothesis

The low-dimensional model parameter count hides the serialized tensor dictionary/metadata overhead. On this tiny 2x2 system, a dense independent matrix serializes more compactly than base-plus-angle state, even though Mirror has fewer learned floating-point values. At larger dimensions, this byte ordering may change.

## U — unresolved

Larger state dimensions, adaptive solvers, nonlinear vector fields, inference throughput, and whether matrix fusion reduces actual payload/compute remain untested. Result is aligned linear-dynamics evidence only.

