# MA-1166 Phase B — trainable Mirror orientation and stronger gradual path weakening

**2026-10-20 JST, exploratory follow-on protocol frozen after Phase A results, BEFORE Phase B training.** Phase A evidence and fresh splits remain immutable; no phase-A fresh samples may tune Phase B.

## New mechanism/question
Phase A retained ~59–63 of 63 logical edges under weak averaged gate pressure. Investigate whether a **learned orthogonal Mirror direction** (two scalar Householder orientation offsets per candidate, shared u) plus **summed** soft-edge pressure can produce a substantially sparser graph without destructive early pruning.

- Physical trainable object remains exactly 253 FP32 scalars: 64 shared core; 63 edge gates; 63×2 learned Mirror orientation offsets. Affine control has 253 scalars with the same 2 per-edge extra coefficient budget.
- The deterministic pair of Householder vectors for each logical edge is perturbed by a scalar times a deterministic orthogonalized rolled basis; normalize to unit length, so the resulting Q remains orthogonal and Q inverse remains exact (floating-point tolerance).
- The generated virtual MLP still uses **orthogonal Mirror → tanh → inverse Mirror** on one native 6-8-6 virtual weight vector; no activation-space Mirror substitution.
- Every 0<=i<j<=6 directed node-event edge × three Mirror addresses is in the training supergraph (63 candidates). Loop reuse across repeated module identities, long skips, and all graph addresses unchanged.
- No hard pruning during training. Sigmoid edge gates remain differentiable, permitted to recover. **L1 pressure λ(sum_e g_e)** warm-started after 100/400 updates, then linearly ramps to λ=0.005; physical θ receives 1e-3 mean-squared norm regularization.
- Frozen variants: `rot_full`, `rot_sample`, `affine_full`, `affine_sample`; sample Bernoulli 0.5 with 1/p scaling, as Phase A.
- CPU FP32, 400 Adam updates, lr0.02, batch128; new world/seed pairs `aligned` and `independent` × **307,311**. Independent train/dev/fresh generation by Phase A's frozen generator using the new seeds; dev-only selection/thresholding from Phase A and one fresh evaluation after selecting.
- Train and evaluate paired full v sample; export/replay one-vector compiled payload bytes; report actual CPU latency and learned edge survival.

## Gates
Structural exploration mechanism PASS only if each corresponding full/partial method retains <=44/63 edges (>=30% pruning) at dev normalized error <= unpruned*1.05 +0.01, and if removal preserves fresh NMSE within 10% relative in >=3/4 rotating-Mirror worlds. Mirror-specific win requires same-budget affine fresh/binary/compute Pareto advantage; otherwise FAIL.

**Uncertainty:** Strong sparsity pressure may collapse graphs to identity; rotation codes may be no better than affine codes; Phase B is exploratory and no broad empirical claim follows from 2 seeds. Phase B comparisons must NOT be pooled with original Phase A frozen baseline for confirmatory significance.
