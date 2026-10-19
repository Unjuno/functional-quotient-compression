# Phase B2 proposal — Trainable step-dependent Mirror within one physical coefficient vector

**Status: PROPOSED, not frozen and not run.** This is a *new hypothesis after opening MA-1165 audit*; no reuse of MA-1165 fresh seeds as new confirmation.

## Central question

The MA-1165 fixed-schedule Q step-switch produces distinguishable virtual weights, but does not win against simple step gain. Does **learned** step-specific orientation and nonlinear activation threshold allow one physical parameter vector to express functions that gain/FiLM/LoRA cannot efficiently reach?

Preserve the exact MA-1163/1164 mechanism: orthogonal Mirror -> tanh -> inverse Mirror -> residual update of the VIRTUAL MLP WEIGHT VECTOR. This is distinct from rotating the MLP hidden activation as in the Phase-A study.

## Proposed hypothesis and architecture

One trainable vector u of exactly 192 FP32 scalars, with storage allocation `u=(u_core,u_codes)` and `u_codes` holding `2T` scalars (one Givens angle offset plus one tanh threshold offset per step). Hence `len(u_core)=192-2T`. All models compared at a total budget of 192 learned coefficients; for controls, allocate the same 2T coefficients to learned stage gain/bias or low-rank residual, not extra parameters.

Use `z_0 = O pad(u_core)`, and `z_{t+1}=z_t + 0.16 Q(α_t)^-1 tanh(Q(α_t) z_t+b_t(δ_t))`, followed by the same virtual MLP execution operator. Only the single physical u is learned; fixed basis programs and reconstruction code have an explicit byte ledger.

**Variable table (dimensionless SI unit 1)**

| Symbol | Meaning | Type, domain, definition |
|---|---|---|
| u | One learned physical coefficient object | vector R^192 |
| u_core | physical shared-base coefficients | vector R^(192-2T) |
| α_t, δ_t | per-loop Mirror angle and threshold codes | real scalars; 0≤t<T |
| z_t | virtual weights after step t | vector R^280 |
| Q(α_t) | orthogonal stage Mirror composed with learned Givens angle | real orthogonal matrix 280×280 |
| b_t(δ_t) | threshold vector with learned scalar offset | vector R^280 |
| O | fixed orthogonal seed program | matrix 280×280 |
| T | execution loops | integer 2/4/8 |

Dimension and unit check: z_t, Qz_t, b_t, tanh(Qz_t+b_t), Q^-1 tanh(...) all inhabit dimensionless R^280; the generated MLP maps R^8 to R^8 and so can update the residual state.

## Frozen-next protocol before new results

H: At same 192 coefficient storage and bounded compute, a learned step Mirror improves independent-teacher T=4 fresh normalized MSE by ≥5% relative to the best of learned step-gain and step-low-rank controls; swapping step codes should significantly damage the learned function.

T: Use new independently sampled teacher families and seeds **101/103/107/109/113**, not MA-1165 31/47/59; at least 3 distinct held-out world generators (unstructured non-commuting independent MLPS, block-structured synthetic composition, and Mirror-family-aligned positive control). Development-only sweep over alpha initialization/range and code regularization; freeze hyperparameters, updates, dropout, data generation and checkpoint selection before any fresh data. Run 1/2/4/8 depth pilot and then near-convergence checks. Record every update, amount of data, parameter storage bytes, code and program bytes, active MAC estimate, isolated CPU latency, functional ablations, and gradient cosine matrices across layers. A matched 192-float no-Mirror control is mandatory.

D: PASS if learned Mirror beats **both** simple learned gain/bias and matched low-rank shared controls by at least 5% relative fresh normalized MSE on **≥4/5 unseen worlds** at near-equal inference bytes AND execution MAC budget; tied-Mirror fixed budget still must be reported. FAIL if it does not, UNCERTAIN if interval/replicates insufficient. Do not claim MLP-function capacity from fixed-update wins.

C: Nonlinearity and orientation codes may simply replicate FiLM or rank-one residual effects, or the core loss from reserving 2T coordinates may outweigh differentiation. Shallow native MLP may solve tasks better at fixed MAC. Different virtual weights might still represent the same input/output function.

U: Factorial interactions of core width with loops, optimizer-dependent collision, teacher distribution alignment, low-precision and operator-code bytes. Report world-level variability plus 95% confidence intervals if enough independent worlds; do not attach formal uncertainty where assumptions fail.

## Mechanism acceptance checks

1. Exact dense-reference Q inverse for FP64, plus autograd/finite-difference gradient of learned codes.
2. Distinct internal MLP functions by per-step activation causal patching and leave-one-stage-out retraining (not just weight Frobenius difference).
3. Per-step Jacobian singular values and inter-step gradient cosine; separate cross-role conflict from intra-loop conflict.
4. Compare function-space deviation with target task performance and teacher-secret removal (no oracle weights/codes during training).
5. Same serialized byte accounting and CPU end-to-end runtime used for all candidates.

Implementation scope: new sibling MA ID only after branch live-registry re-read; do not retune MA-1165 or overwrite its frozen evidence.
