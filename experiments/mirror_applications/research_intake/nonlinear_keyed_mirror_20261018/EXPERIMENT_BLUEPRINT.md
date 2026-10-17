# NKM-INTAKE-001 — work packages and discriminating experiments (design only)

**Status: DRAFT, not a preregistration.** This file defines discriminating mechanisms and controls; it contains no fresh-world result or tuned hyperparameter.

## Common variables and resource accounting

- `theta`: shared learned physical parent FFN/SwiGLU weights.
- `psi`: shared low-dimensional transform generators/Householder directions and any base learned canonical chart.
- `m_i`: role- or condition-specific stored Mirror address; can be **reused over stages**.
- `eta`: shared (or tightly regularized role-specific) activation parameters; fixed ReLU/SiLU/GELU, low-dimensional basis or short continuous spline.
- `a_{i,t}`: activation placement/on-off/stage-depth selection, including index/program storage.
- `c`: **public operational condition** (e.g. green traffic light) selecting role/on-off; `k`: **cryptographic secret** only for decrypt/authorize.
- `E_i`: **optional explicit private residual**, counted and compared against normal LoRA/private FFN.

All physical tensors, codewords, spline knots, coefficients, masks, schedules, router, latent dictionaries, key-management metadata, and compiled kernels are charged to inference bytes and runtime. Separate disk encryption overhead from plaintext functional-code size and never describe one as free.

## Stage 0 — exact math / degenerate gate audit

For `N_m(z)=Q_m^{-1}phi(Q_m z)`, test:
- `Q=I` yields `phi`;
- `phi(z)=z` yields `z` independent of `Q`;
- permutations commute with a uniform elementwise activation, hence no functional gain;
- sign reflection and SiLU give `-SiLU(-z)=z-SiLU(z)` componentwise;
- ReLU with `Q` orthogonal has Jacobian `J=Q^T diag[Qz>0]Q` off boundaries; rank and orientation move, but arbitrary nonzero inputs need not map to zero;
- Householder `H(v)=I-2 vv^T/(v^Tv)` has det -1 for nonzero v, and `H^2=I`; repeated same H **without interleaved nonlinearity** cancels exactly;
- `Givens` rotations det +1: do **not** call them all reflections;
- `phi` may be noninvertible; applying `Q^{-1}` is inverse of `Q` only, not inverse of `N_m`;
- compare naive unfused operators and composition/fusion, and check finite-difference Jacobian and trainable-parameter gradients at identity.

Do not count angle variation as function difference when empirical `||N_i-N_j||` is zero; compute functional Jacobian sensitivity and numerical rank over m.

## MA-1156 — paired nonlinear sandwich

Hypothesis: while fixed budget and strong controls are matched, sandwich `Q^-1 phi(Qz)` can represent high-distance original functions that canonicalize to one shared rule more efficiently than input-only transforms.

Controls: shared FFN; pre-only Q; post-only Q; `Q^-1 phi(Qz)`; two independent Q codes; direct learned diagonal scale/shift FiLM; trainable down/readout; equal-byte residual LoRA; same-compute widened FFN. Distinguish Householder from Givens; test ReLU, SiLU, GELU. 2D exact instances first; then 8/32D and natural FFN teacher functions.

Measure `raw_distance`, `canonicalized_distance`, heldout test MSE/NLL, worst role, serialized bytes, forward/train FLOPs, wall time. A successful aligned teacher does not establish natural-task utility.

## MA-1157 — repeat same m multiple times

Test stage counts `T in {1,2,3,4}` with the **same** m reused across stages, vs distinct per-stage codes with matched bytes, vs one composite Q, vs same FLOPs from conventional deep/wide FFN, vs recurrent shared block, vs a nonlinear native residual block. Interleave `phi` and learn optional stage strength/schedule. Account for Q application each time; reuse of code does not mean zero active FLOPs. Separate `phi=identity` null, trainable phi and fixed phi.

Metrics: function diversity at fixed m bytes, task quality, hidden norm/Jacobian spectral radius, loss landscape, gradient interference, recurrent collapse/explosion, train-FLOPs-to-target, cache consequences if later migrated to a decoder.

## MA-1158 — theta/psi/m/eta/stage joint optimization

Factorize `Q_{i,t}=Q(psi,m_i,a_t)`. Compare:
A. pretrained fixed theta+fixed phi, only m learned;
B. learned theta+learned m; phi fixed;
C. learned theta+psi+m; phi fixed;
D. joint theta+psi+m+eta (low-knot constrained spline / activation mixture);
E. alternating or progressive unfreezing, same total optimizer FLOPs;
F. native learned activation (spline/KAN) with no Mirror.

Track train-vs-val and fresh. Penalize excessive spline knots, nonmonotone growth if safety/conditioning relevant, and poorly conditioned inverse transforms. Include zero-angle gradient and double-zero factor tests. Automatic placement policy must select identity (no Mirror) as an option. Different initial m for related and unrelated roles. Never use test worlds to choose placement.

## MA-1159 — Mirror as nonlinear feature gate

Compare standard `sigmoid(Gx) ⊙ Vx`, learned diagonal/FiLM gate, Householder/Givens-orientated `Q^-1 ReLU(Qz)`, conditioned `Q^-1 SiLU(Qz)`, soft rank-k projector, native sparse/softmax router, and `Mirror + explicit OFF` at **matched bytes and active work**.

A directional feature gate is not automatically a binary switch. Evaluate activation masks/subspace rank, number of distinct valid functions, false suppression/false activation, downstream MSE or CE, worst subgroup, and learned-role stability. Distinguish direct task supervision from role/oracle routing signals.

## MA-1160 — condition-code gate ("green light" scenario)

Separate (1) known public boolean `c`, (2) noisy/inferred `c`, (3) hidden latent condition. Correct outside-domain behavior is **explicitly gated null** or unchanged identity path as registered; not an accidental property of a rotation. Inputs include clean transitions, boundary cases, class imbalance and corrupted condition labels.

Controls: simple if c gating, sigmoid gate with direct c, conditional FiLM, CAST-like condition vector, independent router+Mirror, Mirror-conditioned gate. Count branch predicate/condition extraction cost. Measure false activation rate when `c=OFF`, false negative when ON, worst-case output quality, counterfactual condition-flip behavior and correct-sequence behavior.

## MA-1161 — encrypted Mirror code / conditional authorization

Keep security evidence entirely separate from model accuracy:
- pre-authorized: encrypted `m` at rest (AEAD) with key handled outside model, digest/hash and authorized code loading;
- no-key/corrupted-ciphertext: fail closed (no plaintext fallback); instrument exact access policy;
- key present in untrusted local process: do **not** claim weight secrecy or strong obfuscation; inspect process-memory plaintext availability and output-query leakage;
- optional trusted hardware / enclave as a separate requirement for hiding `theta`, decrypted m, and execution state.

Controls: encrypt ordinary LoRA/adapter under the **same standard AEAD**, normal access controls, protected container/TLS, and standard TEE where available; compare disk bytes, peak plaintext memory, inference latency, key initialization/rekey and unauthorized invocation rate. **Never design a custom cryptographic cipher from rotations.** If runtime cannot protect plaintext, describe at-rest access control only.

## Preregistration / work order

1. Before opening fresh examples: record exact task splits and full model/optimizer/activation version SHA, fixed LR grid, m code dimensions, stage count selection, equal-byte/compute control match, and predeclared utility gates.
2. Use >=3 independent worlds and >=2 seeds per world for initial exploratory fresh replications, then increase if noisy. Worlds are independent units; seeds are paired repeats, not extra populations.
3. Freeze a development-only configuration and record the primary metric at a **fixed update** and a separate **fixed FLOP / near-convergence** budget; account for actual inference bytes and wall-clock.
4. Use natural/independent task families where teacher's construction does *not* assume the tested View. Demonstrate unknown role combinations without leaking those roles into shared theta or selecting on audit data.
5. Treat each MA row as an **independent experimental hypothesis**; do not silently combine five modifications in one experiment and attribute effects to m without controls. Record H/T/D/C/U and keep FAILs.

## Stage gate for eventual Transformer work

L1/L2 must clear a native baseline at matched bytes **and** learning/runtime costs on new natural functional tasks before claims about Transformer capacity. A legitimate mechanism finding can be retained with PROMISING at its narrow evidence lane only if its frozen gate passes; a controlled null result is equally useful.
