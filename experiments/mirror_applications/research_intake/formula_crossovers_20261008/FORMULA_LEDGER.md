# Internal Formula Ledger — audited before combined Mirror experiments

2026-10-08, research-only branch \`research/mirror-internal-formula-crossovers-20261008\`. Baseline inherited from canonical worker HEAD \`c935a903daca5c7d1d48aa50d05b5bd50f239cba\` through isolated formula research. This is **not** a replacement architecture, not a new fundamental compression theorem, and not a worker handoff.

## 1. Shared variable and dimension contract

For the numbered formulae IF01–IF12 below, every symbol is defined here before use. Normalized activations, parameters, and code coefficients are dimensionless unless stated otherwise. Byte is a practical information unit rather than an SI base unit.

| Symbol | 日本語の意味 | 単位（SI） | 定義・条件・範囲 | 型 |
|---|---|---|---|---|
| x,u,h,q | 入力・中間活性・Query | 1 | compatible, finite normalized features | real vector |
| W_up,W_down,W | FFNまたは共有重み | 1 | W_up∈R^{d_h×d_in}, W_down∈R^{d_out×d_h} | real matrices |
| b_up,b_down | FFN bias | 1 | finite vector in matching output space | real vectors |
| φ,Dφ | GELUとそのJacobian | 1 | elementwise smooth GELU; Dφ diagonal | function / matrix |
| Q_m,Q_m^{-1} | Viewの正則変換と逆 | 1 | GL(d_h), invertible; cond finite | real matrices |
| m,a,c_i | Mirror座標・局所方向・離散コード | 1 | m∈R^k, a∈R^p, c_i∈R^r | real vectors |
| J,B | baselineとMirrorの観測Jacobian | 1 | J∈R^{n×p_θ}, B∈R^{n×p} | real matrices |
| δθ,b | baseline更新・目標観測変位 | 1 | δθ∈R^{p_θ}, b∈R^n | real vectors |
| R_θ,R_M | baseline/Mirror更新計量 | 1 | symmetric positive definite | real matrices |
| W_θ,R_B | baseline観測到達行列・模倣コスト行列 | 1 | W_θ=J R_θ^{-1}Jᵀ; R_B=BᵀW_θ^†B | PSD real matrices |
| λ,v | 一般化固有値・固有方向 | 1 | λ>=0, v∈R^p | scalar / vector |
| η,N | 固定学習率・更新回数 | 1 | η>0, N∈N₀ | scalar / integer |
| e_N | 第N回の目標残差 | 1 | b−J δθ_N | real vector |
| r,K_c | code共分散rank・Viewのsupport数 | 1 | positive integers, K_c>=2 | integers |
| C,π,Σ_c | code行列・確率・共分散 | 1 | C∈R^{K_c×r}, π_i>0; sumπ_i=1 | matrix / probability vector / PSD matrix |
| t,o,k(t,o),β_t | token位置・開始位相・phase・位置感度 | 1 | integer positions; β_t∈R^r | integers / real vectors |
| K,V,A_m,B_m | キャッシュK/Vと右作用View | 1 | K,V∈R^{T×d}, A_m,B_m∈R^{d×d} | real matrices |
| T,d | cached token数・head幅 | 1 | positive integers | integers |
| α | attention重み | 1 | softmax vector, sum=1 | probability vector |
| G,e_i | ブロック間PSD計量とブロック誤差 | 1 | symmetric G≽0, e∈R^n | matrix / real scalar components |
| D, D_diag, D_maj | 真の結合歪み・対角近似・上界 | 1 | nonnegative quadratic values | real scalars |
| S, S_raw, H_s | 実直列化費用・logical bits・header負担 | bit or byte as stated | nonnegative integer; conversion always explicit | integer scalars |
| σ_h,E_f | 入力二次モーメントと線形重み誤差 | 1 | Σ_h=E[hhᵀ]≽0, E_f=ΔW−ΔW_hat | real matrix |
| L,ρ | 一観測あたり損失・View振幅 | nat/token (dimensionless), 1 | finite loss; ρ>=0 | real scalars |
| u_c,k_cov,U | 合成標準不確かさ・包含係数・拡張不確かさ | Lと同単位,1,Lと同単位 | all nonnegative, k_cov>0 | real scalars |

**SI/dimension audit:** Matrices/vectors on each side of any equality must have conformable shapes. Every neural operation above is dimensionless after normalization. \`S\` is a physical serialized bit/byte count and cannot be added to quality \`L\` or wall seconds; those axes are constrained separately. All Jacobian/rank/eigen claims refer to a **specified finite observation set**, tolerance and optimizer metric, not a global Transformer function-space theorem.

## 2. Exact equations and limits

### IF01 — fixed FFN View folding, not independent expert capacity

$$F_m(u)=W_{\rm down}Q_m^{-1}\phi(Q_m(W_{\rm up}u+b_{\rm up}))+b_{\rm down}.$$

Set \`W'_up=Q_m W_up\`, \`b'_up=Q_m b_up\`, \`W'_down=W_down Q_m^{-1}\`. Substitution gives \`F_m(u)=W'_down φ(W'_up u+b'_up)+b_down\` identically for every input. For a pure permutation P, componentwise GELU satisfies φ(Pz)=Pφ(z), so conjugation by P alone is an **exact no-op**. Multiple *different* m values trained with **tied original W** remain constrained by that common parameterization; one fixed m does not create Shannon information. Source: \`docs/phase2/REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md\`, \`docs/phase2/MS003_MECHANISM_SENSOR_BRIDGE.md\`. Numerical algebra F01 checked in five fresh worlds.

### IF02 — correct baseline emulation cost and generalized eigenproblem

For a target \`b\` in the range of J and SPD R_θ, define

$$C_B(b)=\min_{J\delta\theta=b}\frac12\delta\theta^\top R_\theta\delta\theta
=\frac12 b^\top(JR_\theta^{-1}J^\top)^\dagger b.$$

Proof: whiten \`z=R_θ^{1/2}δθ\`, \`Jbar=J R_θ^{-1/2}\`; least-norm feasible solution is \`z_*=Jbar^\dagger b\` by SVD. Its squared norm is \`bᵀ(Jbar Jbarᵀ)^†b\`; every other feasible z differs by a nullspace component orthogonal to z_* and cannot lower the norm. If b is outside range(J), the constrained cost is **+∞**, not the pseudoinverse expression evaluated regardless of feasibility.

For b=Ba, \`R_B=Bᵀ W_θ^† B\`, \`W_θ=J R_θ^{-1}Jᵀ\`, and \`C_M(a)=aᵀR_Ma/2\`, the emulation ratio is

$$E(a)=\frac{a^\top R_B a}{a^\top R_M a},\qquad
R_Bv=\lambda R_Mv.$$

Whiten by \`R_M^{1/2}\`; Rayleigh–Ritz yields largest λ as maximal local emulation-cost ratio when \`Ba\` is reachable. This is the **correct direction**; the inverse eigenproblem reverses which direction is most difficult for baseline. Large E is not high task utility, nor rate reduction. Source: \`docs/phase2/REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md\` §2–3; F02 tests the weighted constrained optimum and a diag(1,0.1) Jacobian with cost eigenvalues 1 and 100.

### IF03 — finite-step reachability is not convergence/capacity

For fixed J, R_θ and quadratic residual, the preconditioned iteration gives \`δθ_{j+1}=δθ_j+ηR_θ^{-1}Jᵀe_j\`. Multiplying by J and subtracting from b yields \`e_{j+1}=(I−ηW_θ)e_j\`; inductively

$$e_N=(I-\eta W_\theta)^N b.$$

Here exponent N means **matrix power**, not transpose. An eigencomponent with eigenvalue ω decays by \`(1−ηω)^N\` when \`0<η<2/λ_max(W_θ)\`; a null component remains. An actual AdamW Transformer has changing Jacobians and is not guaranteed by this equality. Source: RA-Mirror §4; F03 exact fixed-J replay.

### IF04 — minimum support for mean-zero full-rank code, not optimal view count

Given positive probabilities π and \`Cᵀπ=0\`, nonzero π lies in the left nullspace of C. Thus \`rank C≤K_c−1\`. Positive diagonal π implies \`rank(Cᵀdiag(π)C)=rank C\`; hence if covariance rank is r, necessarily

$$K_c\ge r+1.$$

Existence: choose an orthonormal basis V of the subspace perpendicular to the all-ones vector in \`R^{r+1}\`, set \`C=\sqrt{(r+1)/r} V\`. Then row norms are 1, off-diagonal inner products \`−1/r\`, mean zero and uniform covariance \`I_r/r\`. This is a support **lower bound** for a selected r, not evidence that K_c=r+1 maximizes quality/capacity or minimizes compute. Source: RA-Mirror §5; F04.

### IF05 — zero-mean token-phase codes do not cancel a model response

With periodic index \`k(t,o)=(t+o) mod K_c\`, first-order loss response is \`Σ_t β_tᵀ c_{k(t,o)}\`. Even when \`Σ_k c_k=0\`, different β_t prevent cancellation. Minimal witness: \`c=(1,−1)\`, \`β=(1,2)\` gives \`−1\`. Uniform average over all starting phases cancels the first-order term at a fixed shared baseline but cross-token Hessian terms remain. Chunk boundaries must not silently reset logical token phase. Source: RA-Mirror §6 and TM001; F05.

### IF06 — exact lazy canonical KV read, subject to prefix provenance

Assume *exactly* \`K_m=KA_m\` and \`V_m=VB_m\` for a physically identical canonical cached prefix. Then

$$\operatorname{softmax}(qK_m^\top/\sqrt d)V_m
=\big[\operatorname{softmax}((qA_m^\top)K^\top/\sqrt d)V\big]B_m.$$

Proof: \`(KA_m)ᵀ=A_mᵀKᵀ\`, and associativity moves the right value transform after the weighted sum. Nothing here says different hidden-state histories satisfy \`K_m=KA_m\`; their caches may be incompatible. RoPE additionally requires a position-commuting transform or explicit position-state handling. A copied tensor is not physical aliasing. Source: \`docs/phase2/MIRROR_KV_CACHE_REUSE.md\`, MA-691 report, GVA Stage-0; F06 verifies exact case and invalid perturbed-prefix counterexample.

### IF07 — coupled task distortion can defeat block-local greedy allocation

For a PSD block metric G and error e, true distortion is \`D=eᵀGe\`. A diagonal-only approximation uses \`D_diag=Σ_i G_ii e_i²\` and may **underestimate** D. By triangle inequality and \`2|e_i e_j|≤e_i²+e_j²\`,

$$D\le \sum_i \Big(G_{ii}+\sum_{j\ne i}|G_{ij}|\Big)e_i^2=D_{\rm maj}.$$

The majorizer is safe but may be loose. For \`G=[[1,0.8],[0.8,1]]\`, \`e=(1,1)\`, true D=3.6 while diagonal gives 2.0. This constructed counterexample is not a real model task-error gap. Source: \`docs/EXACT_RESULTS.md\` E2, \`src/fqc/coupling.py\`; F07.

### IF08 — exact serialized bytes outrank raw bit formulas

A toy codec using \`S_serialized=8 ceil((S_raw+5)/8)\` bits sends a raw 76-bit candidate to **88 serialized bits**. Under an 80-bit hard budget, \`76≤80\` is a **false admission**. Headers, padding, map descriptions, decoder prerequisites and compression framing must be paid once according to physical file format, not estimated after hypothesis selection. Source: \`docs/EXACT_RESULTS.md\` E1, \`src/fqc/pareto.py\`; F08.

### IF09 — safe quotient requires decoder prerequisites

Two optimizer states with identical decoded behavior cannot necessarily be merged if they have different **future decoder prerequisites**. Key the quotient by the pair \`(decoded semantics, prerequisite closure)\`. Under identical key and loss contract, keep the lower charged-bit representative; otherwise preserve both candidates. Counterexample: decoded state A requires future atom α and decoded state B requires atom β. Merging by decoded state alone deletes a potentially distinct executable state. Source: \`src/fqc/quotient.py\`, E3; F09.

### IF10 — jointly optimize structure × precision × private state

In small exact optimizer states, a nonseparable objective can make *each single-axis change look harmful* while a combined bundle wins. An explicit constructed table from the incumbent is \`D(0,0)=10\`, \`D(1,0)=12\`, \`D(0,1)=11\`, \`D(1,1)=5\` at equal paid bytes: one-axis greedy stops at 10, exact joint search returns 5. E7 gives a **different, empirically enumerated toy** with 6.40% true distortion improvement at the same hard budget. This motivates cross-over MA-1171, but does **not** prove Mirror geometry has improved task capacity. Source: \`docs/EXACT_RESULTS.md\` E7; \`src/fqc/joint_codec.py\`; F10.

### IF11 — function-sensitive local error, not raw weight error

For a frozen linear interface and \`E_f=ΔW−ΔW_hat\` with uncentered second moment \`Σ_h=\mathbb E[hhᵀ]\`,

$$\mathbb E_h\|E_fh\|_2^2
=\operatorname{tr}(E_f\Sigma_h E_f^\top)
=\|E_f\Sigma_h^{1/2}\|_F^2.$$

Proof: expand squared Euclidean norm as \`hᵀE_fᵀE_fh\`, use \`tr(uvᵀ)=vᵀu\`, linearity of expectation and cyclic trace. A small-weight-Frobenius error can have **larger** functional error if it lies in a frequent activation direction. This is exact only for the linear interface on the chosen probe distribution; nonlinear downstream NLL/KL still requires independent testing. Source: \`docs/phase2/MIRROR_FUNCTION_SPACE_FALSIFICATION_2026-10-08.md\`, \`src/fqc/functional_sensitivity.py\`; F11.

### IF12 — order-sensitive rules require composition, not summed residuals

For two residual operators A and B,

$$(I+A)(I+B)-(I+B)(I+A)=AB-BA.$$

Proof: expand both products and cancel I, A, B. If the commutator is nonzero, the sequence is order-dependent. A parallel summed residual \`I+A+B\` omits \`AB\`; it does **not** implement a two-step rule executor. SRM002 succeeded with an externally supplied sequential executor; SRM003 showed that a one-forward causal decoder did not automatically discover that algorithm. This is an architectural/credit-assignment boundary, not proof that arbitrary programs are efficiently compositional. Source: \`docs/phase2/SRM002_NONCOMMUTATIVE_COMPOSITION.md\`, \`docs/phase2/SRM003_CAUSAL_DISCOVERY.md\`; F12.

## 3. Empirical constraints transferred with the formulas

| Source experiment | Verified *bounded* result | Required control in combined experiment |
|---|---|---|
| E1/E2/E7 | True serialization, coupled quality and joint layout/precision can change an exact toy optimum | Real serializer, diagonal-vs-coupled error and coordinate-only optimizer |
| SM002 and MS003 | Fixed assigned shear can beat some equal-byte baselines at fixed updates; folded fixed View can be materially faster but consumes deployed RAM | Equal-wall controls, cached/expanded weights, no claim for arbitrary experts |
| MS008–MS012 | Calibration-first/no sensor-ID core improves changed-law transfer; residual View helps only above genuine mismatch and sometimes slows CPU | No-ID native calibrated core; optional View gate vs always-on and FiLM |
| SRM001 | Multi-rule selection succeeds on decomposable synthetic counts, but near-convergence native MoE still improved and parity remained difficult | Native dense/top-k MoE, private-rule load, exact rule retention, independent compute |
| SRM002 | Signed shared operator basis can beat richer Mirror in controlled ordered-task teacher; oracle executor matters | Native signed basis and explicit executor |
| SRM003 | The fixed-depth two-forward external composition recovers 100% on the tested atom-trained synthetic rules while one-forward pair accuracy stays low | Two-forward and one-forward matched-compute controls, no teacher intermediate state |
| TM001 | Parallel deterministic/revealed token packet succeeds, hidden packet branch fails without shared sampled mode | Native AR with KV, native PTP, packet-level shared latent, valid-path and joint NLL |
| MA-253 | Small Mirror-only coordinate fails on independent rank-2 private LoRA teachers | Independent LoRA and private residual rank sweep, aligned/misaligned mix |
| MA-691 | Exact read-only canonical-cache transform is possible under stated algebra | Physical pointer/VRAM alias, incompatible prefix, prefilling cost |
| Real-digit function pilot | Mirror geometry can win over diagonal codes yet lose to dense core and native LoRA at higher runtime | Native BOLT, dense CtS, independent LoRA, actual task accuracy |
| MS005 | ES view averaging has no free fixed-budget variance gain | Equal independent samples, equal total branches, native backprop |

## 4. Definitions of the evidence categories

- **THEOREM / IDENTITY:** algebra valid only under the stated rank, invertibility, PSD or independence assumptions.
- **CONTROLLED FACT:** repository experiment on named finite toy/data/hardware and frozen seeds; it does not automatically generalize to large models.
- **INFERENCE / DESIGN:** using an equation to decide what to benchmark next.
- **UNTESTED:** physically useful Mirror m benefit in a new target method until frozen paired experiment plus strong native controls passes.

## 5. Interdisciplinary transfer map

- **Numerical linear algebra / information geometry:** IF01–IF04 and IF11 -> adaptors, attention/gauge, source-only Jacobian diagnosis.
- **Coding / combinatorial optimization / information theory:** IF07–IF10 -> actual serialized optimizer, decoder DAG, private residual allocation, hard budget.
- **Dynamical systems / causal sequence modeling:** IF05/IF12 -> token-phase protocols, rule execution, packet latent and hidden-state consistency.
- **Systems engineering / computer architecture:** IF06/IF08 -> true KV allocation, folding, stored-bank bytes, active compute and kernel latency.

A formula is only integrated successfully when the target-native baseline and a strictly simpler same-byte non-Mirror control are both beaten on a predeclared practical frontier.
