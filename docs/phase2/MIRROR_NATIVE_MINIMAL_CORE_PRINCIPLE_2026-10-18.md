# Mirror-native minimal shared-core principle — 2026-10-18

**Status: ARCHITECTURE HYPOTHESIS / RESEARCH OBJECTIVE. NOT an achieved Transformer replacement or an unlimited-capacity proof.**

Parent evidence: MA-1162 (causal Mirror heads), MA-691 (exact compatible KV-cache read), MA-253 (invalid cross-layer cache sharing counterexample), MA-1163/1164 (real MLP comparison and negative Mirror-specific adoption results), MA-1156..1159 (nonlinear Mirror ideas). This document is an addendum to the existing doctrine, not a revision of historic evidence.

## 1. First-principles target

Build the smallest useful set of operators from which a sequence network can be composed:

1. **One principal shared physical parameter object:** a learned vector or structured tensor S with m independently updated scalar entries. This is *one physical object*, **not one mathematical degree of freedom**. Embeddings, output heads, normalization coefficients, router, Mirror codes, private deltas and numerical precision must all be counted if stored or learned.
2. **An addressed View/Mirror program:** a low-description role code changes how the same physical information is interpreted and operated upon. The View may include reversible coordinate transforms, nonlinear activation between transforms, structured addressing and program composition.
3. **Horizontal composition:** evaluate K distinct Views on the same state in parallel, then combine their useful outputs. Distinct address values are not automatically distinct useful functions; count head diversity/ablation, function distance and extra active attention FLOPs.
4. **Vertical composition:** reuse the *same physical core* on its newly updated state for R sequential steps. The physical weights need not be copied, but the computation happens again. A sequence of identical orthogonal-ReLU sandwiches can degenerate; residuals, changed coordinates or mixing must be tested.
5. **Explicit information/state semantics:** self-attention, cross-attention, masked decoding, retrieval and an FFN all use a shared execution primitive with role-specific *source, target, mask and state* contracts. It is not enough to apply channel Mirrors without actual token mixing.

The research question is **whether a larger m and a small code/program budget permit more useful functions at competitive serialized bytes, end-to-end quality and runtime**. Do not assume a Transformer is globally optimal; it is a concrete first comparison target. Non-attention mixers/scan/RNN are separate future controls.

## 2. Functional specification (without forced materialization of virtual weights)

For role ρ, step ℓ and logical head j, define one function from the common S and a paid role address:

\[
 \mathcal V_{\rho,\ell,j}(z;S,\mu)
 =Q_{\rho,\ell,j}^{-1}
    \phi_{\eta_{\rho,\ell,j}}\!\left(
      Q_{\rho,\ell,j}\,
      G_{\rho,\ell,j}(z;S,\mu)
    \right).
\]

G is an on-demand shared-core operator (for example structured matrix-vector multiplication, source/key/value projection, or a learned shared nonlinear map); it must not quietly introduce independent full-size role-specific parameters. The Q matrices may be fixed/structured or learned via paid generator coefficients. Distinct left/right Views and non-invertible activations are allowed when clearly modeled; the displayed sandwich is a **minimal example**, not the only permitted Mirror family.

A minimal Attention port has two streams, queries X and memory Y:

\[
\operatorname{A}_{\rho,j}(X,Y;S)
=\operatorname{softmax}\!\left(
\frac{\mathcal Q_{\rho,j}(X;S)
       \mathcal K_{\rho,j}(Y;S)^{\mathsf T}}{\sqrt{d_h}}+M_\rho
\right)\mathcal U_{\rho,j}(Y;S).
\]

- Encoder self-attention: X=Y=encoder state; bidirectional/padding mask.
- Decoder self-attention: X=Y=decoder state; strict causal/padding mask.
- Cross-attention: X=decoder state, Y=encoder output or external memory; source availability and padding mask required.
- Multihead: several j on one source state, independent View codes, a paid shared or specialized output mixer.
- Mirror-FFN/MoE: a separate role port using the same S and nonlinear View programs; routing and real active-K cost are paid.
- Recurrent depth: output of step ℓ becomes input to ℓ+1; evaluate the shared core again. Intermediate heads/readouts may emit output without halting.

### Variable table (all displayed equations)

| Symbol | Meaning (Japanese) / definition | SI unit | Domain / conditions | Type |
|---|---|---|---|---|
| S | 共有する物理パラメータ; m個の成分を持つ | 1 | S∈R^m; precision separately fixed | real vector / tensor |
| m | 物理的に独立更新できるスカラー成分数 | 1 | integer >=1 | integer scalar |
| μ | 実行されるMirror/役割アドレス | 1 | discrete/code or generated from paid state | code / vector |
| ρ | 演算の役割（encoder-self, decoder-self, cross, FFN 等） | 1 | finite role set | discrete value |
| ℓ,j | 反復深さとヘッド識別子 | 1 | bounded nonnegative integers | integer scalars |
| z | View演算の入力状態 | 1 | real d-vector, normalized feature convention | vector |
| G | Sを利用する共通演算 | 1 | compatible real-dimension input/output | operator |
| Q | 役割別の可逆Mirror座標変換 | 1 | d×d invertible; generally structured | matrix |
| η | 活性化関数の形状指定（固定または有料の係数） | 1 | explicit bounded code | scalar / vector |
| φ | 要素ごとの非線形活性化 | 1 | real vector to equal-shape vector | map |
| V with calligraphic symbol | 非線形Mirror View演算 | 1 | output d-vector | map |
| X | Query側の位置列 | 1 | Lq×d_input | matrix |
| Y | Memory/Key/Value側の位置列 | 1 | Lm×d_input | matrix |
| mathcal Q,K,U | 共有核由来のQuery/Key/Value射影 | 1 | outputs Lq×dh, Lm×dh, Lm×dv | maps |
| M_ρ | 役割別Attentionマスク | 1 | Lq×Lm; allowed=0, blocked=-∞ | matrix |
| d_h | head Key/Query次元 | 1 | positive integer | integer scalar |
| A with calligraphic symbol | Attentionによる出力演算 | 1 | Lq×dv | map |
| Lq,Lm,d,d_input,dv | query長、memory長、内部幅、入力幅、Value幅 | 1 | positive integers | integer scalars |

**Dimension/unit check:** (Lq×dh)(dh×Lm) gives Lq×Lm, and multiplication with Lm×dv yields Lq×dv. The Mirror sandwich returns d-vector after d×d transforms; all model-feature dimensions and matrices are dimensionless under the normalized neural-state convention. If physical sensor units are present, preprocess and track units separately.

## 3. Algebraic opportunities AND non-negotiable boundaries

**Exact cache-compatible right actions.** If all Views consume exactly the same historical hidden state and produce cache variants by right actions, then for row Query q:

\[
 K_j=K_0 A_j,\quad V_j=V_0 B_j,\qquad
 \operatorname{softmax}\!\left(\frac{qK_j^\top}{\sqrt{d_h}}\right)V_j
 =\left[
 \operatorname{softmax}\!\left(\frac{(qA_j^\top)K_0^\top}{\sqrt{d_h}}\right)V_0
 \right]B_j.
\]

Here K0,V0 are common prefix cache matrices of shape T×dh, A_j and B_j are compatible right-side transforms, q is 1×dh, and T is cached length (dimensionless); K_j,V_j are T×dh. In the Value term the common output has matching dh for this simplified illustration. The multiplication and the unit check are shape-consistent: (1×dh)(dh×T)(T×dh) -> 1×dh. The identity is distributivity/associativity plus transpose; it does not prove that arbitrarily different hidden histories can share cache. RoPE position placement and producer version/prefix provenance must be checked (MA-691 / MA-253). Fixed A/B can also be folded into ordinary native Query/output projections: **that algebraically equivalent native method is a required control**.

**Parallel/serial boundary.** A shared physical core is not free reuse of *compute*. K heads can run in parallel for the same H but still incur K separate attention score and value reads. R depth recurrences read a changed H and must normally execute serially. Self-attention and FFN inside one block may overlap only if the architecture and dependency graph permit it. Cross-attention on encoder outputs cannot run before those outputs exist. Encoder and decoder may share S and perform independent work concurrently, but cross-attention introduces a true source-to-target dependency.

**No infinite independent Attention.** Unbounded addresses or recursive program length do not imply unbounded independent learned capabilities at finite bytes or finite compute. For a differentiable decoder W_virt=f(S)∈R^N its Jacobian J∈R^(N×m) satisfies rank(J)<=m because an N×m matrix has at most m independent columns. More nonlinear levels can increase reachable global function diversity, not local independent parameter-update rank. This holds even with high precision when the **number of continuously optimized scalar coordinates remains m**.

**Information lower bound.** If N independent weights each have q stored bits and are unconstrained, there are 2^(Nq) possible models. A format with p bits of physical state and c bits of program/model-specific code encodes at most 2^(p+c) possibilities. If p+c<Nq it cannot losslessly cover *every* model (pigeonhole principle). Structured/redundant families may still compress very well. This statement counts all paid side information and excludes arbitrary external oracles.

For these last two arguments: N=virtual weight count (integer), J=Jacobian (dimensionless matrix), q=bits per weight (count, dimensionless), p=physical-state stored bit count (count), c=program/metadata stored bit count (count), and f=deterministic decoder (map). The linear algebra and bit counts are dimensionless; fixed scalar precision must not be confused with increasing m.

## 4. Three independent scales

| Axis | Meaning | How to isolate |
|---|---|---|
| m (physical dimension) | Independently optimized shared vector components | m=32/128/512 first; hold numerical precision, K and R fixed |
| K (horizontal View heads / roles) | Number of distinct executed logical functions | K=1/2/4/8, hold m and R fixed; ablate roles and count FLOPs |
| R (vertical recurrent uses) | Sequential computational depth | R=1/2/4, hold m/K fixed; measure stability and wall time |
| b (numerical precision, separate axis) | Bits for each scalar | FP32/FP64 controlled; FP8000+ only in separate arbitrary-precision experiment |
| P (output/halting schedule) | Intermediate readout and early-exit behavior | Keep fixed until sequence-quality gates pass |

A viable Pareto result requires held-out functional quality, actual total serialized inference bytes, cache/state RAM, measured wall time, active compute, and stability. Virtual logical weight count and K×R are **not** extra independent learned scalar dimensions.

## 5. Exact evidence boundary and related work

**Already established (scoped):**
- MA-1162 L0: causal distinct Views, QK gauge null and gradients; the later aligned multihead teacher code experiment reports a 794-byte compact payload in its restricted family, but unrelated dense off-orbit heads FAILED. That is not general MHA compression.
- MA-691: exact algebraic lazy right-transformed KV View (under compatible shared provenance), not arbitrary hidden-state cache transfer.
- MA-253: final placement can preserve K/V; earlier View changed later K/V.
- MA-1163/1164: full-dimensional nonlinear Mirror decoder can reconstruct a pretrained MLP; low-dimensional Mirror was generally no better than simple orthogonal/low-rank/native controls at matched bytes; generated-weight representation did not yet dominate runtime.
- MA-1164 CPU fusion: on-demand structured Mirror can avoid full matrix materialization and outperform Python eager implementation in tested CPU cases; no natural language/GPU benefit follows.

**Not established:** Encoder+Decoder+Cross-Attention quality parity from one S; arbitrary MLP/MoE functions per code; independent head capacity gain; one forward pass giving both encoder and decoder outputs without dependency work; unlimited heads; unconditional cache reuse; performance scaling in numeric bit-width; global superiority to Transformers.

**Direct literature controls, not new Mirror contributions:**
- Transformer encoder–decoder & cross-attention: https://arxiv.org/abs/1706.03762
- Universal Transformer weight-tied recurrence: https://arxiv.org/abs/1807.03819
- ALBERT cross-layer parameter sharing: https://arxiv.org/abs/1909.11942
- Deep Equilibrium Models / implicit recurrence: https://arxiv.org/abs/1909.01377
- GQA native KV sharing: https://doi.org/10.18653/v1/2023.emnlp-main.298
- MLA latent caching: https://arxiv.org/abs/2405.04434

## 6. Research doctrine for this direction

Do **not** claim success merely because one scalar/object can *describe* a virtual network. The kernel/source decoder can hide uncharged data. Physical vector dimension, Mirror/activation/program bytes, optimizer state, recurrent compute, attention memory, cache source/provenance, and actual model function must be measured.

The core first-principles hypotheses, controls, priorities and exact next steps are in [the MA-1162 execution roadmap](../../experiments/mirror_applications/ma-1162-mirror-native-sequence/MINIMAL_CORE_EXECUTION_ROADMAP_20261018.md). New MA IDs are NOT allocated by this design note. Formal experiment statuses are unchanged.

**H:** one shared m-vector plus small function-bearing Mirror programs can jointly realize useful Attention, nonlinear FFN/MoE and cross-source roles at competitive byte–compute–quality frontier.  
**T:** first verify causal masks, true cross-source dependencies, functional non-equivalence and cache correctness; then train and compare paired fresh worlds.  
**D:** promote only against equal-bit native tied/GQA/FiLM/low-rank and quality-ceiling independent models, at compute-matched budgets.  
**C:** simple tied/depth-conditioned/low-rank models can express the same functions more cheaply and faster.  
**U:** generalization beyond aligned teachers, learned routing, decoder code costs, native kernel latency, and long-context state drift remain unknown.
