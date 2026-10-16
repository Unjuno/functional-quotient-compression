# Mirror research support: share computation, not only parameters

Date: 2026-10-07. Scope: literature synthesis, exact algebra, numerical falsifiers, worker support. **Zero trained language models and zero throughput benchmarks in this support package.**

## Worker brief: read this first

The next opportunity is to separate **parameter sharing**, **intermediate-computation sharing**, **cache sharing**, and **attention-map sharing**. They have different conditions. Do not infer one from another.

1. In a shared-up/shared-down Mirror FFN, evaluate the input projection once and move the expert weighted sum before the shared output projection. Arbitrary nonlinear Views still run separately; this is NOT averaging their transformation codes.
2. For sign-conjugate GELU/ReLU/SiLU Views there is a stronger exact fusion: one activation evaluation plus a channel gate and linear bypass suffices. The matching gate/bypass control is mathematically equivalent, so this is a compute optimization, not evidence of extra independent capacity.
3. Equal averaging of antipodal transforms Q and -Q cancels this branch's nonlinear response. More Views and more averaging can destroy the feature being sought.
4. MA-691 already established the square right-transform cache identity. Do not repeat it as a new discovery. The next questions are language quality, source-token-varying routing, rectangular latent temperature, and real shared-prefix kernels.
5. A common KV cache does not imply a common softmax map. Values-only read transformations can share one attention aggregation when the query/key scores are identical. Different query/key Views generally require separate score maps.
6. Preserve source-token View history. Changing the next token's FFN expert in a standard MoE is normal; reinterpreting the entire old prefix as a different globally conditioned model is a different operation.
7. Check dedicated branches, not just the shared status board. At the inspected snapshots the worker had completed MA-248..251 while the shared baseline still said MA-248 was next. Do not reset any run, parent, seeds, optimizer or acceptance gate.

## Current worker evidence, as reported, not rerun here

Shared baseline inspected: `0f7f3c2a6f5c684263ea2724b50f99bfae292e30` (700 registered MA candidates).

The dedicated `research/ma-251-expert-depth-factorization-20261007` status board at `4ddc2962b53dc621ec6ce30758a3cf527aaa2c18` reports completion through MA-251 and MA-003 next. MA-248 is reported FAIL versus a cheaper broadcast packet code; MA-249, MA-250 and MA-251 are reported PROMISING in aligned synthetic families, with CPU regressions. These are not natural-language results and were not re-executed in this support pass.

MA-691 branch: `research/ma-691-lazy-kv-mirror-20261007`, inspected head `7312cc2713838ab15b0f8b590b2e9be31c8e818a`. It reports exact lazy KV and RoPE/MLA diagnostics; its source-dependent runtime crossover is not a GPU prediction.

**Coordination action:** new proposals in this package use CR subtest IDs linked to existing MA IDs. No new global MA IDs are reserved, no existing registry rows/statuses are reset, and no active protocol is changed. A worker may promote a genuinely distinct proposal after checking the live registry. Source references use paper identifiers rather than mutable PA numbers.

## Definitions and units

All learned representations below are dimensionless (SI unit 1); memory is measured in bytes (B), not an SI base unit. Column vectors are used in the derivations; code stores vectors as rows.

| Symbol | Meaning / 日本語 | Unit | Definition / domain and assumptions | Type |
|---|---|---|---|---|
| d,h,o,T,E | input, hidden, output, cached-token and active-View counts / 各次元・個数 | 1 | positive integers | integer scalar |
| x,v | input and shared preactivation / 入力・共有活性化前状態 | 1 | x in R^d; v=U x+b in R^h | vectors |
| U,D,b,c | shared up/down weights and biases / 共通射影・バイアス | 1 | U in R^(h*d), D in R^(o*h), b in R^h, c in R^o | matrices/vectors |
| phi,Phi | elementwise activation and normal CDF / 活性化・正規累積分布 | 1 | phi acts on R^h; Phi is the standard normal cumulative distribution | functions |
| t | scalar activation argument / 活性化の引数 | 1 | real | scalar |
| Q_e,S_e,I | View transform, diagonal sign matrix, identity / View・符号・単位行列 | 1 | invertible Q_e in R^(h*h); S_e=diag(s_e), s_e in {-1,+1}^h; I is the identity of the dimension of its expression | matrices |
| a_e,beta,G | mixture coefficient, coefficient sum, effective diagonal gate / 混合係数・総和・ゲート | 1 | a_e may depend on x; beta=sum_e a_e; G=sum_e a_e S_e | scalars/matrix |
| Psi_e,F_e | conjugated activation and expert function / 共役活性化・expert | 1 | Psi_e(v)=Q_e^-1 phi(Q_e v), F_e(x)=D Psi_e(v)+c | functions |
| q,K,V,A_e,B_e,p_e,y_e | attention query, caches, read transforms, weights, output / Attention変数 | 1 | row q in R^(1*d_k), K in R^(T*d_k), V in R^(T*d_v); square A_e,B_e for this identity; p_e in R^(1*T); y_e in R^(1*d_v) | vectors/matrices |
| d_k,d_v,d_c,C,L | logical key width, value width, latent width, canonical latent, key up-projection / KV・潜在次元 | 1 | d_k,d_v,d_c positive; C in R^(T*d_c), L in R^(d_c*d_k); K=C L | integer scalars/matrices |
| H,P,W,A | common hidden states and source/target projections / 共通状態と射影 | 1 | H in R^(T*d), P in R^(d*r), W in R^(d*s), A in R^(r*s) | matrices |
| r,s,P^dagger,R,Z,B | source/target widths, pseudoinverse, residual, sidecar basis/coefficient / 補助基底 | 1 | r,s positive; R=(I-P P^dagger)W; R=Z B | scalars/matrices |
| z,w | null-space witness / 情報欠落の証人 | 1 | z in R^d; w target output vector | vectors |
| l_1,l_2,y_1,y_2 | partition log-normalizers and attention outputs / 分割正規化量 | 1 (log values in nat) | l_i=logsumexp(scores_i); y_i=softmax(scores_i)V_i | scalars/vectors |

## Proposition A: shared projections need not run for every expert

Starting from the definition of F_e, distribute the sum:

    sum_e a_e F_e(x)
      = sum_e a_e [D Psi_e(Ux+b)+c]
      = D [sum_e a_e Psi_e(v)] + beta*c,  v=Ux+b.

The second equality uses only linearity of D; it does not move a sum through phi. Thus U is computed once for identical input x, all needed Psi_e are computed, their hidden outputs are accumulated, and D is computed once. This remains valid with input-dependent router coefficients and has the same derivatives wherever those coefficients/activations are differentiable.

Matrix dimensions: Ux+b is h-dimensional; each Psi_e output remains h-dimensional; D maps their weighted sum to o dimensions. All additions are dimensionally valid and dimensionless.

Boundary: expert-specific up/down matrices, post-projection nonlinear branches or independent dropout can invalidate this rewrite. Dense Q_e applications may outweigh the saved projections. Low-description codes do not automatically imply cheap operators.

## Proposition B: sign-conjugate activation mixtures have an exact shortcut

For GELU, phi(t)=t Phi(t) [S01] and Phi(-t)=1-Phi(t). Therefore

    phi(-t) = -t[1-Phi(t)] = phi(t)-t,
    phi(t)-phi(-t)=t.

ReLU and SiLU obey the same difference identity. The tanh GELU approximation also obeys it in real arithmetic because its tanh argument is odd. This does not cover every activation.

For one coordinate of a sign View, the sign is either +1 or -1. The +1 case gives phi(t). The -1 case gives -phi(-t)=t-phi(t). Combining these two cases componentwise yields

    S_e phi(S_e v) = S_e phi(v) + (I-S_e)v/2.

Multiply by a_e and sum:

    sum_e a_e S_e phi(S_e v) = G phi(v) + (beta I-G)v/2.

Consequently, the full expert mixture becomes

    D [G phi(v) + (beta I-G)v/2] + beta*c.

For normalized coefficients beta=1. For signed/unnormalized coefficients the beta terms must remain. One evaluation of phi(v) is sufficient. The expert-dependent information is now an effective channel gate G.

**Interpretation:** this is exact computational factoring. It does not create E independent hidden nonlinear networks. The function class must be compared against the identical gated-activation-plus-linear-bypass model, as well as IA3/BatchEnsemble/shared-basis controls.

Fixed sign codes are assumed in the derivative tests; gradients through discrete code selection are not established. Input/router/up/down gradients are checked. Independent per-View dropout is a deliberate negative control.

## Proposition C: antipodal averaging can erase nonlinearity

For any invertible Q_e (not only orthogonal transforms), (-Q_e)^-1=-Q_e^-1. Hence

    [Q_e^-1 phi(Q_e v) + (-Q_e)^-1 phi(-Q_e v)] / 2
      = Q_e^-1 [phi(Q_e v)-phi(-Q_e v)] / 2
      = Q_e^-1 Q_e v / 2
      = v/2.

An equal pair therefore makes this expert branch affine: D(Ux+b)/2+c. A separate nonlinear backbone/residual remains nonlinear; the conclusion concerns this branch, not the whole Transformer. Unequal/data-dependent pair weights need not cancel.

Numerical example: in one dimension, GELU(1) is about 0.841344746 and -GELU(-1) about 0.158655254; their average is exactly 0.5 in real arithmetic.

A permutation conjugate is another null control: applying the same elementwise activation after a permutation and then undoing that permutation gives the original activation.

## Proposition D: cache reuse and attention-compute reuse differ

For row-stored states K_e=K A_e and V_e=V B_e, transpose the first product and reassociate:

    q K_e^T = q A_e^T K^T,
    y_e = softmax(q A_e^T K^T / sqrt(d_k)) V B_e.

This reuses canonical K,V exactly when the whole prefix truly satisfies the relation. It is the MA-691 primitive, not a new result of this pass.

If all Views use the SAME attention probabilities p, then

    sum_e a_e [p V B_e] = (p V) [sum_e a_e B_e].

One score map and one weighted-value aggregation suffice. Coefficients may vary with the current query. They must not vary across source tokens inside this factorization. If A_e or queries alter p_e, replacing the mixture of softmax outputs with one softmax of averaged keys is generally false. The supplied falsifier detects this.

Independent Q/K compensation can also be a gauge: q->qA and K->K A^-T leaves scores unchanged. Do not call that new retrieval behavior solely because the coordinates differ.

**Rectangular latent warning:** when K=C L, move L to q L^T but retain the scale 1/sqrt(d_k), not 1/sqrt(d_c). A default attention routine that infers temperature from the transformed query width can silently change the function. The test suite contains an exact check and a failing substitute.

## Proposition E: when no cache-only translator can exist

For a common hidden state H, consider old cache H P and desired cache H W. A token-independent linear translator works for every H exactly when W=P A for some A. Sufficiency follows by associativity: H W=(H P) A. For necessity, require the equality for each possible hidden-state row (equivalently stack basis rows as H=I_d); this yields W=P A.

Equivalently, every column of W must belong to the column space of P. If not, there exists a z with z^T P=0 but z^T W!=0. The two hidden states 0 and z^T then have the same old cache and different desired caches, so no deterministic translator of the old cache alone can be exact for both. A two-coordinate counterexample is included.

A sidecar can retain the missing directions. Write W=P A+R with R=(I-P P^dagger)W, and factor R=Z B. Cache H Z in addition to H P; then

    H W=(H P)A+(H Z)B.

The minimum extra linear feature dimension for arbitrary H equals rank(R): with any k-column sidecar Z', projecting W=P A+Z' B' orthogonally off col(P) gives R=(I-P P^dagger)Z' B', whose rank is at most k. Conversely, a rank factorization R=Z B supplies exactly rank(R) columns, attaining the bound. This statement assumes a common H; earlier nonlinear expert changes can make H itself View-dependent, which is a different and harder problem.

## Serving and compression implications from the literature

- aLoRA [S02] preserves the base prefix by activating the adapter only later. This is an architectural reuse guarantee, not arbitrary transport of an already-adapted prefix.
- Hydragen [S03] separates shared-prefix and private-suffix attention, batches queries and merges with log-normalization weights. A naive arithmetic mean of partition outputs is wrong. Its published speedups are NOT imported as Mirror speedups.
- SwitchHead [S04] makes attention projections conditional while reducing attention-map computation. It is a stronger control than comparing only to independent full heads.
- muMoE [S05] computes factorized linear expert mixtures without expanding all weights. It does NOT license commuting an arbitrary nonlinear expert mixture through an activation.
- UMoE [S06] shares expert components across attention and FFN; adding nonlinearity defines an architectural change, not a free algebraic equivalence to ordinary attention.
- xKV v2, revised 2026-05-27 [S07], adds selective reconstruction to shared-subspace KV compression. Test Mirror residual corrections in attention-output space, not only cache cosine similarity.
- FusedKV v3, revised 2026-02-19 [S08], distinguishes key and value sources across layers. Do not force identical sharing schemes for K and V without measurement.
- KIVI [S09] uses different quantization groupings for K and V. A shared compressed cache must be tested under every intended read View; stretch/shear can amplify quantization error. Quantize-then-transform and transform-then-quantize are generally different.
- S-LoRA [S10] and Punica [S11] are strong batching/serving controls. Weight residency sharing alone does not imply activation or KV identity.
- LayerSkip [S12] shares draft/verifier computation under its own training design. A Mirror draft with changed upstream hidden states cannot automatically reuse a target model's deeper cache.

## Numerical audit and limits

19 unit tests passed, including deliberate counterexamples. The separate replay uses 5 fixed diagnostic seeds and two precisions; 200 checks pass (180 equality checks and 20 deliberate-inequality checks). These seeds are numerical fixtures, not independent task-world replications.

Maximum equality discrepancy: float64 1.1102230246251565e-15; float32 6.258487701416016e-7. Maximum sign-fusion gradient discrepancy: float64 2.42861286636753e-17; float32 7.450580596923828e-9. The smallest deliberately incorrect key/temperature substitution differed by 0.05759285390377045.

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5; CPU, one thread, no CUDA. No model-training accuracy, storage-compression multiplier or runtime speedup is claimed. The branch/proposal descriptions are research hypotheses, not newly validated LLM capabilities.

## Reproduction and worker use

In the support directory:

    python -m unittest test_algebra -v
    python measure.py

Start with CR-01/02 for MA-003 and CR-05/06/07 for MA-699/694. Do not rerun completed MA-691 algebra as if it were a novel training result. Use SUBEXPERIMENTS.csv for controlled follow-ups and SOURCES.json for durable source identifiers. No external datasets or network access are needed for these checks.

## ERROR CHECK

Unit dimensions and matrix shapes checked; nonlinear sum/activation interchange rejected; source-token-dependent Views rejected from global pull-out; rectangular latent temperature checked; no independent-model capacity inferred from code counts; numerical exactness separated from task usefulness and GPU speed; worker branch divergence explicitly recorded.
