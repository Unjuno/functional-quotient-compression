# Mirror KV / MoE Generalization Notes

Date: 2026-10-07 JST
Status: research support for follow-up workers
Scope: algebraic design space and systems hypotheses. Only MA-691 exact lazy-read mechanism is experimentally verified here; the extensions below are hypotheses until separately tested.

## 1. Exact affine canonical-cache theorem

Let the physical canonical cache be

- K in R^(T x d_k)
- V in R^(T x d_v)

and logical View m use

K_m = K A_m + 1 a_m^T
V_m = V B_m + 1 b_m^T

where

- A_m in R^(d_k x d'_k)
- B_m in R^(d_v x d'_v)

may be rectangular and do not need to be invertible.

For logical query q_m in R^(d'_k), with standard scale 1/sqrt(d'_k), define the canonical query

q_c = sqrt(d_k / d'_k) * q_m A_m^T.

Then

softmax(q_m K_m^T / sqrt(d'_k)) V_m
=
softmax(q_c K^T / sqrt(d_k)) V B_m + b_m^T.

Reason:
- the key bias contributes the same scalar q_m a_m to every cached position, so softmax removes it;
- attention weights sum to one, so a constant value bias becomes an output bias;
- the linear key/value maps can be moved to current-query and post-attention computation.

Consequences:
- exact zero-copy cache reuse is broader than rotations;
- logical head dimension may differ from canonical cache dimension;
- MLA/Palu-style latent caches are special cases;
- per-View key/value bias does not require per-View cache storage.

## 2. Score-invariant View theorem

For invertible C_m, choose

q_m = q C_m
K_m = K C_m^(-T).

Then

q_m K_m^T = q K^T.

The attention score matrix and softmax weights are therefore identical across Views.

If additionally

V_m = V B_m + 1 b_m^T,

compute once:

alpha = softmax(q K^T / sqrt(d))
y = alpha V.

Each logical View is then only

y_m = y B_m + b_m^T.

This is stronger than cache sharing:
**the expensive QK score/softmax/value aggregation can be shared across logical Views.**

A useful family is:
- shared retrieval pattern;
- View-specific interpretation/output transform.

Do not count a pure q/K gauge with compensated output as new functional multiplicity if the end-to-end function is unchanged.

## 3. Score-affine Views

A broader cheap family satisfies

S_m = lambda_m S + row_constant,

where S = q K^T.

The row constant vanishes under softmax and lambda_m is only a temperature change. Such Views can share the QK dot-product matrix even though their attention distribution differs.

Candidate coordinates:
- per-head temperature;
- additive positional/bias terms;
- low-dimensional score-bias basis.

This should be compared with ordinary attention-temperature/bias mechanisms before any Mirror-specific claim.

## 4. Orbit + private cache residual

Natural specialists will not generally satisfy the exact orbit.

Use

K_m = K A_m + 1 a_m^T + E^K_m
V_m = V B_m + 1 b_m^T + E^V_m.

Do not jump directly from exact reuse to full independent caches.

Test a frontier:
1. exact View only;
2. low-rank per-token residual;
3. sparse-token residual;
4. critical-layer residual/recompute;
5. full independent cache.

Storage can scale like O(T r) per View instead of O(T d) when residual rank r << d.

MiniCache suggests retaining exceptional tokens separately. CacheBlend/KVShare suggest recomputing only high-deviation/high-impact tokens. DroidSpeak suggests contiguous critical-layer recomputation. These are complementary dimensions.

## 5. Quantization gauge x functional View

Separate two coordinates:

g = function-preserving numerical gauge
m = functional logical View.

Canonical cache pipeline:

hidden
 -> quantization-friendly gauge g
 -> one low-bit canonical cache
 -> lazy logical View m at read time.

QuaRot/SpinQuant show that rotations can improve low-bit KV quantization without changing the full-precision function. This should not be counted as functional multiplicity, but it can multiply the storage benefit of canonical cache sharing.

Important comparisons:
- quantize canonical -> lazy View;
- materialize View -> quantize;
- shared gauge for all Views;
- independently optimized gauge per View.

## 6. Canonical writer / specialist reader

Arbitrary earlier specialists normally invalidate downstream KV because they change the hidden state used to write future cache entries.

A structural solution is to separate:

- **canonical memory writer stream** — evolves independent of specialist identity and writes K/V;
- **specialist reader/output stream** — reads canonical memory through Mirror Views and produces task/expert-specific output.

This is analogous in spirit to YOCO/shared-cache and speculative-stream designs, but the View becomes the specialist read interface.

A specialist may be arbitrarily rich on the read/output side without invalidating old cache as long as it does not feed back into the canonical writer. If feedback is needed, test explicit rebase/projection into canonical state.

## 7. Mirror-MoE compute factorization

Suppose logical experts share expensive FFN matrices and differ only in a cheap nonlinear View:

E_m(x) = W2 psi_m(W1 x).

For top-k routing weights a_m,

sum_m a_m E_m(x)
=
W2 [ sum_m a_m psi_m(W1 x) ].

Therefore the exact routed result can use:

1. h = W1 x once;
2. evaluate k cheap View/nonlinear maps psi_m(h);
3. weighted sum in FFN hidden space;
4. W2 once.

Standard top-k independent MoE normally pays k expert W1/W2 matrix multiplies.

For GEMM-dominated FFNs this creates a potentially large **active-compute** advantage in addition to parameter storage reduction. Runtime must be measured; k View transforms and activations are not free.

This factorization is especially natural for:
- Mirror activation/conjugation Views;
- IA3/diagonal Views;
- rank-one / BatchEnsemble-style Views;
- BOFT/ACDC/Hadamard transforms;
- shared low-rank atom mixtures.

## 8. Distributed-MoE implication

Standard expert parallelism routes token activations to GPUs that own selected experts and routes outputs back. Recent systems report all-to-all and expert-weight movement as major bottlenecks.

If logical experts are:
- one/few shared physical experts resident locally;
- plus tiny View codes replicated on every worker,

then logical routing need not imply physical expert routing.

Potential architecture:

router selects logical expert IDs
 -> load tiny local View codes
 -> shared local W1/W2
 -> View-space specialization
 -> no expert-owner all-to-all.

If a small fraction of behavior requires truly private residual experts, route only those residual computations remotely.

This yields an **orbit/private communication frontier**:
- aligned View: zero remote expert traffic;
- partially private: residual-only traffic;
- fully independent: standard expert parallelism.

## 9. HBM / expert-weight bandwidth implication

MoE-Infinity and Fiddler optimize movement of full expert weights under memory pressure.

Mirror-MoE changes the object being cached:
- keep shared base expert weights resident;
- cache/stream tiny View codes and rare private residuals;
- optionally materialize hot logical experts only when dwell/reuse justifies it.

Measure:
- expert-weight HBM bytes read/token;
- PCIe/NVLink/network bytes/token;
- View-code bytes/token;
- materialization cost;
- hot/cold logical expert dwell time.

## 10. Serving policy: lazy vs materialized

MA-691 measured a prefix-length-dependent materialization cost and a small per-token lazy overhead.

For each View estimate

L_break_even = C_materialize / (C_lazy - C_materialized_read).

Use:
- lazy read for rapidly switching Views;
- materialized transformed blocks for long-lived/hot Views;
- a small memo cache for recently materialized (block, View) pairs.

The policy should be workload-driven, not fixed.

## 11. Physical block aliasing

If the physical cached representation is truly View-invariant, the serving cache key should not include logical View/adapter identity.

For a vLLM/PagedAttention-style implementation, hash:
- token-prefix identity;
- canonical model/cache ABI identity;
- quantization/cache-layout identity;

but not the logical View ID.

The View code belongs to read-time metadata.

This is the systems counterpart of the algebraic result and should be tested for actual HBM aliasing, not simulated by copying identical tensors.

## 12. Canonical KV ABI for a model family

Cross-model cache-transfer work learns post-hoc translators.

A stronger co-designed architecture is:

shared canonical latent/cache ABI
  <- write map for model/view A
  <- write map for model/view B
  <- write map for model/view C

and

canonical cache
  -> read map for A/B/C.

Possible uses:
- small->large model escalation without re-prefill;
- draft/verifier cache sharing;
- specialist/agent switching;
- multimodel latent communication;
- one shared soft-prompt/prefix-cache library.

Exact-by-construction should be compared against CacheBridge/C2C/XKV and direct cache reuse.

## 13. Recommended follow-up order

Do not interrupt an already-running MA experiment.

High-information next tests after the existing MA-692..700 queue:

1. score-invariant one-attention/many-View mechanism;
2. affine/rectangular exact lazy-cache theorem;
3. exact View + low-rank/sparse private KV residual frontier;
4. quantization gauge x functional View;
5. physical PagedAttention block aliasing;
6. canonical writer / specialist reader;
7. shared-GEMM top-k Mirror-MoE;
8. residual-only distributed expert routing;
9. canonical KV ABI across small/large models.

## 14. Boundaries

- Algebraic cache reuse does not prove natural specialists lie on the orbit.
- A function-preserving gauge does not count as new functional capacity.
- Shared cache does not make top-k attention compute free.
- Shared expert weights do not guarantee expert specialization.
- CPU eager microbenchmarks are not GPU serving results.
- All-to-all elimination requires a distributed implementation or faithful communication model before a systems claim.
