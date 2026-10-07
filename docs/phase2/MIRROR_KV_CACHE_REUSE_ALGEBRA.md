# Mirror KV Cache Reuse — algebra, compatibility classes, and experiment rules

Date: 2026-10-07 JST  
Status: active worker design note

This note generalizes the verified MA-691 primitive. It is an architecture/design guide, not a natural-language performance claim.

## 1. Core exact identity

Let a canonical cache contain

- (K \in \mathbb{R}^{T\times d_k})
- (V \in \mathbb{R}^{T\times d_v})

and let the current query be (q\in\mathbb{R}^{1\times d_k}).

Suppose logical View (m) is defined by position-independent right transforms

[
K_m = K A_m, \qquad V_m = V B_m,
]

where (A_m\in\mathbb{R}^{d_k\times d_k}) and (B_m\in\mathbb{R}^{d_v\times d_v}).

Then

[
q K_m^\top
= q (K A_m)^\top
= (q A_m^\top)K^\top
]

and

[
\operatorname{softmax}(qK_m^\top)V_m
=
\operatorname{softmax}((qA_m^\top)K^\top)V B_m.
]

Therefore the prefix cache does not need to be transformed or duplicated. Transform only:

1. the current query (q\mapsto qA_m^\top);
2. the attention result (o\mapsto oB_m).

### Important

This identity is **not restricted to Givens rotations or orthogonal matrices**. Any compatible linear (A_m,B_m) satisfy the algebra. Orthogonality/invertibility may still matter for conditioning, parameterization or model quality, but not for the cache-reuse identity itself.

## 2. Shared-latent exact generalization

A broader exact family is

[
K_m = K A_m + Z_K C_m,
\qquad
V_m = V B_m + Z_V D_m,
]

with shared token-side latents

- (Z_K\in\mathbb{R}^{T\times r_k})
- (Z_V\in\mathbb{R}^{T\times r_v})

and small View-specific reconstruction maps

- (C_m\in\mathbb{R}^{r_k\times d_k})
- (D_m\in\mathbb{R}^{r_v\times d_v}).

Then

[
qK_m^\top
=
(qA_m^\top)K^\top
+
(qC_m^\top)Z_K^\top.
]

The attention probabilities (P_m) are computed from the sum, and

[
P_mV_m
=
(P_mV)B_m + (P_mZ_V)D_m.
]

Thus one physical cache may consist of (K,V,Z_K,Z_V), independent of the number of logical Views except for small View codes (A_m,B_m,C_m,D_m).

This covers or connects to:
- MA-691 right-transform cache reuse;
- LRAgent shared base + LR cache;
- MLA latent cache;
- xKV shared token bases;
- shared-rule / atom composition.

It suggests that the right research question is not only "are View caches identical?" but:

> What is the minimum shared token-side latent dimension required to reconstruct the family of specialist caches?

## 3. Multiple shared cache atoms

More generally,

[
K_m = \sum_{j=1}^{J} Z^{K}_{j} C^{K}_{m,j},
\qquad
V_m = \sum_{j=1}^{J'} Z^{V}_{j} C^{V}_{m,j}.
]

If (J,J') are small, a large logical View bank can share the token-dependent state and store only compact View coefficients/reconstruction maps.

This is the KV analogue of sparse shared-rule MoE.

Do not count the number of possible coefficient combinations as independent capacity. Measure:
- attention/output quality;
- cache bytes;
- active cache atoms;
- read bandwidth;
- runtime.

## 4. RoPE compatibility

Assume row-vector notation and post-RoPE canonical keys

[
K^{\text{rope}}_t = K_t R_t.
]

If a View is applied before RoPE,

[
K^{\text{rope}}_{m,t}=K_t A_m R_t.
]

To express it as a fixed right transform of the already-RoPE'd canonical cache,

[
K_t A_m R_t = K_t R_t A_m
]

requires

[
A_mR_t = R_tA_m
]

for all relevant positions (t).

Thus the View must lie in the **commutant of the RoPE rotations**. Pairwise rotations confined to the same RoPE frequency planes are a natural exact family; arbitrary mixing between unequal-frequency planes generally is not.

Alternatives:
1. constrain the View to the commutant;
2. store pre-RoPE canonical content and apply both position and View lazily;
3. decompose a desired View into commuting exact part + compact residual correction.

## 5. Why an FFN/MoE switch usually invalidates later caches

For a residual update (\delta h_l) applied after attention in layer (l):

- layer (l)'s already-generated K/V are unchanged because they were produced before the FFN;
- the layer output changes;
- this changed residual stream enters layer (l+1);
- therefore future-layer K/V generally change.

At one layer, a sufficient local condition for unchanged projections is

[
\delta h\,W_K=0,
\qquad
\delta h\,W_V=0.
]

That is, (\delta h) lies in the intersection of the right null spaces relevant to the projections.

But this only guarantees local projection invariance. Nonlinear downstream layers can move the residual out of later null spaces.

Therefore deep cache-safe specialization needs one of:

1. final-layer-only placement;
2. exact transformable cache geometry propagated across layers;
3. a canonical cache-generating stream separated from specialist computation;
4. compact delta/repair state;
5. re-prefill.

## 6. Canonical-stream / specialist-stream architecture

A strong architecture-level design is:

```text
token
  |
  v
canonical stream -----------------> K/V cache
  |                                    |
  +---------- shared attention <-------+
  |
  +----> specialist Mirror side stream
             |
             +--> task/expert/logit residual
```

The canonical stream alone generates persistent K/V. Specialist state may read canonical memory but does not write into the state used to generate future canonical cache.

This is related to ICaRus and final-FFN DMoE but can be tested at finer granularity.

Tradeoff:
- exact cache sharing;
- potentially reduced specialist expressivity because specialist computation is prevented from modifying future memory.

The experiment must measure that quality cost.

## 7. Compatibility ladder

Classify every cache-related MA experiment before implementation.

### C0 — IDENTICAL
Every logical model uses identical physical K/V state.

Examples:
- ICaRus-style frozen cache encoder;
- final-layer specialist after the last cache-producing operation.

### C1 — EXACT-TRANSFORMABLE
Logical K/V are known transforms of canonical state and View is moved to read-side algebra.

Example: MA-691.

### C2 — SHARED-LATENT EXACT
Logical K/V reconstruct exactly from shared token-side latent state plus small View maps.

Examples:
- shared-A low-rank cache;
- trained canonical latent with View-specific reconstruction.

### C3 — CANONICAL + DELTA
Base state is shared; each View stores a compact residual sidecar.

Example:
- base K/V + low-rank adapter cache.

### C4 — REPAIR / APPROXIMATE
Reuse most cache and correct selected states/blocks using learned maps, low-rank corrections or recomputation.

Examples:
- KVCMAS;
- KVEraser;
- PatchKV;
- generic cross-model cache translators.

### C5 — INCOMPATIBLE
Specialist change produces no useful compact mapping; full logical cache or re-prefill is required.

Always test the highest exact class capable of expressing the target before using an approximate class.

## 8. Physical aliasing is separate from value reuse

A cache can be numerically reusable yet still occupy duplicate HBM pages.

For systems claims verify:
- allocation/page IDs or equivalent physical alias evidence;
- reference counts / copy-on-write behavior;
- cache hash/key semantics;
- peak HBM;
- TTFT;
- decode throughput.

"Values reused" is not enough for a memory-saving claim.

## 9. Switch economics

There are at least three execution modes:

1. **lazy** — leave prefix canonical and transform current read;
2. **materialized** — transform the whole cache once for a long-lived View;
3. **delta/repair** — share canonical state and materialize only residual/dirty pieces.

MA-691 showed a prefix-length-dependent materialization cost and a switch-duration break-even on CPU. A production scheduler should choose the mode based on:
- prefix length;
- expected remaining tokens under the View;
- View transform complexity;
- cache residency;
- GPU kernel characteristics;
- number of concurrent Views.

## 10. Multi-View attention can reuse memory bandwidth

For top-k logical Views over one canonical cache, stack transformed queries

[
Q'=[qA_1^\top;\dots;qA_k^\top]
]

and compute scores against one shared (K^\top) in a batched/fused kernel.

Even though arithmetic generally scales with (k), K/V need not be separately loaded (k) times from HBM if the kernel is designed to reuse tiles.

Special cheap cases:
- same K View, different V/output Views -> attention weights can be shared;
- same V, different K Views -> value storage shared but softmax differs;
- only output-side View -> one attention computation, several tiny transforms.

This is a systems opportunity distinct from storage compression.

## 11. Worker experiment ladder

For a real multi-LoRA/MoE target:

1. native independent cache;
2. unsafe direct canonical reuse;
3. exact right-transform fit / diagnostic;
4. shared-latent exact model if trainable;
5. canonical + low-rank delta;
6. selective repair / learned mapper;
7. full re-prefill upper-quality control.

Report:
- task quality / NLL;
- attention-output divergence;
- physical cache bytes;
- View-code/delta bytes;
- prefill and switch latency;
- decode latency;
- active FLOPs;
- HBM bandwidth when available.

## 12. Immediate high-information experiments

Priority:
- MA-701 exact+low-rank delta frontier;
- MA-702 shared token latent with View-specific up-projection;
- MA-705 canonical/specialist dual stream;
- MA-706 cache-compatible training regularizer;
- MA-707 distillation of arbitrary specialists into cache-compatible Views;
- MA-710 shared-A versus shared-B factor orientation;
- MA-715 physical copy-on-write aliasing;
- MA-721 ICaRus encoder + Mirror decoder compression;
- MA-726 top-k Views over one base/LR cache;
- MA-732 RoPE-commutant learned View;
- MA-737 actual paged-cache aliasing;
- MA-739 cache-compatible speculative draft/verifier pair.

## Evidence boundary

MA-691 proves an exact synthetic algebraic primitive. It does not establish:
- that natural specialists lie on a low-rank/exact cache orbit;
- that GPU kernels deliver the expected bandwidth gains;
- that cache-compatible training preserves language quality;
- that arbitrary upstream Mirror-MoE changes can reuse cache.

Those are the next experiments.
