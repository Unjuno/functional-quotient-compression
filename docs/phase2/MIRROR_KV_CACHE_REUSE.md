# Mirror KV Cache Reuse — canonical-cache design

Date: 2026-10-07 JST
Status: active design + synthetic algebra evidence
Scope: attention/KV mechanics; not yet a natural-language or GPU claim.

## Core idea

A Mirror/View specialist does not necessarily need its own physical KV cache.

If a logical View changes cached keys/values by known right transforms

[
K_m = K A_m,qquad V_m = V B_m,
]

then native attention is

[
operatorname{softmax}!left(rac{qK_m^T}{sqrt d}ight)V_m
=
operatorname{softmax}!left(rac{(qA_m^T)K^T}{sqrt d}ight)V,B_m.
]

Therefore the physical cache can remain the canonical (K,V). The View is applied only to the current query and attention output:

1. (q' = qA_m^T)
2. attention over canonical (K,V)
3. (o' = oB_m)

No prefix-cache materialization is mathematically required.

This is stronger than "convert the whole cache on every switch": the cache can be **zero-copy shared** by many logical Views.

## Why this matters for Mirror-MoE

Ordinary dynamically switched specialists can invalidate a prefix KV cache because the historical hidden states and Q/K/V projections under the new specialist differ.

A cache-compatible Mirror architecture can instead pay:

- one physical canonical cache;
- small View codes/operators (A_m,B_m);
- per-current-token View operations.

This separates:
- **physical cache multiplicity** from
- **logical attention/expert multiplicity**.

It does **not** remove the compute of evaluating several active attention Views. Top-k Views can share cache storage while still paying top-k query/attention/output compute.

## Complexity

Let:
- (T): cached prefix length;
- (d_h): head dimension;
- (H): physical KV heads;
- (N): logical Views.

For a structured View with cost (C(d_h)):

### Materialize a cache for a View
Switch cost:
[
O(THC(d_h)).
]

Keeping (N) materialized caches costs approximately (N) times physical KV storage.

### Lazy canonical-cache View
Per decoded token:
[
O(HC(d_h))
]
in addition to ordinary attention.

Physical cache storage stays at one copy, plus View codes.

For frequent switching, the avoided factor is proportional to prefix length (T). For a long-lived View, materializing once can eventually amortize; measure View dwell length rather than assuming one strategy always wins.

## Exact structured families

Useful (A_m,B_m) families include:
- diagonal / IA3-like;
- rank-one / BatchEnsemble-like;
- Householder products;
- Givens rotations;
- butterfly/BOFT;
- ACDC/AFDF;
- Hadamard/sign/permutation;
- low-rank residual maps where the algebra can be fused.

Dense transforms are valid algebraically but may lose runtime advantage.

## RoPE

General View transforms do not commute with RoPE.

Three clean designs exist.

### A. RoPE-commuting Mirror
Use block rotations aligned to the same 2D RoPE planes. Rotations within one 2D plane commute, so

[
R_{	ext{RoPE}}(t)R_m = R_mR_{	ext{RoPE}}(t).
]

The already-RoPE-encoded cache can remain canonical.

### B. Position-free canonical cache
Cache the content-space key and defer both RoPE and Mirror transforms to the attention kernel.

This is conceptually aligned with LazyAttention's deferred positional encoding.

### C. Strip -> map -> reapply
For an existing cache, strip RoPE, map in content space, then apply the target/View RoPE. Cross-model KV-transfer work uses this idea for position-independent cache mapping.

## MLA generalization

Multi-head Latent Attention already demonstrates the right algebraic pattern.

Cache only a latent

[
C_t = W_D h_t.
]

For View (m),

[
K_m = C U^K_m,qquad V_m = C U^V_m.
]

Then

[
qK_m^T=(q(U^K_m)^T)C^T
]

and

[
alpha V_m=(alpha C)U^V_m.
]

So one canonical latent cache can serve many logical K/V Views. This motivates **Mirror-MLA**: keep the latent physical state fixed and put logical multiplicity in low-description up-projection/View coordinates.

## Cross-layer sharing

YOCO and MLKV show that physical KV can be shared across layers. MA-245 already produced an aligned synthetic signal: one cached pair plus layer-specific Mirror Views recovered the teacher while using half the cache state of two-group MLKV.

A stronger architecture is:

```text
one canonical cache
       |
       +-- layer/View 1: q-transform -> attention -> output-transform
       +-- layer/View 2: q-transform -> attention -> output-transform
       +-- layer/View 3: q-transform -> attention -> output-transform
```

No per-layer cache copy is needed.

CLSA further suggests that the sparse routing/index can be shared together with the cache.

## Important boundary: arbitrary FFN Mirror does not imply cache transformability

For a standard decoder layer, an FFN/expert output changes the hidden state consumed by later layers. If two expert choices produce unrelated hidden states, the later-layer KV cache is not generally obtainable from another cache by a fixed small transform.

Exact cache reuse is guaranteed only when the architecture provides one of:

1. final-attention/final-FFN placement that cannot affect later cached states;
2. a known cache-space transform relation;
3. a canonical latent/cache interface independent of the View;
4. a constrained hidden-state orbit with a known map.

Otherwise cache translation is an approximation problem and should be compared against learned linear/nonlinear translators.

MA-253 confirmed the placement distinction: final-FFN View left prefix K/V exactly unchanged, while an earlier View changed downstream K/V.

## Adjacent prior art

- PA49 Activated LoRA: preserves cache by not adapting pre-activation prefix tokens.
- PA50 standard-LoRA direct prefix reuse: large TTFT benefit but quality equivalence and physical aliasing not established.
- PA51 LazyAttention: one physical cache, deferred logical positional transforms.
- PA52/PA53 cross-model KV transfer / CacheBridge: approximate affine translators.
- PA54 Cache-to-Cache: learned cache projection/fusion across models.
- PA55 KVEraser: functional behavior can be changed through local KV-space edits without full recomputation.
- PA56 MLA: canonical low-dimensional cache with projections absorbed into read-side computation.
- PA57 YOCO: one global cache reused across cross-decoder layers.
- PA58 CLSA: share both cache and sparse routing index.

## Local synthetic proof-of-concept

Environment:
- PyTorch 2.10.0+cpu;
- one CPU thread;
- FP32;
- batch 1;
- 8 heads;
- head dimension 64;
- pairwise Givens View transforms.

### Exact lazy-cache equivalence

Across prefix lengths 128, 512, 2,048, 8,192 and 32,768, the maximum absolute difference between:

- explicitly materialized (K_m,V_m), and
- canonical (K,V) with query/output-side lazy transforms

was approximately (1.7	imes10^{-7}) to (2.8	imes10^{-7}).

This is FP32 numerical error, not an approximate translator.

### RoPE-plane commutation

Pairwise Mirror rotations restricted to the same 2D planes as RoPE produced maximum absolute commutation error (7.2	imes10^{-7}) in the synthetic check.

### MLA absorption

For a toy latent cache with 6 heads, head dimension 16 and latent dimension 24, explicit K/V expansion and latent-space absorbed attention differed by (2.1	imes10^{-7}). The physical latent state used 8x fewer bytes than explicit K+V in that toy shape.

### CPU switch/materialization microbenchmark

The following is **not a GPU performance claim**. It shows the expected context-length scaling of an eager PyTorch implementation.

| prefix T | transform full cache once | pre-materialized View attention | lazy View attention | approximate materialization break-even |
|---:|---:|---:|---:|---:|
| 128 | 0.40 ms | 0.038 ms | 0.094 ms | 7 generated tokens |
| 512 | 1.55 ms | 0.083 ms | 0.141 ms | 27 |
| 2,048 | 9.10 ms | 0.242 ms | 0.304 ms | 146 |
| 8,192 | 50.60 ms | 1.531 ms | 1.570 ms | 1,288 |
| 32,768 | 260.43 ms | 8.323 ms | 7.992 ms | lazy faster in this run |

Interpretation: materializing a View incurs a prefix-length-dependent cost; lazy read-side transforms add a small current-token cost. For token-level/frequent Mirror switching, lazy canonical-cache execution is structurally favored. For a View that remains fixed for a long generation, one-time materialization may amortize.

Kernel fusion, GPU bandwidth, cache layout, quantization and real model dimensions can change the constants substantially.

## New experiment IDs

- MA-691: lazy canonical KV Mirror read;
- MA-692: RoPE-commuting Mirror KV Views;
- MA-693: position-free cache + deferred Mirror/RoPE;
- MA-694: Mirror-MLA canonical latent cache;
- MA-695: YOCO one-cache + Mirror layer reads;
- MA-696: CLSA shared index + Mirror layer reads;
- MA-697: exact Mirror adapter switching with cache reuse;
- MA-698: known Mirror translator vs learned cross-model mapping;
- MA-699: top-k Mirror attention Views over one cache;
- MA-700: quantized canonical cache + Mirror logical decode.

## Recommended MA-691 protocol

### Phase 1 — algebra
Require bitwise/FP-tolerance equivalence between native materialized View attention and lazy canonical-cache attention.

### Phase 2 — switch frontier
Sweep:
- prefix length (T);
- number of logical Views;
- View dwell length (tokens before switching);
- top-k active Views.

Measure:
- physical cache bytes;
- cache-copy/materialization bytes;
- switch latency;
- per-token latency;
- total generation latency.

### Phase 3 — orbit/private frontier
Do not test only a perfectly aligned teacher.

Use:
[
Delta(alpha)=alphaDelta_{	ext{cache-view}}+(1-alpha)Delta_{	ext{private}}
]
for (alphain{0,.25,.5,.75,1}).

Compare:
1. hard cache sharing;
2. diagonal/gate View;
3. rank-one View;
4. learned affine cache translator;
5. structured Mirror View;
6. Mirror + private residual;
7. independent native cache/projection.

### Phase 4 — nanoGPT/language
Only after the mechanics pass:
- retrofit a tiny causal model;
- train cache-compatible logical specialists;
- measure validation NLL;
- compare native specialist prefill, aLoRA-style compatibility, direct shared-prefix reuse, learned translator, and exact Mirror lazy reuse.

## New 2026 cross-model KV prior art — distinguish exact and approximate lanes

The original MA-691 lazy canonical-cache result is an **exact known-View algebra** on a compatible shared state. It does not imply that KV caches from independently trained models are equivalent or exactly transformable.

New direct controls:
- PA236: per-head ridge cache transfer between different-size family members, with source-layer selection and position-free RoPE removal;
- PA237: CacheBridge head-matched attention-sensitive ridge mapping with compact mapper construction;
- PA238: Mixture-of-Translators with context correction to reduce target trajectory shift;
- PA239: heterogeneous context reuse across model sizes/families/tokenizer regimes.

The Mirror-specific research question is whether a *family* of source-to-target, layer-to-layer and head-to-head maps can share one physical translator basis addressed by small `m_source, m_target, m_layer, m_head`, instead of keeping an entire map for each ordered model pair. See MA-876..890 and `MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md`.

Before screening these candidates classify the proposed transform:

1. **Exact algebraic:** identical/compatible original hidden state, known invertible or appropriately absorbable View operations; certify attention output equality numerically.
2. **Approximate within-family:** separately trained layers/heads are mapped by calibrated regression; report target NLL, attention sensitivity and failed model pairs.
3. **Approximate cross-family:** model size, tokenizer and head/layer layout may differ; require explicit token/source-cache provenance and nonlinear/fallback controls.

Charge source prefill, calibration and mapper construction, model/cache bytes, translator execution, metadata/token alignment and target decode. Benchmark end-to-end handoff versus native target re-prefill at multiple lengths/dwell times. Never substitute cache tensor MSE or an algebraic coordinate change for target quality and latency.

## New direct 2026 multi-LoRA KV evidence: LRAgent and PReCache

PA364 [LRAgent (ICML 2026)](https://proceedings.mlr.press/v306/jeon26b.html) and PA365 [PReCache (September 2026)](https://arxiv.org/abs/2609.34054) are exceptionally close to the originally proposed Mirror MoE/specialist cache-reuse mechanism.

- **LRAgent** separates a common pretrained-base cache component from adapter-dependent low-rank terms. It offers Flash-LoRA-Attention to consume the low-rank cache without unnecessary full-dimensional materialization, with further sharing under shared-A multi-LoRA architecture.
- **PReCache** compares low-rank agent-cache precomputation (PreLRShared) and neutral-base cache reconstruction (ReBaseShared) for multi-turn adapter switching. Adapter-specific hidden-state contributions and propagation cannot in general be ignored.
- PA154 aLoRA avoids invalid caching by restricting the activation point; PA155 directly measures standard LoRA prefix reuse including failures of physical aliasing; PA156/MA691 establish **exact algebra for known compatible canonical-cache right transforms**, not arbitrary independently adapted models.

The new Mirror-specific study MA1110..1112 must insert low-description `m_agent` into **the already compressed adapter LR cache** or the physical state-read algebra; a merely common base cache plus low-rank private state is now a published baseline, not a novel Mirror win.

Hard guards:
1. Track producer model, adapter, tokenization and exact source-token prefix for every shared cache.
2. Probe actual storage/pointer alias and allocation, not just identical cached values copied between branches.
3. Verify K, V, RoPE, attention-logit scaling, and target decode quality for each logical consumer.
4. Count initial source prefill, low-rank-cache construction, neutral reconstruction, read transform, switches and resident adapter states; benchmark realistic prompt length and adapter dwell.
5. Compare LRAgent fused low-rank kernels, PReCache precomputation/reconstruction, aLoRA compatible-prefix serving and native re-prefill. Lower tensor error is insufficient for adoption.
6. Report relative quality/TTFT/VRAM/throughput **on the same model, task, workload and hardware**; distinguish learned approximate cache translation from MA691 exact algebra.

## Adoption condition

The important target is not merely smaller cache.

A strong result requires:
- no or negligible quality loss;
- one physically aliased canonical cache across logical Views;
- lower switch/prefill cost than recomputation;
- View operation overhead small enough to improve end-to-end latency at realistic switch frequencies.

