# MA-691 — lazy canonical KV Mirror read

Status: **PROMISING — exact cache-compatible mechanism PASS; language/GPU not established**  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `7589ce171d6aa6743b7b0d43cd424134afb8fbdc`

## H — hypothesis

For logical attention Views with known right-transforms

`K_m = K A_m` and `V_m = V B_m`,

a View-specific prefix cache is unnecessary because:

`softmax(q K_m^T) V_m = softmax((q A_m^T) K^T) V B_m`.

Therefore one physical canonical K/V cache should serve multiple logical Views by transforming only the current query and attention output.

## T — execution

Protocol was frozen before fresh execution. Three fresh seeds (69101–69103) used FP32 random attention states with:
- batch 1;
- 8 heads;
- head dimension 64;
- prefix lengths 128, 512, 2,048, 8,192 and 32,768;
- independent pairwise Givens codes for K and V.

Comparisons:
1. explicit full-cache materialization;
2. pre-materialized View cache;
3. lazy canonical-cache execution;
4. RoPE-plane commutation diagnostic;
5. MLA-style latent-cache absorption diagnostic.

One CPU thread under PyTorch 2.10.0+cpu was used for the exploratory runtime measurements. Runtime is not an adoption/GPU claim.

Prior art: PA51 LazyAttention, PA52 cross-model KV transfer, PA56 MLA.

## D — decision

**PROMISING mechanism result. All preregistered exactness gates passed in 3/3 fresh seeds.**

Maximum errors:
- lazy versus materialized View attention: **3.13e-7**;
- Mirror/RoPE same-plane commutation: **7.15e-7**;
- explicit versus absorbed MLA computation: **4.17e-7**.

The threshold was 2e-6. These differences are consistent with FP32 numerical ordering rather than an approximate learned cache translator.

### Cache storage

At prefix length 8,192 in the tested shape, one canonical FP32 K+V cache is 33,554,432 bytes. One logical Givens View code is 2,048 bytes.

Eight simultaneously materialized View caches would require about 268.4 MB of cache state. One canonical cache plus eight View codes is about 33.57 MB. This is a logical-view storage comparison, not a claim that every application needs simultaneous View caches.

The MLA diagnostic cached 98,304 bytes of latent state versus 786,432 bytes for explicit K+V in its toy shape: **8x smaller**, while matching output within 4.17e-7.

### Switch/runtime scaling

Median across three fresh seeds:

| prefix T | full-cache transform once | pre-materialized attention | lazy attention | materialization break-even |
|---:|---:|---:|---:|---:|
| 128 | 0.397 ms | 0.038 ms | 0.092 ms | ~7 tokens |
| 512 | 1.463 ms | 0.083 ms | 0.140 ms | ~26 tokens |
| 2,048 | 7.434 ms | 0.272 ms | 0.339 ms | ~95 tokens |
| 8,192 | 42.850 ms | 1.639 ms | 1.796 ms | ~342 tokens |
| 32,768 | 250.075 ms | 7.233 ms | 7.610 ms | ~628 tokens |

Interpretation: if a logical View changes frequently, paying a transform over the entire prefix is expensive; lazy View transforms scale with the current token rather than prefix length. If a View remains fixed for a sufficiently long generation, one-time materialization may amortize. Constants are implementation/hardware dependent.

## C — strongest counter-hypothesis

The cache relation is deliberately exact by construction. This experiment proves an architectural algebraic primitive, not that naturally trained LoRA/MoE specialists or arbitrary FFN Views have this relation.

In particular, an earlier FFN/expert that produces unrelated hidden states changes later-layer K/V and cannot generally be repaired by these fixed A/B transforms.

## U — unconfirmed

- natural-language NLL and task quality;
- whether trained specialist differences can be constrained to this cache-compatible family without hurting capability;
- GPU/fused-kernel latency and bandwidth;
- real paged-cache physical aliasing across concurrent requests;
- adapter switching against aLoRA and standard-LoRA direct reuse;
- learned fallback translators when the exact orbit is imperfect;
- top-k View execution cost.

## Prior-art delta

- aLoRA obtains reuse by keeping the pre-activation prefix identical.
- standard LoRA prefix reuse can save TTFT but is not guaranteed quality-equivalent.
- cross-model transfer learns approximate affine cache mappings.
- LazyAttention defers positional transforms.
- MLA caches a canonical latent and moves projections into read-side computation.

MA-691 shows the **Mirror/View transform itself can be deferred exactly** when the architecture is parameterized in the right-transform form.

## Evidence files

- `PROTOCOL.json`
- `RESULTS_CORE.csv`
- `CHECKS.json`
- `VERIFICATION.json`
- `source/engine.py`
- `tests/test_engine.py`

See also `../../../docs/phase2/MIRROR_KV_CACHE_REUSE.md`.
