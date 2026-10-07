# KV / MoE Follow-up Queue

Date: 2026-10-07 JST
Purpose: ordered follow-up after the verified MA-691 lazy canonical-cache result.

Do not interrupt an already-started dedicated experiment. Before starting each ID, search live `research/ma-*` branches.

## Tier 0 — finish the already-registered MA-691 family

1. **MA-692** — RoPE-commuting Mirror KV Views
2. **MA-693** — pre-RoPE canonical cache + deferred Mirror/position
3. **MA-694** — Mirror-MLA canonical latent cache
4. **MA-695** — YOCO one-cache + Mirror layer reads
5. **MA-696** — CLSA one cache/index + Mirror layer reads
6. **MA-697** — exact Mirror adapter switching with cache reuse
7. **MA-698** — analytical Mirror map vs learned cache translator
8. **MA-699** — top-k Mirror attention Views over one cache
9. **MA-700** — quantized canonical cache + Mirror logical decode

These directly extend MA-691 and should generally precede later refinements.

## Tier 1 — stronger exact algebra

10. **MA-703** — score-invariant QK Mirror gauge
11. **MA-704** — one-softmax / one-value-read / many value Views
12. **MA-701** — affine lazy canonical KV View
13. **MA-702** — rectangular latent-to-logical KV View
14. **MA-706** — batched multi-View query packing

Why: these test whether we can share not only cache storage, but parts of attention compute.

## Tier 2 — orbit/private cache frontier

15. **MA-709** — exact View + low-rank private KV residual
16. **MA-710** — exact View + sparse-token private correction
17. **MA-711** — exact View + critical-layer recompute
18. **MA-712** — joint token x layer correction
19. **MA-713** — MiniCache merged state + Mirror layer decode
20. **MA-714** — Palu latent + Mirror logical read maps
21. **MA-715** — Palu latent + private residual

Why: natural specialists are unlikely to be exactly on one View orbit. This tier measures how cheaply off-orbit information can be restored.

## Tier 3 — low-bit canonical cache

22. **MA-717** — quantization gauge x functional View
23. **MA-718** — one SpinQuant gauge across logical Views
24. **MA-719** — quantize-before-View vs View-before-quantize
25. **MA-720** — pre-RoPE low-bit canonical cache + deferred transforms

Required controls: QuaRot, SpinQuant, KIVI/KVQuant where applicable.

## Tier 4 — physical serving proof

26. **MA-721** — PagedAttention physical block aliasing
27. **MA-723** — adaptive lazy/materialized execution
28. **MA-724** — hot transformed-block memo
29. **MA-722** — multi-tenant personalized prefix sharing
30. **MA-725** — unified adapter/KV residency manager

This tier must distinguish actual physical aliasing from storing equal copies.

## Tier 5 — architectural cache stability

31. **MA-731** — canonical writer + specialist reader
32. **MA-732** — specialist state rebase into canonical writer
33. **MA-726** — canonical KV ABI across model family
34. **MA-727** — small-to-large escalation without re-prefill
35. **MA-728** — draft/verifier pair over one canonical cache
36. **MA-729** — Mirror-compressed speculative streams

This is where cache reuse becomes an architecture property rather than an adapter trick.

## Tier 6 — shared attention / MoE compute

37. **MA-733** — shared-score attention MoE
38. **MA-734** — value-only Mirror expert bank
39. **MA-735** — precompose top-k value Views before W_O
40. **MA-736** — single-GEMM top-k Mirror FFN

MA-736 is high priority because it may reduce active expert compute, not only model bytes.

## Tier 7 — distributed MoE systems

41. **MA-737** — local logical experts without expert-owner all-to-all
42. **MA-738** — residual-only expert parallelism
43. **MA-739** — resident base + streamed View codes
44. **MA-740** — fused View transforms around shared GEMM
45. **MA-741** — one large GEMM across routed logical experts
46. **MA-748** — orbit/private communication frontier

Required controls: standard expert parallelism; MegaBlocks/grouped GEMM; MoE-Infinity/Fiddler where relevant.

## Tier 8 — compression controls

47. **MA-743** — MoLAE latent expert + tiny structured View
48. **MA-744** — MoBE basis coefficient compression/factorization
49. **MA-745** — DeepSeek shared expert + routed Mirror residuals

These determine whether Mirror adds anything beyond current expert-compression methods.

## Integration gate

**MA-750** is not an early experiment.

Run it only if several independent components pass:
- canonical cache;
- useful specialist Views;
- private residual frontier;
- shared-compute top-k or local logical expert mechanism;
- runtime/system implementation.

A combined model must beat the strongest component alone, not merely Dense/MoE baselines.
