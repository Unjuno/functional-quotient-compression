# MA-691 next phase — cache-aware Mirror decode (2026-10-18)

**Scope:** extension of the existing **MA-691** exact canonical KV-cache mechanism. It does NOT replace its historical PROMISING verdict, create a new MA ID, or change MA-253's FAIL (expert replacement) / cache placement PASS.

> **Mirror insertion:** use low-description View addresses \`m\` on the read-side Q/output transforms of one canonical physical K/V cache, to expose multiple logical attention computations without separately resident View-specific prefix caches.

## Research question
Does canonical KV Mirror provide a useful **end-to-end** rate–quality–latency advantage over *strong* existing KV-sharing alternatives (MQA/GQA/MLA and algebraically absorbed native query/output Views), rather than only beating intentionally duplicated View caches?

Start from the existing MA-691/MA-253 code and the MA-1164 fused CPU kernel. Keep the evidence lanes separate:
- L0: exact algebra, RoPE placement, negative cache provenance, real allocated bytes.
- L1: complete autoregressive prefill + decode on a small *cache-compatible* causal model; real latency/throughput versus exact and native controls.
- L2: trained workload with quality preservation versus GQA/MQA/MLA/native adapters, and, separately, optimized kernel benchmarking on GPU only if it is actually available.

**Important:** fixed right-transforms can be absorbed into ordinary Q/output projections. This is a compulsory exact-functional non-Mirror control; merely switching a duplicated cache to canonical storage does NOT certify Mirror novelty.

See [Japanese execution plan](EXPERIMENT_PLAN_JA.md), [frozen L0 protocol](PROTOCOL_L0_FROZEN.json), [L1 draft](PROTOCOL_L1_DRAFT.json), and [status](STATUS.md).

References:
- MA-691 prior evidence: \`../README.md\`, \`../PROTOCOL.json\`.
- MA-253 invalid downstream cache counterexample: \`../../ma-253-cache-safe-final-moe/README.md\`.
- GQA: https://doi.org/10.18653/v1/2023.emnlp-main.298
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- PagedAttention: https://arxiv.org/abs/2309.06180
- MLA: https://arxiv.org/abs/2405.04434
- LRAgent (multi-LoRA cache sharing): https://proceedings.mlr.press/v306/jeon26b.html
