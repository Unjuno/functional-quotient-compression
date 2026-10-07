# Mirror Application Research Notes — 2026-10-07

Purpose: document the literature sweep that informed the worker-ready Mirror application registry.

## Search scope

Nine focused search workstreams were run, reviewing 80 search-result candidates in total, followed by full-text reads of selected primary sources.

Workstreams:
1. MoE expert tying, shared experts, shared/low-rank routing;
2. hypernetworks and generated low-rank adapters;
3. attention-head and KV/projection sharing;
4. holographic/vector-symbolic binding;
5. multi-token prediction and speculative/parallel decoding;
6. recursive/shared-depth Transformers;
7. QKV hard sharing and MLKV;
8. modern VSA implementations and binding costs;
9. recent generated-weight / live-adaptation architectures.

The goal was not to prove novelty. It was to identify the strongest adjacent control for each MA family and expose combinations that are not tested by merely repeating established parameter sharing.

## High-signal patterns

### A. Sharing is already a first-class efficiency mechanism

Recent work ties MoE experts across depth, shares routing parameters across blocks, shares KV heads within and across layers, and shares Q/K/V projections.

Implication: "we share X" is not a sufficient Mirror contribution. Mirror experiments must show that a small view coordinate recovers useful functional differentiation that hard tying loses.

### B. Shared basis + small modulation is already a strong design family

HyRA, generated-LoRA methods, and recursive Transformer controllers use shared low-rank bases plus small generated/modulated coefficients.

Implication: Mirror-LoRA/depth results need a generic hypernetwork or static per-step LoRA control. A Mirror-specific result is about the structured coordinate, not generic conditional modulation.

### C. Physical expert count and routing cost are separable

MoE work increasingly addresses both expert-storage redundancy and router/path complexity.

Implication: logical expert multiplication is only useful if the router/address mechanism stays cheap. The registry therefore tracks routing cost separately from expert bytes.

### D. Attention/KV have multiple independent sharing axes

Head sharing (GQA/MQA), cross-layer KV sharing (MLKV), and projection equality (e.g. K=V) are complementary.

Implication: Mirror-KV is especially interesting because a view coordinate could be inserted after aggressive physical sharing to recover layer/head/role differences without restoring full copies.

### E. Multi-token generation has a joint-distribution problem, not only a speed problem

MTP uses several future heads over a shared trunk. PTP explicitly introduces random variables to make dependent futures jointly predictable in parallel.

Implication: TM001's hidden packet-latent failure is expected under factorized slots. The more interesting Mirror question is whether a packet-level view can act as shared randomness/plan while future-offset views provide deterministic roles.

### F. Holographic binding has mature alternatives to FFT-HRR

Modern VSA benchmarks show MAP/sign-permute and Hadamard linear binding can be substantially cheaper than HRR while retaining similar retrieval behavior in tested regimes.

Implication: holographic Mirror experiments must include MAP and Hadamard controls. "FFT holography" is not automatically the efficient choice.

### G. Factorized logical state spaces deserve direct testing

Prior work separately shares experts, routers, depth, KV and future heads. The project-specific opportunity is to test whether small independent coordinates can be factorized:

    m = (m_expert, m_head, m_depth, m_time)

without paying for a Cartesian parameter table.

The number of possible combinations is NOT a capacity measure. The experiment must measure useful quality at fixed bytes/compute.

## Literature-derived MA candidates

The sweep added MA-241 through MA-252:

- MA-241: expert tying across depth + layer-specific Mirror expert views;
- MA-242: PathMoE shared router + Mirror expert views;
- MA-243: low-rank router + Mirror-expanded logical experts;
- MA-244: K=V sharing + Mirror role recovery;
- MA-245: MLKV + per-layer Mirror KV views;
- MA-246: Mirror adapter codes vs generated LoRA;
- MA-247: recursive shared block + Mirror depth modulation;
- MA-248: PTP-style packet randomness as a Mirror code;
- MA-249: one physical future head + Mirror future-offset views;
- MA-250: MAP/Hadamard binding as Mirror expert address;
- MA-251: factorized expert x depth Mirror coordinate;
- MA-252: single attention projection + Mirror Q/K/V role views.

## Closest sources

See `MIRROR_APPLICATION_PRIOR_ART.md` for the maintained PA01–PA13 map and URLs.

## Worker interpretation

Treat the literature map as a control-selection tool, not as a claim that an idea is novel or non-novel. Before a fresh experiment, search again for the selected MA ID's narrow mechanism because the research landscape is moving quickly.
