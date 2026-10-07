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


## Second literature sweep — parameter superposition and structured adaptation

A second focused sweep reviewed **115 additional search-result candidates** across seven broad and nine targeted queries. Combined with the first sweep, the repository's prior-art map now spans the main neighboring families needed to control the next MA experiments.

New territories:
1. Parameter Superposition and context-vector retrieval;
2. BatchEnsemble rank-one fast weights;
3. VeRA shared random low-rank bases;
4. IA3 activation scaling;
5. OFT, BOFT and matrix-free OFTv2;
6. Compacter / Kronecker / hypercomplex adapters;
7. Monarch structured matrices;
8. Fast Weight Programmers and dynamic memory;
9. Task Arithmetic, TIES-Merging and Model Soups;
10. Cheap-LoRA / chained structured column subspaces;
11. SETA shared-vs-unique sparse continual experts.

### H. Parameter superposition is the closest historical analogue

Parameter Superposition already stores several task models in one physical parameter tensor and retrieves them with context operators. It even studies complex/unitary and rotational contexts plus composition.

Implication: the project must not claim that "one weight tensor + a context coordinate gives multiple logical models" is novel. The Mirror question is narrower and testable: do the project's structured, learned or factorized View coordinates give a better useful-multiplicity / interference / byte frontier than PSP contexts?

### I. Rank-one modulation is a very strong cheap baseline

BatchEnsemble represents each ensemble member as one shared weight plus two member-specific vectors. VeRA similarly shares low-rank matrices and learns small scaling vectors; IA3 goes cheaper still by rescaling activations.

Implication: any Mirror-MoE/head/adapter result must survive a progression:
IA3/diagonal -> BatchEnsemble/rank-one -> VeRA/shared-low-rank -> richer Mirror transform.

If a diagonal or rank-one control matches quality, the richer geometry is not justified.

### J. Orthogonal transforms are established PEFT, but their reuse is open

OFT and BOFT establish that orthogonal and butterfly-orthogonal transforms are useful adaptation coordinates. OFTv2 shows the same transform can often be applied input-side instead of materializing changed weights.

Implication: Mirror experiments should separate:
- geometry: orthogonal / shear / stretch / binding;
- address sharing: one transform family across many tasks/experts/heads;
- execution: weight-centric vs input-centric.

The project's high-value delta is logical multiplicity and factorized reuse, not orthogonality itself.

### K. Structured matrices enlarge the Mirror implementation toolbox

Compacter uses shared Kronecker/hypercomplex slow weights plus layer-specific fast rank-one factors. Monarch matrices use block-diagonal factors and permutations to produce expressive hardware-friendly transforms.

Implication: the View need not be a dense matrix. Butterfly, Kronecker, Monarch, circulant and rank-one forms should be compared on both bytes and actual runtime.

### L. Mirror coordinates may be dynamic state

Fast Weight Programmers make effective weights input-dependent and editable during inference.

Implication: (m) need not be a stored expert/task ID. It can be a session- or context-generated state. This creates a separate line:
static address -> generated address -> fast writable address.

This line is directly relevant to packet plans, live adaptation and transient experts.

### M. Model merging supplies interference controls

Task Arithmetic shows task deltas can be added/subtracted; TIES shows sign conflicts and weak directions cause interference; Model Soups shows simple weight averaging can work when models occupy a compatible basin.

Implication: composed Mirror codes must compare against simple task-vector arithmetic and sign-aware merging. A complicated composition mechanism that only reproduces averaging is not useful.

### N. Continual learning should separate discovery from compression

SETA explicitly discovers shared versus task-unique sparse subspaces and protects routing/weights over time.

Implication: a Mirror continual-learning claim should not conflate discovering what should be shared with compressing what was discovered. The clean test is:
1. discover shared/private structure with a strong baseline;
2. apply Mirror coding inside or across those components;
3. measure whether parameter growth is delayed without hurting retention.

## Second-sweep MA candidates

The second sweep added **MA-255 through MA-300**, bringing the registry to **300 candidates**.

Especially direct/high-information:
- MA-255/257/258 — Parameter Superposition crossovers;
- MA-260/261 — BatchEnsemble rank-one controls;
- MA-265/266 — VeRA shared-basis Mirror adapters;
- MA-268 — IA3 versus richer activation views;
- MA-271/272/273 — OFT/OFTv2/BOFT Mirror variants;
- MA-278 — Compacter hypercomplex/Kronecker crossover;
- MA-282 — Monarch structured Mirror transform;
- MA-286 — Cheap-LoRA structured-subspace crossover;
- MA-288 — dynamic fast-weight Mirror address;
- MA-292/296 — task-vector compression/superposition;
- MA-297/299 — SETA shared/private continual-learning crossover.

## Updated closest-source map

The maintained prior-art map now runs **PA01–PA30**. Before implementing any new MA candidate, workers must still perform a narrow current search for that mechanism; this document is a baseline, not a guarantee of novelty.
