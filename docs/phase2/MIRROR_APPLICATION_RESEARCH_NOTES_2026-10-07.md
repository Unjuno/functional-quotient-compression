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


## Third and fourth literature sweeps — subnetworks, tensors, symmetries, supernets and prompts

Two further sweeps broadened the search beyond parameter-efficient weight modulation.

The sweeps issued **60 focused searches and screened 346 search-result candidates** before selecting high-signal primary sources for the maintained prior-art map.

New territories:
1. supermasks, Piggyback/PackNet and fixed-weight subnetworks;
2. intrinsic-dimensional/random-subspace fine-tuning;
3. Tensor-Train/Tucker tensorization and LoRETTA;
4. permutation/sign/scale weight-space symmetries and Re-Basin alignment;
5. personalized/federated hypernetworks and rank-heterogeneous LoRA;
6. rank-one Bayesian models and MIMO implicit ensembles;
7. product-key and modern Hopfield associative memory;
8. ACDC/structured fast transforms and Lie-group equivariance;
9. Mixture-of-Depths and adaptive early exit;
10. universally slimmable, Once-for-All and MatFormer supernets;
11. AdapterFusion / LoRAHub composition;
12. L2P / DualPrompt prompt pools;
13. hash, compositional and adaptive embeddings;
14. one-shot NAS and architecture weight sharing.

### O. A View can be a mask, not only a transform

SupSup/Piggyback show that a fixed weight tensor can express many functions by changing only a task mask.

Implication: every richer Mirror transform should be compared against a binary/sparse mask when the use case permits it. The extra geometry must buy something: fewer stored bits, smoother composition, better routing, or better quality.

### P. A View can live in intrinsic task space

Intrinsic-dimension results show that some language-model adaptations can be described with surprisingly small coordinate vectors projected into the full parameter space.

Implication: do not always apply Mirror in d_model or weight-matrix space. It may be much cheaper and better conditioned to apply structure to the already-low-dimensional task coordinate.

### Q. Tensor banks are direct physical-to-logical decompositions

Tucker compression explicitly represents many Transformer matrices by a shared matrix bank plus per-matrix coefficients. TT/LoRETTA similarly uses small cores/factors.

Implication: tensor factors and their coefficient vectors are natural Mirror-address spaces. This is a strong alternative to low-rank matrices and deserves equal priority.

### R. Gauge/symmetry directions must not be counted as functional multiplicity

Permutation, scaling and sign symmetries can produce different parameter tensors with exactly the same function.

Implication: every Mirror family needs a symmetry audit. If a View is only moving along a parameter-symmetry orbit, it is not a new expert/function and must not be counted as capacity.

### S. Personalization exposes an address-compression problem

pFedHN, HyperLoRA and PreLort all represent many clients using shared global structure plus smaller client-specific state.

Implication: federated/personalized models provide a concrete environment for testing whether Mirror codes are cheaper than generic hypernetwork outputs or independent client adapters, including communication cost.

### T. A View can encode posterior/member identity

Rank-1 Bayesian networks, BatchEnsemble and MIMO show several ways to obtain many predictive members from one physical network.

Implication: Mirror uncertainty work should measure diversity, calibration and posterior quality—not merely accuracy or the number of codes.

### U. A View can address memory combinatorially

Product Key Memory factorizes a large address space into products of smaller sub-key sets; modern Hopfield networks provide associative retrieval.

Implication: the number of Mirror addresses need not imply a dense router/table. Product/address factorization and associative retrieval are direct methods to control routing/storage growth.

### V. The View can be the computation path itself

Mixture-of-Depths, early exit, slimmable networks, OFA, MatFormer and one-shot supernets show that one physical training object can represent many computation graphs.

Implication: Mirror is not restricted to weight transformations. The coordinate can represent:
- depth;
- width;
- active subnetwork;
- architecture;
- hardware/latency target;
- token-specific compute budget.

The interesting test is whether a tiny View-specific correction recovers quality lost by aggressive weight sharing.

### W. Prompt/adapter composition is a separate multiplicity layer

AdapterFusion, LoRAHub, L2P and DualPrompt already store and compose many small modules.

Implication: Mirror can compress the module bank itself, or operate on its composition coefficients. These two hypotheses must be tested separately.

### X. Embedding tables are an unusually large physical-multiplicity target

Hash embeddings, quotient-remainder compositional embeddings, adaptive embeddings and ALBERT factorization show multiple ways to create large logical vocabularies from less physical state.

Implication: embedding-space Mirror experiments should compete against these specialized compression methods, not only a dense embedding baseline.

## Registry growth

- Third sweep: MA-301 through MA-360.
- Fourth sweep: MA-361 through MA-400.
- Registry after these sweeps: **400 candidates**.

The current worker queue is intentionally not reordered around an already-running experiment. New candidates enter subsequent research-expansion queues.


## Fifth and sixth literature sweeps — conditional function generation, neural dynamics, modular routing and editable memories

These sweeps moved beyond static parameter-sharing mechanisms and focused on cases where the small conditioning variable is generated by the input, task data, time/depth, state, or an edit request.

New territories:
1. FiLM / conditional normalization / SPADE;
2. StyleGAN2 per-sample weight modulation and demodulation;
3. CondConv and Dynamic Convolution;
4. DeepSDF and modulated implicit neural representations;
5. Neural ODE, Deep Equilibrium and Universal Transformer;
6. Mamba/S4 state-space dynamics;
7. MAML, LEO and learned optimizers;
8. explicitly conditioned sparse transforms and concept modulation;
9. PathNet and Routing Networks;
10. HyperFormer, AdaMix, UniPELT and modular skill banks;
11. model editing: MEND, ROME, MEMIT, SERAC and GRACE;
12. discrete latent/codebook methods: VQ-VAE and residual vector quantization;
13. sparse coding/LISTA and shared-private dictionaries;
14. error-correcting address codes.

### Y. m can be generated per input, not stored per task

FiLM, SPADE, StyleGAN-style modulation, CondConv and Dynamic Convolution all make the effective function depend on conditioning input.

Implication: distinguish:
- stored address m_task;
- generated address m(x);
- stateful address m(x, history).

These have different storage, runtime and generalization properties.

### Z. Weight modulation is a direct historical analogue

StyleGAN2 shows that a compact style can change effective layer weights by channel modulation and normalization. CondConv/Dynamic Convolution synthesize effective kernels from basis kernels.

Implication: "dynamic Mirror weights" must compare against:
1. diagonal/channel modulation;
2. coefficient mixture of basis weights;
3. richer structured Mirror transform.

### AA. One decoder plus a latent code is already a many-function representation

DeepSDF and modulated neural fields represent many continuous functions using one shared synthesis network and compact latent codes.

Implication: controlled neural-function benchmarks are ideal for measuring useful functions per stored code bit, interpolation and held-out factor composition.

### AB. Depth can be a dynamical coordinate rather than a parameter index

Neural ODE, DEQ and Universal Transformer reuse one rule across continuous, equilibrium or recurrent depth.

Implication: Mirror depth should ask whether a small coordinate recovers genuinely different operations beyond simple time/depth conditioning, while reporting solver/iteration cost.

### AC. State-space selectivity makes m a dynamical-system coordinate

Mamba already conditions state-space behavior on input; S4 provides a structured state-space factorization.

Implication: Mirror-SSM experiments should operate in the native structured dynamics space and separate token-level selectivity from slower task/domain Views.

### AD. Meta-learning makes m an adaptation state

MAML creates task models by short gradient trajectories; LEO optimizes a low-dimensional latent that decodes into parameters; learned optimizers generate updates from gradient/history.

Implication: a Mirror task code competes with both static compact codes and quickly learned codes. Evaluate adaptation steps and support-set compute, not just final bytes.

### AE. Modular paths are another function coordinate

PathNet and Routing Networks represent tasks/inputs by sequences of reusable modules.

Implication: factor "which module?" from "which View of that module?". A path plus a View may replace a much larger physical module bank.

### AF. The adapter bank itself can be generated or mixed

HyperFormer generates adapters from task/layer/position embeddings. AdaMix and UniPELT exploit mixtures of PEFT mechanisms. Polytropon discovers reusable discrete skills.

Implication: Mirror adapter experiments need generic hypernetwork and mixture controls; the high-value question is whether structured codes reduce generator/bank storage or extrapolate compositionally.

### AG. Knowledge editing creates a concrete bytes-per-new-behavior test

ROME/MEMIT modify model weights; MEND generates edits; SERAC/GRACE store edits externally.

Implication: Mirror edit codes can be compared in a clear rate-quality setting:
- bytes per edit;
- locality;
- generalization;
- interference after many edits;
- latency/retrieval overhead;
- reversibility.

### AH. m can be a discrete residual code sequence

VQ-VAE and residual vector quantizers represent signals by codebook indices and progressively refined residual codes.

Implication: Mirror address precision should be an empirical rate-distortion frontier, not assumed floating-point. Easy/shared functions may need few codebooks; rare/private functions may need more.

### AI. Sparse coefficients are a universal simple control

Sparse coding represents each signal/function using a few coefficients over a shared dictionary; LISTA learns fast inference of those coefficients.

Implication: shared-rule/Mirror banks must compare against sparse dictionary codes and learned sparse-code routers.

### AJ. Address robustness is a separate design axis

Error-Correcting Output Codes show that redundant, separated codes can be more robust than minimal IDs.

Implication: when routing/address errors matter, a few extra code bits may improve total system quality. Address compression and address robustness form a rate-error frontier.

## Registry growth

- Fifth sweep: MA-401 through MA-450.
- Sixth sweep: MA-451 through MA-500.
- Registry after these sweeps: **500 candidates**.
- Maintained prior-art map: **PA01–PA95**.

The operational rule remains unchanged: research additions enter later queues and do not interrupt a worker's already-locked experiment.


## Seventh literature sweep — representation-space functions and sparse features

This sweep focused on the cheapest possible place for the functional coordinate: the hidden representation itself.

New territories:
1. ReFT / LoReFT representation finetuning;
2. Representation Engineering;
3. Activation Addition and Contrastive Activation Addition;
4. Function Vectors;
5. Conditional Activation Steering;
6. sparse-autoencoder feature steering;
7. transcoders / sparse feature circuits;
8. sparse/local low-rank knowledge intervention.

### AK. Weight space is not necessarily the cheapest functional space

ReFT shows that downstream behavior can be changed through low-rank interventions on frozen hidden representations.

Implication: every weight-space Mirror family now has a representation-space competitor. The experiment program should eventually map which functions are cheaper in:
- weight space;
- activation/representation space;
- latent-code space;
- external memory.

### AL. Function vectors are unusually direct evidence for the project framing

Function-vector work reports compact activation vectors that causally trigger input-output functions and sometimes compose algebraically.

Implication: "functional coordinate" is not only an analogy. A controlled experiment can extract a function vector, compress it into a Mirror basis/code, and measure whether the same function survives.

### AM. Sparse feature dictionaries offer interpretable functional atoms

SAEs and transcoders decompose dense hidden computations into sparse features; feature steering can causally change behavior.

Implication: shared-rule banks should compare to sparse representation-space atoms. If a task/expert can be expressed by a handful of SAE/transcoder features, a dense Mirror transformation may be unnecessary.

### AN. Dynamic condition x behavior factorization is already practical

Conditional Activation Steering separates a condition vector from a behavior vector.

Implication: factorized addresses such as m=(condition, behavior) can be tested directly with held-out combinations and false-trigger metrics.

### AO. Function storage and execution can be separated in activation space

Function vectors provide function identity, while repeated/tied blocks or packet slots can provide execution structure.

Implication: SRM003's storage-versus-execution split can be revisited without weight-space experts: use function vectors/representation Views as operators and separately test the executor.

## Registry growth

- Seventh sweep: MA-501 through MA-550.
- Registry after this sweep: **550 candidates**.
- Maintained prior-art map: **PA01–PA104**.

The research program now explicitly includes weight-space, representation-space, latent-code, memory, routing/path, architecture, dynamic-state, and optimization-space functional coordinates.
