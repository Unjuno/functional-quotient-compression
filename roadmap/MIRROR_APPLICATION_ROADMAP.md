# Mirror Application Validation Roadmap

Date: 2026-10-07 JST
Status: active backlog roadmap

## Goal

Systematically test whether the extra low-description Mirror parameter m can replace duplicated physical state or improve logical functional freedom in strong existing methods. The registry now has **1115** candidate experiments; MA-876..935 were added as UNTESTED research targets on 2026-10-08.

The unit of work is an MA-xxx entry from:
`experiments/mirror_applications/IDEA_REGISTRY.csv`.

## Stage 0 — Common harness

Use a stable nanoGPT-derived baseline plus synthetic mechanism fixtures.

Every candidate must expose:
- baseline and Mirror model construction;
- actual inference serialization;
- active-compute accounting;
- train/dev/fresh split discipline;
- common result schema.

Do not fork the training loop separately for every idea when a thin module injection is sufficient.

## Stage 1 — P0 family screens

### Literature-derived cross-over experiments

Before broad family screens, prefer the highest-information intersections with strong known baselines:
- MA-241 expert tying + Mirror depth-specific expert views;
- MA-244 K=V projection sharing + Mirror role recovery;
- MA-245 MLKV + Mirror per-layer KV views;
- MA-247 recursive tied block + Mirror depth modulation;
- MA-248 PTP-style packet latent as Mirror code;
- MA-249 shared future head + Mirror future-offset views;
- MA-250 MAP/Hadamard binding as Mirror expert address;
- MA-251 factorized expert x depth coordinates.

These are high-value because failure is also informative: each has a strong non-Mirror method that defines what Mirror must add.



### 1A. FFN / expert / adapter
MA-003, MA-005, MA-009, MA-019, MA-024.

Reason: strongest continuity with SRM results and easiest matched controls.

### 1B. Attention / KV
MA-041, MA-048, MA-061, MA-063.

Question: can physical head/KV multiplicity be reduced while preserving logical roles?

### 1C. Depth / position
MA-076, MA-079, MA-086, MA-116.

Question: can weight tying plus a small Mirror coordinate recover part of independent-layer/head-position flexibility?

### 1D. Temporal
MA-121, MA-129.

Build directly on TM001. Separate packet-level uncertainty from phase-slot parallelism.

### 1E. Compression / holographic binding
MA-156, MA-160, MA-171, MA-173, MA-181.

Question: can one paid representation decode into several useful logical functions more cheaply than independent residuals?

### 1F. Continual / optimization / distillation
MA-186, MA-189, MA-199, MA-208.

Question: can new skills or expert behavior be added mainly through a coordinate rather than duplicated weights?

## Research-expansion P0

The second literature sweep added direct controls from Parameter Superposition, BatchEnsemble, VeRA, IA3, OFT/BOFT/OFTv2, Compacter, Monarch matrices, Fast Weight Programmers, task-vector/model-merging methods, Cheap-LoRA/circulant adapters, and SETA.

Highest-information new P0 candidates:
- MA-255 parameter-superposition Mirror contexts;
- MA-257 compositional context groups;
- MA-258 superposed expert bank + Mirror unbinding;
- MA-260 BatchEnsemble Mirror ensemble;
- MA-261 rank-one logical experts;
- MA-265/266 VeRA-based Mirror adapter views;
- MA-268 IA3 vs richer Mirror activation views;
- MA-271/272/273 OFT/OFTv2/BOFT Mirror views;
- MA-274 BOFT logical experts;
- MA-276 BOFT depth views;
- MA-278 Compacter Mirror adapters;
- MA-282 Monarch Mirror FFN transforms;
- MA-286 Cheap-LoRA Mirror subspace views;
- MA-288 dynamic fast-weight Mirror state;
- MA-292 task-vector Mirror basis;
- MA-296 orthogonalized task-vector superposition;
- MA-297/299 SETA shared/private subspace + Mirror allocation.

These do not preempt an already-started worker experiment. They enter the queue after current active work and earlier locked cross-over candidates.

## Expansion lanes discovered by research

The backlog now contains **875 candidates**. The literature program now spans the following strategic lanes:

### A. Representation selection
Masks/supermasks and intrinsic subspaces ask whether a task needs a new weight transform at all, or only a compact selector/coordinate.

### B. Tensor and symmetry coordinates
Tucker/TT factors provide shared physical banks; Re-Basin and monomial symmetries identify directions that change coordinates without changing function. Symmetry audits are required before capacity claims.

### C. Elastic execution
MoD, early exit, slimmable/OFA/MatFormer and supernet methods make architecture/depth/width a logical coordinate. Mirror is tested as a low-cost correction on top of shared supernet weights.

### D. Module/address compression
Prompt pools, adapter composition, hash/compositional embeddings and associative/product-key memories expose large banks of small logical objects that may be represented by shared bases plus addresses.

These lanes enter only after current locked/active work and should reuse family-level harnesses.

### E. Dynamic functional coordinates

FiLM/StyleGAN/CondConv-style mechanisms make the functional coordinate input-dependent. Neural fields make it an instance latent. Meta-learning makes it an adapted latent. SSMs make it part of the dynamical state.

Research question: which representation of m gives the best storage/quality/runtime frontier for the same family of logical functions?

### F. Modular programs and editable memory

Path/routing networks make the coordinate a program over reusable modules. Model editors and codebooks make it a persistent behavioral memory.

Research question: can Mirror compress module programs or edit memories while preserving locality, routing correctness and lifelong retention?

### G. Discrete/rate-controlled addresses

VQ/RVQ and error-correcting codes expose an explicit number-of-bits axis.

Research question: how many bits of functional address are actually needed at a target quality/robustness, and when is residual/private capacity preferable?

### H. Representation-space functional coordinates

ReFT, activation steering, function vectors, SAEs and transcoders show that useful task/behavior coordinates can live directly in hidden-state space.

Research question: for a fixed behavior/task family, is the most storage-efficient functional coordinate a weight View, a hidden-state View, a sparse feature program, a latent code, or an external memory entry?

This lane must report inference-time intervention cost and off-target effects in addition to stored bytes.

### I. Structured transform coordinates

Householder, scaled-Cayley, GSOFT and low-displacement-rank families expand the View geometry beyond diagonal/low-rank/BOFT transforms.

Research question: which structured family gives the best useful-function quality per serialized byte and realized transform time?

### J. Model-merge geometry

DARE/DELLA sparsification, Fisher/RegMean weighting, KnOTS alignment and Model Stock expose different low-dimensional structures in task deltas.

Research question: after removing redundancy and alignment artifacts, can reusable Mirror capability atoms preserve both individual and composed task behavior?

### K. Posterior and ensemble coordinates

SWAG, Subspace Inference, Laplace, Packed-Ensembles and Snapshot Ensembles provide distinct ways to obtain several predictive models without training fully independent copies.

Research question: can uncertainty/diversity live in a compact View space without collapsing calibration or OOD detection?

### L. Neural-operator and relational coordinates

FNO/DeepONet represent operator families, while R-GCN/CompGCN already represent relation-specific functions with shared structure.

Research question: do factorized Mirror codes generalize to held-out physics/relation combinations rather than merely memorize task IDs?

### M. Diffusion-control coordinates

ControlNet, T2I-Adapter, IP-Adapter, Ctrl-Adapter and CtrLoRA create or reuse condition-specific control functions.

Research question: can control type x depth x timestep x target-backbone structure be represented by a shared control basis and compact composable Views?

### N. Self-organising rule coordinates

Neural Cellular Automata repeatedly apply one local rule, while goal conditioning, attention and online self-organisation make that rule/task state dynamic.

Research question: can a small transient Mirror state select stable global behaviors and adapt online without duplicating the local update network?

### O. Invertible coordinate charts

RealNVP/i-ResNet/INNSteer expose exact or near-exact coordinate changes where a small latent intervention becomes a nonlinear functional change in the original representation.

Research question: when does a shared nonlinear invertible chart beat linear/orthogonal/low-rank Views after inverse cost and cycle error are counted?

### P. Global expert budgets and logical recovery

MoRE/UniPool decouple experts from strict layer ownership, while M-SMoE/REAP/upcycling expose mergeable, prunable and expandable expert structure.

Research question: can the system learn an overcomplete expert population, compact it to fewer physical experts, then recover only behaviorally necessary distinctions through Mirror codes?

### Q. Compositional latent dynamics and control

Low-rank recurrent theories, Vector Networks, Koopman models and motor option banks all expose reusable computation atoms with task/state-dependent coefficients.

Research question: can persistent task coordinates and fast context coordinates be factorized while preserving held-out composition and long-horizon stability?

### R. Programmable architecture and task-matched state rank

Matrix-memory results expose task-dependent rank requirements; programmable neural graphs expose connectivity itself as an executable state.

Research question: can Mirror allocate state rank and graph structure only when demanded, without hiding storage in metadata or dynamic program synthesis?

### S. Functional edges and causal coordinates

KAN/GS-KAN expose learnable edge functions and shared parent functions; NTK/linearization and AI Engram expose tangent or Fisher-geometric task/memory directions.

Research question: is the cheapest functional coordinate a transformed edge function, a local tangent coordinate, or a causal memory trace rather than a weight-space adapter?

### T. Writable state versus stable-state adaptation

Differentiable plasticity, Hebbian fast weights and DeltaNet use online writable state; gain-modulated stable-synapse networks adapt through dynamic activation state without changing synapses.

Research question: how many writable bits and update FLOPs are actually required for Level-3 Mirror adaptation, and when can stable gain/context state replace fast weights?

### U. Learned model manifolds

Learned neural subspaces and Bezier mode-connectivity surfaces produce empirical low-loss coordinate spaces rather than choosing a transform family a priori.

Research question: does a learned low-loss manifold give a better rate-quality quotient than task vectors, random intrinsic dimensions or Model Stock interpolation?

### V. Collective and structural coordinates

Mesh Inference moves the functional coordinate into admission/communication policy over private agents, while Structural Composition encodes how reusable modules should be recombined.

Research question: can the project compress protocols and composition rules when the underlying physical models/modules themselves remain separate?

## Core execution doctrine

Across every strategic lane, the default question is not "is this adjacent method interesting?" It is:

> Where can the extra low-description Mirror parameter `m` be inserted into the native method, and what marginal functional freedom does it buy per byte/compute/interference cost?

Workers should preserve the native method as a direct baseline, insert `m` with minimal surgery, compare against the cheapest ordinary parameter that could provide similar freedom, and then sweep static/dynamic/factorized/shared-private variants where justified.

Broad literature exploration is therefore converted into **Mirror-parameter integration experiments**, not independent research detours.

### W. Cross-model KV translator manifold

New PA236–239 establish calibrated source/target KV transfer, compact head matching, translator mixtures and heterogeneous context reuse. MA-876..890 test whether multiple transfer maps can share a physical basis and cheap factorized Mirror codes. Strong native controls, exact/approximate separation, target model quality and end-to-end handoff costs are mandatory.

### X. Multi-scene neural fields and dynamic Gaussian assets

C-NGP, ReFiNe, Instant-NGP, TensoRF and K-Planes already share or compress scene representations. 4DGS, ADC-GS, CC-4DGS, P-4DGS and MRO-GWM supply canonical geometry/motion priors. MA-891..905 add scene/time/appearance/object/action m; test actual scene assets, PSNR/LPIPS and FPS against these native baselines.

### Y. Multi-speaker speech and acoustic coding

NanoVoice, HyperTTS, MoA and Hyper-MoA are direct shared-speaker adaptation baselines; StableVC, interventional content/speaker subspaces and HybridCodec separate voice semantics from acoustics. MA-906..918 test bytes per added speaker, intelligibility, timbre, prosody and rendering costs.

### Z. Flow-map sampling and style/subject code

Flow Map Matching, Consistency Models, S4S, LoRA.rar and EST-LoRA define already-efficient samplers and style merge operators. MA-919..928 insert factorized Mirror interval, solver, subject, style and timestep codes, measured at equal NFE and output-quality budgets.

### AA. Cross-model block and feature stitching

StitchLLM and cross-model residual-stream SAE feature transfer provide native affine connectors; functional alignment can mislead about shared information. MA-929..935 test source×target×layer View codes over a shared bridge with held-out pair and informational counterexamples.

### AB. Neural video: frame/chunk logical roles and predictive coding

MA-936..950 tests the extra Mirror parameter `m` around **already-shared** video INR and codec structures. NerVast (Fisher-selected partial sharing) and DCVC-UF (one chunk latent with parallel frame-specific decoders) are the closest native baselines; DCVC-RT and DCMVC test operational speed/context costs. Research question: can the remaining chunk/frame-private function state become a compact m at comparable video RD, real coded bits and decode FPS?

### AC. Soft-equivariant logical functions with a symmetry/gauge audit

MA-951..960 tests real task-specific functional differences over G-CNN/steerable/e3nn/EGNN group representations. Learnable symmetry constraints and parameter-free equivariance are existing controls. Exact transformed copies or gauge-equivalent parameterizations **do not** constitute additional independent learned capacity.

### AD. Spiking neural function coordinates

MA-961..970 tests neuron threshold, temporal normalization, leak and neuromodulatory Mirror codes around a common synaptic physical network. STL-SNN, TEBN, TACOS and event-sampling SNNs are direct controls. Target true spike/energy/latency per useful task and no oracle task labels in task-agnostic continual setups.

### AE. Reconfigurable optical matrix hardware as a physical W

MA-971..980 treats an optical MZI/PCM/diffractive substrate as the physical object and a small phase/program configuration m as a candidate shared logical function code. Phase programming itself is prior art (LightPro, coherent nanophotonic MZI, hybrid diffractive architectures). Mirror adoption needs fewer control bits/devices and lower measured switch+compute energy at preserved optical quality.

### AF. Wireless and physical beam-state factorization

MA-981..989 tests site×user×frequency/channel Mirror coordinates over shared physical beamformers, learned beam codebooks, RIS phase surfaces and CSI codecs. Existing NBL, Type-II, CsiNet and RIS optimizers are required controls. Count actual feedback/pilot/control bits, net spectral efficiency, reconfiguration and power.

### AG. Spatial-audio head transfer functions

MA-990..995 tests one shared HRTF neural spectral basis with listener/direction/head-pose m. RANF retrieval and anthropometric listener latents are closest prior methods. Target listener held-outs and physical binaural transfer quality (spectral distortion, ITD/ILD/localization) per stored bit and head-motion update time.

### AH. Constraint-preserving Mirror over atomistic potentials

MA996–1005 add material/chemistry codes to native equivariant MACE-MP/NequIP/MatterSim force fields. Conservation of forces as negative energy gradients, E3 equivariance and long-run trajectory stability are hard gates. Sparse equivariant fine-tuning and frozen transfer are direct controls; see PA296–301.

### AI. MRI unrolled inversion with immutable measurement consistency

MA1006–1015 insert m only into learned VarNet/MoDL/DUNE reconstruction priors or scanner adaptation, preserving k-space data consistency. D2SA test-time adaptation and SSDU split measurements are native comparators; see PA302–307.

### AJ. Quantum circuits under physical resources

MA1016–1023 factor variational ansatz/task gates into small Mirror codes over one compiled circuit. Data reuploading, TensorHyper-VQC, superposed parameter circuits and transpiled device costs are mandatory baselines; count gates, shots, qRAM and postselection; see PA308–311.

### AK. Vision-language prompts and long-video object memory

MA1024–1034 target shared prompt/adapter/memory state, with CoOp, CoCoOp, MaPLe, SAM2, SAM2Long and MoPEFT as native controls. Measure base-to-new class transfer, tracked-object J&F, prompt/memory bytes and FPS; see PA312–317.

### AL. Retrieval indexes and implicit quantizer codebooks

MA1035–1045 test compact task/shard/metric m over fixed ScaNN/RaBitQ/Matryoshka, QINCo/ColBERTv2/PLAID, DiskANN and PGM index. Measure true recall, encoded physical index bytes, SSD reads, P99 and update bounds. Isometric/gauge rank-preserving transformations do not constitute new independent search capability; see PA318–325.

### AN. Universal time-series foundation models

MA1046..1061 investigate an extra low-description Mirror coordinate over *already shared* Chronos, TimesFM, Moirai, PatchTST and related forecasting models. Most decisive comparisons: MA1049 PatchTST variate View, MA1052 TRACE-selected LoRA, MA1057 factorized frequency×horizon and MA1059 natural-series orbit/private frontier. DLinear and seasonal-naive are mandatory inexpensive controls; Chronos/Lag-Llama probabilistic baselines require CRPS/WQL and interval coverage. Chronological split integrity, unseen datasets and measured inference cost are hard gates. See PA326..336.

### AO. Memory-dominant recommendation embeddings and task experts

MA1062..1078 compare Mirror m to DHE's existing table-free hash generator, QR compositional embedding tables, TT-Rec optimized tensor cores/lookup, VQ-Rec semantic IDs, HSTU sequential histories and native MMoE task gates. The goal is additional useful CTR/ranking tasks or cheaper per-ID/field functions at matched AUC/NDCG, true storage and P95/P99 lookup. Simple item IDs are not free functional capacity. MA1063, MA1066, MA1070, MA1072 and MA1078 are highest-information. See PA337..343.

### AP. Physically conditioned Earth-observation sensors

MA1079..1095 stress test Mirror m with continuous wavelength, sensor hardware, ground sampling distance, observation time and EO task against **DOFA's wavelength-conditioned hypernetwork** and native AnySat/CROMA/Prithvi/TerraMind/AlphaEarth. MA1079, MA1080, MA1083, MA1090 and MA1095 separate true spectral/OOD transfer from ordinary sensor-ID conditioning. Compare complete held-out sensors, region/time splits, all filter-generation bytes and GPU runtime; SAR is not merely rotated optical color. See PA344..350.

### AQ. Gauge-invariant natural LoRA task geometry

MA1096–1099, MA1114–1115 evaluate naturally trained adapter deltas, not artificial teachers constructed from the Mirror family. The same `D=B@A` has infinitely many GL(r)-equivalent factor pairs, so all shared-B/shared-A and task-similarity claims require gauge-invariant projector/spectrum analysis. First use `research_intake/natural_lora_orbit_20261008` only as a mathematical/oracle weight-space screen; then train m from new-task examples and measure task quality, serialization and GPU execution.

Native controls: Compress then Serve (PA351), CtM (PA352, single-model merge objective), EigenLoRAx (PA353), VB-LoRA (PA354), MetaTT (PA355), pretrained singular-coefficient tuning (PA356), Pico (PA357), GLoRA (PA358), Share/LoDA (PA361/362) and independent LoRA. Do not infer task-delta alignment from stable pretrained W singular vectors alone.

### AR. Compact task-address versus already optimized adapter serving

MA1100–1109 compare structured m with clustered native adapter basis compression and factorized task×layer×matrix TT cores, natural heldout task adaptation, as well as client/task-specific shared/private residuals and text-conditioned hypernetwork generation. Report logical task quality, real basis+code+cluster+optimizer bytes, resident GPU state, adaptation time and serving throughput, not only parameter ratios.

### AS. Exact cache arithmetic versus low-rank multi-agent baselines

MA1110–1112 compare Mirror canonical-cache Views against *native* ICML 2026 LRAgent low-rank cache decomposition/Flash-LoRA-Attention (PA364), PReCache (PA365), aLoRA/standard-LoRA reuse (PA154–155), and exact MA691 algebra. Require physical cache aliasing, adapter-specific state, exact source-token provenance, target output quality, end-to-end prefill and TTFT. A common base plus small LR cache is already established prior art.

## KV-cache transformation lane

MA-691..700 test whether one physical canonical KV/cache latent can serve multiple logical Mirror Views.

Key distinction:
- final-only placement avoids cache invalidation by construction;
- cache-transformable Views support specialists that differ while reusing one physical cache;
- arbitrary earlier FFN changes are not automatically cache-transformable.

Start with MA-691 exact lazy-cache algebra, then RoPE/MLA/YOCO integration. Only after exact mechanics pass should this lane move to nanoGPT/language and GPU kernels.

## Stage 2 — Replication gate

A candidate moves beyond SCREENING only if:
- it beats or complements the cheapest simple control;
- the effect reproduces on fresh seeds/worlds;
- bytes and compute are reported;
- failure modes are bounded.

## Stage 3 — Combination tournament

Only combine mechanisms that pass individually.

Test factorized addresses before Cartesian-product parameter tables:

    m = (m_expert, m_head, m_depth, m_time)

Priority combinations:
1. Mirror-MoE x Mirror-LoRA;
2. Mirror-head x Mirror-KV;
3. Mirror-depth x Mirror-RoPE;
4. packet phase x Mirror-MoE;
5. holographic binding x sparse private residual;
6. continual Mirror code x shared expert basis.

A combination must beat the stronger component alone, not merely the original Dense baseline.

## Stage 4 — Natural-language nanoGPT gate

Promote mechanisms that survive synthetic controls into the stable nanoGPT testbed.

Initial language tasks:
- tiny Shakespeare for integration/debugging;
- a small BPE corpus for validation NLL;
- targeted held-out probes only when their interpretation is predeclared.

Measure:
- validation NLL;
- actual payload bytes;
- training tokens and updates;
- active compute;
- throughput;
- retained capability after adding new views.

## Stop criteria

- If a simpler low-rank/gate/weight-tying control matches the candidate, do not claim Mirror-specific value.
- If logical multiplicity rises but quality collapses, do not count the addresses as useful experts/heads/layers.
- If storage falls but active compute or latency becomes impractical, keep the result as storage-only.
- Preserve negative results in the registry.

## Status updates

Registry status vocabulary:
UNTESTED -> SCREENING -> PROMISING / FAIL -> REPLICATED -> ADOPTED.

Only development evidence changes priority. Audit/fresh results change scientific status, not hyperparameters.
