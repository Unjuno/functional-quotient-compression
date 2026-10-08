# Worker Context Router

Purpose: give an experiment worker the **smallest sufficient context** for one MA run. Do not read the entire repository before coding.

## Always load — in this order

1. `AGENTS.md`
2. `GOAL.md`
3. `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
4. `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`
5. `docs/phase2/LATEST_WORKER_FINDINGS.md`
6. `experiments/mirror_applications/STATUS_BOARD.md`
7. exactly one selected row from `IDEA_REGISTRY.csv`
8. only the PA headings named by that row's `prior_art_refs`
9. `EXPERIMENT_CONTRACT.md`
10. `TEMPLATE/`

Then load family-specific evidence below.

## Mirror parameter translation rule

Before loading family-specific material, translate the selected candidate into four fields:

1. native method and physical object;
2. exact insertion point for Mirror parameter `m`;
3. cheapest native/non-Mirror parameter that could provide the same freedom;
4. claimed marginal benefit of `m` in bytes, compute, interference, adaptation, reuse, or functional multiplicity.

If these four fields are not clear, scope the candidate before coding.

## Family-specific context

### MoE / experts / shared FFN
Read:
- SRM001 report for multi-selection/routing;
- SRM002 for shared/private decomposition;
- MA-241 finding for aligned tied-expert views;
- MA-253 finding for misaligned private LoRA variation and cache-safe placement.

Primary question: what fraction of expert variation is shared-coordinate versus private?

### Attention / KV / GQA / MLA
Read:
- MA-244 and MA-245 summaries in LATEST_WORKER_FINDINGS;
- PA07, PA08 and PA58 when referenced.

Keep separate:
- projection/model bytes;
- actual KV-cache bytes;
- bandwidth/materialization;
- quality;
- wall-clock.

Do not infer cache reuse from parameter sharing alone.

### Depth / recurrent / equilibrium
Read:
- SRM003 for the executor/state-passing bottleneck;
- MA-241 for layer-view aligned feasibility;
- PA06, PA66, PA67 as referenced.

Always compare:
hard tying -> step embedding/diagonal -> static low-rank -> Mirror -> input-conditioned/generated modulation.

### LoRA / adapters / PEFT composition
Read:
- SRM002 shared/private result;
- PA18/19/21/23/29 and PA91/92 when referenced.

Default baseline ladder:
IA3 -> BatchEnsemble rank-one -> VeRA/shared low-rank -> candidate -> candidate+private residual -> independent LoRA.

### LoRA composition libraries
Read PA148–149 when referenced.

Compare code-space precomposition against executing multiple full LoRAs. Active adapter FLOPs are a first-class metric.

### Temporal / packet / parallel streams
Read:
- TM001 first.
- PA09/10 for multi-token/parallel-token;
- PA153 for multi-stream when referenced.

Joint consistency is mandatory. Per-slot token accuracy alone is insufficient.

### Quantization / gauge
Read:
- symmetry-audit section of LATEST_WORKER_FINDINGS;
- PA112/113/114.

Exact function-preserving rotations/scales count as zero functional multiplicity but may still improve quantization. Full-precision equivalence must be tested.

### Masks / subnetworks / elastic supernets
Read PA31/32/50/51/52/53 as referenced.

Count:
- mask/config bits;
- shared physical weights;
- active parameters/FLOPs;
- quality at each submodel.

Do not treat combinatorial subnetwork count as capacity.

### Tensor / structured matrices
Read PA21/23/24/34/35/36/45 as referenced.

Measure actual runtime: structured asymptotic savings may lose to dense GEMM at small scale.

### Modular execution / function codes
Read PA127–128 as referenced.

Keep routing/composition and function-parameterization errors separate. Neural Interpreter is a direct baseline for compact function code + shared executor.

### Function codes / meta-learning / implicit representations
Read PA128–133 as referenced.

The direct question is not whether small codes can represent functions — prior art establishes that — but whether Mirror structure improves:
- code size;
- composition;
- fitting speed;
- robustness;
- interpolation;
- private-residual need.

### Model editing / episodic memory
Read PA135–141 and PA152 as referenced.

Always report:
- edit efficacy;
- paraphrase/generalization;
- locality;
- sequential retention;
- bytes per edit/episode;
- lookup/update latency.

Explicit memory is a strong control; parameter compression is not automatically superior.

### Test-time writable memory / SSM
Read PA134 and PA144–147.

Separate:
- fixed model parameters;
- writable state;
- per-sequence/session state;
- update compute;
- persistent storage.

### Federated personalization
Read PA38–40.

Separate server/global bytes, per-client state, communication and client compute.

### Representation-space / function-vector / sparse-feature experiments
Read the PA references in the live registry row; these candidates may have been added concurrently by another research agent. Do not assume their current IDs from an older document.

### Cross-model KV translation and handoff (MA-876..890)

Read PA236–PA239 and `docs/phase2/MIRROR_KV_CACHE_REUSE.md`. Compare exact algebra and approximate fitted transfer as **different evidence lanes**. Native per-pair ridge, CacheBridge, MoT and re-prefill are mandatory relevant baselines. Count calibration/construction costs, map bytes, source prefill, target token provenance and target NLL, attention error, switching dwell length and end-to-end latency. Hold out ordered model pairs.

### Neural scenes, NeRF, dynamic Gaussian assets (MA-891..905)

Read PA243–PA252 and PA265. Use native C-NGP, ReFiNe, Instant-NGP, TensoRF, K-Planes, 4DGS, ADC-GS, CC-4DGS, P-4DGS as relevant controls. Report full hash/field/anchor/scene bytes and coded bitstream, PSNR, LPIPS, temporal/cross-view consistency, encoder time, actual FPS and peak VRAM.

### Speech adaptation and neural audio codecs (MA-906..918)

Read PA253–PA259. NanoVoice, HyperTTS, zero-shot MoA and Hyper-MoA already share speaker parameters. Test added Mirror `m` beyond these, with held-out speakers, speaker identity and intelligibility separately, voice/codec marginal bytes, realtime factor, and appropriately licensed/consented audio.

### Flow maps, learned samplers and diffusion LoRA fusion (MA-919..928)

Read PA260–PA264. Preserve native FMM interval conditioning, Consistency, S4S, LoRA.rar and EST-LoRA controls. Count NFEs, time/schedule/solver/LoRA state, measured GPU latency, FID/fidelity, and held-out interval/subject-style compositions.

### Cross-model stitching and information retention (MA-929..935)

Read PA240–PA242. Compare StitchLLM and per-model affine residual-stream stitchers. Evaluate target NLL, transferred SAE/probe feature consistency, connector/router bytes, held-out model/layer pairs, and informational/counterfactual negative controls. Output match alone does not prove shared knowledge.

### Neural video rate-distortion and chunked INR (MA-936..950)

Read PA266..274. Compare NerVast selected shared/private masks, DCVC-UF chunk-parallel decoder, DCVC-RT, native context modulation, HNeRV/CoANeRV/nested video codec. Use actual encoded bpp/BD-rate, PSNR/MS-SSIM, encoder time, seek time, FPS and VRAM. Analyze scene-cut failures and hold-out video segments; preserve same chunk latent and full control costs.

### Group equivariance, irrep and learned symmetries (MA-951..960)

Read PA275..280 and PA294. Native G-CNN/steerable/e3nn/EGNN and learned soft symmetry are strong controls; zero-parameter equivariance penalty is a strong byte control. **Exact gauge/covariant copies are zero independently learned functional multiplicity.** Only m that changes useful task behavior beyond symmetry earns Mirror-specific value.

### Spiking thresholds, time gains and neuromodulation (MA-961..970)

Read PA281..285. STL-SNN thresholds, TEBN time steps, TACOS task-free continual control and EAS-SNN event sampling are native mechanisms. Count spike operations and energy, temporal precision, membrane/synaptic bytes, event latency and retention. No oracle task boundary in task-agnostic comparisons.

### Programmable photonic physical operators (MA-971..980)

Read PA286, PA287 and PA295. LightPro/MZI/diffractive optical programming itself is prior art; only a smaller learned m with reduced reconfiguration/energy/control memory beyond native full programming is a Mirror-specific proposal. Distinguish circuit models and real chips; count loss/crosstalk, programmable devices, thermal drift, total power and switching milliseconds.

### Beam codebooks, CSI feedback and RIS (MA-981..989)

Read PA288..291. Native beamspace codebook, Type-II feedback, CsiNet and RIS phase optimization already give compact controls. Use net spectral efficiency, pilot/feedback bits, RF hardware constraints, dynamic channel variations, energy and reconfiguration latency; hold out wireless site/user/frequency combinations.

### Personalized spatial HRTFs (MA-990..995)

Read PA292..293. RANF/retrieval field and anthropometric latent codes are direct baselines. Use held-out listener/direction pairs, binaural spectral/phase distortion, interaural timing/levels, measurement budget, file bytes and streaming head-pose update cost.

### Atomistic materials and conservative forces (MA-996..1005)

Read PA296..301. MACE-MP, NequIP, MatterSim, equivariant-sparse fine-tuning and frozen transfer are strong native baselines. A Mirror code m enters equivariant energy features; forces must be derived from the same scalar energy and audited for E3 covariance, energy/force errors, MD stability and actual adapter/neighbor compute. Hold out chemical families.

### MRI physics and scan adaptation (MA-1006..1015)

Read PA302..307. Use VarNet, MoDL, DUNE, D2SA and SSDU. Mirror codes modulate learned regularizers/latent priors, **not** acquired k-space measurements or known forward physics. Keep multicoil sensitivity and data consistency, disjoint k-space self-supervision, scanner/contrast held-outs, image metrics, resource bytes and runtime.

### Variational quantum circuits (MA-1016..1023)

Read PA308..311. Quantum data reuploading, TensorHyper-VQC TT generated parameters, superposed parameter circuits, and transpiled hardware-aware controls are prior art. Count circuit postcompilation depth/entangling gates, qubits, classical code/generator state, hardware noise and shots; qRAM/postselection failure costs are not free.

### Visual foundation prompts and object memory (MA-1024..1034)

Read PA312..317. Native CoOp/CoCoOp/MaPLe coupled prompts, SAM2 per-object streaming memory, SAM2Long path tree, and MoPEFT are mandatory nearest controls as relevant. Code only counts if it reduces physical prompt/adapter/object-memory state without losing base-to-new/OOD performance or object J&F, occlusion recovery or FPS.

### Retrieval ANN indexes and ranking policies (MA-1035..1045)

Read PA318..325. Preserve ScaNN/RaBitQ bounded quantizer, Matryoshka nested prefixes, ColBERTv2 residual token vectors, PLAID centroid pruning, QINCo generated codebook, DiskANN disk traversal or PGM correctness guarantee as applicable. Compare ranking recall/nDCG/MRR and complete index+codebook/generator memory, update cost and tail latency. Exact isometric m that leaves rankings unchanged is not independent learned capacity.

### Time-series forecasting and temporal foundation models (MA-1046..1061)

Read PA326..336, the fourteenth research notes and the selected MA hypothesis only. Native Chronos/TimesFM/Moirai/PatchTST/TimeMixer/iTransformer/TRACE are stronger than a freshly trained weak baseline; DLinear/seasonal naive and simple scalar/FiLM/rank-1 are minimum-cost controls. Lock chronological splits, frequency×horizon held-out pairs and OOD series. Report forecast quality (MASE/WQL/CRPS/coverage as applicable), trained/frozen state and real throughput.

### Recommendation and high-cardinality ID embeddings (MA-1062..1078)

Read PA337..343. DHE requires no per-ID embedding table, TT-Rec has tensor-train lookup kernels/caches, and QR/VQ-Rec already give compact item codes. Compare the native source method with native plus Mirror m and a simpler field/task gate. Evaluate cold IDs, CTR AUC/logloss, sequential ranking, full ID/codebook/table/generator bytes, GPU memory and QPS/P99. Per-ID m states count linearly with ID cardinality; no free IDs, item-frequency hindsight or private data.

### Earth-observation multi-sensor foundation models (MA-1079..1095)

Read PA344..350. DOFA is a direct dynamic wavelength-to-filter hypernetwork; AnySat/SatMAE, CROMA, Prithvi, TerraMind and AlphaEarth provide multisensor conditioning and task baselines. Report sensor-response/calibration metadata, generated filters, pixel/GSD/band counts, OOD sensor/held-out wavelength combinations, geographic/time-separated land cover/change/segmentation metrics, bytes and GPU inference wall-time. SAR and optical sensors cannot be assumed information-equivalent. Require actual functional advantage beyond one sensor ID or ordinary spectral mask.

## Historical result loading rule

Do **not** load every SRM/MS/MN document.

Load an old report only if:
1. selected MA row or this router names it; or
2. the current hypothesis repeats the same physical object/mechanism.

This prevents old superseded architecture narratives from consuming context.

## Orbit/private protocol

If the candidate is a functional View intended to replace independent copies, prefer an interpolation teacher:

`Delta(alpha) = alpha * Delta_view + (1-alpha) * Delta_private`

with development sweep over alpha and fresh locked evaluation for the selected frontier points.

Aligned-only screens answer feasibility.
Misaligned-only screens answer a lower bound.
The frontier answers the real compression question.

## Runtime escalation

Eager Python structured transforms have already caused false-negative runtime Pareto positions in MA-241/244/245.

If:
- quality/storage passes;
- analytical compute is modest;
- eager wall time loses;

then run one vectorized/compiled/folded implementation check before declaring a fundamental runtime failure. Preserve the eager result.

## Concurrent agents

Before starting:
1. search existing `research/ma-*` branches;
2. do not duplicate an active MA ID;
3. before adding registry IDs, re-read the live registry and allocate from max+1;
4. re-read after commit and assert no duplicates.

## What to put in the worker's active context

Keep only:
- selected hypothesis;
- selected protocol;
- strongest 3–6 controls;
- direct prior-art facts;
- latest relevant worker findings;
- current raw results/errors.

Do not keep hundreds of candidate ideas in working context after selecting the experiment.
