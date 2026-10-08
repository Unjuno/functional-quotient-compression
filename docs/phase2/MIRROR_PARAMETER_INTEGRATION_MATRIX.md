# Mirror Parameter Integration Matrix

Date: 2026-10-08 JST  
Status: active breadth-first validation map

Read [MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md](MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md) first.

The common operation is:

`F(x; theta) -> F(x; theta, m)`

The table below tells workers where to try the same extra low-description functional parameter `m` in different established methods.

| Family | Native physical object | Minimal Mirror insertion | Mirror question | Strong direct control |
|---|---|---|---|---|
| MoE | expert FFNs / expert pool | expert/layer View `m` around shared/reused expert | fewer physical experts for useful logical specialization? | tied expert + embedding, BatchEnsemble, MoRE/UniPool |
| LoRA / adapters | adapter matrices / low-rank basis | task/layer coefficient or transform `m` | many logical adapters from fewer physical bases? | LoRA, VeRA, IA3, AdapterFusion |
| Attention heads | Q/K/V/O projections | head/role `m` on shared projection | fewer physical head projections? | MQA/GQA, QKV sharing |
| KV cache | canonical K/V or latent cache | view-specific key/value/readout `m` | switch logical views while reusing physical cache? | MLKV, MLA, ordinary cache duplication |
| Tied depth | recurrent/shared block | layer/step `m` | recover layer-specific function without new block weights? | ALBERT/Universal Transformer, timestep embedding, static LoRA |
| FFN roles | shared nonlinear block | hidden-feature / function `m` | multiple logical nonlinear roles from one MLP? | gate, IA3, CondConv |
| KAN | parent/basis edge function | edge/task transform `m` | many edge functions from shared physical function bank? | standard KAN, GS-KAN |
| Embeddings | shared embedding table/basis | domain/token/source `m` | domain-specific logical embeddings without tables per domain? | factorized embeddings, hash embeddings |
| SSM / recurrent dynamics | state transition/read/write operator | task/role/dynamics `m` | logical dynamical experts from shared transition basis? | S4/Mamba native conditioning |
| Fast weights / DeltaNet | writable matrix state | low-rank/state/update-rule `m` | smaller writable state for same online function? | full Hebbian/DeltaNet state |
| Plasticity | slow weights + plasticity rule | task/context plasticity `m` | multiple online learners from one physical network? | differentiable plasticity, learned optimizer |
| Representation steering | activation/intervention basis | behavior/condition `m` | many behaviors from one intervention basis/chart? | ReFT, linear steering, INNSteer |
| Function vectors | causal function-vector basis | task/composition `m` | compact reusable function library? | explicit function vectors |
| Model editing | edit basis / memory bank | fact/edit `m` | more edits per stored physical basis? | ROME, MEMIT, GRACE, SERAC |
| AI Engram | causal memory traces | memory coefficient `m` | causal memory compression and reversible manipulation? | raw engrams, edit-vector baselines |
| Task vectors / merging | task delta basis | task/composition `m` | logical task models from compact shared delta atoms? | Task Arithmetic, TIES, DARE, KnOTS |
| Model manifold | learned subspace/surface | manifold-position `m` | useful logical models addressed by few coordinates? | subspace training, Model Soups |
| Ensembles / posterior | member/posterior subspace | member/sample `m` | diversity/uncertainty without model copies? | BatchEnsemble, SWAG, Packed Ensemble |
| Federated personalization | global model / shared adapter basis | client `m` | many clients with small persistent state? | pFedHN, client LoRA |
| Optimizer geometry | gradient/update subspace | task/layer update `m` | reusable optimization rules/subspaces? | GaLore, Meta-SGD |
| Neural operators | operator basis | PDE/boundary/regime `m` | many physical-system operators from one basis? | FNO, DeepONet, KNO |
| Relational GNN | relation transform basis | relation/schema `m` | many logical relation transforms? | R-GCN basis decomposition, CompGCN |
| Diffusion control | control branch/adapter basis | control/depth/timestep `m` | replace banks of control adapters/branches? | ControlNet, T2I-Adapter, Ctrl-Adapter |
| Robot policy | policy/controller/option adapter | skill/embodiment `m` | add skills without full policy/adapter copy? | option adapter bank, LoRA policy |
| Neural cellular automata | local update rule | goal/rule/damage `m` | many global behaviors from one local rule? | GoalNCA, Attention NCA |
| Programmable graph | reusable node functions / substrate | topology/program `m` | logical architectures as compact state? | GrapNet, routing networks |
| Collective inference | private agents + communication policy | admission/coalition `m` | task-specific collective protocols without model sharing? | Mesh Inference native policy |
| Packet / multi-token decoding | future head / packet state | slot/branch/phase `m` | multiple future roles from fewer heads/state? | PTP / explicit slot heads |
| Quantization / gauge | quantized shared weight | gauge/codebook `m` | lower rate/error through addressable gauge? | QuaRot / codebook quantization |
| Supernets / elastic models | shared supernetwork | width/depth/architecture `m` | cheap correction for many submodels? | OFA, MatFormer, US-Net |

| Cross-model KV translation | source-target head/layer translator maps | model-pair/head/layer `m` over shared translator basis | reduce N-by-N handoff map storage with target quality? | Heo ridge, CacheBridge, MoT, native re-prefill |
| Multi-scene NeRF | per-scene hash/tensor fields and decoder | scene/time `m` on shared field basis | more useful rendered scenes per physical field? | C-NGP, ReFiNe, Instant-NGP, TensoRF |
| 4D Gaussian assets | canonical Gaussian anchors and deformers | scene/time/appearance/SE3 `m` | dynamic logical assets from shared geometry? | 4DGS, ADC-GS, CC-4DGS, P-4DGS |
| Multi-speaker TTS | speaker scales / hypernetwork / MoA adapters | speaker×layer×content×style `m` | new voices with small marginal state? | NanoVoice, HyperTTS, MoA, Hyper-MoA |
| Neural speech codec | semantic and acoustic streams/codebooks | stream/speaker `m` | shared codec functions at fixed bitrate/fidelity? | HybridCodec, SoundStream/EnCodec |
| Flow-map generator | two-time transport operator | start-time×end-time `m` | many flow maps from one backbone? | Flow Map Matching, Consistency Models |
| Diffusion solver | optimized solver coefficient/schedule | sampler/budget/state `m` | multiple useful fast samplers at low code cost? | S4S, S4S-Alt |
| Image personalization | subject/style LoRA banks and merger | subject×style×timestep `m` | adaptive composition without per-pair adapters? | LoRA.rar, EST-LoRA |
| Cross-model stitching | pretrained model blocks/affine feature maps | source×target×layer/feature `m` | portable functions with fewer paid connectors? | StitchLLM, affine feature stitching, information audit |

| Neural video segment codec | Fisher-shared INR parameters / chunk latent frame decoder | scene×chunk×slot/bitrate `m` | many useful frames/chunks without independent group weights? | NerVast, DCVC-UF, DCVC-RT, HNeRV |
| Group-equivariant features | steerable filters / irrep tensor-product weights | task×group/irrep `m` | extra task behavior beyond zero-information group orbit? | G-CNN, e3nn, EGNN, learned symmetry, no-parameter approximate equivariance |
| Spiking neurons | shared synapses and native plasticity | task×time threshold/leak/neuromodulation `m` | distinct learned temporal behaviors without duplicating synapses? | STL-SNN, TEBN, TACOS, EAS-SNN |
| Photonic programmable MVM | one physical MZI/PCM/diffractive processor | sparse phase/coupler configuration `m` | multiple tasks per chip at less programming state/energy? | LightPro, MZI photonic circuits, MDR-HDONN |
| Wireless beam control | physical antenna/RIS and learned beam codebook | site×user×frequency/channel `m` | compact logical beams at fixed signaling and spectral efficiency? | NBL, Type-II CSI, RIS optimization, CsiNet |
| Personalized HRTF | shared neural spectral field and retrieval data | listener×direction×head-pose `m` | individualized binaural functions with fewer measurements/bytes? | RANF, anthropometric HRTF latent |

| Atomistic materials / force fields | equivariant MACE/NequIP potential and sparse fine-tune support | material×element×environment m | domain-specific conservative forces at fewer stored bytes? | MACE-MP, sparse E3 fine-tuning, NequIP |
| Physics-consistent MRI | VarNet/MoDL learned regularizer and data-consistency projection | scanner×contrast×unroll-step m | reconstruct different acquisitions without independent full priors? | VarNet, MoDL, DUNE, D2SA, SSDU |
| Quantum circuit parameters | fixed compiled variational ansatz and classical TT parameter generator | task×gate-block m | logical circuits sharing a paid physical ansatz? | Data reuploading, TensorHyper-VQC, hardware transpiler |
| Visual-language prompts | frozen CLIP and coupled visual/text prompt basis | class×domain×modality m | useful base-to-new visual functions per prompt byte? | CoOp, CoCoOp, MaPLe |
| Video object segmentation | SAM2 temporal memory and mask decoder with per-object states | object×time×memory-path m | more tracked objects per physical memory with stable masks? | SAM2, SAM2Long, MoPEFT |
| ANN retrieval index | shared codebooks, late-interaction centroids, disk graph index | task×query×shard×budget m | useful ranking change without rebuilding index? | RaBitQ, ScaNN, ColBERTv2, PLAID, QINCo, DiskANN |

| Universal time-series forecasting | shared Chronos/TimesFM/Moirai/PatchTST backbone | frequency×variate×horizon×regime `m` | meaningful calibrated forecasts without private heads? | native foundation/TRACE, PatchTST, DLinear, seasonal-naive |
| Categorical recommendation embeddings | DLRM table, DHE generator or TT-Rec tensor cores | field×domain×rare-ID `m` | quality per bit beyond table-free/TT embeddings? | DHE, QR, TT-Rec, VQ-Rec and dense table |
| Multi-objective recommendation | MMoE/HSTU shared expert and history model | task×region×session `m` | new logical objective functions beyond native task gates? | MMoE, HSTU, rank-1 gates |
| Multispectral Earth observation | shared DOFA wavelength hypernet / AnySat encoder | sensor×wavelength×resolution×time `m` | physical multisensor transfer at lower incremental state? | DOFA, AnySat, SatMAE, CROMA, Prithvi |
| Multimodal EO translation | TerraMind multimodal tokenizer/decoder and CROMA fusion | source modality×target modality `m` | cheaper verified cross-sensor functions than native any-to-any? | TerraMind, CROMA, missing-modality controls |

## Required variants after a direct screen

A family that passes a first mechanism screen should usually be expanded in this order:

1. **static `m`** — one learned/stored code per logical function;
2. **factorized `m`** — e.g. task x layer, expert x depth, condition x behavior;
3. **dynamic `m(x,h)`** — generated from context/history when justified;
4. **shared + private residual** — quantify the part of natural variation that does not fit `m`;
5. **composition** — signed/sparse/product/sequential combinations only after individual functions work.

Do not jump directly to the most complicated variant.

## Breadth priority

Prefer families with:
1. a clearly duplicated physical object;
2. a cheap native baseline;
3. a low-cost insertion point for `m`;
4. a measurable storage/state/cache benefit;
5. a controlled aligned test plus a natural/misaligned test.

The purpose is to map where the **same Mirror parameter principle** survives contact with different architectures.
