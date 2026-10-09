# Mirror Application Research Notes — 2026-10-08

Status: literature-grounded expansion and worker handoff; **no new MA training executed in this sweep**. The canonical registry is `experiments/mirror_applications/IDEA_REGISTRY.csv`.

## Eleventh research sweep: scope and conversion to Mirror experiments

The program continues to study the extra low-description functional parameter `m`:

`F(x; theta) -> F(x; theta, m)`.

A paper qualifies as a new Mirror integration target only when we can name: (1) the native physical object, (2) an insertion point for `m`, (3) the strongest native method and a cheap simple control, (4) a falsifiable metric, and (5) a circumstance where the candidate may fail. Generic novelty from adding an embedding, ordinary shared weights or an existing adapter is not claimed.

New primary prior art: **PA236–PA265**. New MA entries: **MA-876–MA-935 (60)**, 46 P0 and 14 P1, all UNTESTED. The existing MA-255 worker priority is **not** preempted.

### BN. Cross-model KV cache is now an empirically studied problem, not merely a Mirror conjecture

- **PA236** (Heo et al., 2026) demonstrates family-level source-to-target KV transfer with per-head ridge maps, source-layer selection and RoPE removal. Reported accuracy retention is **pair-dependent**, and some pairs fail. It reports prefill-reuse latency benefits in its tested systems.
- **PA237** (CacheBridge) tightens this baseline with head-aligned sparse mapper support and attention-sensitive sufficient-statistic calibration; it reports materially reduced mapper storage/construction/application costs.
- **PA238** (Mixture-of-Translators) adds multiple translator modules and a trajectory-correction loss for heterogeneous models, making target-state drift and injection position primary failure modes.
- **PA239** explores transfer across model scales, families, architectures and tokenizers. Differing tokenization creates a separate provenance/alignment problem, not just a new matrix to multiply.

**Mirror insertion:** Instead of keeping a full learned translator for each ordered source–target pair, learn one physical translator basis and codes `m_source`, `m_target`, `m_head`, `m_layer`. A target key/value map could be `T_{s->t,l,h} = T_shared + sum_j c_j(s,t,l,h) B_j`, or a proven structure-preserving product. This expression is an **experimental proposal**, not an established exact identity.

**Most decisive tests:** MA-876, MA-878, MA-880, MA-881, MA-887, MA-889.

**Required controls:** native Heo ridge, CacheBridge, MoT, independent per-pair affine or nonlinear translator, and native target re-prefill. Count mapper matrices, calibration corpus/compute, source-model prefill, transformer-side injection cost, cache layout changes and end-to-end target latency. Include 1K/4K/long contexts, switching dwell lengths, different attention configurations and failing model pairs. Hold out ordered model pairs entirely.

**Critical distinction:**
1. **Exact lazy Mirror sharing** (MA-691): known algebraic right transforms around a common compatible cached state can sometimes be moved to queries/output with FP32-equivalent results;
2. **Approximate cross-model translation** (PA236–PA239): separately trained models usually differ in intermediate hidden states and cache semantics. A fitted mapper may preserve quality, but is not automatically exact, reversible, transitive or cheaper than re-prefill.

Do not promote cache MSE or a source-only speed figure into decoder quality or system throughput.

### BO. Neural radiance fields: many physical scene networks versus one field and tiny scene code

- **PA243 C-NGP** already handles multiple scenes in one continual neural graphics model with scene pseudo-labels and replay. Scene-code conditioning by itself is therefore **not novel**.
- **PA244 ReFiNe** shares recursive hierarchical scene fields and compacts scene latent state.
- **PA245 Instant-NGP** uses trainable multiresolution hash feature tables as a large physical storage object; a small MLP decoder alone does not remove these costs.
- **PA246 TensoRF** compresses a radiance field with CP and vector-matrix tensor factorization.
- **PA247 K-Planes** uses paired space-time/appearance feature planes.

**Mirror insertion:** scene `m` should transform a shared physical field basis (hash features, tensor factors, hierarchical local blocks or shared decoder activations), rather than merely append an arbitrary scene ID to an MLP. Target MA-891–MA-895 and MA-902.

**Strong controls:** C-NGP, native ReFiNe scene latents, independent Instant-NGP hash tables, independent/shared decoder with scene embedding, CP/VM tensor basis of the same rank, K-Planes native factors. Pay for all feature grids/tables, scene coordinates, replay snapshots and all renderer weights.

**Failure mode:** unrelated scenes may share almost no high-frequency geometry or texture. Low scene-code bytes can conceal enormous per-scene feature hash grids or reconstruction time. Aligned synthetic scene morphing cannot certify arbitrary multi-scene compression.

### BP. Dynamic 4D Gaussian assets: canonical geometry is already standard prior art

- **PA248 4DGS** stores canonical 3D Gaussians with neural voxel features and a time-conditioned deformation decoder.
- **PA249 ADC-GS** reuses anchor structures and hierarchical spatial-temporal motion instead of deforming every Gaussian independently.
- **PA250 CC-4DGS** combines compact computational deformation, conditional appearance autoencoding and residual codes with a strong storage/runtime objective.
- **PA251 P-4DGS** uses temporal prediction and actual entropy coding; the rendered object cannot be evaluated by parameter count alone.
- **PA252 MRO-GWM** represents each object in canonical-frame Gaussians with rigid SE(3) action-dependent motion.
- **PA265 GenSplatCodec** combines geometry/appearance coding with one-step generative decoder and cross-view consistency constraints.

**Mirror insertion:** factor object/scene identity `m_scene`, motion `m_motion(t)`, pose `m_SE3`, time `m_t`, illumination/appearance `m_appearance` around canonical Gaussian assets and decoder. Candidate MA-896–MA-905.

**Controls and measurements:** native 4DGS/ADC-GS/CC-4DGS/P-4DGS, independent per-scene or per-motion Gaussian assets, and a normal scene-embedding+deformer. Report PSNR, LPIPS/SSIM, temporal and cross-view consistency, actual bitstream/asset bytes, renderer peak VRAM, random access, training time and measured FPS on named hardware. When rigid pose suffices, do **not** count it as a special Mirror compression discovery.

**Failure mode:** dynamic topology changes and occlusion may break a shared canonical orbit; predictive entropy coding can beat a dense code decoder on actual bitrate; naïve materialization can destroy the real-time FPS frontier.

### BQ. Multi-speaker speech: the native shared-adapter controls are already strong

- **PA253 NanoVoice** shares speaker adaptation parameters with a trainable scale matrix and batch-wise multi-speaker training. This is a direct precursor to speaker-indexed low-description parameter sharing.
- **PA254 HyperTTS** uses speaker-conditioned hypernetwork-generated adapters.
- **PA255 lightweight zero-shot TTS MoA** mixes speaker-conditioned decoder and variance adapters.
- **PA256 Hyper-MoA (2026)** integrates a hypernetwork and MoA for a shared multi-speaker adaptation module. A Mirror speaker-code bank must beat it.
- **PA257** learns interventional speaker/content subspaces for SSL speech models.
- **PA258 StableVC** separates linguistic content, timbre and style under conditional flow matching.
- **PA259 HybridCodec** separates semantic/acoustic streams with joint specialization.

**Mirror insertion:** `m_speaker`, `m_content`, `m_style`, `m_layer` and `m_codec` can replace *some* speaker-specific adapter/decoder parameters if they share a true latent structure. Candidate MA-906–MA-918.

**Controls:** native NanoVoice shared scales, HyperTTS generated adapter, native MoA/Hyper-MoA, StableVC native gated style/timbre conditioning, ordinary speaker embeddings, independent LoRA adapters. Evaluate real held-out speakers and language/accent/style combinations with speaker similarity, intelligibility (WER/CER), reconstruction/acoustic measures, listener ratings where feasible, incremental inference bytes per voice, adaptation cost and real-time factor.

**Guard:** do not assume speaker similarity proves preservation of words or prosody; do not claim a privacy/identity guarantee from a code. Use appropriately licensed/consented speech data.

### BR. The sampler and flow map are additional objects to Mirrorize

- **PA260 Flow Map Matching** learns a two-time transport map and unifies various consistency-model formulations.
- **PA261 Consistency Models** directly address the one/few-step generation tradeoff.
- **PA262 S4S** learns an efficient solver and optionally the discretization schedule for a fixed diffusion network.
- **PA263 LoRA.rar** already learns subject-style LoRA composition with a hypernetwork.
- **PA264 EST-LoRA** already performs timestep-dependent, training-free subject/style LoRA selection.

**Mirror insertion:** rather than duplicate a learned flow map, solver table, or personalization adapter for every task/schedule/budget, use `m_{t0}`, `m_{t1}`, `m_solver`, `m_subject`, `m_style`, `m_layer`. Candidate MA-919–MA-928.

**Controls:** native FMM interval conditioning, Consistency Model one/few-step, optimized S4S/S4S-Alt coefficients/schedules, LoRA.rar and EST-LoRA on the same subject/style source banks. Count true neural function evaluations (NFE), sampling GPU time, solver-controller cost, adapter bank bytes, subject/style fidelity and image quality. Hold out interval pairs and subject-style pairs; enforce flow-map semigroup/consistency checks.

**Failure mode:** a tiny parameter `m` can still require more NFEs, break the semigroup, or degrade perceptual/identity quality. Classical time embeddings and native optimized solvers are very strong simple controls.

### BS. Model stitching connects cross-model cache transfer with portable function coordinates

- **PA240 StitchLLM** dynamically routes pre-trained model blocks through learned stitching layers.
- **PA242** transfers SAE features, probes and steering directions between model residual streams with affine connectors.
- **PA241** warns that successful functional stitching alone can occur between representations with very different information contents, including unrelated tasks or structured noise.

**Mirror insertion:** replace an `O(N_models^2 × layers)` connector bank with `m_source × m_target × m_layer × m_feature` over one shared physical connector basis. Where possible, reuse a common coordinate map for both KV cache transfer and residual-stream feature transfer. Candidate MA-929–MA-935.

**Controls and falsification:** independent affine stitchers, StitchLLM routing, shared+private affine low-rank connectors, held-out source-target model pairs and unseen layer pairings. Measure next-token NLL, transferred probe/SAE feature fidelity, training/serving bytes, active compute and target-informational preservation. Successful output matching **does not prove** the models share the same knowledge or causal mechanism.

### BT. Mechanism hierarchy for worker prioritization

Prioritize an experiment only when it can test the **marginal value of Mirror parameter `m`**, not merely show that a related field has an existing efficient model. Each protocol must include:
1. native method (B(theta));
2. (B(theta,m)) with the minimal View insertion;
3. a cheaper ordinary code/gate/low-rank/native adapter that has comparable freedom;
4. an independent upper control and a naturally learned/misaligned target;
5. actual serializer bytes and runtime;
6. a fixed fresh split and a falsification gate.

For **cross-model KV**, first choose an exact/approximate classification and keep the native 2026 translators as controls. For **graphics**, measure end-to-end renderer/scene asset size and FPS; for **speech**, measure speaker/content/prosody separately; for **generative samplers**, count true NFEs; for **stitching**, include counterexamples to informational alignment.

### BU. Registry and worker handoff

- Literature additions: PA236–PA265 (30 primary sources).
- New experiments: MA-876..935 (**60**, P0 46 and P1 14; all UNTESTED).
- Registered candidate count: **935**.
- Previous verified 47 MA runs (29 PROMISING, 18 FAIL) are unchanged.
- Existing worker next MA-255 is unchanged; new candidates form an appended research family, not a queue preemption.
- The natural next follow-up after the locked MA-255/MA-260 crossover family is MA-876 or MA-880, because cross-model cache translation now has very strong direct empirical controls.

No reported model performance, bitrate, voice quality, rendering FPS or KV transfer speed from cited papers is a Mirror experimental result. Those are baseline observations from the source papers and must be replicated in the relevant harness before any adoption claim.
