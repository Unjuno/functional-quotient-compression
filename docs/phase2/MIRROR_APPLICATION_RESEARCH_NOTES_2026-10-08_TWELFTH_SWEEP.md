# Mirror Application Research — Twelfth Literature Sweep (2026-10-08 JST)

**Type:** research intake, prior-art controls and pre-experiment protocols. No new Mirror model training, fresh-world evaluation or runtime measurement was executed here.

**Baseline:** 935 registered MA IDs and PA01..PA265; 47 completed MA experiments (29 PROMISING, 18 FAIL).  
**Additions:** MA-936..995 (**60**, **47 P0 + 13 P1**, all UNTESTED); PA266..295 (**30** cited primary prior works).  
**Post-sweep totals:** MA-001..995 (**995** candidates), PA01..PA295, 29 PROMISING / 18 FAIL / **948 UNTESTED**.

## Research objective: stress-test the extra Mirror parameter m

This continuation follows `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`.

The hypothesis is **not** that a group rotation, learned temporal embedding, spike threshold, optical phase shifter, beamforming codeword, or personalized audio listener embedding is itself newly discovered. Those are established methods and are native controls.

The central experimental question is: can a low-description extra functional coordinate `m` be integrated into an already-effective native architecture such that:

`B(x; theta) -> B(x; theta, m)`,

and measured quality, useful functional multiplicity, serialized bytes, runtime, interference, or adaptation improve beyond the relevant native code/gate/adapter at matched resource budgets?

The common mistakes to prevent are: comparing only against a dense unshared baseline; treating exact group orbit as independently learned new information; ignoring real encoded bitstreams; counting no device/control energy for optical or wireless switching; and claiming energy/throughput wins from FLOP or parameter counts alone.

## BV. Neural video coding: the most immediate computational application

**Direct prior PA266..274.**

- NeRV (PA266) and HNeRV (PA267) already represent a video as a shared neural decoding function queried by time/feature codes. A time coordinate alone is **not** Mirror novelty.
- **NerVast (PA268, WACV 2026)** is unusually close to the shared/private Mirror idea. It groups temporally similar video chunks, selects partial parameter-sharing masks with approximate Fisher sensitivity and jointly trains shared and private INR parameters. Native research reports average 39.9% parameter reduction over matched compute-efficient INR references. Mirror must be inserted **after** this strong sharing step to replace some of the remaining private chunk weight state. Candidate **MA-936 and MA-938**.
- **DCVC-UF (PA269, CVPR 2026)** encodes several frames in a chunk as **one latent**, then reconstructs frames in parallel with frame-specific decoding and consolidated entropy interactions. The shared chunk latent exists already. Mirror must compress the **frame-specific decoder roles/weights** using a small frame-relative `m`, not claim invention of parallel video coding. Candidate **MA-941, MA-942**.
- **DCVC-RT (PA270)** targets actual memory-I/O and kernel-launch overhead beyond arithmetic. It is a critical runtime control for a structured View that might decrease serialized bytes but create extra materialization.
- DCMVC (PA271) already uses reference-based temporal context modulation. Mirror must beat the native context compensation (MA-944).
- CoANeRV (PA272) already uses per-video tokens + common coordinate decoder; S-NVRC (PA273) already supports nested rate/complexity prefixes; SIEDD (PA274) already shares an encoder and uses lightweight video group decoders. These are direct native controls for MA-945..947.

**Proposed first screen: MA-936 vs NerVast.** Use the exact same fitted video chunk checkpoints and Fisher masks, freeze common weights, then compare three ways of encoding the remaining chunk-private delta: (a) native NerVast private weights, (b) equal-byte rank-r or FiLM code, (c) structured Mirror `m`, plus (d) sparse private residual on m. Hold out scene-change windows; follow with UVG or comparable public content. Run rate-distortion-complexity audit on actual encoded bitstreams, PSNR/MS-SSIM and observed decoding FPS. A result aligned with a generated synthetic Mirror teacher remains MECHANISM-only.

**Second screen: MA-941 vs DCVC-UF.** Preserve chunk latent encoder and entropy coder; replace/reparameterize only frame-slot adaptation. Sweep packet/chunk length and slot factorization, and test unseen motion/scene-cut windows. Compare full frame-slot decoder, shared decoder with ordinary offset embedding, low-rank head, and Mirror View. Joint decoder consistency, actual coded bpp, CPU/GPU decoding latency, VRAM and random access are mandatory. Avoid extrapolating from TM001 token packets to continuous video.

**Failure modes:** A shared (m) cannot encode genuinely new high-frequency private scene content; learned decoder View can increase arithmetic and decoder calls; scene cuts break low-rank temporal orbit; classical/native predictive residual entropy coding can have a lower real bitrate; a low-bit parameter count is not a coded video rate claim.

## BW. Equivariance: necessary falsification of “new logical functions”

**Direct prior PA275..280 and PA294.**

G-CNNs (PA275), Steerable CNNs (PA276), EGNNs (PA277) and e3nn (PA278) already encode group symmetry and shared representations. Learned symmetry/soft-sharing methods (PA279; SEMoLA PA280) can discover a useful group action; PA294 provides a strong **parameter-free approximate equivariance** control.

A group View (g) acting on a function by pre/post-transformation may be exactly known and cost few bits. This **does not** create new independent learned information, and may not change the physical function at all when the target readout is invariant.

**Mirror-specific hypothesis:** `m_task` selectively breaks or modulates symmetry within a shared group/irrep representation without allocating a separate learned full filter/tensor bank per task. Candidate **MA-951..960**. Count all equivariant kernel/irrep basis storage, Wigner/CG execution where relevant, subgroup metadata and conditional code generation.

**First tests:** (a) rotate/permutate inputs under exact equivariant sharing (zero-independent-capacity audit), (b) train tasks that genuinely require different outputs under identical geometry, (c) sweep learned symmetry breaking `m` versus simple task gate/rank-1 residual and native G-CNN, (d) hold out task×transformation combinations. Measure task generalization and equivariance deviation separately. If all benefits can be explained by exact group relabeling or a zero-parameter regularizer, assign no Mirror-specific gain.

## BX. Spiking networks: functional parameters outside synaptic weights

**Direct prior PA281..285.**

- TACOS (PA281) uses task-agnostic synaptic consolidation/metaplasticity/neuromodulation and a **fixed memory size** during continual learning. It explicitly avoids oracle task labels, so a Mirror alternative requiring known task ID at test time has an unfair advantage.
- Synapse-Threshold Synergistic Learning (PA282) jointly learns synaptic weights and spike thresholds. Thus neuron threshold as a functional parameter is **not a novel Mirror discovery**.
- Temporal Effective Batch Normalization (PA283) learns time-step-specific scaling, so per-time gain code is also not novel.
- EAS-SNN (PA284) co-adapts event sampling and recurrent SNN representation. A temporal grouping rule can erase crucial events.
- Nature Communications 2026 event-vision experiments (PA285) show timing precision can carry subpixel discrimination, so packet/period compression must include fine-grained timing ablation.

**Mirror insertion:** factor many native threshold/gain/leak/adaptation settings as `m_task × m_t × m_neuron` around one physical synaptic network, optionally with a compact private synapse residual. Candidate **MA-961..970**.

**First tests:** keep spike weight tensors frozen and measure native independent threshold vectors, simple scalar/diagonal threshold, STL-SNN, per-time TEBN and Mirror codes at equal **persistent bytes and write/state bytes**. For TACOS, test without explicit task ID and include no-task-boundary continual learning. Measure accuracy, forgetting, firing/spike events, event-based sensor latency, energy (hardware-measured or clearly labeled simulator estimate), surrogate training cost and temporal precision.

**Failure modes:** It may require many bits to encode useful thresholds, fewer spikes may reduce correctness, adaptive plasticity may be better than persistent codes, and additional input-conditioned code generators may dominate energy.

## BY. Physical computing: the hardware itself can be the shared “W”

**Direct prior PA286, PA287, PA295.**

Shen et al. (PA295) demonstrate programmable coherent nanophotonic MZI meshes. LightPro (PA286, 2026) uses programmable phase-change couplers and architecture search; MDR-HDONN (PA287) reuses fabricated diffractive modules by reconfiguring system variables. They already implement multiple logical optical transformations on one physical substrate.

**Mirror-specific hypothesis:** if many optical task matrices share a calibrated physical operator, one base configuration plus small **structured phase/coupler code m** might beat storing and applying a full new optical phase table each time. Candidate **MA-971..980**.

The true resource vector here is **not only checkpoint bytes**. Report:
- number and type of physical tunable devices;
- persistent controller/configuration bits and calibration state;
- optical insertion loss/crosstalk/noise/thermal drift;
- programming, settling, and switching latency;
- total laser/thermal/ADC/DAC/programming energy and inference energy;
- matrix/operator fidelity and actual task accuracy;
- chip area and shared/private hardware fraction.

**First simulation then physical prototype:** construct a hardware-constrained calibrated linear optical map and a set of independently trained/optimized task maps. Fit (a) full MZI/PCM phase setting per task, (b) shared configuration plus scalar/diagonal/rank-1 correction, (c) Mirror sparse phase m, (d) private extra couplers. Hold out physical perturbations. A software tensor-based speedup is **not** evidence of optical accelerator efficiency. Report simulation separately from actual photonic measurements.

**Key insight:** this extends the Mirror functional parameter principle to a new *physical* meaning of shared W, but optical phases and programming are clearly substantial established prior art, not proof of Mirror novelty.

## BZ. Wireless beamforming: phase hardware and channel codes as Mirror addresses

**Direct prior PA288..291.**

Network beamspace learning (PA288) already learns beam codebooks across sectors. Site-specific Type-II feedback (PA289) already infers a user beam subspace and sends compact effective-channel coefficients. RIS joint beamforming (PA290) already reconfigures physical propagation using controllable phases. CsiNet (PA291) already compresses channel-state feedback into low-dimensional codes.

**Mirror-specific test:** a shared antenna/RIS base operator plus compact `m_site × m_user × m_frequency × m_time` may reduce duplicated codebook/controller state at fixed net spectral efficiency, feedback bandwidth and power constraints. Candidate **MA-981..989**.

Mandatory controls are native learned beam codebook, Type-II feedback, codeword lookup, CsiNet, RIS phase optimization and model-driven joint precoder. Record channel/coherence model, SNR, RF chain count, UE feedback/pilot bits, control signaling, RF switch delay, weighted sum-rate and energy. Do not compare learned m against a deliberately weak uniform codebook. Report exact phase-only and amplitude constraints.

**Boundary:** a new phase setting alone is not evidence of compression; if the full channel-specific phase table is still needed or signaling dominates, Mirror gains are illusory. Nonstationary multipath channels may demand private/online capacity.

## CA. Spatial personalized audio: one spectral field, many listeners/directions

**Direct prior PA292–293.**

RANF (PA292) retrieves neighboring listener HRTFs and conditions a shared neural field to infer high-resolution direction-dependent binaural transfer functions from few measured directions. The anthropometric HRTF method (PA293) uses a direction-conditioned latent and listener physical shape features. Personalized HRTF latents therefore already exist.

**Mirror-specific hypothesis:** use shared HRTF spectral/phase field atoms with a compact `m_listener × m_direction × m_headpose`, optionally a sparse private filter residual, to improve the storage/accuracy/measurement frontier beyond retrieval and native latent codes. Candidate **MA-990..995**.

Use held-out listeners and measured directions, compare RANF/RANF+, native anthropometric decoder, nearest-neighbor HRTF, shared plus generic listener embedding, and Mirror code. Audit spectral distortion, interaural time and level difference (ITD/ILD), localization/elevation/front-back errors where listeners are available, head-tracking update latency, actual signed/complex coefficient storage, and measurement count. No medical/biometric assumptions should be inferred from the listener code.

## CB. Ranked follow-up selection and worker safety

The existing next worker remains **MA-255 Parameter Superposition**; this document is **not** a new execution instruction. The new 60 candidates are appended to the research queue and may be batched only after the locked direct-prior sequence or a documented reprioritization.

High-information future P0:
1. **MA-936** — direct NerVast shared/private competition with residual Mirror code; best natural-video storage comparison.
2. **MA-941** — DCVC-UF chunk frame-specific decoder View; direct packet/phase reuse transfer.
3. **MA-951** — real task-specific symmetry-breaking Mirror beyond known equivariance; important guard against false capacity.
4. **MA-961** — threshold-only Mirror code vs STL-SNN; relatively cheap CPU mechanistic probe.
5. **MA-971** — physical tunable photonic hardware configuration basis, only after realistic device simulator is available.
6. **MA-981/982** — learned beamspace and limited-feedback subspace code; use public channel models.
7. **MA-990** — listener HRTF personalization vs RANF; test true held-out listener condition.

For each, keep **native method -> native + m -> cheapest matched simple control -> independent/private reference** and report both an intentionally aligned construction and natural/out-of-family variation. No new performance claims are entered until implementation, frozen protocol and verifier exist.

## Registry audit targets

- Previous: 935 MA, 265 PA, 29 PROMISING, 18 FAIL, 888 UNTESTED.
- Added: 60 MA (47 P0, 13 P1), 30 PA.
- Expected: **995 MA, PA01..PA295, 29 PROMISING, 18 FAIL, 948 UNTESTED**.
- Expected priorities: **P0 501, P1 391, P2 103**.
- MA936..995 must all be **UNTESTED**.
- Preserve all original 47 verified experiment directories and claim ledger. `main` is not an integration target.
