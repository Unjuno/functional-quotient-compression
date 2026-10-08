# Mirror Application Research — Sixteenth Sweep (2026-10-08 JST)

Status: **literature-backed experimental intake only; no new Mirror training result**.

## Scope and authoritative counts at intake

Before this sweep: **MA-001..1115** (1115 ideas), **PA01..381**, 47 MA results (29 PROMISING, 18 FAIL), 1068 UNTESTED. A separate natural-digit shifted-task pilot has an explicitly scoped FAIL, but is **not** a completed MA result and must not be promoted as one.

Added here:
- **32 primary published/preprint references, PA382..413**.
- **40 individually scoped, untested experiment candidates MA-1116..1155**: 33 P0 and 7 P1.
- Four natural-function and physical-system branches: KG relation operators, camera ISP/optics, online robot dynamics, and room neural acoustic fields.
- New totals: **1155 MA; P0 630, P1 422, P2 103; 1108 UNTESTED, 29 PROMISING, 18 FAIL; 413 prior art**.

No existing MA status, frozen seed, worker ownership, original 254-entry experiment branch, branch experiment code, or `main` is intentionally modified by this research extension.

The authoritative next worker candidate remains **MA-255 Parameter Superposition**. This research family must remain behind that and other already frozen/registered queues unless its priority is explicitly re-evaluated.

## 1. Scientific invariant: Mirror m is the marginal change

For a native established architecture `B(x; θ)`, the hypothesis is whether an additional low-description parameter `m` can improve useful functional variety or quality per byte:

`B(x; θ) -> B(x; θ, m)`.

The new workstreams provide **direct prior art** to this idea. In several cases, the native model already uses essentially the same compact condition: relation phase, ISO/exposure code, online dynamics latent, or room/source coordinates. Merely renaming that native parameter as `m` is **not Mirror-specific novelty**.

Every protocol must identify:
1. the exact physical object shared (trained relation factor, ISP trunk, robot actor or acoustic field);
2. which specific native parameter/interface receives a new `m`, and how that differs from the native code;
3. the strongest efficient native baseline **without** this new intervention;
4. an even cheaper non-Mirror code, gate, coefficient or rank-1/LoRA control;
5. natural/OOD variation, a private residual upper reference, serialized bytes and actual runtime;
6. a preregistered falsification gate that may conclude **NO MARGINAL MIRROR VALUE**.

## 2. Knowledge graphs: RotatE and TuckER already encode relation as a small operator

Primary prior:
- **PA382 RotatE:** a relation is a complex-space entity rotation; relation reversal, symmetry and composition are explicitly studied.
- **PA383 ComplEx:** entity-relation factors and asymmetric bilinear scores.
- **PA384 TuckER:** one physical tensor core plus relation coefficients, extremely close to a shared-basis-plus-address concept.
- **PA385 QuatE:** quaternion operators for richer noncommutative relations.
- **PA386 PairRE:** independently scaled head and tail relation views.
- **PA387 CompGCN:** compositions of relation and entity embeddings through shared graph convolution.
- **PA388 temporal tensor KGE (TNTComplEx):** time-conditioned relation factors.
- **PA389 5starE:** includes projective relation maps, so rotations/scales/reflections are not a new operator family.
- **PA390 KrausKGE (2026 preprint):** proposes rank-dependent channel operators with structure and fan-out semantics.

Research candidates: **MA1116..1125**.

### KG first-screen MA-1123: naturally fitted relation orbit/private frontier

Build a compact natural-graph benchmark:
- Fit native **RotatE, PairRE, TuckER** or reuse public trained states; retain actual entity and relation embeddings, not a teacher planted from a Mirror rotation.
- Learn one shared physical relation-operator dictionary on training relations; for each target relation, fit native relation coefficients, `m`, `m + private`, and independent native relation operators.
- Evaluate isolated fit and joint link-prediction performance on **filtered** held-out triples. Hold out entire relation×domain or relation×time combinations where data support them.
- Report MRR, Hits@1/10, relation-pattern diagnostics (1-to-N, N-to-1, N-to-N, symmetry, inversion, composition), entity-table + relation-state + dictionary bytes, and scoring examples/s.
- Sweep code dimension and private rank. Include a *same-score gauge transformation* that cannot be mistaken for independent learned functionality.

**Falsification:** if native RotatE phase/PairRE diagonals or TuckER coefficients have equal/better quality at fewer bytes, the extra Mirror view has no marginal benefit. If joint entity/gauge rotation preserves all triple scores, count **zero independent functions** from that apparent multiplicity. The real gain sought is shared relation *families* under time/domain changes, not the use of a relation parameter itself.

**High-value follow-ups:** MA1120 factorized relation×time (TNTComplEx), MA1121 Kraus rank and its physical operator cost, MA1122 native CompGCN edge processing. Avoid scoring held-out inverse triples through train/test inverse leakage.

## 3. Camera ISP: EXIF-conditioned functional transformations already exist

Primary prior:
- **PA391 ParamISP:** converts EXIF camera ISO/exposure conditions into ParamNet modulation of a RAW/sRGB image signal processor.
- **PA392 MetaISP:** one scene-aware network performs device-targeted color rendition.
- **PA393 Uni-ISP:** joint forward/inverse pipeline across real camera devices with explicit device embeddings; FiveCam provides synchronized multi-device capture.
- **PA394 PQDynamicISP:** a lightweight dynamically controlled ISP handles camera/environment variation and even spatially local parameter control.
- **PA395 Modular Neural ISP:** controllable stages, multiple styles and camera generality.
- **PA396 camera-parameter denoising:** imaging-noise-conditioned correction.
- **PA397 OmniLens++:** a pretrained latent optical point-spread-function (PSF) family.
- **PA398 Neural Lens Modeling:** learned differentiable lens optics.
- **PA399 physics-informed low-rank lens correction:** low-rank optics already competes with more general learned correction.

Research candidates: **MA1126..1135**.

### Camera first-screen MA-1127: incremental camera state under native Uni-ISP

Use paired real RAW/sRGB images from **FiveCam** or equivalent appropriately licensed camera data. Freeze or reuse a native Uni-ISP backbone; compare:
- native per-device embedding, ParamISP EXIF conditioning and optional PQDynamicISP controller;
- a plain scalar/FiLM/low-rank camera adapter with byte-matched learned state;
- structured Mirror per-camera `m` on a selected early/middle ISP feature operator, optionally + private residual;
- independent native per-camera ISP as a quality upper reference.

Train on a subset of cameras/ISO/lighting settings and hold out **complete camera device(s)**, not random crops of the same scenes. Evaluate physically meaningful RAW->RGB and inverse conversion separately: PSNR, SSIM/LPIPS, color-chart DeltaE, clipping/noise/HDR robustness, inference FPS, physical serialized state including conditioner and EXIF metadata. Measure latency on a named device.

**Falsification:** if native Uni-ISP/ParamISP conditioner has equal quality and is smaller/faster, Mirror fails. RAW information irreversibly lost by clipping/quantization cannot be reconstructed merely via an invertible View. For MA1131/1134, measured PSFs/lens spatial dependence and modulation transfer must not be replaced by generic spatial color correction.

**High-value follow-ups:** MA1129 local PQDynamicISP, MA1130 modular stage factor, MA1132 bidirectional ISP consistency, MA1133 real ISO-noise conditions.

## 4. Online motor adaptation: RMA latent extrinsics are direct prior art

Primary prior:
- **PA400 UP-OSI (2017):** single universal control policy plus online physical parameter identification.
- **PA401 RMA (2021):** shared legged policy modulated by a compact history-estimated environmental extrinsics vector.
- **PA402 CoRMA (2026):** contact-rich context adaptation based on RMA-style contrastive representations.
- **PA403 A-NC:** recurrent controller implicitly estimates unobserved physics.
- **PA404 graph-operator world model:** adapts to morphology and dynamics changes.
- **PA405 morphology-conditioned quadruped world model:** one world model with body-specific conditions.
- **PA406 CTS:** deployable teacher/student control distinctions and sensor privilege.

Research candidates: **MA1136..1145**.

### Robot first-screen MA-1136: a deployable latent m beyond native RMA

Construct one common simulated control benchmark spanning mass/payload, friction/ground roughness, actuator delay and terrain. Use a frozen pretrained common actor with estimated environmental context. Compare:
- **native RMA** and UP-OSI online adaptation with equal history and causal sensors;
- constant context / plain FiLM / rank-1 view of policy features;
- low-description Mirror `m` predicted **only from deployable action-observation history**;
- Mirror `m` + small private residual;
- independent policy or oracle-physics actor as a separately labeled **upper** control only.

Freeze train/development/test condition sets, then hold out **complete combinations**, including abrupt within-episode changes. Report return, fall/contact failure, task success, time to adapt, online update FLOPs, writable adaptation state, persistent parameters, simulation wall clock and policy inference latency.

**Falsification:** if ordinary native RMA extrinsics or hidden A-NC recurrent state performs as well with fewer bytes and lower delay, no Mirror benefit. Robot dynamics parameter fields must **never be directly supplied at test time** if the native baseline only sees history. Oracle teacher values/privileged sensors can calibrate a known upper bound but not a fair deployment claim.

**High-value follow-ups:** MA1138 CoRMA friction/contact, MA1140–41 morphology×skill factorization on withheld bodies, MA1142 shared/private extreme dynamics, MA1144 online code update amortization.

Do not infer real robot safety or battery power improvement from simulated trajectories.

## 5. Acoustic room fields: spatial coordinates and room adaptation already exist

Primary prior:
- **PA407 Neural Acoustic Fields (NAF):** continuous source/receiver-conditioned acoustic response field, generally fitted to room-specific measurements.
- **PA408 Real Acoustic Fields:** measured real-room audio-visual dataset and benchmark, essential for naturally distributed sound fields.
- **PA409 retrieval-augmented NAF:** pretraining and room enrollment/retrieval for few-shot unseen room adaptation.
- **PA410 TA-RIR:** topology-aware room impulse response (RIR) synthesis with geometry/propagation.
- **PA411 NAMS:** learned multipole acoustic field and pruning with physical structure.
- **PA412 direction-aware NAF:** Ambisonic spatial fields with a native few-shot LoRA-style adaptation study.
- **PA413 few-shot multimodal acoustic flow matching:** room-consistent generative acoustic synthesis from sparse examples.

Research candidates: **MA1146..1155**.

### Acoustic first-screen MA-1147: room m beyond native retrieval-pretraining

On the **Real Acoustic Fields** benchmark or licensed measured-RIR rooms:
1. Train or reuse common shared field and the native geometry-aware **retrieval+room enrollment** baseline.
2. Infer a low-dimensional room `m` using an identical few-RIR budget (e.g. 3/5/10 measured source/receiver pairs; adjust if benchmark cannot support).
3. Compare native per-room fine-tuning/LoRA, plain room embedding/gate, Mirror code, Mirror + sparse private reflection residual, and independent per-room NAF reference.
4. Evaluate unseen room(s) and unseen source/receiver locations **jointly**, plus long reverberation and occluded configurations.
5. Record predicted RIR waveform/time-frequency error, RT60, direct-to-reverberant ratio (DRR), clarity (C50), energy decay, spatial/phase coherence, measured sound-field consistency, audio state+retrieval bank bytes, adaptation computation and RIR generation runtime.

**Falsification:** the native retrieved-room embeddings or few-shot LoRA win at comparable storage/runtime; Mirror code is unable to reproduce private echoes or reflection geometry; phase-incoherent STFT magnitude matching gives high apparent score but invalid physical waveform. A room latents-only parameterization is not new compared with NAF family.

**High-value follow-ups:** MA1150 multipole-basis sharing under actual pruning, MA1151 direction-aware Ambisonic room View, MA1152 same-NFE flow model, MA1153 physics/causality audit.

## 6. Four-domain protocol rule and triage priority

Across all candidates, follow the project's **native + m** marginal value principle:

| Family | Mandatory natural/OOD gate | Native control | Physical cost that cannot be ignored |
|---|---|---|---|
| Knowledge graph | held-out relation/domain/time and filtered triples | RotatE / TuckER / PairRE / KrausKGE | entity+relation storage, link scorer runtime |
| Camera | held-out camera/ISO/exposure/lens | Uni-ISP / ParamISP / PQDynamicISP | EXIF, color/noise pipeline and inference FPS |
| Robot dynamics | held-out physics/morphology and online drift | RMA / UP-OSI / CoRMA / A-NC | estimation latency, writable state, deployable sensor budget |
| Room acoustics | held-out measured room and source/receiver | NAF / retrieval NAF / TA-RIR / NAMS | RIR phase, microphone geometry, room/field bytes and generation latency |

Top future candidates among this family: **MA1123**, **MA1127**, **MA1136**, **MA1147**; then 1121/1131/1140/1150/1151. Their priority reflects strong falsifiability and native baseline quality, **not** successful Mirror training.

Universal stop conditions: if a code merely reproduces a native relation phase, camera EXIF embedding, RMA extrinsics, or NAF source/receiver coordinate, it has no Mirror-specific gain. If quality only passes on an artificially candidate-aligned teacher, label **mechanism feasibility**, not real task compression.

Worker safety:
- Do not change MA-255 or any worker-owned running branch.
- Do not run experiments on the MA-1116..1155 IDs until a protocol and fresh split have been frozen.
- Cite PA382..413 by ID; do not read the entire 400+ prior art file into an experiment worker's context.
- Any new repository experiments should use existing `EXPERIMENT_CONTRACT.md`, byte-exact verifier, reproducible reference models, and a separate fresh-world decision.
