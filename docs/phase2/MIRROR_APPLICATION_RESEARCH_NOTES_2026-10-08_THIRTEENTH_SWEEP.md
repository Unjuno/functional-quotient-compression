# Mirror application research — thirteenth literature sweep (2026-10-08 JST)

**Scope:** verified primary literature and new controlled research hypotheses. **No new Mirror model training, fresh audit, or runtime measurement occurred during this sweep.**

Previous authoritative registry: 995 MA proposals, PA01–295, with 29 PROMISING and 18 FAIL.
New additions: **PA296–325 (30 primary sources)** and **MA996–1045 (50 UNTESTED proposals; 40 P0 + 10 P1)**.
Post-sweep expected: **1045 MA (541 P0, 401 P1, 103 P2), 325 PA, 998 UNTESTED, 29 PROMISING, 18 FAIL**.
Next locked worker remains **MA-255**, and the prior 47 verified experiments remain unchanged.

## Core finding: physics-/interface-constrained Mirror functional freedom

The research objective remains the extra low-description Mirror parameter:
`F(x; theta) -> F(x; theta, m)`.

A recurring weakness of generic weight-/activation-space View experiments is that they may create logical variation by violating a structure that the original model was designed to enforce. This sweep finds several **hard-constrained interfaces** that offer a more decisive falsification opportunity:

1. **Atomistic force fields:** let `m` enter a scalar, Euclidean-invariant **energy** predictor and derive force from its negative position gradient. Do not independently alter the force head in a way that destroys conservative force consistency.
2. **MRI reconstruction:** let `m` change the learned **prior / regularizer**, then retain the acquired k-space measurement/data-consistency projection, sensitivity model and forward operator. A visually plausible result without physical measurement fidelity is a failure.
3. **Quantum circuit:** let `m` change gate angles or physically valid unitary/measurement controls inside a **fixed compiled circuit**. Charge gate count, qubits, shots, qRAM and postselection, not only trainable parameters.
4. **Video visual segmentation:** let `m` change object/task memory but preserve native prompt/correction semantics and evaluate drift, occlusion and real-time interactive quality.
5. **Nearest-neighbor retrieval:** let `m` change useful **ranking behavior** in a controlled way; a shared orthogonal transform of both query and documents can leave exact Euclidean/dot-product rankings unchanged and does not create new learned retrieval capability.

A new broad hypothesis is **constraint-preserving Mirror**: identify the degrees of freedom *outside* a domain's hard physical/operational constraint, and learn compact `m` only there. This is an experimental synthesis of existing methods, **not** a literature-established Mirror performance claim. No material, MRI, quantum or ANN win is asserted.

## CC. Materials and conservative atomistic potentials (MA996–1005, PA296–301)

**Strongest primary controls**:
- PA296 **MACE**: higher-body-order E3-equivariant atomic message-passing force field.
- PA297 **MACE-MP**: pretrained shared interatomic potential across many materials.
- PA298 **NequIP**: high-data-efficiency equivariant atomistic force field.
- PA299 **MatterSim**: broad material, temperature and pressure potential.
- PA300 **sparsity-promoting material foundation-model adaptation**: existing structured sparse updates can match full or equivariant low-rank tuning while modifying only a small subset of physical weights. Reported percentages are the paper's native result, not Mirror.
- PA301 **partially frozen transfer learning for interatomic potentials**: direct few-shot and reactive-surface baseline.

**Mirror integration:** apply a compact `m_material`, `m_element`, `m_environment`, `m_reference-theory`, and optionally `m_T × m_P` inside an equivariant feature/energy basis, leaving a shared pretrained physical potential. Test scalar/diagonal gate, LoRA/equivariant LoRA, native sparse mask, full fine-tune, and Mirror+private sparse exceptions. Measure energy MAE, force MAE, conservative-gradient residual `||F_m + ∇_x E_m||`, E3 invariance/equivariance, molecular dynamics energy drift/rollout stability, actual payload and atoms/s. Hold out full chemical families and distinct atomic environments.

**First cheap falsification:** reuse 1–2 simple 3D conservative toy potentials with known energy and analytic force. Compare code-only scale, low-rank/FiLM, tiny Mirror rotation and private exceptions under an aligned interpolation teacher and an unrelated pair-interaction potential. The efficient Givens teacher is only feasibility; freeze fresh seeds, test force gradients, then use actual MACE data. If a simple energy-scale parameter recovers the target, no Mirror-specific benefit is established.

**Failure modes:** a View may be symmetry-equivalent, introduce spurious chirality or coordinate artifacts, produce nonconservative forces, destabilize long MD even at low one-step MAE, or be dominated by sparse material fine-tuning. Do not treat an energy-only fit as a valid force-field model.

## CD. Physics-correct MRI and multi-scanner inverse operators (MA1006–1015, PA302–307)

**Native controls:**
- PA302 **VarNet**: multicoil learned reconstruction and sensitivity maps with cascaded data consistency.
- PA303 **MoDL**: iterative inverse-problem CNN already sharing weights across unrolled iterations.
- PA304 **DUNE (2026)**: learned representation-space unrolling retaining physical data consistency.
- PA305 **D2SA**: scanner-distribution and slice-specific test-time MRI adaptation.
- PA306 **multi-contrast INR reconstruction**: native contrast-conditioned image field.
- PA307 **SSDU**: physics-guided reconstruction using genuinely held-out acquired k-space subset for self-supervision.

**Mirror insertion:** `m_scanner`, `m_coil`, `m_contrast`, `m_iteration`, `m_slice` should modulate the prior or representation chart *without bypassing* the physical forward operator. A single-coil orthonormal masked Fourier projection can be expressed as replacing acquired k-space samples with measurements after each update; a general multicoil operator must use the correct sensitivity/encoding equations, and that simple overwrite formula does **not** apply unchanged.

**First falsification:** 2D synthetic phantoms with a known masked Fourier forward model. Compare native tied MoDL, scalar data consistency, a matched low-rank prior, a Mirror step-specific prior and independent per-step prior. Monitor reconstructed PSNR/SSIM, `||A(x_m)-y||` on acquired samples, computational cost, and exact data-consistency behavior. Then validate on public retrospective datasets with true multicoil forward models; do not infer clinical diagnostic adequacy from phantom metrics.

**Data leakage:** never reuse the same acquired k-space coordinates as both adaptation target and independent SSDU audit split; never tune on held-out patient/scan. A plausible-looking hallucinated structure or anatomy mismatch is a failure even at good aggregate PSNR.

## CE. Quantum circuits as a different notion of “physical shared object” (MA1016–1023, PA308–311)

- PA308 **Data Re-uploading**: repeated rotations of input/classical data are already a standard trainable circuit construction.
- PA309 **TensorHyper-VQC**: a classical TT hypernetwork generates gate parameters, and is a very strong compact-parameter native baseline.
- PA310 **Superposed Parameterised Quantum Circuits**: explores a family of parameter sets via qRAM/postselection/repeat-until-success. It is **not** free exponentially-many model capacity; success probability, preparation and repetition must be charged.
- PA311 **hardware-aware compilation**: physical transpilation can change optimization statistics and gate cost.

**Mirror insertion:** `m_task × m_gateblock × m_axis` over one fixed physical ansatz. Fit multiple task-specific angle vectors via compact Mirror coefficients and compare against data-reuploading independent parameter lists, classical TT generation, native full ansatz and physical postselection architectures where appropriate. Measure circuit task accuracy, noise robustness, real gate count/depth **after compilation**, two-qubit count, shot/measurement repetitions, classical conditioner bytes, qRAM memory where relevant, and total execution time on a declared simulator or device.

**Failure modes:** simple per-task angle storage can be smaller than a shared code decoder; compilation may undo a theoretical circuit depth saving; changing measurement basis may not add learned independent functions; low postselection probability multiplies resource costs. No claim of quantum hardware advantage follows from software simulation.

## CF. Vision-language prompts and video object memory (MA1024–1034, PA312–317)

- PA312 **CoOp** already learns frozen-backbone task/class context prompts.
- PA313 **CoCoOp** already generates an instance-conditioned prompt token, improving base-to-new transfer.
- PA314 **MaPLe** already couples prompts between frozen vision and text branches across layers.
- PA315 **SAM 2** already offers streaming memory, object prompts and interactive video masks.
- PA316 **SAM2Long** already manages alternative segmentation memory paths to suppress long-horizon drift.
- PA317 **MoPEFT** already mixes multiple PEFT adapters for domain-specific SAM segmentation.

**Mirror insertion:** `m_task`, `m_domain × m_class`, `m_vision × m_text`, `m_object × m_frame` over shared, *native* CLIP/SAM feature/prompt/memory interfaces. A single new prompt vector is not novel; Mirror's test is whether a physically compact **shared function basis plus codes** outperforms native prompts/conditional generator/memory branching. Compare zero-shot CLIP, CoOp, CoCoOp, MaPLe, SAM2 and MoPEFT. Evaluate seen-to-novel class harmonic mean, zero-shot domain shifts, J&F/boundary quality, long-video object ID swaps, occlusion recovery, click/correction counts, all prompt/memory/adapter bytes, and measured FPS.

**Failure modes:** per-object m may entangle identities; image-conditioned generator costs may outweigh code savings; a cheap View may forget boundary details; oracle object IDs/ground-truth masks at inference invalidate an apparent gain. Multiobject streaming memory is a state/cache cost independent of backbone weights.

## CG. Approximate nearest-neighbor and late-interaction retrieval (MA1035–1045, PA318–325)

- PA318 **ScaNN** anisotropic vector quantization for efficient MIPS.
- PA319 **RaBitQ** error-controlled high-dimensional vector coding.
- PA320 **Matryoshka Representation Learning** nested embeddings for variable dimension/budget.
- PA321 **ColBERTv2** residual-compressed multi-vector late-interaction indexing.
- PA322 **PLAID** optimized centroid pruning/interaction over ColBERTv2.
- PA323 **QINCo** neural implicit residual codebooks conditional on previous codeword state.
- PA324 **DiskANN** SSD-based graph search with real storage/latency benefits.
- PA325 **PGM-index** compact dynamic piecewise learned indexing with correctness bounds.

**Mirror insertion:** `m_domain`, `m_query`, `m_shard`, `m_budget`, `m_codebookstage`, `m_update` over a **single paid physical index, codebook or retrieval scorer**. Evaluate both (a) function-preserving gauge/rotation controls and (b) genuinely changed ranking policies, carefully separating them. Quantized approximate ranking may change due to round-off after a rotation; that is a quantization-specific outcome, **not** new information in the exact index.

**Required comparisons:** native ScaNN, RaBitQ, Matryoshka prefixes, ColBERTv2 residuals, PLAID centroid pruning, QINCo, DiskANN and PGM as relevant. Report Recall@k/nDCG/MRR, actual disk+RAM index bytes, encoder/query generator bytes, SIMD operations, SSD reads, build/update costs, QPS and P95/P99 latency; use fixed precision/hardware and held-out query domains. For PGM-style ordered indexes, correctness and bounded lookup under updates are hard requirements.

**Failure modes:** compression may improve MSE without changing or improving ranking; a View can destroy the index metric/pruning guarantee; index rebuild and query-specific decoding can dominate latency; the full document-token residual store can dwarf small task code bytes.

## CH. Highest-information next screens and anti-duplication rule

The native worker queue remains **MA-255 Parameter Superposition**, not MA-996. New MA996–1045 are a later research intake.

Within this sweep, the initial high-information tests would be:
- **MA-997**: sparse shared support + Mirror task coefficient vs the 2026 sparse materials adaptation baseline.
- **MA-1000**: exact conservative force and E3 equivariance audit — a cheap, strong falsifier of the validity of `m`.
- **MA-1007/1014**: MoDL step View and exact acquisition-consistency audit on synthetic MRI.
- **MA-1016**: TensorHyper-VQC TT generator + tiny Mirror code, with physical gate/shots accounting.
- **MA-1026**: MaPLe coupled multimodal prompts vs factorized `m`, base-to-new shift mandatory.
- **MA-1027/1028**: SAM2 object memory and SAM2Long alternative paths, focused on actual memory duplication.
- **MA-1035/1039**: QINCo codebook-stage Views and PLAID query-role Views, end-to-end ANN latency/recall rather than weight saving alone.

Do not benchmark each candidate as a separate empty proof that an arbitrary code exists. Start from the **best existing native parameterization** and test the **marginal utility of m** at matched physical bytes and compute. A positive aligned-teacher mechanism screen does not establish natural task compression.

## Scientific source policy and reviewer cautions

All 30 PA references were traced to original arXiv, journal, conference or institutional primary source pages. Scientific assertions about baseline improvements belong to those sources and should not be repeated as Mirror results.

For user-facing progress:
- **Registered ideas:** 1045;
- **Prior-art map:** PA01–325;
- **Completed existing MA:** 47 (29 PROMISING, 18 FAIL);
- **Newly completed Mirror tests from this sweep:** zero;
- **New untested:** 50, bringing total UNTESTED to 998.
The next worker MA-255 and default main branch were not preempted by research expansion.

**Repository ID warning:** first ID above 999 appears in this sweep. The canonical `MA-1000`–`MA-1045` IDs are **four digits**. Workers/scripts must parse `^MA-[0-9]{3,}$` and use integer ID ordering / max+1, **not** `^MA-[0-9]{3}$` or fixed-column slices. Likewise directories should use full numeric IDs (e.g. `ma-1000-...`). This is important for future result recording and status reconciliation.
