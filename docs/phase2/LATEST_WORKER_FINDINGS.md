# Latest Worker Findings — active Mirror application evidence

Date: 2026-10-07 JST
Purpose: concise operational context for every new MA worker. Dedicated experiment branches are the source of truth for measurements.

## Reconciled program and new research intake — 2026-10-08

This file's detailed five experimental reports are still valid, but not the complete MA count. The canonical worker-ready branch now indexes **47 completed MA experiments**: 29 PROMISING, 18 FAIL. The eleventh through fourteenth literature sweeps added **220 entirely UNTESTED** candidates, bringing the registry to **1095**, with **1048 UNTESTED** total.

Read `experiments/mirror_applications/STATUS_BOARD.md` and `docs/phase2/MIRROR_MA_EVIDENCE_INTEGRATION_2026-10-08.md` for all 47 verified historical experiment reports and their scoped claims. New prior art PA236..265 and MA876..935 are detailed in `docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md`. Additional prior art PA266..295 and MA936..995 are detailed in `docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md`. PA296..325 and MA996..1045 are documented in `docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_THIRTEENTH_SWEEP.md`. New PA326..350 and MA1046..1095 (time series, recommender embeddings, multisensor EO) are documented in `docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_FOURTEENTH_SWEEP.md`. The read-only `experiments/mirror_applications/check_registry_integrity.py` audits MA-1000+ IDs, PA/claim consistency and the status board.

MA-255 is reconciled as **PROMISING only for a post-fit aligned representation screen**: 734B Mirror state reached 6.68e-17 fresh MSE across seeds 101/211/307/401, versus 6,490B implemented PSP, 14,650B rank-2 task code and 27,906B independent. A different 1,200-update protocol variant failed its development quality gate and stayed sealed on fresh worlds. The two protocols are retained separately and are not a replication pair; see `experiments/mirror_applications/ma-255-mirror-context-superposition/RECONCILIATION.md`.

MA-260 is **FAIL** for its registered byte gate: the four-seed aligned case used 890B versus 1,226B BatchEnsemble (27.4% saving, below the required 75%); the independent-task stress case fell to 0.6018 accuracy versus 0.8332 controls. This was a post-fit one-layer linear screen, not a deep BatchEnsemble reproduction. See `experiments/mirror_applications/ma-260-batchensemble-mirror/README.md`.

MA-261 is **FAIL** against the literal frozen gate. Its source branch claimed 4/4 PASS, but raw fresh per-seed Mirror/independent MSE ratios are 1,390x, 302,057x, 11.7x and 0.227x; the <=1.10x criterion passes only 1/4. A separate fixed-update two-expert protocol failed during development. See `experiments/mirror_applications/ma-261-batchensemble-logical-experts/RECONCILIATION.md`.

MA-265 is **FAIL**: four fresh aligned rotation worlds had numerical-zero task error, but the 3,494B Mirror payload saved only 4.4% versus 3,654B VeRA, below the frozen 20% storage gate. Independent scale codes failed, and a separate 1,000-update diagonal-code protocol also lost to native VeRA at development. See `experiments/mirror_applications/ma-265-vera-mirror-scaling/README.md`.

MA-268 is **PROMISING only for its trained nonlinear aligned task screen**: it passed the frozen quality/byte gate in 3/3 fresh worlds at 4,167B versus 4,273B IA3 and 7,933B independent, but used 1.5x IA3 active MACs and had about 0.29–0.34x IA3 eager CPU throughput. A separate post-fit linear orbit screen saved 15.7% versus IA3, below its 20% byte gate. See `experiments/mirror_applications/ma-268-ia3-mirror-views/`.

MA-271 is **FAIL for Mirror-specific value**. A trained aligned cross-over reported 3,309B vs 7,253B dense OFT on 3/3 fresh worlds, but it omitted the exact simple control. A second four-seed screen found that ordinary rank-one task-code × shared-angle factorization matches the Mirror's functions and 1,086B payload exactly. See `experiments/mirror_applications/ma-271-oft-mirror-views/RECONCILIATION.md`.

MA-272 is **FAIL for Mirror-specific/runtime Pareto value**. The corrected post-fit audit found that scalar-times-shared-skew exactly matches the Mirror function and 3,786B payload; exact input-side execution averaged 0.335ms vs 0.138ms materialized. A separate trained screen improved bytes/quality against dense OFTv2 but had slower CPU throughput. See `experiments/mirror_applications/ma-272-oftv2-mirror-views/RECONCILIATION.md`.

MA-273 is **FAIL for Mirror-specific value**. A corrected post-fit fresh audit (163/269/367/463) found Mirror identical in function and 1,150B payload to ordinary scalar-times-shared-angle factorization; the aligned orbit used 27.3% fewer bytes than independent BOFT but the stress family required independent angle state. A distinct neutral-initialization trained screen failed development (3,593B vs BOFT 3,787B, lower quality) and kept fresh worlds sealed. See `experiments/mirror_applications/ma-273-boft-mirror-bank/RECONCILIATION.md`.

MA-274 is **FAIL (development screen)**. Across two fixed-update dev worlds with oracle routes, Mirror saved 3.4% bytes vs native BOFT (5,430B vs 5,624B) but lost in MSE both worlds; independent FFNs were much better. Mirror beat rank-one with 19% fewer bytes, but eager CPU throughput was 0.33–0.36M examples/s versus 3.58–4.32M rank-one and 6.35–7.20M IA3. Fresh worlds remained sealed. Alongside MA-273, this pauses the current task/expert shared scalar-angle BOFT family, without generalizing to other insertion points. See `experiments/mirror_applications/ma-274-boft-logical-experts/FAMILY_BOFT_MIRROR_DIAGNOSTIC.md`.

MA-276 is **PROMISING only for aligned serialized-state compression**: the 3 fresh seeds reconstructed the four-step butterfly-aligned depth map at 396B vs 1,193B untied. The 2,048B prepared-operator workspace exceeds the untied serialized payload; unrelated layer maps fail, and no language/training/capacity claim is established. Five tests and exact replay passed.

MA-278 is **FAIL on this unaligned task-factor screen**: Mirror and scalar modulation both serialized to 2,525B, but Mirror had higher MSE in all four development world/LR pairs; native Compacter also beat Mirror. Fresh worlds stayed sealed. This does not test an aligned Compacter task family.

MA-282 is **PROMISING on a deliberately aligned synthetic Monarch FFN family**: four scalar Views reproduced exact outputs 3/3 at 925B, vs 1,145B free-angle Monarch and 3,279B independent FFNs, with 13.6M vs 10.7M examples/s. Unrelated task maps collapse to hard-tie quality; runtime workspace is 1,024B. Seven tests and all 42 fresh rows replayed exactly. No learned Transformer, language, capacity or runtime-RAM claim.

**The next candidate is MA-286.** MA-275/277/279–281 remain P1 UNTESTED. MA-876..1115 remain appended research-intake hypotheses and must not preempt the registered P0 crossovers. Natural variation and benchmark-level runtime remain unproven; aligned synthetic PROMISING must not be described as real-world Mirror adoption.

## MA-241 — layer-specific Mirror views over tied experts

Branch: `research/ma-241-expert-tying-mirror-20261007`
Verified result: `0ee183668285231d825e853c69c4791b9d252bf2`
Status: **PROMISING**

Aligned synthetic teacher: layer variation was generated by the same Givens-view family available to the candidate.

Observed:
- untied payload 45,681 B;
- hard tied 23,967 B;
- Mirror 24,286 B;
- Mirror passed the preregistered aligned quality/storage gate in 3/3 fresh worlds;
- Mirror beat matched gate/rank-1 controls on quality;
- unfused CPU implementation was slower (median inference throughput ~0.61x hard tying).

Boundary: this proves aligned coordinate variation can be recovered cheaply. It does not show naturally trained expert layers lie on a Givens orbit.

## MA-244 — shared MQA K/V projection + Mirror role/head views

Branch: `research/ma-244-kv-role-view-20261007`
Verified result: `85de2fd65618d72bd0bf6a091b558a0dda57b741`
Status: **PROMISING**, strict model-payload gate missed.

Aligned synthetic teacher: one shared projection plus role/head Givens transforms.

Observed:
- Mirror fresh MSE ~9e-12 to 2e-11;
- equal-size scalar gate remained ~0.056–0.130 MSE;
- Mirror payload 2,100 B vs separate-K/V MQA 2,349 B and MHA 3,885 B;
- model-payload saving vs MQA was 10.6%, below the preregistered 20% gate;
- physical cache state 128 B vs MQA 256 B and MHA 1,024 B;
- unfused CPU inference was slower, median ~0.67x MQA.

Boundary: aligned role geometry can make one physical cached projection serve several logical roles. Natural-language Q/K/V structure, large-d serializer amortization and optimized kernels remain untested.

## MA-253 — cache-safe final-layer Mirror-MoE

Branch: `research/ma-253-cache-safe-final-moe-20261007`
Verified result: `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`
Status: **FAIL for Mirror expert replacement; cache placement PASS**.

Misaligned teacher: per-domain variations were independently generated rank-2 LoRA updates.

Observed:
- rank-2 LoRA MSE ~3.6e-14 to 6.1e-14;
- Mirror-only MSE ~0.0161–0.0173, essentially base-only;
- Mirror 2,956 B vs LoRA 4,110 B, but quality failed decisively;
- adding rank-1 private residual improved MSE by ~71–78% but remained far from rank-2 LoRA;
- final-FFN-only View left all prefix K/V cache values exactly unchanged;
- applying a View before later attention changed downstream cache as expected.

Boundary: narrow coordinate views do not replace genuinely independent private variation. Placement and parameterization are separate hypotheses.

## MA-245 — MLKV shared cache + per-layer Mirror views

Branch: `research/ma-245-mlkv-layer-views-20261007`
Verified result: `76d91b7a662b4227e7f25e4733e06d6735cf1cd2`
Status: **PROMISING**, strict model-payload gate missed.

Aligned synthetic teacher: one physical K/V base transformed by layer/role Givens views.

Observed:
- Mirror fresh MSE ~6.7e-12 to 6.8e-11;
- scalar gate and hard one-group MLKV remained ~0.015–0.028 MSE;
- Mirror payload 2,541 B vs two-group MLKV 2,864 B: 11.3% smaller, below the preregistered 25% gate;
- actual cache state 256 B vs two-group MLKV 512 B and MHA 1,024 B;
- eager CPU Mirror throughput ~0.85x two-group MLKV and training ~2.1–2.2x slower.

Boundary: all layers saw the same memory hidden state and the teacher was exactly on the candidate orbit. Real layer-to-layer hidden-state evolution is untested.



The five results form a coherent pattern:

1. **aligned functional variation** -> compact Views can work very well; this has now repeated across tied experts, head/role K/V views, and cross-layer K/V views;
2. **misaligned independent variation** -> a narrow View can fail completely;
3. **aligned variation can still fail to learn when the View is embedded inside recurrent/sequential credit assignment**;
4. **private residual capacity** helps when variation leaves the shared orbit;
5. **placement** can determine cache/runtime properties independently of quality;
6. **unfused structured operations** can lose wall-clock Pareto position even when analytical MAC overhead is small.

The next question is therefore not "does Mirror work?" but:

> What fraction of real functional variation lies on a cheap shared coordinate orbit, and how much private residual is needed for the remainder?

# Mandatory synthetic interpolation axis

When practical, future mechanism screens should not use only an all-aligned or all-independent teacher.

Define teacher variation as:

    Delta(alpha) = alpha * Delta_view + (1 - alpha) * Delta_private

with alpha spanning at least:
- 0.00 — fully private/misaligned;
- 0.25;
- 0.50;
- 0.75;
- 1.00 — fully view-aligned.

The exact construction may differ by family, but the experiment should identify:
- quality vs alpha;
- serialized bytes vs alpha;
- required private residual rank/capacity vs alpha;
- active compute and runtime.

This produces an **orbit/private frontier** rather than a binary win/fail result.

# Mandatory baseline ladder

Use the cheapest meaningful controls before richer geometry:

1. hard tying / no adaptation;
2. scalar/diagonal modulation (IA3/FiLM where relevant);
3. rank-one modulation (BatchEnsemble-style);
4. shared low-rank basis / VeRA-style;
5. structured Mirror candidate;
6. structured Mirror + private residual;
7. independent object upper control.

Skip irrelevant rungs, but document why.

# Symmetry audit

Before counting logical functions, ask whether the View is merely function-preserving gauge symmetry.

Relevant controls:
- neuron permutations;
- sign/scale monomial symmetries;
- exact orthogonal computational invariances.

Exact symmetry-equivalent parameterizations count as **zero new functional multiplicity**. They may still be valuable for quantization, alignment or numerical conditioning.

# Runtime rule

Never infer runtime from MACs alone.

For structured transforms report:
- analytical active compute;
- eager wall-clock;
- vectorized/compiled wall-clock when available;
- transform materialization or cache bandwidth.

Prefer cheap/vectorizable comparison families:
- diagonal/IA3;
- rank-one;
- Householder/Givens in fused form if available;
- butterfly/BOFT;
- ACDC/AFDF;
- Hadamard/MAP;
- low-rank.

# Storage rule

Small synthetic models have serializer/config overhead large enough to distort percentage savings. Report both:
- actual serialized inference bytes (authoritative);
- raw tensor payload / learned parameter bytes (diagnostic only).

Do not replace the authoritative serialized-byte gate after seeing results.

# Worker handoff

Before starting a new MA experiment:
1. read this document;
2. read the selected registry row and prior-art refs;
3. identify whether the experiment is aligned, misaligned, or an alpha-interpolation frontier;
4. declare the cheapest baseline rung and strongest upper control;
5. declare whether runtime is part of the adoption gate;
6. only then freeze PROTOCOL.json.
