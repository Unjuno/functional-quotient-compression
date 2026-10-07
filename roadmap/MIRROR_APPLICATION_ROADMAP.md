# Mirror Application Validation Roadmap

Date: 2026-10-07 JST
Status: active backlog roadmap

## Goal

Systematically test whether a low-description Mirror/View coordinate can replace physical duplication in existing model structures.

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

The backlog now contains **550 candidates**. Recent literature sweeps add four new strategic lanes:

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
