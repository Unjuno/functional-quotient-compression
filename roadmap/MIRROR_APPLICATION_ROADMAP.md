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
