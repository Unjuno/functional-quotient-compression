# Worker Context Router

Purpose: give an experiment worker the **smallest sufficient context** for one MA run. Do not read the entire repository before coding.

## Always load — in this order

1. `AGENTS.md`
2. `GOAL.md`
3. `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
4. `docs/phase2/LATEST_WORKER_FINDINGS.md`
5. `experiments/mirror_applications/STATUS_BOARD.md`
6. exactly one selected row from `IDEA_REGISTRY.csv`
7. only the PA headings named by that row's `prior_art_refs`
8. `EXPERIMENT_CONTRACT.md`
9. `TEMPLATE/`

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
