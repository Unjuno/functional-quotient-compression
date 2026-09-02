# Research Roadmap

The roadmap is gated. A phase does not count as complete because more experiments were run; it is complete only when its exit criteria are satisfied.

## G0 — Canonical repository reconstruction

**Goal:** turn the handoff collections into one validated research codebase rather than preserving their directory layouts.

Work:

- classify each source artifact as theory, executable code, specification/pseudo-code, raw data, derived result, duplicate, obsolete, or missing;
- merge duplicate theory into canonical documents;
- preserve origin hashes and transformations in provenance metadata;
- keep invalidated claims in a retraction/history ledger rather than active documentation;
- separate Compression Core, Exact Codec Optimizer, and Synthetic Scheduler evidence.

Exit criteria:

- every canonical claim has an evidence lane and origin;
- all files under canonical `src/` compile;
- no pseudo-code is represented as executable Python;
- known missing artifacts are documented;
- historical numbering is metadata, not repository architecture.

## G1 — Exact experiment reconstruction

**Goal:** rebuild the important E1–E7 codec experiments as clean, executable tests.

Priority reconstruction:

1. serializer-aware Bellman / Pareto frontier;
2. cross-block coupling and false-pass test;
3. decoded-state quotient and tree co-design;
4. description-only tree rotations and conditional-frontier transfer;
5. operation-local Pareto cache;
6. joint tree × precision local bundle.

Exit criteria:

- `pytest` reproduces the declared exact optima/state counts within the declared numeric tolerance;
- every small solver is checked against brute force;
- reconstructed runners are labeled `RECONSTRUCTED` until original source identity is established.

## G2 — Quotient Codec Optimizer v0

**Goal:** unify the durable optimizer ideas into one solver.

Minimum state dimensions:

- root / shared representation selector;
- private/shared choice;
- support/topology;
- precision state;
- decoder prerequisites.

Required mechanisms:

- decoder-DAG paid-state closure;
- decoded-state equivalence/quotienting;
- exact local Pareto islands;
- joint local codec bundles;
- branch-and-bound or another certifiable global mechanism where tractable;
- serializer-authoritative final acceptance.

Exit criteria:

- exact agreement with brute force on a registered suite of small problems;
- no logical-bit-only acceptance path;
- deterministic experiment records and reproducible fixtures.

## G3 — Real Transformer structural audit

**Goal:** measure whether the central compression hypothesis exists in a real trained model before building a large codec stack.

Use one small pretrained Transformer first. Do not start with multiple architectures.

Measure:

- functional/gauge equivalence dimensions;
- shared-root / shared-atom coverage;
- private residual fraction;
- task-sensitive spectra versus raw-energy spectra;
- cross-layer/block coupling;
- structured support complexity;
- metadata and decoder-prerequisite cost;
- low-energy/high-task-value exceptions.

Required baselines:

- standard quantization;
- magnitude/structured pruning as applicable;
- SVD/low-rank approximation;
- task-aware method without Mirror-specific operations;
- full FQC proposal.

Exit criteria:

- a predeclared structural report answering whether FQC-specific structure adds value beyond standard baselines;
- enough measured rate components to estimate a defensible codec-family lower bound.

## G4 — Actual FQC codec v0

**Goal:** implement an encoder/decoder whose measured file size is the rate.

Paid components must include, where used:

- roots/dictionaries;
- maps and selectors;
- support/topology descriptions;
- codebooks/scales;
- coefficients;
- private exceptions;
- headers, alignment, padding, and other metadata.

Exit criteria:

- deterministic round trip;
- exact bit ledger agrees with actual file size;
- no decoder dependency cycles;
- task evaluation can be run from the decoded model alone plus declared external-fixed state.

## G5 — Hard 64× feasibility test

**Goal:** decide the 64× question rather than assume success.

For a 16-bit scalar baseline with `N` unique paid scalars:

`B64 = floor(N / 4)` bits.

Procedure:

1. compute a structural lower bound for the fixed codec family;
2. compute/test optimistic candidate upper bounds;
3. serialize only candidates that remain feasible;
4. evaluate under the predeclared task-quality witness.

Final decision must be one of:

- `FEASIBLE` — actual payload is within budget and quality gate passes;
- `IMPOSSIBLE_UNDER_CODEC_FAMILY` — a safe lower bound or exhaustive/certified result rules it out;
- `UNCERTAIN` — current lower/upper bounds do not decide the question.

## G6 — Scaling and generalization

Only after G5 yields useful real-model evidence:

- test small/medium/larger models;
- decompose bits/parameter into root, selector, support, coefficient, metadata, and private-residual terms;
- determine whether metadata amortization beats increasing specialization/private structure;
- test architecture dependence (attention variants, GQA, RoPE, MLP structure).

Exit criteria:

- scaling curves with explicit rate decomposition and task-quality uncertainty;
- no extrapolation from model size without measured evidence.

## G7 — Runtime and publication

Measure separately from storage compression:

- peak memory;
- load/decode time;
- inference latency/throughput;
- memory bandwidth;
- decoder/kernel overhead.

Publish both positive and negative results with the claim ledger and reproducible experiment suite.

## Parallel lane — Synthetic Research Scheduler

The D70–D120 scheduler line may continue only as a supporting lane when it reduces the cost of G1–G6 experiments.

It should not become the main research frontier until real-model compression uncertainty has been reduced.

## Stop / pivot criteria

- If real-model shareability/quotient structure adds no meaningful benefit over strong standard baselines, reduce or retire FQC-specific machinery.
- If mandatory codec state alone exceeds the target rate, stop optimizing payload details and redesign the codec family.
- If task-aware geometry does not outperform simpler sensitivity baselines, remove the additional complexity.
- If decode/runtime overhead dominates storage benefit, restrict the claim to offline/storage compression rather than inference acceleration.
