# Functional Quotient Compression

> Compress task-relevant structure; quotient redundant degrees of freedom.

Functional Quotient Compression (FQC) is a research project on neural-network compression through **functional equivalence**, **quotient representations**, **shared low-description structure**, and **task-aware codec optimization**.

The project grew out of earlier “Vector Mirror” experiments. The current framework is broader: mirror transformations are treated as one possible structural tool, not as the fundamental source of compression.

## Research question

Can a trained neural network be represented by coding only the degrees of freedom that remain distinguishable under the target task, while deriving or sharing the rest through a decoder with exact serialized-bit accounting?

The long-term target is to determine—by proof, exact optimization, and real-model experiments—whether **64× compression relative to a 16-bit parameter baseline** is feasible under a pre-registered task-quality constraint. A quality-preserving 64× real-Transformer result has **not** been demonstrated.

## What is established so far

### 1. Corrected compression principle

Known invertible transforms (including mirror/orthogonal changes of basis) do not by themselves create fundamental rate-distortion gains. Compression must come from actual non-redundant structure: hard sharing, quotienting task-null degrees of freedom, low-description decoder structure, structured private exceptions, or restricted codecs.

### 2. Functional / decision equivalence

Several research lines converge on the same principle: states that produce the same relevant decoded or decision behavior can be merged. This includes gauge freedoms, decoded-signature equivalence, and decision-null common-mode constructions.

### 3. Task-aware geometry matters

Small parameter or spectral energy does not imply small task importance. Later real-checkpoint experiments also show that lower parameter-space reconstruction error need not improve task KL/NLL.

### 4. Exact structural codec optimization

Deterministic toy experiments show that serializer effects, cross-block coupling, tree topology, precision, selectors, and decoder prerequisites must be optimized jointly.

### 5. Real Transformer codec path now exists

The T266-T282 real-checkpoint lane crossed the earlier engineering boundary: learned Transformer weights have been serialized into actual codec artifacts, independently decoded, and executed end-to-end. Reproduction and corruption/configuration checks were expanded through this phase.

This does **not** establish quality-preserving 64× compression. The tested simple 64× family failed quality, and FQC-specific functional sharing has not yet been shown to beat a strong non-sharing baseline at matched final bytes.

### 6. Search work is not codec bits

Scheduler/controller experiments study how to reduce experiment, query, and validation work. These are useful optimization results, but they are **not evidence of model-bit compression** unless they change the serialized decoder DAG.

## Evidence status

| Evidence lane | Status |
|---|---|
| Mathematical / structural framework | Active |
| Exact deterministic toy optimization | Available |
| Synthetic scheduler / decision-geometry experiments | Available |
| Real Transformer structural / codec engineering | **Available through T282** |
| Actual serialized Transformer codec | **Demonstrated as an engineering artifact** |
| Official TinyStories validation advantage | Not yet demonstrated |
| FQC sharing advantage over strong non-sharing control | Not yet demonstrated |
| Quality-preserving 64× real-model compression | **Not demonstrated; tested simple family failed** |
| MPS/CUDA runtime advantage | Not yet measured |

## Repository structure

```text
claims/        machine-readable claim/evidence ledger
docs/          canonical research state, theory map, and evidence policy
roadmap/       gated research plan
provenance/    reconstruction rules and source mapping
src/           canonical implementations
tests/         exact/reproducibility tests
experiments/   normalized experiment suites
```

The repository is being **reconstructed from prior handoff packages**, not used as a dump of those packages. Duplicate, obsolete, and pseudo-code artifacts are normalized into a canonical structure while preserving provenance. Large checkpoints and duplicated binary artifacts are intentionally kept out of normal git history.

## Current central experiment

The next falsifiable comparison is at matched **final serialized bytes**:

1. strong activation-aware non-sharing control;
2. control + FQC functional sharing;
3. sharing + selected private exceptions;
4. private exceptions + joint byte allocation / QCO.

The comparison must use held-out evaluation, token NLL and KL, and a multi-rate frontier. The purpose is to determine whether FQC-specific structure adds rate-distortion value beyond a strong conventional control.

## Core research lanes

1. **Compression Core** — functional equivalence, shared roots, task geometry, decoder DAG, exact bit accounting.
2. **Exact Codec Optimizer** — Bellman/Pareto methods, state quotients, branch-and-bound, local joint bundles, serializer-aware optimization.
3. **Synthetic Research Scheduler** — experiment/query/validation work optimization; intentionally separated from codec evidence.
4. **Real-model validation** — official evaluation path, strong controls, FQC ablations, multi-rate frontiers, and later MPS/CUDA measurements.

## Non-negotiable claim boundaries

- Invertible representation changes are not themselves compression evidence.
- Logical bits are not accepted as a 64× result; actual serialized bits are authoritative.
- Synthetic work savings are not codec-bit savings.
- Toy optimality is not real-model optimality.
- Low energy is not assumed to mean low task value.
- Parameter-space reconstruction error is not treated as task quality.
- KL-only improvement is not promoted to an NLL-quality claim.
- A real 64× claim requires exact serialization and a pre-registered task-quality witness.

See [`docs/RESEARCH_STATE.md`](docs/RESEARCH_STATE.md), [`docs/REAL_MODEL_T266_T282.md`](docs/REAL_MODEL_T266_T282.md), [`docs/EVIDENCE_POLICY.md`](docs/EVIDENCE_POLICY.md), and [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
