# Functional Quotient Compression

> Compress task-relevant structure; quotient redundant degrees of freedom.

Functional Quotient Compression (FQC) is a research project on neural-network compression through **functional equivalence**, **quotient representations**, **shared low-description structure**, and **task-aware codec optimization**.

The project grew out of earlier “Vector Mirror” experiments. The current framework is broader: mirror transformations are treated as one possible structural tool, not as the fundamental source of compression.

## Research question

Can a trained neural network be represented by coding only the degrees of freedom that remain distinguishable under the target task, while deriving or sharing the rest through a decoder with exact serialized-bit accounting?

The long-term target is to determine—by proof, exact optimization, and real-model experiments—whether **64× compression relative to a 16-bit parameter baseline** is feasible under a pre-registered task-quality constraint. A 64× real-Transformer result has **not** been demonstrated.

## What is established so far

### 1. Corrected compression principle

Known invertible transforms (including mirror/orthogonal changes of basis) do not by themselves create fundamental rate-distortion gains. Compression must come from actual non-redundant structure: hard sharing, quotienting task-null degrees of freedom, low-description decoder structure, structured private exceptions, or restricted codecs.

### 2. Functional / decision equivalence

Several research lines converge on the same principle: states that produce the same relevant decoded or decision behavior can be merged. This includes gauge freedoms, decoded-signature equivalence, and the D116 decision-null common-mode result.

### 3. Task-aware geometry matters

Small parameter or spectral energy does not imply small task importance. In the D117 line, a mode carrying only about 0.019% of coefficient Frobenius energy was still decision-critical under hard removal; soft shrink was substantially better.

### 4. Exact structural codec optimization

Deterministic toy experiments show that serializer effects, cross-block coupling, tree topology, precision, selectors, and decoder prerequisites must be optimized jointly. A constructed tree/precision example exhibits strict complementarity: neither coordinate alone improves the incumbent, while the joint move does.

### 5. Search work is not codec bits

The D70–D120 scheduler/controller line studies how to reduce experiment, query, and validation work. These are useful optimization results, but they are **not evidence of model-bit compression** unless they change the serialized decoder DAG.

## Evidence status

| Evidence lane | Status |
|---|---|
| Mathematical / structural framework | Active |
| Exact deterministic toy optimization | Available; canonical reconstruction in progress |
| Synthetic scheduler / decision-geometry experiments | Available |
| Real Transformer structural audit | Next major phase |
| Actual serialized Transformer codec | Not yet demonstrated |
| 64× real-model compression | **Not demonstrated** |

## Repository structure

```text
claims/        machine-readable claim/evidence ledger
docs/          canonical research state, theory map, and evidence policy
roadmap/       gated research plan
provenance/    reconstruction rules and source mapping
src/           canonical implementations (to be reconstructed)
tests/         exact/reproducibility tests (to be reconstructed)
experiments/   normalized experiment suites (to be reconstructed)
```

The repository is being **reconstructed from prior handoff packages**, not used as a dump of those packages. Duplicate, obsolete, and pseudo-code artifacts are normalized into a canonical structure while preserving provenance.

## Core research lanes

1. **Compression Core** — functional equivalence, shared roots, task geometry, decoder DAG, exact bit accounting.
2. **Exact Codec Optimizer** — Bellman/Pareto methods, state quotients, branch-and-bound, local joint bundles, serializer-aware optimization.
3. **Synthetic Research Scheduler** — experiment/query/validation work optimization; intentionally separated from codec evidence.

## Non-negotiable claim boundaries

- Invertible representation changes are not themselves compression evidence.
- Logical bits are not accepted as a 64× result; actual serialized bits are authoritative.
- Synthetic work savings are not codec-bit savings.
- Toy optimality is not real-model optimality.
- Low energy is not assumed to mean low task value.
- A real 64× claim requires exact serialization and a task-quality witness under the real-pilot / Transformer-extraction contract.

See [`docs/RESEARCH_STATE.md`](docs/RESEARCH_STATE.md), [`docs/EVIDENCE_POLICY.md`](docs/EVIDENCE_POLICY.md), and [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
