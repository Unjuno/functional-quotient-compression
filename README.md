# Vector Mirror Research — FQC and Mirror-native models

**Research in progress. No quality-preserving 64x Transformer result or general Mirror-native capacity advantage is claimed.**

## Current organization (2026-10-04)

| Phase | Scope | Status |
|---|---|---|
| [Phase I: Functional Quotient Compression](docs/phase1/STATUS.md) | Post-training compression, task-sensitive shared/private codecs, exact byte accounting | **Paused; results, code and history preserved** |
| [Phase II: Mirror-native conditional models](docs/phase2/README.md) | Native shared-state models, global Mirror control, dynamic state growth | **Active through MN010; MN011 pre-registered** |

The previous Phase-I README remains preserved verbatim in [the Phase I snapshot](docs/phase1/README_before_phase2_20261003.md). Historical source, tests, claims and experiment directories remain preserved.

## Start here

- [**A9/A9R optimization reachability experiment**](experiments/mirror_capacity/A9/README.md) — 324 locked synthetic models and proof that the successful fixed scaffold is equivalent to ordinary direct shift, not a general Mirror-specific compression gain.

- [**A8/A8b rank-certified function capacity experiment**](experiments/mirror_capacity/A8/README.md) — 336+32 locked synthetic models, exact theoretical rank barrier and Mirror realizations, but no trained general Mirror advantage.

- [**2026-11-15 A7 compositional-image capacity experiment**](experiments/mirror_capacity/A7/README.md) — 336 locked models, 16 AND/XOR tasks, seven controls, negative Mirror-specific rate-quality result. Six mapping worlds share one digit dataset.

- [**2026-11-15 A6 real handwritten-digit, 16-function capacity benchmark**](experiments/mirror_capacity/A6/README.md) — 336 models, locked IID/OOD audit and negative Mirror-specific result; same underlying digit images across six task mappings.
- [**A1–A9 conditional function capacity evidence ledger**](experiments/mirror_capacity/README.md) — prior synthetic controls and new real-image follow-up; none establishes a general Mirror capacity multiplier.

- [**Phase II research state through MN010**](docs/phase2/RESEARCH_STATE_THROUGH_MN010.md)
- [**Global Dense Mirror architecture**](docs/phase2/GLOBAL_DENSE_MIRROR_ARCHITECTURE.md)
- [**MN011 pre-registered breadth × count plan**](experiments/mirror_native/MN011_PLAN.md)
- [**Mirror-native experiment index**](experiments/mirror_native/INDEX.md)
- [MN002 runnable public experiment](experiments/mirror_native/mn002_20261003/README.md)

Historical conceptual side lane:

- [Semantic Mirror hypothesis](docs/phase2/SEMANTIC_MIRROR_HYPOTHESIS.md)

## Current main hypothesis

The active architecture is:

```text
context
  -> one context-aware router
  -> global Mirror state / mixture
  -> one wide Dense model
```

The Dense path remains primary. The research question is whether a small number of sufficiently broad Mirror states can provide useful conditional specialization without duplicating full expert matrices.

MN010 indicates that **Mirror count and Mirror breadth are separate resources**. Increasing the number of 8-dimensional states produced little split-specific gain on the tested task, while a 256-dimensional global modulation produced a positive signal in one discovery world. A fresh-world replication was mixed, so no general advantage is claimed.

## What Phase II has established at small scale

- shared-state conditional computation can be trained without materializing a full expert matrix per token;
- router heads can be packed rather than executed as a sequential routing chain;
- the tested MLP-centered designs preserve ordinary causal KV-cache reuse under fixed past states;
- functional recombination supervision can organize reusable semantic factors, although additive/low-rank controls remain strong;
- a wide shared FFN can outperform a near-byte-matched collection of narrow independent experts on one synthetic task;
- global Mirror states can be added function-preservingly during training;
- context information supplied to the router is a first-order architectural constraint.

## What remains unresolved

The evidence does **not** establish that:

- Mirror-native models generally outperform Dense;
- Mirror states can replace a strong sparse-MoE at equal quality;
- more states monotonically improve useful capacity;
- the MN010 rich-Mirror improvement is stable across data worlds;
- semantic factorization is intrinsically better with phase Mirrors;
- CPU synthetic measurements predict GPU or production-LLM behavior.

## Reproduce the public MN002 baseline

```bash
cd experiments/mirror_native/mn002_20261003
python -m pip install -r requirements.txt
python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python reproduce.py --output /tmp/mn002-reproduction
```

MN003-MN010 were run in the project container. Their large evidence bundles are intentionally not committed to ordinary Git history; verified SHA-256 provenance is listed in the [experiment index](experiments/mirror_native/INDEX.md) and current research-state document.

## Reuse and evaluation

Code and extensions are welcome under **Apache License 2.0**; see [LICENSE](LICENSE). Please cite this repository when using the prototype or reporting independent evaluations. Citation is requested, not an additional license restriction.

Independent replications, including negative results, are welcome. The next useful target is the MN011 breadth × count experiment at matched serialized bytes.
