# Vector Mirror Research — FQC and Mirror-native models

**Research in progress. No quality-preserving 64x Transformer result or general Mirror-native capacity advantage is claimed.**

## Current organization (2026-10-03)

| Phase | Scope | Status |
|---|---|---|
| [Phase I: Functional Quotient Compression](docs/phase1/STATUS.md) | Post-training compression, task-sensitive shared/private codecs, exact byte accounting | **Paused; results, code and history preserved** |
| [Phase II: Mirror-native conditional models](docs/phase2/README.md) | Native shared-state models, parallel routing, sparse/state modulation, semantic parameter sharing | **Active through MN007** |

The previous Phase-I README is preserved verbatim in [the Phase I snapshot](docs/phase1/README_before_phase2_20261003.md). Its baseline was commit `41e084440e5c8525c1e1adeefd172be7f05acb10`. Historical source, tests, claims and experiment directories remain preserved.

## Start here

- [**Phase II research state through MN007**](docs/phase2/RESEARCH_STATE_THROUGH_MN007.md)
- [**Semantic Mirror hypothesis**](docs/phase2/SEMANTIC_MIRROR_HYPOTHESIS.md)
- [**Mirror-native experiment index**](experiments/mirror_native/INDEX.md)
- [**MN002 runnable public experiment**](experiments/mirror_native/mn002_20261003/README.md)

## What Phase II has established

Small CPU experiments demonstrate that:

- shared FFN/state models can learn input-dependent functional states without materializing a full expert matrix per token;
- independent routing heads can be packed into batched projections rather than executed as a sequential routing chain;
- the tested MLP-centered designs preserve ordinary causal KV-cache reuse under fixed past states;
- candidate-region routing can reduce routing metadata/parameter cost;
- shared-base/state representations can be serialized more compactly than fully independent embeddings in some controlled settings.

## What remains unresolved

The evidence does **not** establish that:

- more Mirror states automatically increase useful capacity;
- combinatorial routing-path count equals independent-expert capacity;
- dynamic overlap or averaging is generally superior to structured disjoint subspaces;
- router collapse is the main optimization bottleneck;
- task learning automatically organizes semantic relatives into one shared parameter identity;
- Mirror transforms beat strong additive or low-rank factorization controls;
- the CPU measurements predict GPU/production-LLM speed.

MN007 makes the current representation question explicit: can a model learn **shared parameter identity + low-description view/state coordinates** for related concepts, rather than only placing separate vectors near one another? That hypothesis remains open.

## Reproduce the public MN002 baseline

```bash
cd experiments/mirror_native/mn002_20261003
python -m pip install -r requirements.txt
python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python reproduce.py --output /tmp/mn002-reproduction
```

MN003-MN007 were run locally in the project container. Their large evidence bundles are not committed to ordinary Git history; verified SHA-256 provenance is recorded in the [Phase II research state](docs/phase2/RESEARCH_STATE_THROUGH_MN007.md).

## Reuse and evaluation

Code and extensions are welcome under the existing **Apache License 2.0**; see [LICENSE](LICENSE). Please cite this repository when using the prototype or reporting independent evaluations. Citation is requested, not an additional license restriction.

Independent replications, including negative results, are welcome. Especially useful tests are stronger additive/low-rank controls, non-saturating compositional tasks, semantic state-swap consistency, matched-byte comparisons, and later real-language/model-scale evaluation.
