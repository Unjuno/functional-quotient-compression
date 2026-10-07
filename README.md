# Vector Mirror Research — Functional sharing, sparse rules, and Mirror residuals

**Research in progress. No quality-preserving 64x Transformer result, general LLM capacity multiplier, or general Mirror-native advantage is claimed.**

## Current organization — 2026-10-07

| Phase | Scope | Status |
|---|---|---|
| [Phase I — Functional Quotient Compression](docs/phase1/STATUS.md) | post-training functional compression, exact byte accounting, codec design | **Paused; preserved** |
| [Phase II — shared-backbone conditional models](docs/phase2/README.md) | native sharing, residual specialization, sparse compositional rules | **Active** |

Historical source and negative results are preserved rather than rewritten into the current architecture.

## Start here

1. [**Current Phase II state**](docs/phase2/CURRENT_STATE_2026-10-07.md)
2. [**Current architecture — Sparse Compositional Shared-Rule Transformer**](docs/phase2/ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md)
3. [**Phase II experiment registry**](docs/phase2/EXPERIMENT_REGISTRY.md)
4. [**Experiments index**](experiments/README.md)
5. [**Active Phase II roadmap**](roadmap/PHASE2_SPARSE_RULE_ROADMAP.md)
6. [**Known negative results**](docs/KNOWN_NEGATIVE_RESULTS.md)
7. [**SRM001 — strongest current synthetic Transformer evidence**](docs/phase2/SRM001_SHARED_RULE_MOE.md)

## Current research question

The active hypothesis is no longer "Mirrorize the whole model."

It is:

> **Store common Transformer computation once, represent only specialist residual behavior with a sparse bank of reusable rule atoms, and select multiple atoms per token/context to construct virtual experts.**

Mirror geometry is currently one candidate low-description parameterization of those rule atoms. It must beat simpler low-rank/shared-rule controls before any Mirror-specific claim is made.

```text
token/context
    |
shared Transformer backbone
    |
shared FFN base
    |
    +-- sparse residual router
           |
           +-- rule atom A
           +-- rule atom B
           +-- rule atom C
           |
        compose multiple rules
    |
output
```

## Current evidence boundary

Small synthetic experiments support:
- reusable shared factors;
- separating shared world/core computation from condition-specific residuals;
- sparse multi-rule composition;
- promising LM-loss-only address discovery on a controlled synthetic task.

They do **not** establish:
- a natural-language LLM advantage;
- near-convergence capacity superiority over strong MoE baselines;
- a universal Mirror-specific benefit;
- a general compression factor inferred from synthetic tasks.

Actual serialized bytes are required for storage claims.

## Repository map

```text
docs/
  phase1/                   historical FQC state
  phase2/
    CURRENT_STATE_2026-10-07.md
    ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md
    EXPERIMENT_REGISTRY.md
    SRM001_SHARED_RULE_MOE.md
    ... historical Phase II records
  KNOWN_NEGATIVE_RESULTS.md

experiments/
  README.md                 experiment-lane navigation
  mirror_native/            historical MN experiments
  analytic_mirror/          reachability / local geometry
  sensor_mirror/            sensor/world residual experiments
  shared_rule_moe/
    srm001_20261007/        current shared-rule code/results/protocol

roadmap/
  PHASE2_SPARSE_RULE_ROADMAP.md
  ROADMAP.md                 broader historical FQC roadmap

src/fqc/                    canonical reusable implementation
tests/                      repository-level tests
```

## Historical records

Important historical architecture documents are intentionally retained:
- [Research state through MN010](docs/phase2/RESEARCH_STATE_THROUGH_MN010.md)
- [Global Dense Mirror architecture](docs/phase2/GLOBAL_DENSE_MIRROR_ARCHITECTURE.md)
- [2026-10-06 Mirror Transformer proposal](docs/phase2/MIRROR_TRANSFORMER_CURRENT_ARCHITECTURE.md)
- [Reachability-adjusted Mirror analysis](docs/phase2/REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md)

Their status should be read from their headers. A historical document may be superseded without being invalid as an experimental record.

## Reuse

Apache License 2.0; see [LICENSE](LICENSE). Independent replications, including negative results, are welcome.
