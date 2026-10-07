# Vector Mirror Research — Functional sharing, sparse rules, and Mirror residuals

**Research in progress. No quality-preserving 64x Transformer result, general LLM capacity multiplier, or general Mirror-native advantage is claimed.**

## Current organization — 2026-10-07

| Phase | Scope | Status |
|---|---|---|
| [Phase I — Functional Quotient Compression](docs/phase1/STATUS.md) | post-training functional compression and exact byte accounting | Paused; preserved |
| [Phase II — shared-backbone conditional models](docs/phase2/README.md) | native sharing, residual specialization, sparse compositional rules | Active research; SRM003 execution complete, adoption gate failed |

Historical source and negative results are preserved rather than rewritten into the current architecture.

## Start here

For autonomous / worker execution:

1. [**GOAL.md**](GOAL.md) — autonomous iteration loop and stop rules
2. [**WORKER_START_HERE.md**](WORKER_START_HERE.md) — repository navigation and evidence rules
3. [**Mirror Application Status Board**](experiments/mirror_applications/STATUS_BOARD.md) — exact next MA candidate
4. [**550-candidate registry**](experiments/mirror_applications/IDEA_REGISTRY.csv)

For research context:

5. [Current Phase II state](docs/phase2/CURRENT_STATE_2026-10-07.md)
6. [Mirror application design space](docs/phase2/MIRROR_APPLICATION_DESIGN_SPACE.md)
7. [Mirror application prior-art map](docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md)
8. [Mirror application research notes](docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-07.md)
9. [Mirror application validation roadmap](roadmap/MIRROR_APPLICATION_ROADMAP.md)
10. [Latest result — TM001 parallel period token mixing](docs/phase2/TM001_PARALLEL_PERIOD_TOKEN_MIXING.md)
11. [Runnable SRM003 source, tests, protocol and counts](experiments/shared_rule_moe/srm003_20261007/README.md)
12. [Architecture hypothesis — Sparse Compositional Shared-Rule Transformer](docs/phase2/ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md)
13. [Experiment registry](docs/phase2/EXPERIMENT_REGISTRY.md)
14. [Known negative results](docs/KNOWN_NEGATIVE_RESULTS.md)

## Latest result and current question

SRM001 showed strong fixed-update signals on explicitly factorized synthetic tasks. SRM002 learned ordered operators efficiently when the execution order was supplied by the model's control flow. SRM003 removed oracle routing/private masks in a conventional two-layer causal decoder: all 24 atomic rules could be retained exactly, yet direct unseen two-rule composition remained about 13–19% even after 6,400 updates. Calling the same trained model twice with its own predicted intermediate state gave 100%, but costs two forwards and supplies the execution procedure externally.

This is evidence for separating **rule storage**, **rule selection**, and **ordered execution**. It is not a natural-language capacity or same-compute result.

The project now has a second explicit lane: systematically apply the Mirror/View coordinate to existing multiplicities (experts, adapters, heads, KV, layers, packet slots, quantizers, memory, etc.) and measure whether physical duplication can be replaced by low-description logical multiplicity. The machine-readable registry currently contains 550 MA-xxx candidates.

The research hypothesis remains:

> Store common Transformer computation once, compose multiple reusable specialist residuals, and allocate additional private capacity only when it gives measurable quality/storage benefit.

Mirror geometry is one candidate parameterization, not a mandatory whole-model transformation. It must beat simpler shared-basis controls before any Mirror-specific claim is made.

## Evidence boundaries

Actual serialized bytes are required for storage claims. Equal updates or tokens are not equal FLOPs or wall time. A flat learning curve is not a theorem about capacity. Stable routing is not proof that semantic rules were discovered. An explicitly supplied executor is not the same as learning to execute from endpoint loss. Removing half a residual bank does not halve the whole model or its active top-k work.

Small synthetic findings do not establish a natural-language LLM advantage, general quality-preserving compression multiplier, or universal Mirror benefit.

## Repository map

```text
docs/
  phase1/                         historical FQC state
  phase2/
    CURRENT_STATE_2026-10-07.md
    ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md
    EXPERIMENT_REGISTRY.md
    SRM001_SHARED_RULE_MOE.md
    SRM002_NONCOMMUTATIVE_COMPOSITION.md
    SRM003_CAUSAL_DISCOVERY.md
  KNOWN_NEGATIVE_RESULTS.md
experiments/
  README.md
  mirror_native/                  historical MN experiments
  analytic_mirror/                reachability and local geometry
  sensor_mirror/                  MS-series fixtures
  shared_rule_moe/
    srm001_20261007/
    srm002_20261007/
    srm003_20261007/
  token_mixing/
    tm001_20261007/
  mirror_applications/
    IDEA_REGISTRY.csv             550 candidate applications
    FIRST_QUEUE.md                first 25 P0 candidates
third_party/
  nanoGPT/                       compact shared A/B baseline
roadmap/
  PHASE2_SPARSE_RULE_ROADMAP.md
  ROADMAP.md                      broader historical FQC roadmap
src/fqc/                          historical reusable implementation
tests/                            historical repository tests
```

The SRM003 experiment has its own tests. Its verification report does not claim that the unchanged historical repository-wide suite was rerun.

## Historical records

- [Research state through MN010](docs/phase2/RESEARCH_STATE_THROUGH_MN010.md)
- [Global Dense Mirror architecture](docs/phase2/GLOBAL_DENSE_MIRROR_ARCHITECTURE.md)
- [2026-10-06 Mirror Transformer proposal](docs/phase2/MIRROR_TRANSFORMER_CURRENT_ARCHITECTURE.md)
- [Reachability-adjusted analysis](docs/phase2/REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md)
- [Experiment-lane index](experiments/README.md)

A historical document may be superseded without invalidating its original experimental record. Current amendments are recorded in the current-state document, not retroactively substituted into old protocols.

## Reuse

Apache License 2.0; see [LICENSE](LICENSE). Independent replications, including negative results, are welcome.
