# Phase II experiment registry

Date: 2026-10-07, updated through SRM003.

This is a navigation index, not a replacement for original protocols. Status refers to the stated tested regime, not a universal theorem.

| Series | Question | Scoped result | Record |
|---|---|---|---|
| MN007 | shared semantic bases | weak spontaneous grouping | [MN007 state](RESEARCH_STATE_THROUGH_MN007.md) |
| MN008 | relation-supervised factor reuse | trainable; not Mirror-specific | [MN010 state](RESEARCH_STATE_THROUGH_MN010.md) |
| MN009 | shared wide vs independent narrow | shared path useful; strong simple controls | [MN010 state](RESEARCH_STATE_THROUGH_MN010.md) |
| MN010 | count vs breadth | count weak; broad signal mixed | [MN010 state](RESEARCH_STATE_THROUGH_MN010.md) |
| MT001 | paired multi-view training | repeated-exposure confound | [historical summary](RESEARCH_STATE_2026-10-06.md) |
| MT002 | router vs representation | both bottlenecks | [historical summary](RESEARCH_STATE_2026-10-06.md) |
| MT003 | gain-Mirror breadth | too narrow in tested conditions | [historical summary](RESEARCH_STATE_2026-10-06.md) |
| RF1 | deterministic rotation views | robustness, no stable capacity win | [historical summary](RESEARCH_STATE_2026-10-06.md) |
| RA | reachability | analytical diagnostic, not capacity proof | [analysis](REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md) |
| MS005 | Mirror and ES | no free variance reduction at fixed sample budget | [report](MS005_MIRROR_ENSEMBLE_ES.md) |
| MS006 | BP vs ES | BP better for the tested shared-weight update | [report](MS006_BACKPROP_VS_ES.md) |
| MS008 | sensor identity and residual views | conditional view utility, not universal identity ban | [report](MS008_VIEW_DECOMPOSITION.md) |
| MS009–MS014 | view optimization and storage | transfer strong, quality-equivalent compression unresolved | [report](MS009_MS014_VIEW_OPTIMIZATION.md) |
| SRM001 | sparse multi-rule causal task | controlled fixed-update and routing signals; parity failed | [report](SRM001_SHARED_RULE_MOE.md) / [experiment](../../experiments/shared_rule_moe/srm001_20261007/) |
| SRM002 | non-commutative operators | shared/private success with supplied executor and controlled teacher | [report](SRM002_NONCOMMUTATIVE_COMPOSITION.md) / [experiment](../../experiments/shared_rule_moe/srm002_20261007/) |
| SRM003 | no-oracle causal discovery/pruning | atomic retention complete, direct composition weak; adoption FAIL | [report](SRM003_CAUSAL_DISCOVERY.md) / [runnable experiment](../../experiments/shared_rule_moe/srm003_20261007/README.md) |

## Branches

Latest SRM experiment and integrated state: `research/srm003-causal-discovery-20261007`.

Preserved branches include `research/repo-reorg-20261007`, `research/srm001-shared-rule-moe-20261007`, `research/srm002-noncommutative-20261007`, `research/ms013-ms014-shared-world-frontier-20261007`, `research/ms011-ms012-view-optimization-20261007`, `research/ms008-view-decomposition-20261007`, `research/ms006-backprop-vs-es-20261007`, `research/ms005-mirror-ensemble-es-20261007`, and `research/ms003-sn001-sensor-bridge-20261007`.

## Evidence policy

Execution completion is not hypothesis success. A numerical gate passed at unusably low task accuracy is not useful-quality compression. Original preregistrations remain unchanged; amendments identify whether they preceded fresh audit access. New SRM source lives under `experiments/shared_rule_moe/`. Preserve negative results and distinguish logical sparsity, active compute, total storage and private-rule semantics.

See [current state](CURRENT_STATE_2026-10-07.md) for scoped corrections to older interpretations.
