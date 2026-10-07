# Phase II experiment registry

Date: 2026-10-07
Status: canonical navigation index

This file is an index, not a claim ledger. Exact claims remain in the experiment reports.

| Series | Question | Result | Canonical record |
|---|---|---|---|
| MN007 | can shared semantic bases organize reusable factors? | weak spontaneous grouping | historical Phase II records |
| MN008 | can relation supervision train reusable factors? | yes; not Mirror-specific | historical Phase II records |
| MN009 | shared wide path vs independent narrow experts | shared path useful; Dense/simple controls strong | historical Phase II records |
| MN010 | Mirror count vs breadth | count alone weak; breadth signal mixed | RESEARCH_STATE_THROUGH_MN010.md |
| MT001 | paired multi-view learning | robustness / repeated-exposure confound | RESEARCH_STATE_2026-10-06.md |
| MT002 | router vs representation bottleneck | both matter | RESEARCH_STATE_2026-10-06.md |
| MT003 | gain-Mirror breadth | too narrow vs full experts | RESEARCH_STATE_2026-10-06.md |
| RF1 | router-free rotation views | no stable Dense capacity advantage; robustness signal | RESEARCH_STATE_2026-10-06.md |
| RA | reachability-guided directions | useful analysis, weak model-level continuation | REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md |
| MS005 | Mirror + ES | no free sample-efficiency gain | MS005_MIRROR_ENSEMBLE_ES.md |
| MS006 | Backprop vs ES | Backprop dominates for shared differentiable parameters | MS006_BACKPROP_VS_ES.md |
| MS008 | view decomposition | sensor ID should stay out of shared core; residual View only when needed | MS008_VIEW_DECOMPOSITION.md |
| MS009–MS014 | residual View optimization / pruning / shared-world frontier | adaptive residual capacity promising; quality-equivalent compression not reached | MS009_MS014_VIEW_OPTIMIZATION.md |
| SRM001 | sparse compositional shared-rule / Mirror-MoE | multi-rule composition essential; LM-loss-only routing promising | [report](SRM001_SHARED_RULE_MOE.md) / [experiment](../../experiments/shared_rule_moe/srm001_20261007/) |\n| SRM002 | ordered non-commutative composition; shared/private decomposition | weighted shared basis PASS; sparse private hybrid + autoprune PASS in controlled family; Mirror main family FAIL | [report](SRM002_NONCOMMUTATIVE_COMPOSITION.md) / [experiment](../../experiments/shared_rule_moe/srm002_20261007/) |

## Status vocabulary

- **PASS**: predeclared narrow hypothesis passed in the tested regime.
- **PROMISING**: repeated positive signal, but important controls / scaling remain.
- **FAIL**: tested hypothesis did not meet its gate.
- **NOT ESTABLISHED**: not enough evidence to make the claim.

## Active branch lineage

Current organization branch:
- `research/repo-reorg-20261007`

Current SRM experimental branch:
- `research/srm001-shared-rule-moe-20261007`

Important preserved branches:
- `research/ms013-ms014-shared-world-frontier-20261007`
- `research/ms011-ms012-view-optimization-20261007`
- `research/ms008-view-decomposition-20261007`
- `research/ms006-backprop-vs-es-20261007`
- `research/ms005-mirror-ensemble-es-20261007`
- `research/ms003-sn001-sensor-bridge-20261007`

## Documentation policy

1. Never delete a negative result merely because the architecture changed.
2. Historical architecture files keep their original claims and receive a superseded banner when needed.
3. New readers should enter through:
   - repository README;
   - this registry;
   - CURRENT_STATE_2026-10-07.md;
   - current architecture document.
4. Large raw evidence bundles stay outside ordinary Git history when appropriate; checked summaries, protocols, hashes, and core results belong in Git.
5. New SRM experiments live under `experiments/shared_rule_moe/`; do not create parallel top-level `experiments/srmXXX_*` directories.
