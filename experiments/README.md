# Experiments

This directory contains experiment-specific code, protocols, and checked result summaries.

## Active lane

### Shared-rule / sparse Mirror-MoE

Canonical directory:

- [shared_rule_moe/srm001_20261007](shared_rule_moe/srm001_20261007)

Files:
- `mirror_rule_core.py` — core structured Mirror-rule implementation used for the current SRM line;
- `PROTOCOL.json` — consolidated task, routing, training, storage, and evidence boundary;
- `CORE_RESULTS.csv` — consolidated checked result table.

The corresponding narrative report is:
- [../docs/phase2/SRM001_SHARED_RULE_MOE.md](../docs/phase2/SRM001_SHARED_RULE_MOE.md)


### Token mixing / parallel period generation

- [TM001 parallel period token mixing](token_mixing/tm001_20261007/) — one-forward P-token phase slots, hidden packet-latent boundary, and CPU cached-AR benchmark.

## Preserved historical lanes

- `mirror_native/` — MN-series shared-state / Mirror experiments;
- `analytic_mirror/` — reachability and local-geometry analysis;
- `sensor_mirror/` — sensor/world-core and residual-View experiments;
- `transformer/` — earlier Transformer/FQC experiments;
- codec and scheduler directories — Phase I / supporting research.

Do not infer current architectural status from directory age. Use:
- [Phase II current state](../docs/phase2/CURRENT_STATE_2026-10-07.md);
- [Phase II experiment registry](../docs/phase2/EXPERIMENT_REGISTRY.md).

## Storage / evidence rule

Result claims should point to actual serialized bytes when storage matters. Large raw bundles may remain outside ordinary Git history; checked protocols, summaries, code required for interpretation, and provenance should be committed.
