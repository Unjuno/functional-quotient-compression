# MA-330 status

- Status: PROMISING for aligned shared-cache views; no runtime gain established
- Branch: `research/ma-330-tensorized-kv-cache-mirror-20261008`
- Protocol/source frozen before fresh: `32bf014`
- Fresh seeds: 33011, 33012, 33013
- Development complete: yes
- Fresh/audit opened: yes
- Results committed: yes (dedicated branch commit `c2d2aafa726daa791fca73c8f46d20319ceef0b6`)
- Verification committed: yes; 15 payloads replay exactly, 4 tests pass
- Registry row updated: yes on integration branch

## Next action

Integrated on `research/mirror-application-current-evidence-20261008`; registry, claim ledger and status board record narrowly scoped PROMISING. No trained LM/runtime claim.

## Decisions / rulings

Mirror phase cache payload is 72.3% smaller than independent FP16 caches with near-zero attention-output error in all fresh worlds. The nearest direct cos/sin code control is only 30B (0.082%) larger and has the same quality; throughput does not consistently improve. The phase coordinate saves half the per-view rotation-code bytes but not a material fraction of total cache state. Without private cache, the deliberately unrelated fourth layer has nMSE about 0.92–0.99, so unrelated functions need private state.
