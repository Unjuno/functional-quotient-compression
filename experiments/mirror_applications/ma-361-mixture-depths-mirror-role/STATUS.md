# MA-361 status

- Status: FAIL at amended development gate; fresh sealed
- Branch: `research/ma-361-mixture-depths-mirror-role-20261008`
- Base commit: `c935a90`
- Development: 2 seeds x 4 controls rerun after A1; exact replay passed
- Fresh/audit: not opened; 36111–36113 remain sealed
- Result bundle: 24116ef7edf9500c5128ddb7c0f08d03f8d41974
- Registry/claim: registry and status board now carry the amended-run metrics; claim provenance needs binding

## Decision

Mirror and direct scalar gate have identical payload hashes and metrics in both development seeds. Shared MoD is about 52% smaller than native MoD, but seed 36102 misses the frozen +0.01 NLL margin (0.11502 vs 0.10227). The registered hypothesis fails; no Mirror-specific gain is established.

## Next action

Bind verification and claim provenance to the amended result bundle, push, then recheck the latest registry and branches before selecting another candidate.
