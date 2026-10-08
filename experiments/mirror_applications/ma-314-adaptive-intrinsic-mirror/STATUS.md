# MA-314 status

- Status: PROMISING only for aligned coordinate-only storage/quality; broad private-fallback screen missed its byte gate
- Branch: `research/ma-314-adaptive-intrinsic-mirror-20261008`
- Broad protocol base commit: `e037a79`
- Aligned-only variant base commit: `9ca5c968b5ec9a285b11140f08a0f573c8e07cf3`
- Last verified commits: broad development `a1d5737`; aligned-only fresh screen `4bd89532bb79d9639bb02d29fa608e16236b2f71`
- Broad development complete: yes; its matched adaptive direct payload is smaller than Mirror
- Broad fresh/audit opened: no (remains sealed after byte-gate miss)
- Aligned-only development/fresh complete: yes; both passed that variant's frozen gates
- Results, verification, registry, status board and claim ledger: recorded on this branch

## Next action

Continue with MA-315, which tests explicit sparse private coordinates against raising the global intrinsic dimension.

## Blockers

None.

## Decisions / rulings

- Preserve the broader 160-task private-fallback protocol as the root experiment. Development quality and median dimensions passed, but adaptive Mirror payloads (7,082/7,084B) exceeded adaptive direct controls (6,730/6,734B); keep that protocol's fresh seeds unopened.
- Preserve the independently completed aligned-only coordinate variant under `protocol_variants/aligned_coordinate_only/`. It has different task count, predictor/basis dimensions, world generator and controls; do not pool its fresh results with the root protocol.
- The aligned-only variant's initial angle fitter omitted cross-pair residual terms. This implementation issue was corrected to four coordinate-descent sweeps; the corrected development worlds were rerun before that variant's fresh seeds were opened.
- No private-state or capacity claim follows from the aligned-only variant.
