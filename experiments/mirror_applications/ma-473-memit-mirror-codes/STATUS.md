# MA-473 status

- Status: PROMISING for aligned update storage only; additive multi-fact utility unestablished
- Branch: `research/ma-473-memit-mirror-codes-20261009`
- Base commit: `2e394171`
- Protocol freeze: `2a8d0407`
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze
- Results committed: yes (`6c82b0aa`)
- Verification committed: yes (`6c82b0aa`)
- Registry row updated: yes

## Next action

MA-473 is complete; continue with MA-475 on a dedicated branch.

## Limitations

- The screen stores a known analytic layerwise edit bank, not a learned MEMIT result.
- At N=64 the merged edit error is large (mean 4.80 NRMSE), despite exact representation equivalence to direct factors.
- Shared orbit and bases are hand specified; no natural factual-key geometry was tested.

## Decisions / rulings

- The initial fresh invocation stopped on a query-shape bug before writing result rows; no partial output is included. The implementation was fixed and the entire frozen fresh split was rerun.
- Factor controls, generic coordinates and Mirror share the same teacher and composition rule; differences in merged interference are numerical only.
- Small CPU timing probes are diagnostic and are not serving throughput claims.
