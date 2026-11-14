# MA-1009 status

- Status: FAIL (bounded synthetic development gate)
- Branch: `research/ma-1009-dune-representation-chart-20261114`
- Base commit: `db8f25f35`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Commit checked results and status updates; then continue at MA-1010.

## Blockers

No blocker for this synthetic mechanism screen. Official DUNE source at pinned commit is README-only, with no declared code license or checkpoint; the DUNE-z equations were implemented independently. Acquired and clinical data claims are not established.

## Decisions / rulings

The final amendment-07 run is decision-bearing. Amendments 01, 03, and 04 encountered respectively a fully determined operator, a zeroed unmeasured prediction, and a non-exact image projection; their exploratory metrics are separately retained and excluded. Amendment 02 uses an underdetermined 2-coil operator. Amendment 05 uses a QR orthogonal image-space projection. Amendment 06 measures cached inference separately from one-time QR setup. Exact replay and the acquired-sample residual tests pass. Fresh remains sealed after the quality gate miss.


## Amendment 07 ruling

A final audit found that standalone and two-scanner payloads used different serialization metadata wrappers. The interim byte rows are preserved separately but excluded from the final storage conclusion. All alternatives are reserialized with identical metadata and object keys before comparing complete payload bytes. Quality outcomes are unchanged; fresh remains sealed.
