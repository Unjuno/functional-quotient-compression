# MA-476 status

- Status: PROMISING for synthetic aligned value storage; no broad GRACE claim
- Branch: `research/ma-476-grace-value-mirror-20261009`
- Base commit: `7b1ae9f6`
- Protocol freeze: `5b178cdd`
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze
- Results committed: yes (`581bae90`)
- Verification committed: yes (`581bae90`)
- Registry row updated: yes (same report commit)

## Next action

MA-476 is complete; continue with MA-478 on its dedicated branch.

## Limitations

- Keys/radii are synthetic and unchanged; only value representation is tested.
- The two-dimensional value orbit and uniform VQ codebooks are specified analytically, not learned from real edits.
- Continuous angle coding is equivalent to a polar parameterization; the generic coefficient control is close in bytes.
- Retrieval timings are CPU microbenchmarks and do not include serving infrastructure or initial payload load.

## Decisions / rulings

- K=8/16/32/64 codebooks are a preregistered fixed sweep, not selected using fresh outputs.
- Every codebook tensor and per-entry index is included in actual payload bytes.
- Per-key radii are stored and identical across all methods, isolating value compression.
