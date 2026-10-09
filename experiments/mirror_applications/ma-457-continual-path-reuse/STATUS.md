# MA-457 status

- Status: FAIL
- Branch: `research/ma-457-continual-path-reuse-20261008`
- Base commit: `2264c54`
- Protocol frozen: yes; SHA-256 25fb9ce4341f410582cc674a16a01eedf66b5688a002c0c00073e10aa9c9803d
- Development complete: yes; seeds 45701, 45702
- Fresh/audit opened: no; sealed by frozen gate
- Results committed: c770221e79e6973fa16c3aff64ce8902ba98569d
- Verification record: points to result commit c770221e79e6973fa16c3aff64ce8902ba98569d
- Registry row: FAIL

## Decision

FAIL: Mirror reduced full-module births to two but its actual payload (2,222 bytes) exceeded both PathNet (1,442) and independent modules (1,180); it also exactly aliased native rank-one conditioning and had higher error and much slower training. Fresh data stayed sealed.

## Preserved implementation records

Three incomplete preflights and the superseded initial control-timing report are retained in `runs/`; they are excluded from the terminal result and were not used for tuning. The full corrected runs pass serialization and metric replay.
