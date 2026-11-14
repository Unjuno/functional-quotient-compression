# MA-1003 status

- Status: SCREENING
- Branch: `research/ma-1003-reactive-surface-mirror-20261114`
- Base commit: `6463b1676`
- Last verified commit: pending
- Development complete: no
- Fresh/audit opened: no (12 initial diagnostic structures are burned development data)
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Implement the shared-basis/readout controls on the parsed development data, tune rank on stages 3–4, use stage 5 only for few-shot support, and freeze configs before querying stage 6 archives.

## Blockers

The curated extxyz lacks trajectory/path IDs and is excluded from primary scoring. NOMAD mainfile paths provide run-group split keys; the stage-6 audit manifest is frozen and hashed before raw access (SHA-256 `7be703b...f0d82c2`); stage-6 raw files remain unopened. Stage-1 selected AIMD outputs are nonconverged and excluded by Amendment 02. Barrier labels/path pairing are not explicit. PA301 training recipe has not yet been operationally reproduced on this checkpoint.

## Decisions / rulings

- Do not use the domain-trained H2/Cu checkpoint as the shared prior; it overlaps the target domain.
- Exclude the 12 initial per-cell prior diagnostic frames from all future fresh scoring.
- Do not claim barrier MAE from this data absent verified reaction paths/barriers.
