# MA-1003 status

- Status: FAIL (development gate; fresh sealed)
- Branch: `research/ma-1003-reactive-surface-mirror-20261114`
- Base commit: `6463b1676`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no; stage-6 metadata manifest is locked but no raw files/metrics were downloaded because the development gate failed.
- Results committed: pending commit
- Verification committed: pending commit
- Registry row updated: pending commit

## Next action

Commit the checked development failure and proceed to MA-1004 on its own research branch.

## Blockers

The curated extxyz lacks trajectory/path IDs and is excluded from primary scoring. NOMAD mainfile paths provide run-group split keys; the stage-6 audit manifest is frozen and hashed before raw access (SHA-256 `7be703b...f0d82c2`); stage-6 raw files remain unopened. Stage-1 selected AIMD outputs are nonconverged and excluded by Amendment 02. Barrier labels/path pairing are not explicit. PA301 training recipe has not yet been operationally reproduced on this checkpoint.

## Decisions / rulings

- Do not use the domain-trained H2/Cu checkpoint as the shared prior; it overlaps the target domain.
- Exclude the 12 initial per-cell prior diagnostic frames from all future fresh scoring.
- Do not claim barrier MAE from this data absent verified reaction paths/barriers.

## Decision

FAIL on development quality and Mirror-specific attribution. Rank-2/4-shot improves offset-corrected energy error versus frozen MACE-MP, but force error worsens; independent readouts are substantially better. Native low-rank factorization has the exact same inference payload/function. Fresh remains sealed.
