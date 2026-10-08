# MA-288 status

- Status: FAIL (registered direct-path runtime gate missed; aligned state/quality point retained)
- Source verified commit: `4c7610ddbd99e734bd5dd5f7806d511bcd35ef15`; local tests/replay rerun passed
- Branch: `research/ma-288-fastweight-reconciled-20261008`
- Base commit: `e259f27`
- Development complete: yes
- Fresh/audit opened: yes, locked seeds only
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

No further action; complete scoped result indexed on cumulative branch.

## Blockers

None. NumPy/SciPy CPU execution can assess this small mechanism; GPU is unnecessary.

## Decisions / rulings

- Candidate pool, entropy-backed selection, and live branch check are recorded in PROTOCOL.json.
- Aligned state/quality result is strong; registered eager per-query throughput missed in all fresh worlds. Long-session cache diagnostic is supplementary and requires 1,024 B workspace for four cached contexts.
- Each active context's float32 coordinate and session record are included in inference bytes.
