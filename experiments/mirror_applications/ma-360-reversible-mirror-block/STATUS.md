# MA-360 status

- Status: FAIL for Mirror-specific value; reversible tied-block memory point is narrow PROMISING
- Branch: `research/ma-360-reversible-mirror-block-20261008`
- Base commit: `c935a90`
- Last verified commit: `7d0bf37c4e008a804f32a2b94a8e6baa7aa21a86`
- Development complete: yes; checkpoint-only attempt excluded and documented
- Fresh/audit opened: no; seeds 36011–36012 sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes on integration branch (FAIL for Mirror-specific value)

## Decision summary

Reversible tied block saved 41.1% peak live activation bytes at ~1.8× wall time. Mirror/direct scalar views had identical outputs and payload hashes, adding no Mirror-specific value.

## Next action

Integrated from dedicated branch commit `7d0bf37c4e008a804f32a2b94a8e6baa7aa21a86`; 10 payloads replay exactly and three tests pass. The 41.1% activation point is attributed to reversible execution, not Mirror; fresh remains sealed.
