# MA-361 status

- Status: FAIL at development gate
- Branch: `research/ma-361-mixture-depths-mirror-role-20261008`
- Base commit: `c935a90`
- Last verified commit: `fb7c8ec6f0df0c9678e58b6bc766d2f8b0037cba`
- Development complete: yes
- Fresh/audit opened: no; 36111–36113 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: no

## Decision summary

Mirror/direct gate outputs and payload hashes matched exactly. Shared MoD reduced bytes against native MoD by ~52%, but seed 36102 missed the 0.01 NLL margin. Fresh sealed.

## Next action

Verify replay/tests, record FAIL in registry and board, then proceed to next P0.
