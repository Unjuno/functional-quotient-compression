# MA-494 status

- Status: **FAIL** for Mirror-specific attribution
- Branch: `research/ma-494-error-correcting-expert-ids-20261008`
- Base commit: `9e5eb1de16b0e4e5982fd28beb666a27f18d489d`
- Protocol frozen: yes
- Development complete: yes (49401, 49402; five corruption rates)
- Fresh/audit opened: no (49411–49413 sealed)
- Verification: serialized replay passed; tests 3 passed

## Next action

Proceed to MA-498 code-distance regularization, next untested P0 after MA-494.

## Decisions / rulings

Hamming(7,4) passed the p=.05 reliability/bytes feasibility gate but exactly implements standard native ECOC. FAIL for Mirror-specific attribution; fresh remains sealed.
