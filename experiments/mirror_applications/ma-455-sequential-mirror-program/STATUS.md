# MA-455 status

- Status: FAIL
- Branch: `research/ma-455-sequential-mirror-program-20261008`
- Base commit: `04554b5`
- Protocol frozen: yes; SHA-256 f26455ad6123f78f587d688e849741947d5eb316d219a6fdef4c6edea91744bf
- Development complete: yes; seeds 45501, 45502
- Fresh/audit opened: no; sealed by frozen gate
- Results committed: pending
- Verification committed: pending
- Registry row: SCREENING until terminal metadata update

## Decision

FAIL: actual payload exceeds the independent three-block bank by 31.6% in both seeds, native Givens exactly aliases Mirror, and seed 45501 misses the 0.05 RMSE gate. Reverse View ordering increases error on both seeds. See README.md for H/T/D/C/U and the bounded interpretation.
