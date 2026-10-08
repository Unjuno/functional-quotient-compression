# MA-742 status

- Status: FAIL (fresh quality gate passed 2/3 worlds; all-world criterion failed)
- Branch: `research/ma-742-rgcn-attribute-mirror-20261008`
- Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`
- Pre-fresh freeze commit: `74e0d32`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes; seeds 74201, 74202 and 74203
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Result

Mirror rank 2 uses 104 B of coordinate payload vs 264 B for native relation coefficients, and 1,128 B total vs 1,288 B. Fresh quality passed in worlds 74202 and 74203; world 74201 had 3.90× native seen MSE and 4.67× additive held-out MSE. The all-world quality gate therefore fails.

## Next action

Commit the report and verification, then update the registry, claim ledger and status board. Continue with a new random draw after MA-742 is recorded.

## Blockers

None for this scoped CPU mechanism screen.
