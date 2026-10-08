# MA-257 status

- Status: FAIL for Mirror-specific value at development
- Branch: `research/ma-257-compositional-mirror-context-20261008`
- Base commit: `2be81372d70e1bdcaab706c932df66e17603b6d7`
- Result source commit: `ad5edd503357c74f04bfff917dd1ff83af1ee5b9`
- Development complete: yes; selected support_residues=1 (64/512 combinations)
- Fresh seeds 25711–25713: locked and **unopened**
- Protocol amendment A1 added native PA16 rotational composition before fresh access; A2 froze selected support before any fresh access
- Strongest control: native PA16 rotational context; exact function and payload-length equality
- Tests: 7 pass; selected-support 54-row metric/byte/compute/hash replay: exact

## Result

The aligned teacher supports held-out composition from 1/8 task-combination support. But Mirror FP16 angles and native PA16 rotational contexts have identical function and 296B payload; test nMSE is 6.58e-9 for both. Estimated peak transform workspace is 14,336B; this is not stored payload. Mirror fails the Mirror-specific gate. On independent tasks, Mirror mean test nMSE is 8.64 and coefficient pairs are 1.036; independent full weights are near zero.

## Next action

Verify replay, record FAIL in registry/claim ledger/status board, and advance to the next untested P0 candidate MA-258. Fresh remains sealed because the exact native-control equivalence decided the preregistered claim.

## Blockers

None for this scoped synthetic post-fit mechanism screen.
