# MA-709 status

- Status: FAIL (scoped synthetic; no Mirror-specific attribution)
- Branch: `research/ma-709-toeplitz-logical-experts-20261109`
- Base commit: `eae6e926`
- Frozen configuration: yes; serialization accounting amendment documented
- Development complete: yes
- Fresh/audit opened: yes; corrected same-seed audit replay due accounting amendment
- Results committed: yes (`948225ea`)
- Verification committed: pending

## Result

Aligned (ρ=0) full-model payload is 1,348B and exact; independent Toeplitz is 4,614B and dense is 262,662B. At ρ=.05 Mirror NRMSE rises to .0488–.0533 while independent Toeplitz remains numerical zero. At ρ=.2 Mirror NRMSE is .1908–.2139. Native shared circular-convolution control is an exact payload/function alias. The per-task NPZ accounting attempt is retained under `source/attempts/` and is invalid for storage claims.

## Next action

Update registry and advance to MA-710, learned LDR atoms with Mirror composition/pruning.
