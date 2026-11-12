# MA-879 status

- Status: FAIL
- Branch: `research/ma-879-attention-sensitive-cache-code-20261111`
- Base commit: `cc3b71cc4`
- Protocol freeze: `0e4fbfc1704cc842d280f02e3e28afe3f86da440`
- Dev settings frozen and committed: `aeac3d4e8`
- Dev settings frozen: selected learning rate `0.002`; dev evidence in `dev_results.json`
- Fresh/audit opened: yes, after freeze commit `aeac3d4e8` (fresh execution begins after this freeze commit)
- Fresh/audit opened: yes, after freeze commit `aeac3d4e8`
- Results: `RESULTS.md`; all three fresh seeds miss quality and pass bytes
- Verification: `VERIFICATION.json`

This experiment fixes source-head support to [0,4,7] and tests target-loss optimization of shared-basis codes. It must not be reported as a Mirror-specific gain if the ordinary low-rank coefficient optimizer explains the result.
