# MA-357 status

- Status: FAIL at development
- Branch: `research/ma-357-hopfield-mirror-address-reservoir-20261008`
- Base commit: `c935a90`
- Last verified commit: `dd10a811caae56111dd2d88494f31dbb7d39df81`
- Development complete: yes
- Fresh/audit opened: no; 35711–35713 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes on integration branch (FAIL)

## Decision summary

Hopfield used the same packed address payload as explicit Hamming, had higher query MACs, and produced larger decoded-code error due to soft mixture interference. K=256 retrieval accuracy fell to 0.925–0.941. See README and results.

## Next action

Integrated from dedicated branch commit `dd10a811caae56111dd2d88494f31dbb7d39df81`; 27 development rows replay exactly and three tests pass. Fresh remains sealed.
