# MA-357 status

- Status: FAIL at development
- Branch: `research/ma-357-hopfield-mirror-address-reservoir-20261009`
- Base commit: `c935a90`
- Last verified commit: `b383738e`
- Development complete: yes
- Fresh/audit opened: no; 35711–35713 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: no

## Decision summary

Hopfield used the same packed address payload as explicit Hamming, had higher query MACs, and produced larger decoded-code error due to soft mixture interference. K=256 retrieval accuracy fell to 0.925–0.941. See README and results.

## Next action

Verify replay and tests, record the FAIL in registry and evidence index, then continue to next P0.
