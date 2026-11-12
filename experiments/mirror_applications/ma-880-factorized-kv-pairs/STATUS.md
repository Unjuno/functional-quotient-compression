# MA-880 status

- Status: SCREENING
- Branch: `research/ma-880-factorized-kv-pairs-20261112`
- Base commit: `00402001a`
- Four model snapshots downloaded and cached before any model-quality development run
- Protocol freeze: `ecf6f00e6`; amendments `40095e836`, `294690119`, and RoPE correction pending commit
- Development: RoPE-corrected; selected CP rank 1 at 600 updates
- Corrected dev factor NLL is 4.34 nat and 15.63 nat worse than full maps on the two dev pairs; 6.58/16.05 nat worse than native
- Fresh/audit opened: yes; unrotated implementation deviation is retained; corrected fresh pending; unrotated implementation deviation retained
- Results: pending
- Verification: pending

The experiment tests whole ordered-pair generalization from a source×target×role factor model. The ordinary CP factorization alias is explicit.

- Amendment 01: rank-code standardization from training pairs only, before fresh access.

- Post-fresh accounting amendment 02: diagnostic replay only; model/pair/prompt/Q/update settings unchanged.

- RoPE correction follows the original frozen protocol; fresh deviation outputs are preserved separately.

- Corrected Q and dev outputs frozen in `FROZEN_SETTINGS.json` and `dev_results.json` before corrected fresh replay.
