# MA-880 status

- Status: SCREENING
- Branch: `research/ma-880-factorized-kv-pairs-20261112`
- Base commit: `00402001a`
- Four model snapshots downloaded and cached before any model-quality development run
- Protocol freeze: `ecf6f00e6`; amendments `40095e836`, `294690119`, and RoPE correction pending commit
- Development: initial unrotated run archived as deviation; corrected development pending
- Initial unrotated development is not used for corrected setting selection
- Fresh/audit opened: yes; unrotated implementation deviation is retained; corrected fresh pending
- Results: pending
- Verification: pending

The experiment tests whole ordered-pair generalization from a source×target×role factor model. The ordinary CP factorization alias is explicit.

- Amendment 01: rank-code standardization from training pairs only, before fresh access.

- Post-fresh accounting amendment 02: diagnostic replay only; model/pair/prompt/Q/update settings unchanged.

- RoPE correction follows the original frozen protocol; fresh deviation outputs are preserved separately.
