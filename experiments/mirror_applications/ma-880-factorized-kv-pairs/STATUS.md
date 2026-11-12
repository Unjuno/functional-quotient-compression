# MA-880 status

- Status: SCREENING
- Branch: `research/ma-880-factorized-kv-pairs-20261112`
- Base commit: `00402001a`
- Four model snapshots downloaded and cached before any model-quality development run
- Protocol freeze: `ecf6f00e6`
- Development: complete; selected CP rank 4 at 600 updates (lowest held-out dev NLL)
- Development quality screen: factor NLL remains 7.2–14.3 nat worse than independent full maps across dev pairs; settings frozen regardless
- Fresh/audit opened: no (settings frozen before fresh run)
- Results: pending
- Verification: pending

The experiment tests whole ordered-pair generalization from a source×target×role factor model. The ordinary CP factorization alias is explicit.

- Amendment 01: rank-code standardization from training pairs only, before fresh access.
