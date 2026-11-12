# MA-880 status

- Status: SCREENING
- Branch: `research/ma-880-factorized-kv-pairs-20261112`
- Base commit: `00402001a`
- Four model snapshots downloaded and cached before any model-quality development run
- Protocol freeze: `ecf6f00e6`; code amendment `40095e836`; frozen settings `1cfae376b`
- Development: complete; selected CP rank 1 at 600 updates (lowest held-out dev NLL)
- Development quality screen: factor NLL is 5.68 and 19.99 nat worse than independent full maps on the two dev pairs, and 7.59/20.59 nat worse than native; settings frozen regardless
- Fresh/audit opened: no (settings frozen before fresh run)
- Results: pending
- Verification: pending

The experiment tests whole ordered-pair generalization from a source×target×role factor model. The ordinary CP factorization alias is explicit.

- Amendment 01: rank-code standardization from training pairs only, before fresh access.

- Post-fresh accounting amendment 02: diagnostic replay only; model/pair/prompt/Q/update settings unchanged.
