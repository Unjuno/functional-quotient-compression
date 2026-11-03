# MA-545 branch reconciliation

Two independently frozen MA-545 protocols exist and are retained separately; do not pool their metrics.

- Canonical registered experiment: `research/ma-545-fv-residual-moe-20261009`. It is the claim-ledger result: few-shot routed function vectors lose the primary held-out accuracy metric to the shared-mean FV on all five seeds, although they improve gold likelihood and use fewer bytes than the matched rank-one weight experts.
- Alternative activation-only screen: `research/ma-545-fv-activation-moe-20261009`. It also ends in FAIL for Mirror-specific attribution: activation FVs improve gold likelihood but match the same-router output-bias control within the frozen tolerance and lower top-1 accuracy. Its tests passed 3/3 on its dedicated branch. It is retained there as an alternative MA-545 protocol; its helper dependency `ma-516-function-vector-mirror-compression/source/run_screen.py` is branch-local.

The registry and claim ledger point to the canonical residual-router experiment. The alternative branch is evidence of the same high-level failure family, not a second independent registry outcome.
