# MA-511 status

- Status: **FAIL for Mirror-specific attribution**
- Branch: research/ma-511-hierarchical-condition-behavior-codes-20261008
- Protocol frozen before development: yes (freeze commit bc853bb)
- Amendment 1: unit-test tolerance only; runner/protocol/data/gates unchanged
- Development: complete (51101/51102)
- Fresh/audit opened: **no** (51111–51113 remain sealed)
- Verification: 3 tests passed; 54 payload hashes and metrics replay exact; split manifests exact

## Decision

Rank4/rank4/rho0 passes the frozen synthetic composition quality and byte gates, but native additive factorization plus PCA exactly aliases Mirror. CAST additive reaches zero error at 5,414 B and one-third the rank-4 decode operation proxy. FAIL for Mirror-specific attribution; fresh stays sealed.

## Fact / interpretation / hypothesis

- Fact: Mirror rank4 uses 4,136 B vs 16,932 B full pair table (0.244x); held-out RMSE .0244/.0452.
- Fact: rho=.1 errors .1115/.1200; rho=.25 errors .2673/.2749.
- Fact: native additive/PCA payloads and outputs exactly match Mirror across all fixed conditions.
- Interpretation: ordinary additive condition-behavior factorization supports zero-shot held-out pair reconstruction in this synthetic world, with a storage/compute frontier.
- Hypothesis: trained-LM semantic function compositions may have a different frontier; not measured here.

## Next action

After ledger and registry integrity checks, push this branch and proceed to MA-516.
