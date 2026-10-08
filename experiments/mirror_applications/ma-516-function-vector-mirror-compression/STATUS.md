# MA-516 status

- Status: **FAIL**
- Branch: research/ma-516-function-vector-mirror-compression-20261008
- Protocol frozen before development: yes (75a80cd)
- Amendments: rank-1 dispatch correction; support-token accounting correction; initial outputs preserved
- Development: complete (51601/51602)
- Fresh/audit opened: **no** (51611–51613 remain sealed)
- Verification: 5 tests passed; 22 payload hashes, split manifests, extracted FVs and metrics replay exact

## Decision

Explicit FVs pass the frozen learnability signal, and rank4 passes accuracy and byte thresholds. It misses the gold-logprob preservation gate by .535/.606 nats. Native PCA exactly matches all Mirror ranks. FAIL; fresh stays sealed.

## Fact / interpretation / hypothesis

- Fact: explicit FVs improve held-out gold logprob by 2.62/2.97 nats over no intervention.
- Fact: rank4 uses 12,688 B vs 34,454 B explicit; held-out accuracy .188/.219 versus .219/.188 explicit, but gold logprob is .535/.606 nats worse.
- Fact: native PCA payload/output aliases Mirror exactly at all ranks.
- Interpretation: task-vector compression produces a measurable byte-quality curve, but rank4 loses ranking confidence and adds code-decoding work; this is ordinary PCA.
- Hypothesis: task-conditioned nonlinear or natural task bases may improve, but untested.

## Next action

After ledger and registry integrity pass, push this branch and proceed to MA-517.
