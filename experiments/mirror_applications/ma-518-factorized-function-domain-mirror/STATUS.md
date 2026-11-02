# MA-518 status

- Status: FAIL (quality gate); factorization claim NOT ESTABLISHED due implementation defect
- Branch: `research/ma-518-factorized-function-domain-mirror-20261009`
- Model: pinned GPT-2 `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32
- Fresh: 51810–51812 × 0–2; 9 world-seed banks; 45 result rows
- Verification: 2 tests pass; payload hash/size and fresh split checks pass

## H / T / D / C / U

**H:** Factorized procedure×domain residual codes preserve held-out task quality and reduce serialized payload versus generic low-rank codes.

**T:** Frozen GPT-2; four deterministic procedures × four token domains; 12 training pairs and four diagonal held-out pairs; direct ICL, query-only, explicit prompt-delta, generic PCA, and factorized conditions. Six development direct-ICL checks were 31.25%; fresh opened only after source freeze.

**D: FAIL for task quality; factorization comparison NOT ESTABLISHED.**

**Fact:** Across 9 fresh banks, direct ICL accuracy was 31.25% (mean target NLL 4.1976), query-only accuracy 6.25% (NLL 6.4163), and explicit, generic, and factorized interventions each had 0% accuracy. Their mean NLLs were 6.4878, 6.5116, and 6.5856 respectively. Actual payloads were 50,921B explicit, 17,633B generic PCA, and 11,993B factorized.

**Implementation defect:** The factorized inference reconstruction used a flat rank-2 projection of each task vector. It did not decode from the serialized procedure/domain coefficient tables; those coefficients were stored but unused. Therefore the 11,993B number is not a valid quality-matched factorized representation and cannot support a factorization/storage claim.

**Interpretation:** The tested prompt-delta interventions failed the registered task-utility gate even though direct ICL worked. This supports a negative result for this extraction/injection construction, not for causal head FVs generally. The Mirror factorization hypothesis remains unestablished.

**C:** Residual prompt-minus-query vectors may not represent reusable causal operators; single-next-token synthetic task formatting may still favor direct demonstrations.

**U:** Corrected factorized decoder, causal head-specific extraction, natural language task utility, and any Mirror Pareto improvement. Fresh data were not reused for tuning.
