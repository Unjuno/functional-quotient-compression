# MA-520 status

- Status: FAIL
- Branch: `research/ma-520-function-vector-mirror-distillation-20261009`
- Model: pinned GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32
- Fresh: worlds 52010–52012 × seeds 0–2 (9 banks; 45 rows)
- Verification: 2 tests pass; payload byte/hash replay passes

## H / T / D / C / U

**H:** A product-quantized Mirror code preserves useful task behavior while reducing FV payload bytes.

**T:** Direct ICL, query-only, explicit vectors, generic rank-4 PCA, K=8 PQ Mirror.

**D: FAIL.**

**Fact:** Direct ICL 25% accuracy; query-only 6.25%. Explicit/PCA/PQ all 0%. Mean NLL explicit 6.4128, PCA 6.4988, PQ 6.5109. Actual payloads 50,921B, 17,505B, 14,117B. Vector NRMSE PCA .1522, PQ .1755.

**Interpretation:** Large payload reduction does not preserve logical task behavior. No useful-function compression is established.

**C:** The extracted prompt-delta vectors may not encode the desired operations; vector reconstruction is not causal function retention.

**U:** Runtime, natural-language transfer, causal head FVs, and learned quantizer variants.
