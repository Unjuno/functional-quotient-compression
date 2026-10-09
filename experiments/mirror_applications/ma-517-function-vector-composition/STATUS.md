# MA-517 status

- Status: FAIL
- Branch: `research/ma-517-function-vector-composition-20261009`
- Model: pinned `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, 124,439,808 parameters, CPU float32
- Fresh worlds/seeds: 51710–51712 × 0–2 (9 banks; protocol locked before fresh run)
- Tests: 2 passed; serialized artifact hash/byte and fresh split checks passed.

## H / T / D / C / U

**H:** Role-specific layer placement of two extracted task vectors would improve held-out composition accuracy over raw same-layer addition at equal payload cost.

**T:** Four fruit→color and four color→shape support pairs per task; four held-out fruit queries; query-only, direct combined ICL, raw `f1+f2` at layer 6, factorized `f1@4/f2@8`, and individual-vector ablations. Nine fresh banks; all intervention payloads and metadata serialized and charged.

**D: FAIL.**

**Fact:** Direct ICL, query-only, raw sum, factorized, and each single-vector condition scored 0/32 correct held-out targets in each of the 9 banks (0% accuracy). Mean target NLL across banks was 6.538 for direct ICL, 11.184 query-only, 9.967 raw sum, and 9.878 factorized. Serialized two-vector payloads were 99,945 B for both raw and factorized methods; each single-vector ablation was 50,793 B.

**Interpretation:** The registered quality gate fails before any layer-factorization advantage can matter. Factorized placement slightly reduced mean NLL relative to raw sum but did not change exact task accuracy and had equal payload bytes.

**C:** The task vectors are prompt-minus-query residual states, not causally isolated reusable operators; direct ICL itself was at chance on this tokenizer/task/model setup. Target-token prediction may also be mismatched to GPT-2's learned behavior.

**U:** Other prompts, token scoring, causal head-specific function vectors, and larger pretrained models are not tested. This is not evidence that all function-vector composition fails.
