# MA-521 status

- Status: FAIL
- Branch: `research/ma-521-function-vector-demo-code-extraction-20261009`
- Model: pinned GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32
- Fresh: 52110–52112 × seeds 0–2; 9 banks; 36 rows
- Tests: 2 passed; payload hashes/sizes verified

## H / T / D / C / U

**H:** A 4D code predicted from demonstrations selects useful FV behavior with lower state/context cost.

**T:** Direct ICL, query-only, oracle FV, ridge-fitted linear code predictor and full FV bank.

**D: FAIL.**

**Fact:** Code identity accuracy averaged 25%; direct ICL accuracy 37.5%; oracle and compiled FV task accuracy 0%. Compiled state 61,017B vs explicit FV+ID 50,921B. Context tokens and inference latency not measured.

**Interpretation:** Code prediction did not produce function utility and increased paid state.

**C:** Task identity is recoverable from demonstrations, but prompt-delta FV representation is not a useful function.

**U:** Natural-language transfer, causal FV extraction, token savings, latency, and generic HyperFormer control.
