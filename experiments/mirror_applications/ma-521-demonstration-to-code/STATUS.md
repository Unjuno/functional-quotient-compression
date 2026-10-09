# MA-521 status

- Status: **FAIL**
- Branch: `research/ma-521-demonstration-to-code-20261009`
- Protocol frozen before development.
- Development seeds 52101/52102: complete; deterministic replay exact.
- Fresh seeds 52111–52113: sealed and unopened.
- Registry/status board: FAIL after reconciliation.

## H / T / D / C / U

- **H:** rank-three demo-to-code compilation would preserve explicit-FV causal quality, fit within half the FV bytes, save >=80% ICL context tokens, and beat a same-rank linear compiler.
- **T:** pinned Pythia-70m; 16 functions; 8 support examples; 12 fit and 4 held-out tasks; two dev seeds; 2,000 updates; direct-ICL, explicit-FV and linear controls.
- **D:** FAIL. Tanh compiler loses 1.301/.729 nats vs explicit FV, is .120/.026 nats worse than linear, and misses the byte cap by 23 B. It saves 89.7%/90.1% query tokens. Fresh remains sealed.
- **C:** pooled frozen embeddings do not capture the task function, while direct ICL is more accurate in gold likelihood and uses a smaller token-bank payload.
- **U:** other encoders, task families, model sizes and query-count amortization remain untested.

## Evidence classification

- **Facts:** three tests pass; eight payload hashes, splits, teacher vectors and demo features replay exactly; no fresh seed directory exists.
- **Interpretation:** token savings do not compensate for missed causal quality, storage and attribution gates.
- **Hypothesis:** richer demo representations may alter this result but require a new protocol.
