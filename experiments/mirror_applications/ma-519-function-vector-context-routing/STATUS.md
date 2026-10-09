# MA-519 status

- Status: FAIL
- Branch: `research/ma-519-function-vector-context-routing-20261009`
- Model: pinned GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32
- Fresh: 51910–51912 × seeds 0–2; 9 banks; 45 rows
- Result commit: `10195e6b`; verification replay tests: 2 passed; serialized payload hash/size checks passed

## H / T / D / C / U

**H:** A content-derived address selects an FV bank entry with high accuracy and useful task behavior, replacing explicit task IDs at lower storage.

**T:** Direct ICL, query-only, oracle FV, cosine content-routed FV and explicit-ID serialized bank; 16 synthetic procedure×domain tasks.

**D: FAIL.**

**Fact:** Direct ICL averaged 25% accuracy, query-only 6.25%, and both oracle and routed FV interventions 0%. Router accuracy averaged 95.83%; six of nine banks reached only 93.75%. Routed payload was 75,685B versus 50,921B for explicit bank plus IDs.

**Interpretation:** High routing accuracy did not yield useful functions and cost 48.5% more bytes. The registered success gate fails. The symbolic keyword router control was not implemented, so the routing gain is not Mirror-specific.

**C:** Prototype matching may only recognize the visible task words; the prompt-delta vector bank does not encode executable functions.

**U:** Symbolic routing, causal FV extraction, natural-language task generalization and runtime profiling.
