# MA-522 — persistent function code across conversation

Status: FAIL.

> **Mirror insertion:** this experiment adds a persistent session code `m_session` that is written once from task demonstrations and reused across later turns, replacing repeated task demonstrations with a small decoded residual intervention.

PA99 motivates function vectors; PA25 motivates session fast state. We tested an 8-turn frozen-GPT-2 session bank with direct ICL, query-only, explicit FV, rank-4 PCA, and K=8 PQ state. All state and context token-ID payloads were serialized and charged.

## H / T / D / C / U

**H:** Persistent state amortizes repeated task demonstrations while keeping per-turn quality within 5pp of repeated ICL and lowering total state+context bytes.

**T:** 16 synthetic procedure×domain sessions, 8 turns, 3 fresh worlds×3 seeds; pinned GPT-2 CPU float32. Context bytes are serialized token-ID payloads, with base model bytes reported separately.

**D: FAIL.**

**Fact:** Across 9 fresh banks, repeated ICL averaged 42.19% accuracy (NLL 3.4936); explicit, PCA, and PQ persistent FV interventions each averaged 1.56% (NLL 7.0880/7.1590/7.1786). Query-only was 1.56%. Per-turn ICL accuracy ranged 37.5–47.9%; intervention methods ranged 0–6.25%.

Repeated ICL used 4,864 context tokens / 14,957B serialized context. Query-only and persistent conditions used 1,312 tokens / 5,485B. Adding state produced totals of 56,406B explicit, 23,118B PCA, and 32,018B PQ; all exceed the 14,957B repeated-ICL total. Mean full-bank CPU inference time was 7.87s for repeated ICL, 2.03s explicit FV, 1.98s PCA, and 1.95s PQ for 128 queries. Pinned model/tokenizer payload was 550,959,861B for all methods and is excluded from these incremental totals.

**Interpretation:** Codes save context and inference time, but fail quality by over 40pp; at the registered 8-turn horizon they also increase combined state+context bytes. No persistent useful function is established.

**C:** Prompt-delta FVs are not causal task operators; persistence only reuses a low-utility intervention. Longer horizons could change cost amortization but cannot repair the observed quality gap.

**U:** Natural conversation tasks, causal head-specific FVs, horizons beyond eight turns, and online state updates.
