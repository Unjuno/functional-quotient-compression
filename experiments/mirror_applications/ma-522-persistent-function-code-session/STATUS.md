# MA-522 status

- Status: FAIL
- Branch: `research/ma-522-persistent-function-code-session-20261009`
- Model: pinned GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32
- Fresh: 52210–52212 × seeds 0–2 (9 banks; 45 rows)
- Result commit: `42b813e2`; tests: 2 passed; state/context bytes and hashes verified

## H / T / D / C / U

**H:** Session-stored Mirror code retains repeated-ICL quality while reducing combined state+context bytes.

**T:** 16 task sessions × 8 turns; repeated ICL, query-only, explicit FV, rank-4 PCA, K=8 PQ; context serialized as actual token-ID payload.

**D: FAIL.**

**Fact:** Repeated ICL accuracy 42.19%; explicit/PCA/PQ each 1.56%. Repeated ICL total context 14,957B; persistent totals 56,406B explicit, 23,118B PCA, 32,018B PQ. Context-only query stream dropped from 4,864 to 1,312 tokens, but paid state erased the byte saving. CPU wall-clock for 128 queries: ICL 7.87s; PCA 1.98s; PQ 1.95s.

**Interpretation:** Fails quality and the registered combined-byte gate at 8 turns.

**C:** Prompt-delta extraction did not make executable functions; the saved state is a compressed nonfunctional intervention.

**U:** Natural dialogue, causal head FVs, longer horizons and online updates.
