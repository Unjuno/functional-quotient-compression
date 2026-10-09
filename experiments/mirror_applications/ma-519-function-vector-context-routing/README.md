# MA-519 — function-vector routing from context

Status: FAIL.

> **Mirror insertion:** this experiment adds a content-derived router coordinate `m(x_context)` to a frozen function-vector bank so an input context selects the functional intervention without an externally supplied task ID.

Native control is direct in-context learning. Mirror uses a cosine nearest-prototype router at GPT-2 block 6 to select an extracted task-vector intervention. Router accuracy and downstream function execution are assessed separately.

PA99 reports causal head-linked function vectors and partial composition. This experiment tests whether a task-ID lookup can be replaced by context-derived selection.

## H / T / D / C / U

**H:** Context routing replaces explicit task IDs while preserving function behavior and reducing payload.

**T:** Pinned GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; 16 procedure×domain task labels; 3 fresh worlds×3 seeds. Controls: direct ICL, query-only, oracle FV, content-routed FV, explicit-ID FV payload.

**D: FAIL.**

**Fact:** Mean direct-ICL accuracy was 25%; query-only 6.25%; oracle and routed FV both 0%. Cosine routing averaged 95.83% ID accuracy, but six of nine banks were 93.75%. Routed bank payload was 75,685B versus 50,921B for explicit FV bank plus task labels; the router therefore increased bytes by 48.5%.

**Interpretation:** Routing is mostly accurate but does not deliver useful execution because the prompt-delta FVs do not encode the behavior. A zero-cost symbolic keyword router could explain the address selection result; this run did not implement that preregistered control, so no Mirror-specific routing benefit is established.

**C:** Activation prototype similarity identifies the known prompt family, but that identity is already present in task words and the activation deltas are not causal functional operators.

**U:** Symbolic router comparison, causal head-specific FVs, natural tasks, and measured per-method runtime. Router-accuracy-only is not evidence of useful logical functions.
