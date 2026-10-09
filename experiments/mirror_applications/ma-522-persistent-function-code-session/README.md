# MA-522 — persistent function code across conversation

Status: SCREENING; protocol fixed before fresh evaluation.

> **Mirror insertion:** this experiment adds a persistent session code `m_session` that is written once from task demonstrations and reused across later turns, replacing repeated task demonstrations with a small decoded residual intervention.

PA99 motivates stored functional vectors; PA25 covers fast-weight/session state. Compare repeated direct ICL, explicit persistent FV, compact code, and query-only. Measure per-turn quality and cumulative context plus actual state bytes.

H: A persistent Mirror session code amortizes task specification across turns while retaining direct-ICL quality, with lower combined context+state cost.

C: Existing prompt-delta FVs have failed functional execution; persistence may amortize a useless representation.
