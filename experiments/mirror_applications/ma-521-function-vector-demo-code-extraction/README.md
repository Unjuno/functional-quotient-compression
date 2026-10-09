# MA-521 — Mirror code extraction from demonstrations

Status: SCREENING; protocol fixed before fresh evaluation.

> **Mirror insertion:** this experiment adds a demonstration-conditioned encoder `m(demos)` to a frozen LM so demonstrations compile into a small functional code that selects/decodes an intervention without replaying the full prompt at inference.

PA99 motivates function vectors; PA82 establishes shared task-conditioned hypernetworks as a strong native control. Compare full ICL, a frozen encoder plus linear code head, a prototype router, and explicit task IDs. Report routing, code reconstruction and downstream target behavior separately.

H: a compact context encoder predicts a useful FV code, retaining target quality while reducing stored prompt/context bytes and per-task FV bytes.

C: task identity may be encoded accurately while the underlying prompt-delta FV fails to execute the task (MA-516–520).
