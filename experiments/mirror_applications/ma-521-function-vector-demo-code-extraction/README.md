# MA-521 — Mirror code extraction from demonstrations

Status: FAIL.

> **Mirror insertion:** this experiment adds a demonstration-conditioned encoder `m(demos)` to a frozen LM so demonstrations compile into a small functional code that selects/decodes an intervention without replaying the full prompt at inference.

PA99 motivates function vectors; PA82 is the generic hypernetwork/task-code control. This screen used a frozen GPT-2 hidden-state feature and a 4D linear code predictor, with an FV bank and query-time intervention.

## H / T / D / C / U

**H:** A compact demonstration encoder predicts a useful FV code and reduces stored/context cost while preserving task behavior.

**T:** 16 synthetic procedure×domain tasks; 3 fresh worlds × 3 seeds; direct ICL, query-only, oracle FV, compiled Mirror code.

**D: FAIL.**

**Fact:** Fresh code identity accuracy was 25% (chance 6.25%). Direct ICL averaged 37.5%; query-only 6.25%; oracle and compiled FV task accuracy were both 0%. Compiled payload was 61,017B; it contains the full explicit FV bank plus encoder/codes and therefore exceeds the 50,921B explicit bank+IDs by ~19.8%.

**Interpretation:** The encoder predicts some task identity, but the underlying FVs remain nonfunctional and the compiled state is larger than the explicit bank. No useful code extraction or context savings established.

**C:** Same prompt-delta extraction failure as MA-516–520; task classification does not imply the intervention executes the task.

**U:** HyperFormer-quality trainable encoder, symbolic router, natural ICL, context token accounting, runtime and causal head vectors. Token savings/runtime were not measured in this run.
