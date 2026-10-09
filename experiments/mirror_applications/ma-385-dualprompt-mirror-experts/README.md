# MA-385 — DualPrompt expert prompt bank with Mirror views

Status: protocol frozen before development. Dedicated branch: `research/ma-385-dualprompt-mirror-experts-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens coordinate `m_i` to a shared expert-prompt basis so task-specific expert prompt functions can vary without storing eight independent expert vectors; the common general prompt stays shared and paid.

## H — Hypothesis

A Mirror-coded expert bank can retain task-incremental quality and old-task retention at no more than 60% of the full expert-prompt pool bytes, beyond the common general prompt shared by all methods.

## T — Frozen design

PA57 DualPrompt separates task-invariant general prompts and task-specific expert prompts. This proxy fixes one shared general prompt, then reveals eight expert tasks sequentially. Task ID is available at test, consistent with the specified task-incremental setting. Compare independent experts, hard tying, scalar gates, unrestricted two-coordinate basis, and Mirror angles. Earlier tasks are reevaluated after every new task. Seeds, schedule and gates are frozen in `PROTOCOL.json` before implementation.

This differs from MA-383's L2P retrieval screen: MA-385 tests the task-specific expert side of DualPrompt, not prompt-key retrieval.
