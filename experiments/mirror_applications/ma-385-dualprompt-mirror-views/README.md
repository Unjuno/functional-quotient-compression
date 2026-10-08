# MA-385 — DualPrompt general prompt + Mirror expert prompt views

Status: **FAIL — aligned development gates missed; fresh seeds remain sealed.**
Evidence lane: QUALITY / STORAGE / COMPUTE / CONTINUAL PROMPT
Base commit: `6bbb19c`
Prior art: PA57, DualPrompt separates task invariant general prompts from task specific expert prompts.

## H — hypothesis

When task expert deltas lie on a low dimensional Givens orbit, one shared expert basis with a task-specific Mirror coordinate, added to one general prompt, preserves continual task quality with fewer actual bytes than explicit expert prompts and beats scalar and low rank private residual controls. Unrelated task deltas should require private state.

## T — protocol and execution

Protocol was frozen before development. Two development seeds (38501, 38502), aligned and unrelated task prompt families, eight tasks, 32D inputs, four classes, and 1,600 Adam updates per method. Training was sequential at 200 updates per task. After each task block, test accuracy was recorded for all tasks seen so far. All models use the same nearest centroid retrieval derived from training inputs; task identity is hidden at inference. Six controls were evaluated: explicit DualPrompt, tied expert, scalar basis, Givens Mirror, shared rank 4 private residual basis, and a prompt hypernetwork. The classifier, retrieval keys, prompt state, metadata, and archive headers are charged in the serialized FP16 ZIP/NPY payload.

Final quality below is measured after reloading the actual FP16 payload. Mean positive forgetting is an exploratory descriptive statistic: for each task, subtract final payload accuracy from its accuracy after its first training block, clamp at zero, then average over eight tasks. It was not a preregistered gate.

## D — decision

**Fact:** Retrieval accuracy was 1.0 throughout. In aligned seed 38501, final task accuracy was explicit 0.8682, hypernetwork 0.8398, scalar 0.6077, rank 4 private 0.6111, Mirror 0.4907, and tied 0.3806. In aligned seed 38502 it was explicit 0.8081, scalar 0.7354, rank 4 private 0.7358, Mirror 0.7275, tied 0.6294, and hypernetwork 0.7078. Mirror missed the registered explicit quality margin in both seeds and was 11.7 percentage points below scalar in seed 38501; in seed 38502 it was 0.8 points below scalar. Its 2,269 B payload was 91.8% of the 2,471 B explicit payload, missing the registered <=80% byte gate. Prompt generation proxy was 64 MACs/query for Mirror versus 32 for scalar. Mean positive forgetting in the aligned worlds was Mirror 0.379/0.158, scalar 0.227/0.124, and explicit 0.029/0.061. All 24 serialized method payloads reloaded and replayed final test metrics with zero reported NLL difference. Four tests passed. Fresh seeds 38511–38513 were not opened.

In unrelated worlds, Mirror also trailed scalar and rank 4 private: 0.4844 vs 0.6338/0.6348 and 0.5498 vs 0.6946/0.6953. The explicit pool reached 0.7576/0.7722. Detailed per method/per task outcomes are in `RESULTS_CORE.csv` and `results/development/`.

**Interpretation:** This registered synthetic continual-learning protocol FAILs for the Mirror expert-view hypothesis. Mirror did not improve over the scalar or rank 4 residual control, added prompt-generation work, and its byte savings were too small after retrieval keys and the frozen classifier were charged. In the harder aligned seed, even the shared controls showed substantial forgetting, while independent expert prompts retained much more quality.

**Hypothesis:** A possible mechanism is optimization sensitivity of angles when the shared basis starts at zero; the angle derivative is initially zero at that point. This was not isolated by an initialization ablation and is not evidence that a better initialization would pass the gates.

## C — strongest counter-hypothesis

Only 200 updates were allocated per task, and the frozen synthetic classifier makes prompt optimization sensitive to the seed. The hypernetwork was competitive with explicit prompts in one aligned seed but cost 5,823 B; this points to optimizer/task-structure effects and does not establish Mirror's representational ceiling.

## U — unresolved

Natural DualPrompt benchmarks, longer streams, rehearsal baselines, initialization ablations, and near-converged fixed-byte frontiers are untested. No general prompt-view or capacity claim is made.
