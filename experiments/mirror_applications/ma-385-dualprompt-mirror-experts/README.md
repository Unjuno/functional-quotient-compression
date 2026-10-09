# MA-385 — DualPrompt expert prompt bank with Mirror views

Status: **FAIL on the frozen development quality and storage gates; fresh seeds remain sealed.** Dedicated branch: `research/ma-385-dualprompt-mirror-experts-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens coordinate `m_i` to a shared expert-prompt basis so task-specific expert prompt functions can vary without storing eight independent expert vectors; the common general prompt stays shared and paid.

## H — Hypothesis

A shared expert basis with one Mirror angle per task can retain task-incremental quality and old-task retention using at most 60% of the independent expert-prompt payload, beyond one common general prompt.

## T — What was run

PA57 DualPrompt separates general and task-specific expert prompts. This synthetic task-incremental proxy uses one fixed common general prompt, eight sequential expert tasks, a fixed 8×8 residual decoder, and explicit task ID at evaluation. Independent experts learn one prompt per task; hard tying is updated sequentially; scalar trains one shared vector on task 0 and freezes it; generic and Mirror bases train jointly on tasks 0/1 then freeze while new task codes are added. Each task receives 1,200 Adam updates at LR .01, batch 64. Two development worlds (38501/38502) were run under protocol commit `0e099ad8`.

## D — Decision

**Fact:** Independent final mean NRMSE was <1.5e-7 in both worlds, with zero measured old-task forgetting. Mirror used 2,238/2,241B versus independent 3,447/3,437B (64.9%/65.2%, above the frozen ≤60% limit). Its final mean NRMSE was .0509/.2692; generic two-coordinate controls were 2.6e-7/1.9e-7 at 2,267/2,262B. Scalar controls were .3944/.2585 at 1,967/1,968B; hard tying was 1.0588/.5764 and forgot old tasks (mean NRMSE increase 1.209/.658). Mirror itself had zero measured old-task forgetting because its learned basis was frozen and task codes were stored independently. Ten payloads pass bytes/hash/final output replay; all sequential retention stages replay exactly; three tests pass.

**Interpretation:** Hard tying shows the expected continual interference, but frozen shared-basis Mirror codes do not recover all expert functions: seed 38502 lost substantial quality on tasks 2–7. Generic coefficients fit the aligned orbit and cost only 21–29B more than Mirror. Mirror also misses the total-payload gate and the scalar byte margin. This is a registered FAIL, not a useful DualPrompt frontier.

## C — Strongest counter-hypothesis

The teacher expert bank was deliberately generated on the Mirror SO(2) orbit, but learning a shared basis from only the first two tasks is still initialization-sensitive; the sharp world-to-world quality difference points to optimization, not a representation impossibility. A new protocol could test basis initialization/continual basis updates, but it must preserve an independent expert bank and matched generic coefficients.

## U — Unresolved

No pretrained DualPrompt model, image benchmark, task-free evaluation, forgetting benchmark, private expert residual, or production latency was tested. Fresh worlds 38511–38513 stayed sealed after development misses. No natural prompt-capacity claim follows.

## Evidence labels

- **Fact:** actual payload bytes, final task errors, every task-stage retention table, MAC proxies, timing and hashes are recorded in `RESULTS_CORE.csv` and `results/development/`.
- **Interpretation:** task-specific codes prevent hard-tie forgetting, but the frozen Mirror training schedule does not reliably fit new expert tasks and does not beat the generic coefficient representation.
- **Hypothesis:** a better-initialized or incrementally updated basis may preserve both past and new expert functions; that is a new frozen experiment, not a post-hoc repair of these worlds.
