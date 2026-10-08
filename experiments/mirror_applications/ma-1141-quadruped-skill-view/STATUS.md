# MA-1141 status

Status: NOT ESTABLISHED — development task-controller blocker; no held-out audit was run.

## H

Factorized embodiment and skill Givens views preserve quality on held-out embodiment × speed combinations within 5% of native-policy payload and are not dominated by matched FiLM.

## T

MuJoCo Ant-v5 on CPU. One development world (23), one model initialization (31), eight even-parity seen pairs, 100 behavior-cloning updates. Four methods: native one-hot conditioned policy, no-code shared policy, factorized FiLM, factorized Givens Mirror. Teacher diagnostics on those same seen pairs only. No odd-parity held-out pair was evaluated.

## D

**NOT ESTABLISHED.** The original phase-controller teacher had 0.639 mean fall fraction over development pairs. A 25-setting amplitude/frequency sweep on those same pairs achieved a best normalized return of 0.3488 at zero falls, worse than the zero-action reference at 0.3570. Since the teacher did not define useful locomotion behavior, no audit run was made. This is a task/controller validity blocker, not a Mirror FAIL.

## C

Native morphology/skill conditioning and FiLM may already provide compositional generalization at similar state cost; Mirror rotations may add inference cost without increasing useful task quality.

## U

The development metrics do not establish Mirror quality on held-out pairs. A valid controller/RL teacher, more natural task set, and audit replay remain required. This synthetic screen cannot establish real-robot transfer or safety.

## Fact / interpretation / hypothesis

- **Fact:** the frozen teacher fell in 63.9% of seen-pair development episodes. In the recorded development sweep, best return was 0.3488 with no falls; zero action returned 0.3570 with no falls. Four model methods were trained for 100 updates and only seen-pair rollouts were run.
- **Interpretation:** teacher stability and useful target behavior were not sufficient to test cross-pair generalization.
- **Hypothesis:** a learned native controller or a validated locomotion reference may make the factorized embodiment × skill question measurable, but it requires a new amendment before any held-out evaluation.
