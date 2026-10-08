# MA-1141 — Quadruped embodiment × locomotion skill Views

Status: NOT ESTABLISHED (development task-controller blocker)  
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Prior art: PA401 (RMA), PA405 (morphology-conditioned world model)

**Mirror insertion:** this experiment adds factorized robot-embodiment and locomotion-skill coordinates `m` to a shared ant policy, so unseen embodiment × speed-task combinations can be expressed without storing a separate policy for every pair.

## H — falsifiable hypothesis

A shared policy with small per-embodiment and per-skill Givens views will retain locomotion quality on held-out embodiment × speed pairs within 5% of a native condition-input policy payload, and will not be dominated by a same-code-budget FiLM control.

## T — frozen screen

Use Gymnasium's MuJoCo Ant model with four declared torso/limb/actuator morphologies and four forward-speed skills. Eight even-parity morphology-skill pairs provide expert-controller demonstrations; the eight odd-parity pairs are held out entirely for audit. The expert is a deterministic phase-based torque controller. Train behavior-cloning policies on fixed train trajectories and choose checkpoints from held-in development trajectories only. Evaluate each policy in MuJoCo on the held-out pairs, recording return, target-speed error, survival/fall fraction, actual safetensors payload bytes, training examples/updates, adaptation-free inference cost, and wall-clock.

Methods: native morphology/skill condition-input MLP; no-code shared policy; factorized learned-code FiLM; factorized per-factor Givens Mirror; independent upper reference is omitted because no training data exists for the held-out pairs.

The randomized MA selection and exact replay pool are in `source/draw27_exclusions.json`. The full protocol is `PROTOCOL.json`.

## D — decision

**NOT ESTABLISHED.** The frozen phase-controller teacher is not a useful locomotion target: in the one-world/one-init development run it averaged 0.639 fall fraction over the eight seen pairs. A development-only 25-point amplitude/frequency sweep had a best return of 0.3488 with zero falls, below the zero-action reference return of 0.3570. The held-out eight pairs were not evaluated. Do not interpret the short behavior-cloning rollouts as Mirror quality evidence. The implementation and protocol are retained; a valid locomotion teacher or native RL control is needed before this ID can decide its hypothesis.

## C — strongest counter-hypothesis

Ordinary morphology-conditioned policies or FiLM already provide compositional generalization. Givens coordinates add inference operations without useful held-out-pair quality gains.

## U — boundaries

Four synthetic morphology settings and four target speeds do not represent the space of quadruped embodiments or locomotion skills. The phase controller is an oracle teacher for this screen, not an RMA/PPO reproduction. No held-out audit, real hardware, safety, energy, or cross-simulator claim is made.
