# MA-846 result: NOT ESTABLISHED

## H / T / D / C / U

- H: a shared Hebbian update basis plus task codes retains heldout adaptation at lower total storage and update cost.
- T: synthetic D=12, eight-task aligned rank-2 task coefficient bank; 24 support and 128 query vectors per task; development seed 11, frozen fresh seeds 29 and 47. Compared common rule, independent coefficients/full tensor, native low-rank bank, Mirror bank. No optimizer training; 192 support examples/world.
- D: NOT ESTABLISHED. All task-specific methods have heldout nRMSE 1.0047 (dev), 1.3169 (seed29), 1.2519 (seed47), far outside the preregistered .02 gate. Native low-rank and Mirror have exactly identical outputs/bytes. Fresh opened only after the screen was frozen; no tuning followed.
- C: implementation mismatch: the rule writes `outer(y,x) * plasticity_tensor`, but the target matrix was defined as the plasticity tensor. The resulting expected fast map is not the target. Therefore poor quality cannot adjudicate the registered hypothesis.
- U: any claim about differentiable-plasticity learning, useful online adaptation, interference, latency, or Mirror-specific value.

## Facts

- Mirror/native low-rank payload: 1,777 B persistent + 4,608 B writable fast state = 6,385 B total.
- Independent coefficient/full tensor: 4,940 B persistent + 4,608 B writable = 9,548 B total.
- Common rule: 908 B persistent + 4,608 B writable = 5,516 B, but does not approach target maps.
- Each method uses 331,776 Hebbian update MAC proxy per world.
- Mirror and native low-rank rows have equal metric values and payload bytes; no Mirror attribution.

## Interpretation

The synthetic code representation compresses the coefficient table by 33.1% versus independent coefficients, but because the update-rule/teacher pairing is invalid, this is only a storage arithmetic observation. It is not evidence of a useful plasticity mechanism or a quality-preserving Pareto gain. Status is NOT ESTABLISHED rather than FAIL for the registered scientific hypothesis.

## Hypothesis for a new protocol

Generate targets from an explicitly realizable plastic learner (e.g. recurrent delta-rule with task-specific Hebbian rates), and verify one-task convergence before comparing coefficient coding. Any rerun after fresh access needs a new MA/amendment and new fresh seeds.
