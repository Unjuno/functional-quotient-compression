# MA-449 status

**FAIL — factors are identifiable and the held-out combination works, but Mirror has no byte advantage over ordinary task vectors or LEO.**

## H / T

Tested support-inferred two-factor Mirror coordinates on a held-out joint concept combination. Each world supplied separate paid single-factor intervention samples to identify the shared decoder; held-out tasks used independent support/query samples.

## D — Fact

Across 3 fresh worlds × 3 seeds × 20 held-out instances, mean NRMSE: Mirror 2.91e-07, direct task-vector 2.94e-07, LEO 0.352, full-space fit 2.48e-06. Intervention recovery was near identity for both factors. Actual serialized N=20 payload bytes/task: Mirror 107.45B, task-vector 107.45B, LEO 107.45B. Mirror saves no bytes versus either code-based control.

## Interpretation

The aligned linear concept factors are identifiable and compose on the held-out pair. The quality comes from the shared decoder and direct factor coordinates; Mirror provides no unique compression over ordinary task-vector or LEO coordinates in this screen.

## C — Strongest counter-hypothesis

The task is exactly additive and the support decoder recovers both factors, so task-vector arithmetic already solves the held-out combination without needing a Mirror-specific representation.

## U — Unknown

Natural concepts, noisy/interacting factors, neural backbones, and generalization beyond one held-out joint combination remain untested. A1 narrows this to a mechanism screen, not full MAML/LEO reproduction.
