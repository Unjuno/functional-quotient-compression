# Low-rank recurrent task-View lane diagnostic (2026-11-11)

## Decision

Pause MA-798 and MA-799 pending a Mirror-specific mechanism that is functionally distinct from ordinary conditional low-rank recurrent parameterization. MA-796 and MA-797 are consecutive scoped FAILs for Mirror attribution with the same structural cause: exact native low-rank/control alias. Continue the queue at the unrelated compositional-weight-atoms family, MA-800.

## Evidence

| Experiment | Scoped fact | Attribution result |
|---|---|---|
| MA-796 | Synthetic stable 8D linear RNN; two heldout task compositions reached rollout nRMSE 3.21e-8 at 2,431 B, 28.8% below the independent matrix bank; either component ablation raised mean task nRMSE to about .062; max spectral radius .557. | Native additive low-rank recurrent conditioning had identical payload/hash and metrics. |
| MA-797 | Synthetic 8D three-stage system; heldout +1 task/locus combinations reached output nRMSE 4.50e-8 at 2,018 B, 30.1% below independent. Correct-locus code reconstructed output; wrong-locus intervention nRMSE was .0281. | Native low-rank locus conditioning had identical payload/hash and metrics; rank-1 update added active MACs. |

Both results use synthetic linear functions, supplied task coordinates and zero optimizer updates. Neither establishes trained RNN performance or broad task composition. MA-796/797 fresh splits remained sealed after exact control alias.

## Family interpretation

The evidence supports the narrower mechanism that known task coordinates can compose shared low-rank dynamics and that locus placement can matter in designed systems. It does not support Mirror-specific value: ordinary native low-rank factors and task/locus IDs express the same function family. Repeating flat or locus-factored low-rank codes would not distinguish the cause.

Resume MA-798/799 only after a preregistered design adds a functional degree of freedom absent from native conditional low-rank controls (for example, a state-dependent recurrent intervention with stability and runtime measurements), and after PA212 is compared against the resulting native alternatives. This pause does not cover the separate Koopman operator or compositional weight-atom families.
