# MA-846 — differentiable-plasticity coefficient bank

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `9ffcb1e6`

## Hypothesis

A shared Hebbian update basis and compact task coefficients can reproduce useful task-specific online learners at lower total storage and update cost than independent per-task plasticity coefficients.

## Mirror insertion

- Native method: shared slow weights plus task-specific plasticity coefficients.
- Insertion: task code `m_t` selects a linear combination of shared Hebbian update matrices.
- `m_t` persists across episodes; fast state is writable per episode.
- Controls include common-rule, independent coefficient, ordinary low-rank and independent full-matrix banks.
- Cheapest ordinary explanation: native low-rank factorization of the task-by-rule coefficient table.

## Physical-to-logical claim

- Shared physical object: Hebbian update basis and slow readout.
- Coordinate: task-specific coefficient vector.
- Logical multiplicity: synthetic online learners.
- Failure mode: arbitrary task rules need private coefficient or full update state, erasing savings.

## Prior art

PA228 (Miconi et al., Differentiable Plasticity) meta-learns slow weights and plasticity coefficients to permit Hebbian online changes. This screen tests only the storage/update-state implication on synthetic rules.

## Protocol

One development world (seed 11), then frozen fresh worlds (29, 47). Shared slow model receives support examples; task rule adapts through Hebbian fast-state writes; evaluation is on heldout query examples. Report persistent bytes, writable fast-state bytes, combined bytes, update MACs, inference MAC proxy and wall time. Actual serialized arrays plus a JSON header are counted. Fresh results remain sealed until implementation and gates are fixed.

## H / T / D / C / U

- H: compact shared rule coordinates retain heldout online adaptation with material total-state savings.
- T: synthetic task-coded linear learners, development seed 11 and fresh seeds 29/47; common, independent, native low-rank, Mirror and full-matrix controls.
- D: pending.
- C: ordinary low-rank factorization explains any savings without a Mirror-specific effect.
- U: trained differentiable-plasticity networks, robustness to sequential interference, and real deployment write latency.
