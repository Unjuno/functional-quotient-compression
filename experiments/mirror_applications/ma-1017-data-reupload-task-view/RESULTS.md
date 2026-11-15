# MA-1017 results

## H — Hypothesis

A task-specific phase on repeated data uploads can express multiple logical classifiers through one physical ansatz with fewer actual bytes than a bank of independent VQCs, and may add functional value beyond ordinary scalar schedule conditioning.

## T — Test

- Two-qubit ideal statevector, three data-reupload blocks; each block applies two input rotations, two trainable rotations per qubit and one CNOT.
- Four fixed phases `[0.1, 0.85, 1.75, 2.9]`; 16-point grid of inputs in `[-1,1]^2`; 1,024 binomial measurement shots per input.
- Compared one shared no-view ansatz, independent full task angle schedules, Mirror task phase and ordinary scalar schedule phase.
- A development-only seed adjustment selected four distinct task sign-label functions; initial three-function run excluded under protocol amendment 01. Fresh seeds 1017301, 1017307 and 1017313 remain sealed.
- Actual inference payload uses the same JSON-header plus contiguous float32 format for every row. Each method uses 21 logical gates per circuit, including 3 two-qubit gates, at scheduled depth 12; no hardware compiler or QPU was used.

## Results

| Method | Teacher agreement | 1,024-shot accuracy | Actual bytes | Qubits | Gates / two-qubit gates / depth | Shots per method |
|---|---:|---:|---:|---:|---:|---:|
| Shared ansatz, no view | 53.13% | 51.56% | 308 B | 2 | 21 / 3 / 12 | 65,536 |
| Independent task schedules | 100% | 98.44% | 523 B | 2 | 21 / 3 / 12 | 65,536 |
| Mirror task phases | 100% | 98.44% | 379 B | 2 | 21 / 3 / 12 | 65,536 |
| Native scalar schedule control | 100% | 98.44% | 379 B | 2 | 21 / 3 / 12 | 65,536 |

All four tasks have distinct label functions, with pairwise disagreement from 18.75% to 93.75%. Mirror saves 144 B (27.5%) versus the independent bank and recovers the teacher functions. But the native scalar schedule has byte-identical payload and identical predictions. The simulator run took 0.125 s.

## D — Decision: FAIL for Mirror-specific value

**FACT:** Mirror and the native scalar schedule both achieve 100% teacher agreement and 98.44% shot accuracy with 379 B. They use the same 21-gate, 3-entangler, depth-12 logical circuit and 65,536 shots. Both are 27.5% smaller than the independent task bank. The no-view control scores 51.56% shot accuracy. Tests and deterministic replay pass; fresh remains sealed.

**INTERPRETATION:** Re-upload phase conditioning allows shared circuit weights to express multiple functions with fewer bytes than independent schedules in this aligned fixture. The effect is exactly reproduced by ordinary scalar input-angle conditioning, so the Mirror-specific gate fails.

**HYPOTHESIS:** The task code adds a useful native conditional degree of freedom in this constructed task family; the evidence does not show that a Mirror coordinate improves on native scalar conditioning.

**C — Strongest counter-hypothesis:** The teacher is generated from the same phase schedule, so this is aligned feasibility only. Natural task circuits may require task-specific variational gates or more private parameters.

**U — Unconfirmed:** Learned task adaptation, natural task generalization, noise robustness, hardware compilation, QPU execution and fresh seeds.
