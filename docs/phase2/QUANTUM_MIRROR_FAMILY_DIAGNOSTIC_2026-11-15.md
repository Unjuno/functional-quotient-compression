# Quantum Mirror family diagnostic — 2026-11-15

Status: **PAUSED pending a Mirror-distinct mechanism**. MA-1016 and MA-1017 are consecutive scoped synthetic FAILs for incremental Mirror attribution. MA-1018..1023 remain UNTESTED; this diagnostic is not a result for those candidates.

## Evidence

### MA-1016 — TensorHyper-VQC task code

- Four aligned TT-generated task functions matched the teacher exactly; all controls had 95.31% accuracy at 1,024 shots.
- Mirror task phase used 567 B, versus 587 B for the rank-2 TT task core (3.4% fewer, below the frozen 10% threshold).
- Ordinary native scalar-phase conditioning produced byte-identical payloads and outputs. An independent angle bank was 48 B smaller than Mirror in the declared serializer.
- Four qubits, 26 logical gates, six two-qubit gates, depth 11; 65,536 shots per method. Ideal simulator only; no hardware compilation or QPU.

### MA-1017 — data re-upload schedule

- Four distinct task label functions were recovered at 98.44% shot accuracy from a shared two-qubit, three-block ansatz.
- Mirror phase reduced payload from 523 B independent schedules to 379 B (27.5%), but native scalar input-phase conditioning had identical bytes, predictions, gates and shots.
- Shared no-view accuracy was 51.56%, showing task conditioning is useful in this deliberately aligned family; it does not show Mirror-specific value.
- Each circuit used 21 logical gates, three two-qubit gates, scheduled depth 12 and 65,536 shots per method. Ideal simulator only; no hardware compilation or QPU.

## Fact / Interpretation / Hypothesis

**FACT:** In both tested families, the proposed Mirror phase maps exactly to a native scalar phase coordinate. MA-1016 also misses its byte threshold against native TT; MA-1017 compresses the independent bank but ordinary scalar conditioning recovers the entire gain.

**INTERPRETATION:** The shared quantum ansatz can express multiple functions with a small task coordinate on aligned teachers. The measured function and storage effects are fully explained by ordinary task-conditioned gate/input angles. These screens do not support a quantum Mirror-specific claim.

**HYPOTHESIS:** A quantum candidate should resume only after it identifies a noncommutative or hardware-constrained insertion where the Mirror coordinate is not reducible to native angle embeddings, TT codes, data-upload offsets, or ordinary compiler/calibration parameters, and freezes a matched native control before implementation.

## Family disposition

Pause MA-1018..1023 as **UNTESTED**, not FAIL. Resume criteria:

1. State the algebraic/function-space difference from a native scalar/vector angle code.
2. Predefine a byte-near native TT/angle control and independent circuit upper bound.
3. Use a natural or held-out gate/task composition rather than only a teacher-generated address family.
4. When making device claims, charge compilation, calibration, two-qubit gates, noise and shots. Simulator-only evidence remains scoped to simulator mechanisms.

Next eligible P0 outside this paused quantum family: MA-1024 (CoOp prompt-bank code).
