# MA-1016 results

## H — Hypothesis

For a deliberately aligned task family whose VQC angles come from a shared Tensor-Train generator, one scalar Givens phase per task can preserve classification with fewer actual inference bytes than storing a rank-2 task core, while providing functional value beyond an ordinary scalar phase coordinate.

## T — Test

- Four-qubit ideal statevector VQC; two variational layers; input encoded by RY(pi × bit); each layer has RY/RZ on all qubits and a three-CNOT linear entangler.
- Shared TT cores have shapes `(2,2,2)`, `(2,4,2)`, `(2,2)`. Four task codes are unit-circle rank-2 vectors decoded from fixed phases `[0.1, 0.9, 2.0, 3.3]` radians.
- The teacher tasks are aligned to the same TT generator. The development-only seed/scale adjustment selected a family with four distinct binary label functions; the initial collapsed-label trial is excluded and recorded in `PROTOCOL.json` amendment 01.
- Compared a materialized 64-angle independent bank, native rank-2 TT task cores, Mirror phase codes, and the matched native scalar-phase control.
- Evaluated all 16 four-bit inputs for each task. Each expectation was sampled with 1,024 seeded Z shots: 65,536 shots per method. No training data or target optimization; all task labels come from the fixed synthetic teacher.
- Payload uses a common JSON header plus contiguous little-endian float32 arrays, with header and array descriptors included in bytes. No quantum device or compiler was available; report logical gate schedule only, not compiled hardware depth.

## Results

| Method | Teacher agreement | 1,024-shot accuracy | Actual payload | Qubits | Logical gates / 2Q gates / depth | Shots per method |
|---|---:|---:|---:|---:|---:|---:|
| Independent full angle bank | 100% | 95.31% | 519 B | 4 | 26 / 6 / 11 | 65,536 |
| Native TT rank-2 task cores | 100% | 95.31% | 587 B | 4 | 26 / 6 / 11 | 65,536 |
| Mirror task phases | 100% | 95.31% | 567 B | 4 | 26 / 6 / 11 | 65,536 |
| Native scalar-phase control | 100% | 95.31% | 567 B | 4 | 26 / 6 / 11 | 65,536 |

All six task pairs disagree on 50–100% of the 16 binary labels; there are four distinct task functions. Mirror saves 20 B (3.41%) versus the rank-2 TT task-core payload, below the 10% gate. The ordinary scalar-phase control serializes to byte-identical payload and produces identical expectations. The independent table is also 48 B smaller than Mirror in this tiny serializer, despite holding 64 rather than 32 float values, because its header contains fewer tensor descriptors. Simulator wall time for all four model variants was 0.163 s.

## D — Decision: FAIL

**FACT:** Exact statevector teacher agreement is 100% for every variant. With 1,024 shots per pattern, all variants have the same 95.31% accuracy. Mirror and the native scalar-phase control have identical 567-byte payloads and function values. Mirror saves 3.41% versus native TT, below the frozen gate, and is 48 B larger than the independent table under the declared actual serializer. Tests and exact replay pass; fresh remains sealed.

**INTERPRETATION:** A scalar phase can encode the aligned TT task-core orbit, but this screen finds no Mirror-specific benefit beyond ordinary polar coordinates. At this four-task scale, the native TT payload is larger than the independent angle table once metadata is charged. Quantum gate count, depth and shots are unchanged because all variants decode to the same circuit skeleton.

**HYPOTHESIS:** MA-1016 is a scoped synthetic FAIL for incremental Mirror value over TensorHyper-style TT plus a native scalar phase code. It does not falsify all task-conditioned VQC compression.

**C — Strongest counter-hypothesis:** The family is intentionally low-rank and aligned, and four tasks are too few to expose TT sharing at larger scales; an optimized/noise-trained TT model on natural tasks could differ.

**U — Unconfirmed:** Official TensorHyper-VQC reproduction, trained task generalization, noisy simulation, transpiled circuits, hardware gate calibration, QPU execution and fresh seeds.
