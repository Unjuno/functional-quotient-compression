# MA-980 — ideal photonic operator simulation frontier

Status: NOT ESTABLISHED. Fresh results are retained, but native-vs-Mirror byte attribution is confounded by serialization schema lengths.

## H — hypothesis

Eight phase-related optical linear tasks can share one physical 4×4 Hadamard operator and small per-task phase Views, preserving the exact transformation while storing fewer bytes than eight independent dense complex matrices. The native direct phase-bank control may explain the entire result.

## Mirror insertion

`F(x; H, m_out, m_in) = diag(exp(i*m_out)) H diag(exp(i*m_in)) x`. One Hadamard core is shared across tasks; each role has input and output phase coordinates. A native MZI-style phase bank stores the same phase values as the cheapest control.

## Prior-art delta

PA286 and PA295 establish programmable photonic couplers/MZI meshes. This screen measures ideal simulated operator fidelity, serialized state, and algebraic operation counts. It does not measure hardware behavior.

## Controls

1. Shared H with no task coordinates.
2. Native direct input/output phase bank.
3. Mirror phase View using the same angle table.
4. Independent dense complex operators.

## T / execution

Three fresh deterministic task worlds (98011/13/17), eight tasks per world. Development world 98001 is reserved for pipeline checks; no hyperparameter selection is needed. Actual payloads use `torch.save` and include all task state and metadata. Fresh results compare reconstructed operators and fixed complex probe outputs.

## D — provisional boundary

Only simulated fidelity/storage/operation counts are admissible. Any ideal phase-factor storage benefit that also appears in the native direct phase bank is an ordinary structured optical parameterization, not Mirror-specific evidence. Physical energy, switch time, loss, footprint, and device-level performance remain unmeasured.
