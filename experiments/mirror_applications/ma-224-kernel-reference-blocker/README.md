# MA-224 — convolution-kernel Mirror reference preflight

Status: UNTESTED (reference mismatch blocker)  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Random draw: #16, pool size 1041, zero-based index 186, selected MA-224.

## Registered hypothesis

The registry describes sharing a state-space/kernel/execution object and adding a state/kernel/execution coordinate to expose logical state machines or runtime views.

> **Mirror insertion:** this experiment would add `m` to a state/kernel/execution interface so that logical state-machine or runtime views can be expressed without duplicating the physical state or kernel object.

## Reference mismatch

FACT: The MA-224 registry row cites PA11. PA11 is *Practical Lessons on Vector-Symbolic Architectures in Deep Learning-Inspired Environments*, covering MAP, HRR and Hadamard binding. The registered MA-224 object is SSM/convolution/kernel execution. PA11 does not supply the native SSM/kernel method or direct controls needed for this candidate.

INTERPRETATION: Implementing a convolution or state-space study now would require selecting the task and native baseline by guesswork. A VSA binding experiment would instead change the registered candidate.

HYPOTHESIS: No scientific hypothesis was evaluated.

## Execution

No code, dataset, development split or fresh/audit data was opened. No result or status change is claimed. The registry remains UNTESTED. Resume only after the registry/prior-art mapping is corrected in a future worker-ready baseline.

Draw pool ID-list SHA-256: `6201dcdabf93c6bc99aa962dccd44307064a8352d6046b230819ef9bb1acd808`. Selection-pool CSV SHA-256: `9377d6da9fea73269a087f7f8b61fe167facfa8a347884d2af9278a904dd1105`.
