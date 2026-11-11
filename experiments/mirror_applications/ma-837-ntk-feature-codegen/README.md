# MA-837 — Input-conditioned NTK feature generator with Mirror task coordinate

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / NTK FEATURE GENERATOR  
Base commit: `513c3af7`  
Prior art: PA225, NTK-CL.

## Hypothesis

H: A small Mirror coordinate generated from task descriptors produces adaptive features without task-specific parameters, preserving seen and unseen descriptor quality while reducing complete bytes or compute versus the native no-task-storage NTK-CL baseline.

## Controls and insertion

The candidate computes `m = descriptor` and decodes a task readout from a shared basis. Controls are the native no-task-storage generator, ordinary direct descriptor conditioning, a cached code table, independent readouts and a shared mean. Every stored descriptor/code is charged.

## Boundaries

This is a synthetic linear task family, not trained NTK-CL or PEFT evidence. Descriptors are request inputs; we do not test generating codes from raw support examples. Unseen descriptors test interpolation only. The protocol freezes all gates and accounting before execution. Result: native no-task-storage and Mirror-generated coordinate are identical at 630 B, with the same 4.30e-15 / 3.83e-15 seen/unseen error and 36,864 MAC proxy. Cached codes cost 936 B plus 576 decode MACs. Fresh remained sealed.
