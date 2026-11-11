# MA-871 — module reuse structure as a task coordinate

This PA234 mechanism screen tests whether tasks built by recombining a small fixed bank of modules can be adapted from fewer examples using a learned low-dimensional recombination coordinate.

**Mirror insertion:** this experiment adds a two-scalar task coordinate `m` at the module-output recombination interface so that new task functions can be selected without storing a full coefficient vector over all modules.

Prior art: PA234, *Structural Composition Enables Very Fast Learning*. Required controls include support-fit full module coefficients, a nearest training-task composition codebook, a random two-dimensional basis, and the native PCA/tensor-factorization equivalent. Mirror-specific attribution requires outperforming the native rank-2 basis at comparable bytes and compute.

The hypothesis is falsified if the learned structure fails to reduce new-task support requirements or serialized inference bytes while preserving heldout function quality, or if native rank-2 PCA reproduces Mirror exactly. See PROTOCOL.json for frozen data generation, seeds, gates, bytes and measurements.
