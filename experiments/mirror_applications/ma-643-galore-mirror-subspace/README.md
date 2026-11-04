# MA-643 — GaLore Mirror subspace codes

Status: SCREENING. This test asks whether reusable Mirror coordinates reduce the projector portion of multi-task GaLore resume state.

## H

For 16 tasks whose gradients use only K=1/2/4/8 recurring rank-4 left subspaces, one shared projector dictionary plus a task code should reduce serialized optimizer checkpoint size versus one projector per task without losing held-out fit across the task bank.

## Comparisons

Full Adam stores full matrix moments. Standard GaLore stores per-task rank-4 projectors and low-rank moments. Mirror GaLore stores a projector dictionary and per-task selected codes. A native dictionary-ID method uses the same projectors, codes and decoder as Mirror, to test whether any gain is simply ordinary projector sharing.

## Storage and compute

Actual `.npz` inference and resumable optimizer payloads are both measured. All matrices, moments, projector state, codes, task IDs and metadata are charged. Dense gradient, projection and discovery costs are reported separately. The synthetic dataset is deterministic from each world seed and excluded from optimizer resume bytes.

See `PROTOCOL.json` for the frozen conditions and gates. Fresh seeds 64311–64313 remain sealed unless every development gate passes.
