# Mirror application experiments

This lane tests where the extra Mirror/View coordinate can replace physical duplication with logical multiplicity.

Files:
- IDEA_REGISTRY.csv — 400 candidate applications with status and first control.
- FIRST_QUEUE.md — 25 P0 candidates spanning different physical objects.
- EXPERIMENT_CONTRACT.md — common byte/compute/quality rules.

New experiments should use stable IDs from IDEA_REGISTRY.csv and live in a subdirectory such as:

    experiments/mirror_applications/ma003_mirror_topk_moe/

Do not silently reuse an ID for a different hypothesis. Update status only after the corresponding report is committed.

The design rationale is documented in ../../docs/phase2/MIRROR_APPLICATION_DESIGN_SPACE.md.


## Worker navigation

Before starting an MA experiment, read:
- [worker start guide](../../WORKER_START_HERE.md)
- [prior-art map](../../docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md)
- [research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-07.md)
- [worker queue](WORKER_QUEUE.md)
- [experiment template](TEMPLATE/)

The registry is authoritative for IDs/status. The queue is operational guidance only.
