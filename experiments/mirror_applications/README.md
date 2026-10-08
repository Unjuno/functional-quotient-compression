# Mirror application experiments

This lane tests where the extra low-description Mirror/View functional parameter `m` can be inserted into existing methods to replace physical duplication or add useful logical functional freedom at worthwhile marginal cost.

Files:
- IDEA_REGISTRY.csv — 1045 candidate applications with status, prior-art links and first control.
- FIRST_QUEUE.md — 25 P0 candidates spanning different physical objects.
- EXPERIMENT_CONTRACT.md — common byte/compute/quality rules.

New experiments should use stable IDs from IDEA_REGISTRY.csv and live in a subdirectory such as:

    experiments/mirror_applications/ma003_mirror_topk_moe/

Do not silently reuse an ID for a different hypothesis. Update status only after the corresponding report is committed.

The design rationale is documented in ../../docs/phase2/MIRROR_APPLICATION_DESIGN_SPACE.md.


## Worker navigation

Before starting an MA experiment, read:
- [worker start guide](../../WORKER_START_HERE.md)
- [Mirror parameter integration doctrine](../../docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md)
- [cross-method Mirror parameter integration matrix](../../docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md)
- [latest verified worker findings and design rules](../../docs/phase2/LATEST_WORKER_FINDINGS.md)
- [minimal-context router](CONTEXT_ROUTER.md)
- [prior-art map](../../docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md)
- [research notes through 2026-10-07](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-07.md)
- [2026-10-08 research notes](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08.md) — cross-model cache, neural graphics, multi-speaker TTS, generative flows and model stitching
- [twelfth sweep: video, symmetry, spiking, photonic, wireless, HRTF](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_TWELFTH_SWEEP.md)
- [thirteenth sweep: materials, MRI, quantum circuits, visual memory and ANN](../../docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-08_THIRTEENTH_SWEEP.md)
- [run registry integrity check](check_registry_integrity.py) — supports MA-1000 and beyond
- [worker queue](WORKER_QUEUE.md)
- [experiment template](TEMPLATE/)

The registry is authoritative for IDs/status. The queue is operational guidance only.

- [Mirror KV cache reuse design](../../docs/phase2/MIRROR_KV_CACHE_REUSE.md) — canonical-cache algebra, prior art, POC and MA-691..700.
