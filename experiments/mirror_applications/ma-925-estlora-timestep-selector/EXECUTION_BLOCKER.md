# MA-925 execution blocker

## Fact

- PA264 is *Subject or Style: Adaptive and Training-Free Mixture of LoRAs* (arXiv 2508.02165). Its abstract says EST-LoRA considers matrix energy, style discrepancy and timestep, and provides code at `anonymous.4open.science/r/EST-LoRA-F318`.
- Direct requests to the cited anonymous endpoint return HTTP 401 `not_connected`; bounded exact-title GitHub searches did not identify a public code mirror.
- The registered control requires the same subject/style LoRA banks and generation comparisons across all steps. None are present locally.
- SDXL generation requires CUDA; runtime is CPU-only.

## Decision

Keep UNTESTED. A synthetic selector or random LoRA tensor test would not establish fidelity, NFE, or the registered comparison. Do not substitute a different LoRA bank or generation model.

## Resume condition

Obtain the cited source or verified official release, identical LoRA banks, held-out subject/style references and GPU SDXL runtime; preregister quality, per-step selector overhead, bytes and NFE.
