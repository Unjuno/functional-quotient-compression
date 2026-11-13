# MA-925 — EST-LoRA timestep Mirror selector

Status: UNTESTED
Evidence lane: DIFFUSION_PERSONALIZATION / QUALITY / NFE / STORAGE

## H — falsifiable hypothesis

A compact structured timestep × layer × subject/style Mirror selector can improve fidelity per byte or inference overhead over EST-LoRA's training-free energy/style-discrepancy heuristic when selecting between the same subject and style LoRA banks at every denoising step.

## T — source and asset review

Reviewed PA264, arXiv 2508.02165. Its cited anonymous source endpoint returns HTTP 401 `not_connected`; bounded exact-title GitHub search found no public mirror. No matched subject/style LoRA bank or image evaluation set is available locally. Stable Diffusion inference needs CUDA; runtime is CPU-only.

## D — UNTESTED

No selector, image generation, fidelity, NFE, storage-byte or runtime comparison was run.

## C — strongest counter-hypothesis

EST-LoRA's training-free energy/style discrepancy and timestep heuristic may already select the right LoRA at each layer; extra Mirror control may add overhead without better balance.

## U — unresolved

Accessible official code, identical subject/style LoRA banks, CUDA-capable SDXL inference, held-out combinations and a frozen quality/latency protocol. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
