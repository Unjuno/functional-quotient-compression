# MA-926 — add new subject with a Mirror code on a shared LoRA bank

Status: UNTESTED
Evidence lane: DIFFUSION_PERSONALIZATION / QUALITY / STORAGE / RETENTION

## H — falsifiable hypothesis

A new subject can be added with a small Mirror code over a frozen shared LoRA bank, reducing actual bytes per subject while preserving subject fidelity and prior-subject retention versus independent LoRA and LoRA.rar controls.

## T — source and asset review

Reviewed PA263–PA264 and MA-924 assets. The LoRA.rar repository has hypernetwork weights only; no subject/style LoRA bank or evaluation images. Workspace has no relevant local LoRA/image assets. A bounded HF check returned an unrelated community LoRA collection without matched references or required controls. Runtime is CPU-only and cannot run SDXL inference.

## D — UNTESTED

No new subject adaptation, generation, fidelity, retention, or per-subject byte measurement was run.

## C — strongest counter-hypothesis

A small code cannot recover fine-grained subject identity; independent LoRAs may be necessary, while storing subject exemplars or embeddings erases byte savings.

## U — unresolved

Frozen base and shared style bank, consenting held-out subjects/reference images, GPU SDXL inference, and matched independent-LoRA/LoRA.rar controls. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
