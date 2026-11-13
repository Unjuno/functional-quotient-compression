# MA-924 — LoRA.rar subject × style Mirror merge code

Status: UNTESTED
Evidence lane: PERSONALIZATION / QUALITY / STORAGE / MERGE_RUNTIME

## H — falsifiable hypothesis

Explicit factorized subject and style Mirror parameters can recover useful held-out personalization combinations with better fidelity per deployed byte and merge cost than native LoRA.rar hypernetwork composition and independent merged LoRA pairs.

## T — source and asset review

Inspected the official SamsungLabs repository at `54ae5578702ad9856cfb919ae2a4a2f487cf0cc6`. It includes two trained hypernetwork checkpoints (~1.9 MB each) but no subject/style LoRA weights or evaluation images. README requires downloading images, training subject/style LoRAs and running SDXL inference; its inference path hardcodes CUDA. Current runtime is CPU-only.

## D — UNTESTED

No subject/style merge, held-out image generation, fidelity metric, actual deployment bytes or latency comparison was run. The small hypernetwork checkpoint alone is not the registered experiment.

## C — strongest counter-hypothesis

Native LoRA.rar already composes unseen subject/style pairs efficiently; separate factorized Mirror codes may not preserve fine-grained subject identity or style fidelity and may duplicate information already in LoRA factors.

## U — unresolved

GPU-capable SDXL runtime, subject/style LoRAs, held-out image/reference sets, and a frozen equal-storage comparison against native LoRA.rar. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
