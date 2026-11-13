# MA-945 — CoANeRV shared coordinate code

## H — hypothesis

Factorized video×time×spatial-region codes may compress CoANeRV’s per-video token state while preserving RD quality, random-query throughput and peak memory versus its native shared coordinate decoder and ordinary factor controls.

## T — work completed

The official CoANeRV repository was found and cloned at commit `3afe1122146819d5854eca2376e5c0faabb72818`. It includes architecture/configuration code and Kinetics/UCF/Something-Something split manifests, but no trained `.pth/.pt/.ckpt` weights. Dataset directories are also absent. The workspace has no CUDA device. No model or metric run was performed.

## D — UNTESTED

The registered quality, token-rate, random-query and memory measurements require a trained checkpoint and actual video dataset.

## C — strongest counter-hypothesis

The official code might be trainable from scratch on Kinetics, but that dataset is absent and the expected training/throughput workload is not available on this CPU-only worker.

## U — unconfirmed

Token-code bytes, frame PSNR/SSIM, random coordinate-query throughput, peak GPU memory and Mirror-specific benefit over native token factorization.
