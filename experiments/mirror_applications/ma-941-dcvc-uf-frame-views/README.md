# MA-941 — DCVC-UF frame-slot Views

## H — hypothesis

A small relative-frame coordinate can replace some DCVC-UF frame-specific decoder state while preserving the actual chunk-coded rate, reconstruction quality and parallel decode throughput. The native chunk latent and entropy coder are fixed.

## T — work completed

Read PA269 and its CVPR 2026 paper/supplement. The paper confirms a single chunk latent, parallel frame reconstruction and frame-specific decoders. Repository inventory and public GitHub repository searches found no official implementation/checkpoint/bitstream. The CVF paper is accessible, but its page exposes only paper and supplement links. `torch.cuda.is_available()` is false. No experiment run.

## D — UNTESTED

Without the official codec and its entropy-coded native baseline, the registered coded-bit/PSNR/FPS comparison cannot be reproduced.

## C — strongest counter-hypothesis

A CPU reimplementation might demonstrate a generic slot factor, but it cannot validate native entropy-code cost or the claimed parallel decode path.

## U — unconfirmed

All registered quality, coded-rate, latency, frame-buffer and Mirror-attribution outcomes.
