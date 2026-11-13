# MA-950 — video Mirror marginal-byte tournament

## H — hypothesis

Across matched video contents and coded-rate points, Mirror methods should improve marginal quality per actual byte without sacrificing decode throughput or peak VRAM relative to native codec controls.

## T — comparability audit

The only current video result is MA-939’s single HNeRV Bunny frame-embedding screen. It measures actual serialized model/embedding payload, PSNR on the last 36 frames and CPU FPS; it is not entropy-coded bpp. MA-936/938 NerVast and MA-941..944 DCVC native codecs are UNTESTED because their official artifacts/runtime are unavailable. These outcomes cannot form the required identical-content, equal-bitrate tournament. See `source/COMPARABILITY_AUDIT.csv`.

## D — UNTESTED

No tournament metric or cross-codec frontier was computed.

## C — strongest counter-hypothesis

A descriptive table of existing heterogeneous results could be useful as an inventory, but it would not establish a Pareto comparison across codecs or contents.

## U — unconfirmed

Matched bitstream BD-rate, MS-SSIM, decode FPS, peak VRAM and cross-family Mirror marginal bytes.
