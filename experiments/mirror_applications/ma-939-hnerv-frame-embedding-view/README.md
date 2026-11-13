# MA-939 — HNeRV frame embedding views

Status: **FAIL (scoped Bunny video)**. The official HNeRV embedding has 128 values/frame across 132 frames. Rank-2/4/8/16 coordinate payloads reduced embedding bytes by 59.6–78.1%, but held-out PSNR fell 13.09–15.44 dB. Because decoder/model state dominates, full-deployment savings were only 0.16–0.20%. The frame-coordinate implementation exactly aliases ordinary PCA/low-rank coding.

See [RESULTS.md](RESULTS.md), frozen [PROTOCOL.json](PROTOCOL.json), and [VERIFICATION.json](VERIFICATION.json). This is a one-video model-payload result, not an entropy-coded video rate or multi-scene claim.
