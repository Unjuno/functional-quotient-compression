# MA-939 results

## H — hypothesis

A shared basis plus per-frame coordinate `m` can reduce HNeRV content-adaptive embedding bytes while preserving held-out PSNR within 0.25 dB and at least 90% of native CPU decode FPS. A Mirror-specific benefit must beat the ordinary PCA/low-rank coordinate control.

## T — trial

Pinned official HNeRV commit `4872129c8d004a25477e0c1ffbbff4ba71943ad5`, bundled Bunny video, native quantized frame embedding shape 132×16×2×4 (128 values/frame). Basis fitted only on frames 0:96; held-out audit frames 96:132. Four frozen ranks (2,4,8,16), uint8 affine quantization for basis/codes. CPU TorchScript decoder used at 640×1280 after the official center crop. No optimizer updates; SVD and coordinate projection only. Native checkpoint, shared decoder/model state, basis, all 132 codes, scales and metadata are charged by actual serialized payload bytes.

## D — FAIL (scoped)

### Fact

| Representation | Rank | Embedding/code bytes | Full payload bytes | Held-out mean PSNR | Delta vs native | Observed CPU decode FPS |
|---|---:|---:|---:|---:|---:|---:|
| Native quantized embedding | 128 | 19,762 | 7,576,370 | 35.04 dB | 0.00 dB | 0.282 |
| Mirror coordinate / PCA alias | 2 | 4,319 | 7,560,927 | 19.60 dB | -15.44 dB | 0.485 |
| Mirror coordinate / PCA alias | 4 | 4,831 | 7,561,439 | 20.23 dB | -14.81 dB | 0.313 |
| Mirror coordinate / PCA alias | 8 | 5,855 | 7,562,463 | 20.80 dB | -14.23 dB | 0.323 |
| Mirror coordinate / PCA alias | 16 | 7,980 | 7,564,588 | 21.95 dB | -13.09 dB | 0.337 |

Native audit minimum PSNR was 33.12 dB. Coordinate payloads save 59.6–78.1% of native embedding bytes, but full payload saves only 0.16–0.20% because common decoder/model state is 7,556,608 B. The best rank16 PSNR remains 13.09 dB below native. All four Mirror coordinate rows exactly alias PCA in function and serialized bytes (max output difference 0.0).

CPU decode FPS is a sequential one-process measurement over 36 frames, including code synthesis for rank modes. The observed variation is not a stable throughput win; the common decoder dominates. No entropy-coded video bitrate was measured.

### Interpretation

This single-video low-rank coordinate compresses the embedding table, but the PSNR loss is far beyond the frozen 0.25 dB tolerance and the total model payload barely shrinks. It is ordinary PCA/low-rank representation coding, not Mirror-specific value. The experiment demonstrates a poor storage-quality frontier at the tested ranks for this HNeRV checkpoint.

### Hypothesis

The native 128-D content-adaptive embedding carries frame-specific information that a rank-16 subspace learned from the first 96 frames does not preserve for later frames. Retaining private residual dimensions or training a richer factorized encoder could improve quality but would add paid bytes and is untested.

## C — strongest counter-hypothesis

This is one Bunny sequence with the final 36 frames as audit; scene distribution and motion may be unusually varied or shifted. The fixed linear PCA-style code is not a learned nonlinear frame encoder. Results do not generalize to HNeRV datasets or video compression broadly.

## U — unconfirmed

No second scene/video, NerVast native masks, learned code encoder, private residual, entropy-coded bitstream, GPU path or broader perceptual metric was tested. The reported payload is serialized model inference state, not bpp.
