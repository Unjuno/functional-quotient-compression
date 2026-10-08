# MA-945 — CoANeRV factorized token Mirror screen

Status: FAIL (development gate)  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-945-coanerv-mirror-20261008`  
Base commit: `379a9417cb32c9f96c68c779315f90381151eed1`  
Prior art: PA272, CoANeRV arXiv:2608.13938v1.  
Upstream implementation audited at commit `3afe1122146819d5854eca2376e5c0faabb72818`.

## H — hypothesis

For a small video library sharing one frozen coordinate-conditioned decoder, a product-factor Mirror code over video × temporal token role × spatial region can replace much of each per-video token bank while retaining reconstruction quality and query throughput. The product View must also improve the byte-matched quality frontier over low-rank and additive token-code controls to support a Mirror-specific claim.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m_video` to the video-token interface of a shared coordinate-query decoder so each token can be generated from video, temporal-role, and spatial-region factors without storing a full `N × d` token bank for every video.

- Native method: one shared CoANeRV coordinate-conditioned decoder and one per-video token bank.
- Insertion point: generate token `i=(time role, spatial region)` using `(m_video P_v) ⊙ (E_time P_t) ⊙ (E_region P_r) D`, then pass those tokens to the same decoder attention.
- `m_video` is persistent per-video code; shared time/region factors and decoder are charged once per library.
- Ranks `{2,4,8}` were evaluated on development only. A product code plus rank-2 private token residual checked the shared/private boundary.

## Prior-art delta and controls

PA272 establishes CoANeRV's feed-forward complete-clip token formation and shared coordinate-conditioned decoder. Its Appendix A.5/Table 7 freezes the decoder and optimizes only per-video tokens; this is the direct native control. The paper configuration uses a 384 × 72 video-token bank.

This screen uses the native coordinate-query interface and decoder-only token-fitting control at reduced dimensions on CPU. It does not reproduce the full convolutional tokenizer, six-layer token former, or full paper results. The tested delta is factorizing video-token state, beyond storing the native tokens or using a low-rank/additive code.

## T — experiment

- Xiph/Derf Y4M sequences, first 8 frames, center-cropped and resized to 64 × 64 RGB.
- Shared-decoder training clips: `akiyo`, `coastguard`, `foreman`, `news`.
- Development adaptation clips: `carphone`, `container`; seeds 9451 and 9452.
- Fresh clips locked but not downloaded/opened: `salesman`, `silent`, `grandma`; seeds 94501 and 94502.
- Reduced decoder: 32 tokens (8 temporal roles × 4 spatial regions), token width 24, axis-adaptive coordinate Fourier embedding, temperature-scaled cross-attention (τ=0.4), and shared RGB MLP.
- Shared decoder training: 800 updates. Shared factor fitting: 600 updates. Per-video code fitting: 600 updates. Batch size 256 coordinates; Adam at 0.001; CPU with one thread.
- Every inference artifact is a real `torch.save` payload. The 22 K=2 artifacts are retained in `source/payloads/`; source video hashes and split provenance are in `source/dataset_manifest.json`.

## D — decision: FAIL

The strict development gate required one product-Mirror rank to pass on both dev clips and both seeds. **No rank did.** All ranks passed complete K=2 payload size, marginal video-state size and query-throughput clauses. Rank 8 passed the byte-matched simple-control clause in all four clip/seed conditions. It still missed the native full-token PSNR tolerance on `container` by 1.15–1.35 dB, so the conjunction failed and fresh stayed unopened.

### Fact

Mean held-out metrics across the two clips and two seeds:

| Method | PSNR (dB) | SSIM | K=2 complete bytes | Added video code bytes |
|---|---:|---:|---:|---:|
| Native full token bank | 15.019 | 0.3867 | 27,436 | 4,380 |
| Shared low-rank, rank 4 | 13.198 | 0.3532 | 37,222 | 1,276 |
| Additive factors, rank 4 | 13.018 | 0.3582 | 24,026 | 1,308 |
| Product Mirror, rank 4 | 13.481 | 0.3655 | 24,026 | 1,308 |
| Product Mirror, rank 8 | 14.162 | 0.3748 | 24,602 | 1,308 |
| Product Mirror + rank-2 private residual, rank 4 | 14.549 | 0.3803 | 25,300 | 1,816 |

Rank-8 Mirror reduced the complete two-video payload by **10.3%** and the marginal serialized video code by **70.1%** versus the native token bank. Its measured random-query throughput was about 1.15 million pixels/second, close to the native control at 1.14 million pixels/second. Rank-8 Mirror exceeded the additive rank-8 control by 1.21 dB at the same K=2 serialized size.

The quality gap depended on content. Rank-8 Mirror was within 0.29–0.63 dB of the native control on `carphone`; it was 1.15–1.35 dB below native on `container`. The private-residual control improved mean PSNR by 1.07 dB over rank-4 product Mirror, at 1,274 more K=2 payload bytes and 508 more marginal bytes per video. It remained a development diagnostic and did not replace the frozen product-Mirror gate.

The 44-row table reports every clip/seed/method/rank, actual payload sizes and hashes, training/inference compute proxies, wall time and memory figures. `source/metric_replay.json` reloads the saved inference payloads and replays every held-out metric from saved payloads; maximum floating-point difference was 8.94e-7 under a frozen 1e-5 absolute tolerance. Fresh clips were not accessed.

### Interpretation

The product factorization compresses token state and beats matched additive controls on these two clips, but its held-out reconstruction did not retain native-token fidelity on the more difficult `container` sequence. The private residual recovers part of that gap, consistent with a need for per-video exceptions. This is a **FAIL** under the preregistered development gate.

### H / T / D / C / U

- **H:** A product factor over video, temporal role and spatial region can replace per-video CoANeRV token state with a useful quality/bytes/runtime frontier and beat simple low-rank/additive controls.
- **T:** Two shared-decoder seeds; four source training clips; two development clips; 32-token × 24-feature decoder; ranks 2/4/8; native full token, low-rank, additive, product Mirror and product-plus-private controls; 800 decoder updates and 600 code updates; 44 development fits.
- **D:** **FAIL**. No product-Mirror rank passed the full quality, storage, runtime and simple-control conjunction on both clips and both seeds. Fresh was not opened.
- **C:** The reduced decoder is underfit and the two development clips are narrow evidence. In addition, the full token bank remains better on `container`; the apparent advantage over additive factors may not generalize beyond these files.
- **U:** Fresh replication, the full CoANeRV feed-forward encoder/token former, standard entropy-coded video rate-distortion, high-resolution GPU memory/FPS, broader clip families and near-convergence fixed-byte capacity remain untested.

BOUNDARY: reduced CoANeRV coordinate-decoder mechanism screen only. Neural-payload bits/pixel are not conventional codec bitrate; CPU results do not establish GPU or energy gains.
