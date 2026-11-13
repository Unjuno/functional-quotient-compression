# MA-897 execution blocker

## Fact

- PA249 identifies ADC-GS as an anchor-driven dynamic Gaussian codec with hierarchical deformation and rate-distortion controls.
- Official repository inspected: `H-Huang774/ADC-GS`, commit `5501b8a163b564a583cca2edf251868e795c3040`.
- Its README installs `torch==1.12.1+cu116` / `torchvision==0.13.1+cu116` and builds `submodules/diff-gaussian-rasterization` and `submodules/simple-knn`.
- Source calls `GaussianRasterizer`, `distCUDA2`, direct `.cuda()` allocations, and `torch.cuda.Event` in train/render/initialization paths.
- Current runtime is `torch 2.14.1+cpu`; CUDA and `nvidia-smi` are unavailable. Workspace search found no HyperNeRF, Neural 3D Video, ADC-GS scenes, or checkpoints.
- No training, evaluation, serialized payload measurement, or throughput measurement was performed.

## Interpretation

The official ADC-GS native control cannot be reproduced in this environment. A CPU replacement renderer would not be a faithful quality/runtime test of the registered method. No Mirror-specific conclusion can be drawn.

## Hypothesis

Whether anchor-group × time Mirror coordinates can provide distinct useful trajectories at a better storage/quality/compute frontier remains untested. Missing hardware and data are not evidence of model failure.

## Requirements to resume

- CUDA GPU/runtime compatible with the official 11.6-era stack and the two compiled extensions.
- Fixed HyperNeRF or Neural 3D Video scene/config, with held-out timestamp × viewpoint cells locked before training.
- Freeze native ADC-GS, ordinary motion/time embedding, Mirror coordinate, and practical independent-motion controls before fresh evaluation. Count all anchor, decoder, codebook, appearance and metadata bytes; report PSNR/LPIPS, temporal stability, actual payload bytes, FPS, fit time, random-access latency and VRAM.
