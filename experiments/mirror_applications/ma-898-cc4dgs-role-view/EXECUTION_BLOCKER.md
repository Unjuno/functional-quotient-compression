# MA-898 execution blocker

## Fact

- PA250 describes CC-4DGS as a compact computational deformation field (CDF) plus conditional canonical attribute compression (CCA), including quantization and residual codebooks.
- Official implementation inspected: `KyungdaePark/CC-4DGS`, commit `1ba6c8f2bf76352e1ce444560996e7ff325cb412`.
- Official README reports validation with PyTorch 2.0.0+cu118, CUDA 11.8 and an NVIDIA RTX 4090. It compiles differentiable Gaussian rasterization and simple-KNN CUDA extensions; `train.py`, `render.py`, metrics and model initialization directly use CUDA. The rasterizer builds a `CUDAExtension`.
- The README says dataset files are not included and points to N3DV and Technicolor datasets. Workspace search found no scene assets or evaluation frames. The README links pretrained checkpoints, but those checkpoints do not supply the missing test frames or remove the CUDA renderer requirement.
- Current runtime is `torch 2.14.1+cpu`, CUDA unavailable, and `nvidia-smi` absent. A reduced-config random-weight smoke of the untouched official `scene/cdf.py` successfully returned finite `[16,12]` CPU output; it does not test trained weights, Mirror coordinates, scenes, or image quality. No training, rendering, scoring, serialized payload measurement or FPS measurement was performed.

## Interpretation

The registered test requires quality and runtime comparison over already-compressed CDF+CCA. A CPU-only decoder microbenchmark or substitute rasterizer cannot establish the registered PSNR/FPS frontier or evaluate a useful scene/motion function set. The official renderer and dataset-based control are unavailable.

## Hypothesis

Whether a compact Mirror scene/motion coordinate improves the native CC-4DGS quality/bytes/runtime frontier remains untested. Missing hardware/data are not model evidence and do not count as FAIL.

## Requirements to resume

- CUDA GPU/runtime compatible with the released CC-4DGS version and compiled rasterizer/simple-KNN extensions.
- Fixed official N3DV or Technicolor scene/config plus locked held-out timestamp × viewpoint cells.
- Freeze native CDF+CCA, shared CDF with ordinary scene/motion embedding, Mirror coordinate and practical independent-asset controls before fresh evaluation. Count all decoder, codebook, CCA, quantization, code, metadata and reconstruction bytes; report PSNR/LPIPS/SSIM, temporal consistency, render FPS, peak VRAM, fit time and random-access latency.
