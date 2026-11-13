# MA-898 environment review

- Official upstream: `KyungdaePark/CC-4DGS`, inspected at `1ba6c8f2bf76352e1ce444560996e7ff325cb412`.
- README validation stack: PyTorch 2.0.0+cu118, CUDA 11.8, RTX 4090; CUDA toolkit must match PyTorch. Requirements install editable rasterizer and simple-KNN extensions.
- `submodules/diff-gaussian-rasterization/setup.py` declares `CUDAExtension`; official train/render/metrics and model initialization use `.cuda()` and `torch.cuda` operations.
- Dataset files are explicitly excluded; README points to N3DV and Technicolor. No matching scene assets were found locally.
- Checkpoints are linked from Google Drive, but they do not include test frames; quality/FPS evaluation still depends on the supported renderer and held-out dataset.
- Current runtime: PyTorch `2.14.1+cpu`, `torch.cuda.is_available() == False`, no `nvidia-smi`.
- A reduced-config random-weight smoke test executed the untouched `scene/cdf.py` on 16 CPU points and returned finite `[16,12]` output (688 parameters, 32 buffer elements). It confirms only pure-PyTorch CPU execution of the CDF primitive, not model quality or trained checkpoint behavior.
- No model run or registered benchmark was attempted.
