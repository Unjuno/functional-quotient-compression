# MA-896 environment review

- Official upstream: `hustvl/4DGaussians`, inspected at commit `843d5ac636c37e4b611242287754f3d4ed150144`.
- Upstream README setup pins its environment to PyTorch 1.13.1 + CUDA 11.6 and installs the `depth-diff-gaussian-rasterization` and `simple-knn` submodules as extensions.
- Upstream `.gitmodules` identifies those extensions; `train.py` allocates CUDA background tensors and CUDA timing events, `scene/gaussian_model.py` allocates model state on CUDA and calls `distCUDA2`, and rendering imports the Gaussian rasterizer.
- Available environment: PyTorch `2.14.1+cpu`; `torch.cuda.is_available()` is false; `nvidia-smi` is unavailable.
- Workspace search found no D-NeRF, HyperNeRF, dynerf, or 4DGS scene assets/checkpoints.
- No training, rendering, scoring, serialization, or FPS measurement was attempted.
