# MA-897 environment review

- Official upstream: `H-Huang774/ADC-GS`, inspected at `5501b8a163b564a583cca2edf251868e795c3040`.
- Official install instructions specify `torch==1.12.1+cu116`, CUDA differentiable Gaussian rasterization and simple-KNN extensions.
- `gaussian_renderer/__init__.py` imports `GaussianRasterizer` and calls CUDA tensors; `train.py` uses CUDA events; `scene/gaussian_model.py` uses `distCUDA2` and allocates CUDA model state.
- Current runtime: PyTorch `2.14.1+cpu`, `torch.cuda.is_available() == False`, no `nvidia-smi`.
- No HyperNeRF or Neural 3D Video scene assets/checkpoints were found in the workspace.
- The separate GPU-Control connector only exposes a disabled paid CPU TinyStories workload and cannot execute ADC-GS. A Hugging Face Jobs tool was listed but unavailable from the active MCP tool registry; no remote job was submitted.
- No training, rendering, scoring, or payload measurement was attempted.
