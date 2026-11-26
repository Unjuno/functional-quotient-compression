# TT-EmbeddingBag build notes

`vendor/FBTT-Embedding` is an unmodified snapshot from `facebookresearch/FBTT-Embedding` commit `b95947ccd32cf9ab851242cc606562f2a74d9efa` (MIT license). It includes the official CUDA extension sources and the upstream setup script.

The runner requires a CUDA-enabled PyTorch runtime and a CUDA compiler/toolkit compatible with that runtime. The upstream README documents CUDA 10.1 and PyTorch 1.6+ as its tested environment, and requires NVIDIA CUB. Build the vendored extension in the reviewed GPU image using the upstream setup instructions; retain compiler/runtime versions in provenance. The current CPU-only worker cannot build or run the GPU kernels.

Do not edit the vendored upstream files. Experiment changes live in `train.py` and `views.py`.
