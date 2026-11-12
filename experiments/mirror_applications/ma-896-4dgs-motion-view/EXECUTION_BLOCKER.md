# MA-896 execution blocker

## Fact

- PA248 describes canonical 3D Gaussians, decomposed 4D neural voxels and a time-conditioned deformation decoder.
- Official repository inspected: `hustvl/4DGaussians`, commit `843d5ac636c37e4b611242287754f3d4ed150144`.
- Its README setup installs `depth-diff-gaussian-rasterization` and `simple-knn` native extensions and reports PyTorch 1.13.1 + CUDA 11.6. `train.py` uses CUDA tensors and CUDA timing events; the Gaussian model allocates trainable state on CUDA and invokes `distCUDA2`.
- Current runtime is PyTorch `2.14.1+cpu`, CUDA unavailable, and `nvidia-smi` absent. Workspace search found no D-NeRF/HyperNeRF/dynerf assets or 4DGS checkpoints.
- No training, scoring, payload measurement, or render timing was performed.

## Interpretation

The official renderer and registered native 4DGS control cannot be executed in this environment. A substitute CPU renderer or synthetic motion field would test a different method and cannot establish PSNR/LPIPS or measured rendering throughput.

## Hypothesis

Whether compact motion Views can share canonical Gaussian geometry while recovering useful distinct motion functions remains untested. Missing hardware/data are not model evidence and do not count as FAIL.

## Requirements to resume

- CUDA GPU/runtime compatible with the pinned 4DGS repository and its rasterizer/simple-knn extensions.
- A fixed official dynamic scene (e.g. D-NeRF) with locked held-out timestamp × viewpoint cells.
- Freeze native 4DGS, ordinary scene/motion-code deformation, Mirror motion View and practical independent-motion controls before fresh evaluation. Count all canonical attributes, decoder state, codes and metadata in actual serialized payload; report PSNR/LPIPS, FPS, fit time and peak VRAM.
