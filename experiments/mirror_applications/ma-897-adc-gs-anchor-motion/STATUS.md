# MA-897 status

**UNTESTED — blocked before training.** The official ADC-GS repository specifies PyTorch 1.12.1 + CUDA 11.6 and installs its differentiable Gaussian rasterizer and simple-KNN extensions. Training, rendering and initialization use CUDA tensors/ops. The current runtime has PyTorch 2.14.1+cpu, no CUDA runtime/device, and no HyperNeRF or Neural 3D Video dataset assets. No candidate or control was trained or scored.

MA-896 remains separately `UNTESTED` and is being skipped per the user's direction so the registry can be traversed. MA-897 is likewise preserved as `UNTESTED`, not FAIL. Advance to MA-898 while retaining this evidence.
