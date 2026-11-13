# MA-898 status

**UNTESTED — blocked before training/evaluation.** Official CC-4DGS was validated on PyTorch 2.0.0+cu118, CUDA 11.8 and an RTX 4090. Its install builds Gaussian rasterizer and simple-KNN CUDA extensions; training/rendering use CUDA directly. Dataset files are excluded. The current runtime has PyTorch 2.14.1+cpu, no CUDA device/runtime, and no N3DV/Technicolor scene assets. No candidate or control was trained or scored.

A separate smoke check instantiated the official CDF source with a reduced random-weight config and ran 16 CPU inputs: finite output with shape [16,12]. This confirms only that the CDF primitive can execute on CPU; it is not trained-model evidence. Official checkpoints alone do not provide held-out scene frames or the CUDA renderer needed for the registered PSNR/FPS frontier. Preserve MA-898 as UNTESTED, not FAIL.
