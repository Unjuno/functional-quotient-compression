# MA-897 — ADC-GS anchor motion Mirror bank

Status: UNTESTED (execution blocked before training)
Evidence lane: NEURAL_GRAPHICS / SPACE_TIME / QUALITY / STORAGE / RUNTIME

## H — falsifiable hypothesis

**Mirror insertion:** this experiment adds a compact motion coordinate `m(anchor_group, t)` to the native ADC-GS anchor-to-Gaussian deformation interface so multiple logical dynamic motion fields can share anchor-based physical deformation state instead of storing separate motion/deformation parameters.

The hypothesis is falsified if the View fails the preregistered held-out-time/view quality gate, costs more actual serialized bytes or active runtime than ADC-GS's native anchor deformation and ordinary motion-code controls at matched quality, or is functionally equivalent to a simpler native coefficient/code parameterization.

## Prior-art delta and controls

PA249 ADC-GS is the target method, not the Mirror contribution. Required controls are official native ADC-GS anchor deformation, the same shared anchors with an ordinary learned motion/time embedding and decoder, the proposed Mirror coordinate, and independent motion/deformation assets when practical. Existing rate control, anchor allocation, pruning, and reconstruction are held fixed across matched conditions.

## Evidence limits

The official implementation requires CUDA 11.6-era PyTorch, compiled differentiable Gaussian rasterization and simple-KNN CUDA extensions, CUDA-bound training/rendering, and HyperNeRF or Neural 3D Video scene data. The available runtime is CPU-only and no registered scene assets are present. No method was trained or evaluated; this package records an execution blocker only.
