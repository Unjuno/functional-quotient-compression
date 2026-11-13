# MA-898 — CC-4DGS compressed deformation Mirror role code

Status: UNTESTED (execution blocked before training/evaluation)
Evidence lane: NEURAL_GRAPHICS / SPACE_TIME / QUALITY / STORAGE / RUNTIME

## H — falsifiable hypothesis

**Mirror insertion:** this experiment adds a compact scene/motion coordinate `m` at the deterministic dense-hash-feature input of CC-4DGS's Computational Deformation Field (CDF), so multiple logical deformation functions can share the paid CDF decoder and canonical attributes rather than adding independent CDFs or motion assets.

The hypothesis is falsified if Mirror does not improve the actual serialized quality/byte/runtime frontier over native CDF+CCA and an ordinary scene/motion code at matched conditions, or if a simpler control reproduces the same functions at equal or lower deployed cost.

## Prior-art delta and controls

PA250 CC-4DGS is the already-compressed native baseline, not the Mirror result. Required controls are official CC-4DGS CDF+CCA, the same shared CDF with an ordinary scene/motion embedding, the proposed Mirror coordinate, and independent per-scene/motion CDF assets when practical. Pay for the CDF decoder, deterministic feature metadata, CCA autoencoder, quantizers, residual codebooks, all addresses and reconstruction state.

## Evidence limits

The official implementation was validated on CUDA 11.8 and an RTX 4090, compiles Gaussian rasterizer/simple-KNN CUDA extensions, uses CUDA across training/rendering, and excludes dataset files. The CDF module itself is pure PyTorch and a small random-weight CPU forward smoke succeeded, but that is not trained-model evidence. The available runtime is CPU-only; N3DV/Technicolor assets are absent. Official checkpoints exist, but without the supported renderer and dataset test frames the registered PSNR/FPS/frontier cannot be evaluated. No model was trained or scored.
