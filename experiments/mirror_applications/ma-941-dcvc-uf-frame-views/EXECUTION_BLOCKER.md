# Execution blocker

- CVPR 2026 paper is accessible and describes one chunk latent, cross-frame interaction, frame-specific decoders and consolidated entropy coding (PA269). Its CVF page exposes paper/supplement links but no code link.
- Public GitHub repository searches for “DCVC-UF” and “Ultra-Fast Neural Video Compression” returned no repositories; local checkout has no DCVC-UF source, checkpoint, entropy coder, or bitstream.
- CUDA/NVIDIA is unavailable.
- MA-941 requires the native chunk-parallel decoder and entropy-coded bitstream. A substitute implementation would not provide the registered comparator.

Disposition: **UNTESTED**, not FAIL. Revisit when official artifacts and compatible runtime are available.
