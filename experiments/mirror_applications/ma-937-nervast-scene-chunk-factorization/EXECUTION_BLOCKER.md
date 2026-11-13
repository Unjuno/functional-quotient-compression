# Execution blocker

The official HNeRV upstream is available and runs CPU decoding for its included Bunny asset (132 frames, finite 640×1280 output; 7,574,217 B combined quantized embedding checkpoint + TorchScript decoder). However, that upstream does not provide multiple fitted scenes. MA-937 requires scene-factor × chunk-factor held-out combinations; using time windows from the same Bunny scene as multiple scenes would invalidate the test.

The direct NerVast baseline from PA268 is also unavailable: no source/checkpoints/Fisher masks, and no CUDA/NVIDIA hardware for its required rate-distortion/decode-FPS path. The experiment remains **UNTESTED**, not FAIL.
