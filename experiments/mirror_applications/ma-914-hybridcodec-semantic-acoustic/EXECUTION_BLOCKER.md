# MA-914 execution boundary

## Fact

- PA259 is arXiv:2606.06743, titled *HybridCodec: Fast Dual-Stream, Semantically Enhanced Neural Audio Codec*. Its abstract describes separate semantic and acoustic branches.
- The public repository `hybridcodec/hybridcodec`, commit `16b0158932f5d3f6735fa45595684d846f83bfbc`, is installable. Its README describes a discrete semantic stream and optional continuous residual stream and cites *HybridCodec: Modeling Discrete and Continuous Representations for Efficient Speech Language Models* (Interspeech 2026).
- Its HF quantizer files are available (195,901,108 bytes each), and its hyperparameters load a separate pretrained FocalCodec model (568,554,604-byte safetensors file).
- Initial installation into the shared runtime pulled a CUDA 13 torchaudio wheel incompatible with its CPU-only PyTorch 2.14.1. An isolated `/tmp/ma914-venv` using torch/torchaudio 2.7.1+cpu avoided this environment conflict.
- Upstream tests passed: 5 passed in 8.13 seconds. They are synthetic shape/checkpoint-format tests, not codec evaluation.
- LibriTTS test-clean is available as four large parquet shards (about 1.47 GB total); it is not present locally.

## Interpretation

A dependency mismatch is avoidable. However, the upstream model/paper mapping is unresolved and its described stream structure differs from the registered PA259 description. Running Mirror comparisons on it risks testing a different method. Full inference also requires additional pretrained weights and a large speech evaluation set.

## Decision

Keep MA-914 UNTESTED. Do not reinterpret passing unit tests as evidence for the registered Mirror hypothesis, and do not replace the PA259 baseline with a merely similarly named codec. Resolve provenance and freeze a matched protocol before any quality/byte comparison.
