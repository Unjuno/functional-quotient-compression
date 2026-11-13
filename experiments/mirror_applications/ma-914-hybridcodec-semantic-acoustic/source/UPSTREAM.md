# Upstream provenance

- PA259 arXiv: https://arxiv.org/abs/2606.06743 (dual-stream semantic/acoustic formulation).
- Repository: https://github.com/hybridcodec/hybridcodec
- Inspected commit: `16b0158932f5d3f6735fa45595684d846f83bfbc`.
- Repository README cites the Interspeech 2026 paper *HybridCodec: Modeling Discrete and Continuous Representations for Efficient Speech Language Models* and describes discrete semantics plus optional continuous residuals. The mapping to PA259 has not been verified.
- HF quantizer model: `hybridcodec/hybridcodec_finetune_12_5hz`, model revision `a64d9efe891f40317087bd675cd9b5758b37a3e9`; weights file size 195,901,108 B.
- FocalCodec dependency: `lucadellalib/focalcodec_50hz`, revision `d6d6da...` (API revision); model.safetensors 568,554,604 B.
- Local test checkout: `/tmp/ma914-hybridcodec`; not copied into this repository.
- Isolated CPU environment: `/tmp/ma914-venv`; not committed.
