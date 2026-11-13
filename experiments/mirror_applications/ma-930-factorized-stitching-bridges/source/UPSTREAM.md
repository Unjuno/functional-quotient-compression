# Upstream provenance

- PA240 StitchLLM: https://aclanthology.org/2025.acl-long.1305/; official implementation/checkpoint not found in bounded review.
- PA242 cross-model residual transport: https://proceedings.neurips.cc/paper_files/paper/2025/file/4569a868e7aa891248832ec08445d071-Paper-Conference.pdf
- TinyStories models pinned in `PROTOCOL.json`; all use GPT-Neo/GPT-2 vocabulary with 8 layers and widths 64, 128 and 256.
- TinyStories dataset revision and exact parquet shards pinned in `PROTOCOL.json`; license is CDLA-Sharing-1.0.
- Model weights and datasets are downloaded to external cache/temp storage only; not committed. Actual file hashes and bytes will be reported after download.
