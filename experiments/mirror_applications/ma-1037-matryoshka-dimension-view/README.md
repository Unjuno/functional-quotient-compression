# MA-1037 — Matryoshka dimension-budget Mirror readout

Status: SCREENING. The original FP32 CPU encode attempt produced no embeddings; AM-01 switches to the official ONNX INT8 export before dev/audit, with all gates unchanged. This experiment tests the P0 registry candidate on Nomic Embed v1.5 official INT8 ONNX export and the official BEIR SciFact corpus/qrels. Per-budget Mirror coordinates rotate the two embedding dimensions at a prefix boundary before truncation.

The native nested prefix, a normalized two-coordinate learned boundary readout, and the full 768-D embedding are controls. Payload accounting includes the complete source vector table, each materialized candidate table, model/tokenizer/code, coordinates, corpus identifiers, and manifests. See `PROTOCOL.json` for the frozen train/test split and gates, and `ARTIFACT_PROVENANCE.json` for pinned sources and hashes.
