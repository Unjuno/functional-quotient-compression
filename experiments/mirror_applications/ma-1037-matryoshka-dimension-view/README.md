# MA-1037 — Matryoshka dimension-budget Mirror readout

Status: FAIL (scoped Nomic/SciFact test). The original FP32 CPU encode attempt produced no embeddings; AM-01 switches to the official ONNX INT8 export before dev/audit, with all gates unchanged. This experiment tests the P0 registry candidate on Nomic Embed v1.5 official INT8 ONNX export and the official BEIR SciFact corpus/qrels. Per-budget Mirror coordinates rotate the two embedding dimensions at a prefix boundary before truncation.

The native nested prefix, a normalized two-coordinate learned boundary readout, and the full 768-D embedding are controls. Payload accounting includes model and tokenizer assets, the complete source vector table, each materialized candidate table, coordinates, corpus identifiers, and resolvable manifests; pinned architecture-code provenance is recorded separately. See `PROTOCOL.json` for the frozen train/test split and gates, and `ARTIFACT_PROVENANCE.json` for pinned sources and hashes.


On 300 disjoint test queries, the best nDCG gain is only +0.315 pp at 64 dimensions; the matched two-weight boundary control is bit-exact to Mirror at every dimension. The complete five-budget bank is 174,126,539 B vs 174,126,309 B native. See `RESULTS.md` for H/T/D/C/U and the full byte, quality, and compute map.
