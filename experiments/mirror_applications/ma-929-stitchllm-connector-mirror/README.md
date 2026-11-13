# MA-929 — shared StitchLLM connector Mirror code

Status: UNTESTED
Evidence lane: MODEL_STITCHING / TRANSFER / CONNECTOR_STORAGE / THROUGHPUT

## H — falsifiable hypothesis

A shared connector basis plus small source/target block Mirror codes can recover StitchLLM's useful cross-model block transitions with lower actual connector bytes while preserving NLL and serving throughput versus independent native stitching layers.

## T — source and asset review

Reviewed PA240 and the ACL anthology paper page. Exact-title GitHub and Hugging Face searches found no public StitchLLM implementation or checkpoint. Workspace contains no matched source/target LLM checkpoints or StitchLLM connector bank. The registered comparison requires native StitchLLM throughput control.

## D — UNTESTED

No stitching, transfer, NLL, connector-byte or throughput run was performed.

## C — strongest counter-hypothesis

Native stitching layers may encode pair-specific representation transport; a shared basis may increase drift and lose the throughput/quality tradeoff.

## U — unresolved

Official StitchLLM source, matched source/target models, native connectors and a serving benchmark runtime. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
