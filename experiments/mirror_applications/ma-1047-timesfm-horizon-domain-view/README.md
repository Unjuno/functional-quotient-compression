# MA-1047 — TimesFM horizon × domain factorized output head

**Decision: FAIL** for the registered quality gate. The factorized output code is matched by a smaller native additive control and worsens MASE on ETTm1. Actual complete payloads, fresh time windows, and CPU runtime are reported in [RESULTS.md](RESULTS.md); the frozen protocol and post-audit reducer amendments are in [PROTOCOL.json](PROTOCOL.json).

The checkpoint is official TimesFM 2.5 PyTorch. Fresh targets are the final 20% chronological windows of ETTh1 and ETTm1; neither series data nor weights are committed raw. H100 runtime remains unmeasured in this CPU-only environment.
