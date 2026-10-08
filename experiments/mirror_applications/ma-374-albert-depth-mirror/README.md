# MA-374 — ALBERT shared layers + depth Mirror

Status: **PROTOCOL FROZEN BEFORE DEVELOPMENT**

## H — hypothesis

A small depth-specific Givens coordinate on a fully tied residual block restores useful layer-role differentiation better than hard tying and byte-near FiLM, while using <=60% of an untied three-block network.

## Prior art delta

PA61 establishes ALBERT cross-layer parameter sharing and reports separate attention/FFN sharing ablations. This experiment compares those sharing patterns with full tying plus a depth-specific functional coordinate. The digits residual-block proxy is not a Transformer reproduction.

## T — frozen protocol

Three depth applications of a 64-wide block with attention-like and FFN-like residual maps. Compare untied A/F, fully tied, A-only shared, F-only shared, tied+depth Givens, and tied+depth FiLM. Digits v1.8.0; per-world stratified 60/20/20. 600 updates, 100 code updates, LR grid {0.003,0.01}. Development worlds 37400/37401; fresh worlds 37410/37411/37412 remain sealed. Actual complete torch.save inference bytes are authoritative.

See `PROTOCOL.json` for gates and metric contracts.


## Results

Fresh accuracies: untied 97.22%, tied 96.48%, A-only shared 97.13%, F-only shared 97.50%, tied+Mirror 96.39%, tied+FiLM 96.67%. Mirror and FiLM payloads are both 89,282 bytes; FiLM is slightly more accurate. **Decision: FAIL** for the depth-view claim. The large storage reduction versus untied comes from block sharing, while partial component sharing gives better quality. See `STATUS.md` and `VERIFICATION.json`.
