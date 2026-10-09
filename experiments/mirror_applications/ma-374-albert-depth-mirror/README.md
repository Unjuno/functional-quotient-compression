# MA-374 — ALBERT shared layers + depth Mirror

Status: **FAIL on the frozen registered protocol**

## H — hypothesis

A small depth-specific Givens coordinate on a fully tied residual block restores useful layer-role differentiation better than hard tying and byte-near FiLM, while using <=60% of an untied three-block network.

## Prior art delta

PA61 establishes ALBERT cross-layer parameter sharing and reports separate attention/FFN sharing ablations. This experiment compares those sharing patterns with full tying plus a depth-specific functional coordinate. The digits residual-block proxy is not a Transformer reproduction.

## T — frozen protocol

Three depth applications of a 64-wide block with attention-like and FFN-like residual maps. Compare untied A/F, fully tied, A-only shared, F-only shared, tied+depth Givens, and tied+depth FiLM. Digits v1.8.0; per-world stratified 60/20/20. 600 updates, 100 code updates, LR grid {0.003,0.01}. Development worlds 37400/37401 selected LR 0.003; fresh worlds 37410/37411/37412 were evaluated once at that locked setting. Actual complete torch.save inference bytes are authoritative.

See `PROTOCOL.json` for gates and metric contracts.


## Results

Fresh accuracies: untied 97.22%, tied 96.48%, A-only shared 97.13%, F-only shared 97.50%, tied+Mirror 96.39%, tied+FiLM 96.67%. Mirror and FiLM payloads are both 89,282 bytes; FiLM is slightly more accurate. **Decision: FAIL** for the depth-view claim. The large storage reduction versus untied comes from block sharing, while partial component sharing gives better quality. See `STATUS.md` and `VERIFICATION.json`.


## Final H / T / D / C / U

**H:** a four-angle depth coordinate on one fully tied residual block could recover useful layer roles at <=60% of untied bytes and beat hard tying and equal-code FiLM.

**T:** CPU sklearn-digits residual-block proxy, 64-dimensional input/hidden state, three block applications; 600 updates plus 100 code updates. Development worlds 37400/37401 selected LR 0.003 from the frozen grid. Fresh worlds 37410/37411/37412 ran once at that setting. Controls: untied, fully tied, attention-only shared, FFN-only shared, tied+Mirror, tied+FiLM. Total paid inference state was measured from serialized `torch.save` payloads.

**D — FAIL:** fresh mean accuracy was 97.22% untied, 96.48% tied, 97.13% attention-shared, 97.50% FFN-shared, 96.39% Mirror, and 96.67% FiLM. Mirror used 89,282B versus 257,447B untied (34.7%), but its payload equaled FiLM and its accuracy was lower. It did not restore a measurable layer-specialization frontier; component-wise sharing performed better. Tests: 3 passed. The current-workspace verifier replayed 42 serialized rows with max metric difference 4.63e-9 and zero logit roundtrip difference.

**C:** this is a small digits residual-block proxy, not ALBERT or a language model; the post-block Givens insertion may miss useful within-block layer roles. Component-wise sharing also shows the quality loss is not forced by all sharing patterns.

**U:** natural language, a trained ALBERT/BERT checkpoint, long-context behavior, near-convergence, and device-specific fused latency remain untested. The fixed-update screen is not a capacity result.

## Duplicate-ID branch resolution

A separate earlier branch, `research/ma-374-albert-shared-depth-mirror-20261008`, tested a different synthetic recurrence-sequence Transformer and reported a development-only final-depth NLL signal. Its protocol and result first appeared together in commit `52fd6b69` at 14:46 UTC, with no separate pre-development freeze commit; it also missed the multi-depth development gate and kept fresh seeds sealed. The registered digits protocol was frozen separately at `f6dec0b7` at 21:57 UTC, before its development/fresh runs. The earlier sequence screen is preserved under [`protocol_variants/unregistered_sequence_probe/`](protocol_variants/unregistered_sequence_probe/) as exploratory evidence; it does not override this registered FAIL or supply fresh replication.
