# MA-965 — EAS-SNN adaptive sampler Mirror event View

Status: NOT ESTABLISHED (deferred; registry remains UNTESTED)
Evidence lane: RUNTIME / QUALITY
Base commit: `59fa0e0d3`

## Hypothesis

H: A compact sensor/time/context coordinate can make one EAS-SNN adaptive event sampler behave as multiple useful event-window samplers while retaining fine event timing and improving the full detector storage/quality/runtime frontier over native adaptive sampling and simple conditioning.

## Mirror insertion

The proposed `m` conditions the shared `AdaptiveRSNNEmbedding` sampler. The closest simple controls are direct per-condition threshold/gain and equal-byte FiLM/low-rank modulation. The required upper control is independent sampler copies. The native upstream module uses recurrent convolution, sigmoid gating, threshold/reset dynamics, spike-triggered segment assignment, and residual readout.

## Decision

No experiment result is claimed. This candidate is deferred to continue the registered queue. Missing local data and unavailable GPU prevent the required detection quality and runtime/energy comparison in this environment. A synthetic sampler-only surrogate would not answer the registered mAP/full-backbone hypothesis, so none was substituted.

Official upstream implementation is available at `https://github.com/Windere/EAS-SNN`, pinned to `423cb8be283af75426c908e94b723456f961913d`; paper: `https://arxiv.org/abs/2403.12574`. The paper's README links Gen1/1Mpx/N-Caltech datasets and pretrained checkpoints, but they are not present in `/workspace`.
