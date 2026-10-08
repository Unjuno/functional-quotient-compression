# MA-333 — Mirror sign/scale orbit audit

Status: protocol frozen before development. Dedicated branch: `research/ma-333-sign-scale-orbit-audit-20261008`.

## H — hypothesis

ReLU positive hidden scaling and tanh hidden sign flips, when applied to all coupled tensors, change parameter coordinates but preserve the function. These Views are not new logical functions.

## Mirror insertion

> **Mirror insertion:** apply a per-hidden-unit monomial coordinate to incoming weights/bias and the reciprocal or matching action to outgoing weights.

PA47 motivates the exact symmetry audit. This is a capacity-counting control, not a compression design.

## Frozen protocol

See `PROTOCOL.json`; fresh seeds are sealed. Both coupled symmetry and uncoupled negative controls are included.

## Results

Pending.

Development exposed FP16 quantization effects in the ReLU scale transform. Before opening fresh, the protocol now records exact FP32 symmetry error separately from base and transformed FP16 rounding errors. Fresh seeds and training settings are unchanged.
