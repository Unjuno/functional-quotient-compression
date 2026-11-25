# MA-1058 — Chronos cold-start series identity View

This screen asks whether held-out traffic-sensor series can be adapted from a small support set with one scalar coordinate on a shared rank-1 Chronos output LoRA basis. The audit series are not used to train that basis.

PA326 establishes Chronos probabilistic tokenized forecasting; PA332 establishes time-series LoRA module adaptation. The Mirror-specific test is a per-series coordinate on a shared low-rank output basis, compared with pooled LoRA, a direct diagonal gain, and independent per-series rank-1 LoRA.

This is separate from MA-1057: it uses Chronos and held-out series identities at one fixed frequency/horizon, not frequency×horizon calibration. The data source/network/calendar overlap is disclosed in the protocol. Actual model, adapter, code, metadata and private exception bytes are charged.

The per-series full two-coefficient control is the private-state boundary measurement. This screen does not claim to validate a conditional fallback selector.
