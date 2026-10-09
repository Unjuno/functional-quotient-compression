# MA-261 — BatchEnsemble-style rank-one Mirror experts

**H:** A shared expert plus per-expert rank-one Mirror coordinates can preserve top-1 routed task quality with much less storage than independent experts; compare native BatchEnsemble and a generic coefficient control.

**T:** Four-domain Gaussian mixture regression with shared learned router. Compare four untied experts, tied expert, BatchEnsemble rank-one factors, bounded Mirror scalar over a shared rank-one direction, and unbounded generic coefficient control. Fresh worlds 26110–26112 × seeds 0–2. Actual serialized model and router bytes are charged.

**D:** Pending.

**C:** The task is intentionally aligned with rank-one shared-basis adapters; a generic coefficient code may exactly match the Mirror with simpler decoding. Router errors may dominate expert-view differences.

**U:** Fresh routed quality, bytes, active computation, training and inference time.
