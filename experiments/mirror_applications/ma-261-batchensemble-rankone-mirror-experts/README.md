# MA-261 — BatchEnsemble-style rank-one Mirror experts

**H:** A shared expert plus per-expert rank-one Mirror coordinates can preserve top-1 routed task quality with much less storage than independent experts; compare native BatchEnsemble and a generic coefficient control.

**T:** Four-domain Gaussian mixture regression with shared learned router. Compare four untied experts, tied expert, BatchEnsemble rank-one factors, bounded Mirror scalar over a shared rank-one direction, and unbounded generic coefficient control. Fresh worlds 26110–26112 × seeds 0–2. Actual serialized model and router bytes are charged.

**D:** Pending.

**C:** The task is intentionally aligned with rank-one shared-basis adapters; a generic coefficient code may exactly match the Mirror with simpler decoding. Router errors may dominate expert-view differences.

**U:** Fresh routed quality, bytes, active computation, training and inference time.

## Results

**D: FAIL for Mirror-specific routed expert improvement.** Across 3 fresh worlds × 3 seeds, standard MoE routed NRMSE averaged 0.0394 at 2,149 B; native BatchEnsemble 0.1158 at 2,213 B; Mirror 0.0595 at 2,149 B; generic rank-one coefficient control 0.0595 at 2,149 B; hard tying 0.4994 at 2,085 B. Router accuracy was 0.9978 across methods. Mirror did not improve quality or bytes over generic rank-one coefficients; standard MoE was better at the same payload bytes.

**Fact:** 45 fresh rows, full router and expert payloads serialized into the same flat format; 500 expert updates and shared router state. Fresh result reproduced development ordering. Mean train time was 0.265 s for Mirror, 0.254 s for generic rank-one, 0.187 s standard MoE, and 0.317 s BatchEnsemble. Decode was about 0.5 ms.

**Interpretation:** Rank-one shared-basis coefficients are a viable compact control for this deliberately aligned synthetic task, but the nonlinear bounded Mirror code did not add a functional degree of freedom beyond generic coefficients. Standard independent experts remained the strongest quality point.

**C:** Tiny 2D linear experts and a near-perfect shared router favor the full MoE; no natural task or meaningful expert compute scaling was measured.

**U:** Natural tokens, top-k>1, large expert FFNs, non-rank-one tasks, and hardware-fused inference remain untested.
