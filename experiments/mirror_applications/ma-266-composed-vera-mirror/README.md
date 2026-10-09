# MA-266 — Composed VeRA Mirror views

**H:** A nonlinear one-scalar composition law over shared VeRA views may preserve sequential composition more compactly than ordinary scale-vector addition.

**T:** Shared rank-1 idempotent basis K, two adapter maps I+alpha*K, and held-out input functions. Compare identity, scale addition, exact sequential application, Mirror product-law composition, generic product-law scalar, and explicit composed matrix. Fresh worlds 26610–26612 × seeds 0–2; actual payload and apply latency measured.

**D:** Pending.

**C:** The generic scalar product law is the same algebra with simpler terminology and may exactly match Mirror; for idempotent K, the law follows elementary matrix multiplication.

**U:** Fresh reconstruction error, byte savings and one-step compute savings.

## Results

**D: FAIL for Mirror-specific storage/composition advantage; composition algebra itself works.** On 9 fresh worlds/seeds, simple scale addition had NRMSE 0.03670. Sequential VeRA application, the one-code Mirror law, generic scalar product law, and explicit matrix all reconstructed at ≤2.1e-7 NRMSE. The packed payload was 140 B for Mirror and generic product law, 144 B for two sequential VeRA codes, and 4,104 B for an explicit matrix. Thus one-code composition saved only 4 B (2.8%) vs two codes and missed the registered 20% gate. One-step application took ~15 μs vs ~31 μs for sequential execution.

**Fact:** 54 fresh rows, exact custom-binary decode, payload hashes and lengths checked. Fresh results match the development algebra.

**Interpretation:** The product term alpha1*alpha2 matters: ordinary scale addition loses function quality. A single code evaluates the composed map in one step. The generic scalar product law is identical to Mirror and has identical bytes/compute, so this is not Mirror-specific evidence.

**C:** The idempotent rank-one operator makes the composition law reducible to elementary scalar arithmetic; richer noncommuting task updates may require more coordinates.

**U:** Multiple independent basis operators, noncommutative adapters, natural tasks, and realistic model-scale storage/latency remain untested.
