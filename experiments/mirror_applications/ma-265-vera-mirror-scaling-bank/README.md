# MA-265 — VeRA Mirror scaling code bank

**H:** A per-task angle over shared VeRA scaling bases can reduce bank bytes while preserving held-out task outputs on aligned scaling orbits; independent scaling vectors may need private state.

**T:** Frozen linear base plus shared VeRA random A/B factors and scale basis; compare full VeRA scales, one-angle Mirror, generic two-coefficient code and independent LoRA upper control. Aligned and independent task strata. Fresh worlds 26510–26512 × seeds 0–2; actual serialized state charged.

**D:** Pending.

**C:** The generic two-coefficient basis is the simpler control and may match all Mirror behavior; per-task fitting noise may dominate tiny code savings.

**U:** Fresh output quality, serialized bank bytes and fit/decode/apply compute.

## Results

**D: FAIL for the preregistered Mirror scaling claim.** On aligned fresh tasks, Mirror, generic two-coefficient basis, and VeRA all achieved NRMSE 0. Their payloads were 9,925 B, 10,178 B, and 10,690 B. Mirror saved 7.2% vs VeRA and 2.5% vs the simpler generic code, below the registered 30% VeRA storage gate. Mirror fitting averaged 4.89 s/bank, vs 0.27 s generic and 0.34 s VeRA. Apply latency was ~40 μs per held-out task bank for all methods.

On independent scaling tasks, Mirror NRMSE was 0.3335, generic basis 0.3049, VeRA 0.0002, and independent LoRA 0.0021. Thus the view coordinate does not replace private scaling vectors for arbitrary tasks.

**Fact:** 72 fresh rows across worlds 26510–26512 × seeds 0–2, two task strata, four methods. A1 timing replay preserved every NRMSE and payload hash.

**Interpretation:** A one-angle code is a small compression for a known scale orbit, but its marginal byte saving over a generic two-coefficient basis is slight and fitting is much slower. VeRA’s independent scales retain substantially more capacity for unrelated tasks.

**C:** This small linear adapter task may not reflect neural-network transfer; the fixed scale orbit is deliberately aligned with the code family.

**U:** Natural task checkpoints, learned shared bases, transformer-scale throughput, quantized scales, and private-residual allocation remain open.
