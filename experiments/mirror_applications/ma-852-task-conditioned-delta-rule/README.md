# MA-852 — task-conditioned Delta-rule decay coordinate

H: task-specific decay on a shared Hebbian update rule preserves retention behavior across context regimes at lower storage than independent per-task full update operators.

PA230 motivates matrix-valued writable state; PA73 motivates selective state propagation/forgetting. The Mirror insertion is task coordinate m selecting a decay rate in W <- lambda(m)W + v kᵀ. Controls are fixed decay, a native scalar task gate, and full per-task decay matrices. Every task's writable W is separately charged.

Synthetic D=16 streams use one-hot keys, random values, 8 task-specific retention regimes, and both short/long prefixes; development seed 23, fresh 43/71. Measure actual serialized rule plus state payload, output error, update MAC, and reset isolation. No learned language model or sequence-model claim.

C: ordinary task-indexed scalar gates exactly implement the coordinate with fewer bytes than any structured Mirror decoder. U: trained DeltaNet/Mamba, context inference, noisy/repeated keys, and hardware throughput.
