# MA-392 Status

**FAIL — development screen. Fresh seeds remain sealed.**

H: one domain Givens coordinate over shared quotient/remainder tables would recover domain-specific embeddings for held-out token-domain pairs beyond tied composition and simple FiLM/additive controls.

T: two synthetic worlds (39201, 39202), five methods, 512 Adam updates each; 25% of token-domain pairs held out. Scores and benchmarks use serialized FP16 state.

D: FAIL. Mirror held-out accuracy beat tied concatenation by 1.74 points in seed 39201 (0.7911 vs 0.7737), below the +5 point gate; in seed 39202 it beat tied concat by 21.90 points (0.7762 vs 0.5572). Mirror seen accuracy was 0.9012 vs FiLM 0.9751 in seed 39201, missing the 2 point margin. Mirror payload was 1,906B vs the smallest shared control at 1,676B (1.137x, gate <=1.10x). CPU throughput was 0.241–0.334x tied concatenation (gate >=0.90x). Mirror yielded 256/256 distinct FP16 role embeddings; tied concat yielded 64/256.

C: FiLM has more state but materially better seen quality and CPU throughput; tied concatenation is smaller and faster. The Mirror view improves held-out results versus FiLM in both worlds, but does not meet the preregistered end-to-end tradeoff.

U: natural language/recommendation data, accelerator kernels, fresh worlds, near-convergence capacity, and broader domain shifts remain untested.

Fact / Interpretation / Hypothesis: exact metrics, bytes, throughput, and replay checks are in `RESULTS_CORE.csv` and `VERIFICATION.json`. The development gate fails seen quality, held-out margin in one seed, relative bytes, and runtime. A hypothesis for later work is that fused Givens kernels may reduce the CPU penalty, but this would need a separately frozen protocol.
