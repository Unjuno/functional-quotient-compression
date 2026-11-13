# MA-981 results — synthetic sector beam codebook

Status: **FAIL.** The preregistered byte gate is missed, and the Mirror View matches a native steering-code control exactly. The shared codebook is a synthetic structured-code feasibility result, not evidence from measured channels or a deployed network.

## H — hypothesis

A shared 8-antenna beam basis plus compact sector phase Views preserves throughput at lower serialized bytes than independent sector codebooks. A native sector-indexed steering code tests whether the result needs Mirror.

## T — execution

**Fact.** The CPU simulation generated one LOS plus three clustered paths for four sector-centered angles, 256 users per sector, 8 ULA antennas, and 10 dB SNR. Development seed 98101 was separate. Fresh channel worlds were 98111, 98113, and 98117. Controls were a global 8-beam DFT codebook, an 8-beam Mirror sector phase View, the equivalent native sector steering-code control, 16-beam independent sector codebooks, and an unquantized matched-filter upper bound. Metrics were decoded `torch.save` bytes, beam-index feedback bits, spectral efficiency, and operation proxies.

**Fact.** Fresh mean spectral efficiency averaged across worlds was 2.447 bits/s/Hz global shared, 2.869 Mirror, 2.869 native sector steering, 2.879 independent dense, and 2.974 continuous matched-filter upper. Mirror/native retained 99.66% of independent dense mean in each world (2.8279/2.8375, 2.9551/2.9653, 2.8239/2.8340). Their 5th-percentile rates were identical to each other in all worlds.

**Fact.** Actual payloads were 10,401 B for both Mirror and native steering code, 18,277 B for independent dense codebooks, and 14,181 B for the global DFT bank. Thus Mirror/native used 43.1% fewer bytes than the 16-beam independent bank, but the simpler native angle representation had the same payload size and output. Feedback used 3 bits per selected beam for the 8-beam code and 4 bits for the 16-beam independent code. The simulation omits pilot, coherence, interference and control-channel overhead.

**Fact.** Mirror and native steering controls decode the same sector centers and offsets and produce identical beams/predictions. Both require the same phase-ramp synthesis work; an initial CSV counted zero phase operations for native control. This accounting was corrected to 64 phase ops/user for both because both evaluate eight 8-element steering beams. The stored throughput/bytes are unchanged.

## D — decision

**FAIL.** The preregistered <=50% byte gate is missed: 10,401/18,277 = 56.9%, despite meeting the quality threshold. Mirror also exactly aliases the native sector steering-code implementation at equal bytes and quality. The apparent gain over independent dense codebooks is explained by an ordinary parametric phase-steered codebook with half as many beams. That is a useful synthetic structured-codebook result, not a Mirror-specific result.

## C — strongest counter-hypothesis

**Interpretation.** Smooth sector geometry means beam directions can be represented by a shared offset grid and four sector centers. Standard steering-angle parameters already provide this sharing without a Mirror mechanism. The quality/byte gain also comes with fewer codewords, so it is not a fixed-feedback comparison to the dense upper control.

## U — unresolved

- Public measured channel traces and comparison against PA288 learned codebooks.
- Multi-user interference, pilot/CSI overhead, coherence time, feedback signaling and RF switch delay.
- Equal codeword count or equal-feedback comparison with independent learned codebooks.
- Measured hardware energy/latency and OOD sector generalization.

## Evidence separation

- **Fact:** synthetic channel setup, fresh rates, decoded payload bytes, feedback bit counts, and exact native/Mirror equivalence.
- **Interpretation:** compact steering codebooks can reduce bytes against independent dense banks in this aligned toy setting; Mirror adds no demonstrated benefit over native phase-angle encoding.
- **Hypothesis:** learned nonuniform sector codebooks on measured channels may change the frontier, but this run does not establish it.
