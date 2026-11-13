# MA-981 — shared ULA beam codebook with sector Views

Status: SCREENING. Protocol frozen before fresh worlds.

## H — hypothesis

A shared physical 8-antenna beam basis with compact sector phase-ramp addresses can provide multiple logical sector codebooks, retaining spectral efficiency with fewer serialized bytes than independent sector-specific beam tables.

## T — planned/executed conditions

The CPU simulation generates clustered narrowband channels for four sectors. It compares one global DFT beam bank, a Mirror phase-ramp sector View, the equivalent native sector-indexed steering code, independent sector beam tables, and a continuous matched-filter upper control. The metric is ideal single-user spectral efficiency at fixed SNR plus actual decoded payload bytes and beam-index bits.

## Prior-art delta

PA288 already learns network beam codebooks across sectors. This screen is a deliberately limited synthetic mechanism check; it does not reproduce PA288 or establish practical network gains.

## Boundary

No measured channel traces, pilot overhead, multi-user interference, RF energy, switch latency or field generalization. The native steering-code control determines whether any result is Mirror-specific.
