# State-space Mirror family diagnostic — 2026-10-08

## Decision
Pause current role/expert state-space proposals MA-434..438 after the two completed P0 screens MA-434 and MA-436. They use different View axes but share the same demonstrated structural failure: both Mirror coordinates are ordinary native state-space parameter conditioning with exact equivalent controls and no Mirror-specific function.

## Evidence

| MA | View tested | Measured result | Exact native control | Outcome |
|---|---|---|---|---|
| 434 | Scalar role code added to shared SSM log-decay | NRMSE 0.000237/0.000212; diversity 0.069/0.078; actual payload 0.930x full-copy bank; throughput 0.964/0.911x | Native per-role log-decay bias, byte/hash/output-identical | FAIL; actual bytes and no-alias gates missed |
| 436 | Four Givens angles over transition A with shared B/C | NRMSE 0.000434/0.000739; diversity 0.0206/0.0252; actual payload 0.743/0.738x full-copy bank; throughput 0.382/0.645x native | Native generated A, byte/hash/output-identical | FAIL; throughput/diversity and no-alias gates missed |

MA-436's first unamended screen is invalid: it used different input sequences for each expert and transformed A/B/C together, a pure state-coordinate gauge change that preserves transfer behavior. The initial run is retained and explicitly excluded. Protocol amendment 1 broadcast identical sequences and transformed A only; canonical metric and payload replay passed. Neither MA tests a trained Mamba/S4 model or natural sequence quality.

## Shared structural cause
A compact role/expert code can select a state-space parameter family, but the tested freedoms are standard native conditioner coordinates (log-decay bias or generated transition matrix). The exact native controls have identical paid state and outputs. Actual bytes also show that nominal full copies can compress; in MA-434, compressed copies were only 7% larger than the role-view payload. Eager state rotations were slow in MA-436.

## Next design requirements
Do not continue MA-435/437/438 unchanged. Resume only with an input-dependent, state-dependent, or structured-kernel View that survives an explicit native parameter generator, uses identical inputs across roles, and measures long-sequence quality plus optimized recurrent/FFT runtime. This pause does not cover meta-learning or other non-SSM families.

## Evidence labels
- **Fact:** two different role-state coordinates were exactly reproduced by their strongest ordinary native controls; MA-436 actual artifact replay passed after its explicit amendment.
- **Interpretation:** the tested state-space role views add no Mirror-specific functionality; the full-copy storage frontier is also narrower after serialization compression.
- **Hypothesis:** a non-native recurrent computation View may still offer value after redesign and realistic sequence validation.
