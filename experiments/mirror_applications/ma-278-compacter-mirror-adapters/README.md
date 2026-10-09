# MA-278 — Compacter Mirror hypercomplex adapters

## H — Hypothesis

Task codes over shared Compacter fast-factor atoms may recover multiple functions with lower storage than native Compacter, while independent tasks may need private capacity.

## T — Conditions

Frozen shared 16×16 linear map, 8 tasks, 256 support / 512 audit examples. Fresh worlds 27810–27812 × seeds 0–2, 90 rows. Compared no adapter, native Compacter (shared slow/fast Kronecker factors plus 4 coefficients/task), shared single-direction scalar code (Mirror and generic coefficient), and independent rank-2 LoRA. Development audits fixed target-axis broadcasting, payload double counting, and matched the aligned teacher to the one-scalar code family before fresh. Payloads store W, slow/fast factors and codes once; expanded Kronecker atoms are deterministic reconstruction. CPU only.

## D — FAIL against storage and Mirror-specific gates

| Stratum | Method | Mean NRMSE | Payload B | Fit seconds |
|---|---|---:|---:|---:|
| shared_atoms | none | 0.6232244 | 1,070 | 0.000 |
| shared_atoms | compacter | 8.060233e-08 | 1,737 | 0.274 |
| shared_atoms | mirror_scalar | 5.788433e-09 | 1,643 | 0.170 |
| shared_atoms | generic_coeff | 5.788433e-09 | 1,643 | 0.172 |
| shared_atoms | lora2 | 0.4616571 | 3,137 | 0.199 |
| independent | none | 0.5429372 | 1,070 | 0.000 |
| independent | compacter | 0.5391782 | 1,737 | 0.191 |
| independent | mirror_scalar | 0.5417491 | 1,643 | 0.176 |
| independent | generic_coeff | 0.5417491 | 1,643 | 0.174 |
| independent | lora2 | 8.623656e-05 | 3,137 | 0.198 |

Fact: shared-atom aligned tasks: Mirror and generic coefficients were exactly equal in output (mean NRMSE 5.79e-9), using 1,643 B; native Compacter reached 8.06e-8 at 1,737 B. The 5.4% byte reduction misses the registered 30% gate, and there is no Mirror-specific advantage.

Fact: independent tasks: Mirror/generic NRMSE 0.542, close to no adapter 0.543 and Compacter 0.539; rank-2 LoRA reached 8.62e-5 at 3,137 B. The shared coordinate family did not capture independent task updates.

Interpretation: a shared atom direction can encode a matching task orbit, but generic scalar coefficients do the same. Private low-rank factors are needed for independent updates. The result does not show Mirror-specific compression and does not meet the storage gate.

## C — Strongest counter-hypothesis

The aligned teacher is generated from the same shared atom direction used by the scalar code; this is an intentionally favorable orbit. The slight byte advantage over Compacter reflects fewer per-task coefficients, not a Mirror operation.

## U — Unknown

No hypercomplex neural architecture, Transformer adapter, end-to-end task quality, learned basis across natural tasks, or GPU runtime was measured. No broad Compacter capacity claim is made.
