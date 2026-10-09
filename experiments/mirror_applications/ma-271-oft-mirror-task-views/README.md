# MA-271 — OFT versus Mirror task views

## H — Hypothesis

A one-angle orthogonal View can represent aligned task transforms over a shared frozen function with lower task-state bytes than independent OFT matrices. Independent rotations test the private-state boundary.

## T — Conditions

Frozen 16→32→8 feature network with eight task identities; 128 support and 512 audit examples per task. Fresh worlds 27110–27112 × seeds 0–2, 576 rows. Compared no view, support-fitted Cayley OFT, oracle-projected OFT upper, one-angle Givens Mirror, generic fixed-plane scalar and shared identity. The protocol was amended and pushed before fresh after development revealed support-fitted OFT residual (~0.23 NRMSE after a 2,000-step diagnostic) on independent transforms. Fresh results were not used to retune. Actual inference payload uses a canonical flat float32 serializer including shared model and all transform state. CPU only.

## D — FAIL against registered gates; no Mirror-specific result

| Stratum | Method | Mean NRMSE | Max NRMSE | Geometry distortion | Payload bytes | Fit seconds | Apply μs |
|---|---|---:|---:|---:|---:|---:|---:|
| aligned_plane | oft_independent | 0.00043460742 | 0.012286434 | 2.54e-07 | 3,413 | 0.06602 | 19.6 |
| aligned_plane | oft_oracle_upper | 3.7513651e-07 | 5.834151e-07 | 3.36e-07 | 3,414 | 0.00000 | 20.1 |
| aligned_plane | mirror_angle | 2.6631998e-09 | 3.1855937e-08 | 9.29e-08 | 3,156 | 0.02237 | 14.8 |
| aligned_plane | generic_plane_scalar | 2.6631998e-09 | 3.1855937e-08 | 9.29e-08 | 3,164 | 0.02237 | 15.1 |
| aligned_plane | shared_identity | 0.17676941 | 0.45528939 | 0 | 3,413 | 0.00000 | 17.4 |
| independent_orthogonal | oft_independent | 0.1508708 | 0.34403434 | 2.8e-07 | 3,413 | 0.06648 | 28.0 |
| independent_orthogonal | oft_oracle_upper | 4.6171988e-07 | 6.6564417e-07 | 3.59e-07 | 3,414 | 0.00000 | 19.7 |
| independent_orthogonal | shared_identity | 1.4275114 | 1.7148832 | 0 | 3,413 | 0.00000 | 16.1 |

Fact: aligned Mirror and generic fixed-plane scalar have identical predictions, task state, and fit time by construction; their mean NRMSE is ~2.7e-9. Mirror payload is 3,156 B versus 3,413 B for support-fitted independent OFT (7.5% less), failing the registered ≤50% gate. It also fails the ≤80% failure cutoff for a storage reduction claim. Oracle OFT upper has ~3.8e-7 NRMSE at 3,414 B. Pairwise geometry distortion is near numerical zero for orthogonal transforms.

Fact: on independent orthogonal tasks, oracle-projected OFT upper reconstructs with mean NRMSE ~4.6e-7 at 3,414 B. Support-fitted OFT mean NRMSE is ~0.151 at 3,413 B and therefore is not an adequate learned-task control. Fresh aligned Mirror max error is below 3.2e-8.

Interpretation: a one-angle coordinate is sufficient for an exactly matching one-plane orbit, but that gain is not Mirror-specific: an ordinary scalar multiplying the same fixed-plane generator is the identical parameterization. Independent orthogonal maps need much more state to represent; this experiment does not establish support-only OFT can learn those maps in the given adaptation budget.

Hypothesis: larger orthogonal rank / butterfly coordinates may give useful intermediate storage-quality points, but need a properly optimized native OFT and generic factor controls.

## C — Strongest counter-hypothesis

The aligned teacher was generated from the same one-angle plane used by Mirror, and the generic scalar control is mathematically equivalent. The observed byte advantage over full OFT comes from deliberately restricting the task family, not a Mirror-specific operation. In the independent stratum, oracle target leakage is only an upper bound and support fitting did not converge.

## U — Unknown

No natural vision/diffusion task, end-to-end OFT training, larger task families, butterfly controls, or well-tuned support-only optimizer comparison was measured. The independent-stratum capacity comparison is oracle-only; adaptation quality there remains not established.
