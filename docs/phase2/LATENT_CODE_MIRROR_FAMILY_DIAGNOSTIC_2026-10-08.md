# Latent-code Mirror family diagnostic — 2026-10-08

## Decision
Pause the current latent-code function family (MA-416 through MA-421) after four consecutive development screens. Resume only after redesigning the hypotheses so the Mirror coordinate is not algebraically the same state as a standard native representation. This follows the worker stop rule: repeated FAILs share one demonstrated structural cause, even though their quality and byte metrics differ.

## Shared structural cause
Each candidate's proposed coordinate is an ordinary native representation under a different name or parameterization: affine geometry coefficients (MA-416), nearest-centroid VQ code selection (MA-417), additive factor tables (MA-418), and direct harmonic amplitude/frequency/phase conditioning (MA-419). Their strongest ordinary controls reproduce the candidate outputs exactly. This removes the proposed Mirror-specific degree of freedom. Good compression or aligned synthetic fit can still occur, but cannot be attributed to Mirror.

## Evidence

| MA | Candidate | Main measured result | Exact ordinary/native alias | Outcome |
|---|---|---|---|---|
| 416 | DeepSDF geometry code | Query NRMSE 0.085–0.093; payload 0.934–0.937x independent codes; quality and byte gates missed | Native affine coordinate warp | FAIL; fresh sealed |
| 417 | Function VQ codebook | 0.778–0.780x payload; query/interpolation gates failed, including zero private residual | Native nearest-centroid VQ | FAIL; fresh sealed |
| 418 | Additive object/style/domain factors | 0.691–0.693x payload; heldout NRMSE 0.292–0.332 versus DeepSDF 0.151–0.168 | Native additive factorization | FAIL; fixed-budget quality miss; fresh sealed |
| 419 | Periodic activation view | NRMSE 0.000134–0.000160 and 0.177–0.178x generic concatenation payload; throughput 0.979–1.007x | Native direct harmonic conditioner, exactly same output and bytes | FAIL under Mirror-specific gate; aligned synthetic harmonic result; fresh sealed |

Per-experiment reports and serialized replay artifacts are retained in their MA directories. The MA-419 task is deliberately aligned to its periodic basis and does not establish broad natural-signal generalization. MA-418 did not approach the frozen quality gate; its oracle did, so fixed-budget optimization remains an alternative explanation for that quality result. These caveats do not change the native-control alias conclusion.

## Next design requirements
A redesigned latent-code candidate must establish a concrete nonlinear or task-dependent view operation whose complete paid state cannot be represented by the tested native code/control at equal quality and bytes. It must first validate function generalization on a task where the native parameterization is not the teacher itself, and separately report near-convergence capacity if it makes a capacity claim. Do not continue MA-420/421 unchanged while this family pause is active.

## Evidence labels
- **Fact:** each of the four controls matched its Mirror candidate algebraically or at serialized prediction replay; the quantitative gates differ by candidate.
- **Interpretation:** the family has not shown incremental Mirror-specific latent-code functionality.
- **Hypothesis:** a non-native functional view may still add value after the task and control are redesigned.
