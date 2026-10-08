# MA-418 status

Status: **FAIL — development screen; fresh seeds sealed.**

## H
Learned additive object/style/domain codes can transfer to unseen combinations while reducing per-combination latent state.

## T
Fixed shared decoder, 256 combinations, 80/20 split with each factor seen in training; additive factors versus independent support-adapted DeepSDF codes and exact native additive control. Seeds 41801/41802.

## D
Both frozen development seeds failed the quality gate. Factor NRMSE was 0.3323 and 0.2918; support-adapted independent DeepSDF NRMSE was 0.1676 and 0.1510. Factor payloads were 13,286/13,312 bytes versus 19,244/19,209 bytes (0.691/0.693x), so the byte gate passed, but query quality did not. The native additive control produced exactly the same predictions. Serialized-payload and metric replay passed; fresh seeds 41811–41813 were not accessed.

## C
The teacher was generated from additive latent factors, and the oracle table reconstructed it with NRMSE 0.00577/0.00168. Thus the observed gap does not show an additive representation-capacity limit. The stronger counter-hypothesis is optimization/identifiability under the 600-update frozen budget; the native additive factorization is also an exact algebraic control, so no Mirror-specific advantage is established.

## U
Near-convergence capacity, other factor geometries, natural compositional tasks, and whether a better optimizer recovers the oracle are untested. This fixed-budget FAIL does not establish a capacity limit. The factor payload retains a 31% storage saving, but that point is not quality-competitive at this budget.

## Evidence separation
- **Fact:** fixed decoder; synthetic additive teacher; 2 development seeds; all payloads FP16/uint8 compressed NPZ; native alias exact; verifier replay passed.
- **Interpretation:** factor tables compress state, but do not recover held-out outputs to the frozen quality gate at 600 updates.
- **Hypothesis:** conditioning/identifiability or optimization budget explains the gap; a converged capacity comparison is still needed.
