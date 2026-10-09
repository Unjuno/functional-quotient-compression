# MA-516 status

- Status: FAIL
- Branch: `research/ma-516-function-vector-mirror-compression-20261009`
- Protocol/source freeze: `e6d6dd31`
- Model: pinned GPT-2 124,439,808 parameters, CPU float32; inference runtime files 550,959,861B
- Fresh: worlds 51610-51612 × seeds 0-2; corrected timing replay completed

H: Extracted function vectors and PCA Mirror coordinates preserve held-out function accuracy with fewer bytes.

T: Arbitrary object-to-color mappings; four support and four held-out queries. Query-only, direct ICL, explicit activation deltas, PCA ranks 4/8/16.

D: FAIL. Direct ICL accuracy 16.7/18.8/13.5% by world, barely above 12.5% chance. Query-only and explicit-vector accuracy were 0%; PCA all ranks 0%. Rank-4 uses 17,569B vs explicit 50,793B (34.6%) at vector NRMSE .0097, but the compressed vector does not preserve useful task accuracy.

C: This support-delta extraction/injection did not create a causal task vector; direct ICL remains weak.

U: Natural ICL, larger models, other extraction sites/scales, and downstream quality.

## Next action

Commit MA-516 results/verification/registry, then continue to the next executable P0 outside the deferred LoReFT family.
