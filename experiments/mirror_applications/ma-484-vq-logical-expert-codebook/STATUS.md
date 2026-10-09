# MA-484 status

- Status: FAIL
- Branch: `research/ma-484-vq-logical-expert-20261009`
- Base commit: `5dff91c2`
- Protocol frozen before fresh: yes
- Development and fresh: complete; 3 fresh worlds × 3 seeds
- Results and verification: pending commit

H: VQ Mirror expert codes preserve useful functions under actual-byte compression.

T: 64 synthetic 32D linear experts with four shared residual directions; K=8/16/32/64; independent, low-rank FP32, generic VQ and Mirror VQ controls.

D: FAIL. At K64, Mirror 3,933B, NRMSE .3068, code collision fraction .863. Shared-basis FP32 low-rank control is exact at 3,681B. No Mirror codebook meets <=.05 NRMSE / <=50% bytes / <=2% collisions.

C: A rank-four teacher is best represented by unquantized shared-basis coordinates; generic VQ fitting was not coordinate-optimized.

U: nonlinear learned experts, direct coordinate codebook fitting, routing, and natural task distributions.

Next: artifact tests, registry update, integrity check, push, continue at next untested P0.
