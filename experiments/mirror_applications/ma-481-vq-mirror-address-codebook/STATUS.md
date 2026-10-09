# MA-481 status

- Status: FAIL for the frozen N64 address-quality gate; continuous address storage frontier measured
- Branch: `research/ma-481-vq-mirror-address-20261009`
- Base commit: `3bc48dec`
- Protocol freeze: `2f9d8213`
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze
- Results committed: pending
- Verification committed: pending
- Registry row updated: yes (branch tip)

## Next action

Run integrity checks, record the claim, and push the dedicated branch.

## Limitations

- The key manifold is hand-specified and crowded; the explicit baseline also misses paraphrase recall.
- Continuous Mirror and generic coefficients have identical retrieval behavior; Mirror is only 5.4% smaller than generic at N64.
- Fixed VQ codebooks are not learned VQ-VAE models.

## Decisions / rulings

- Fresh results are retained even though the direct explicit baseline misses the 99% per-world paraphrase gate; no fresh noise or radius retuning was performed.
- All values, keys/radii, codebooks, indices and metadata are serialized and charged.
- Address decode is separated from query lookup timing; neither is an end-to-end service claim.
