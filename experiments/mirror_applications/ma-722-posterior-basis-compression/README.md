# MA-722 — compress SWAG/PCA posterior basis with Mirror factors

Status: SCREENING

## Hypothesis

H: Factorizing the SWAG posterior basis into compact Mirror factors preserves predictive NLL/calibration while reducing complete posterior bytes beyond native SVD basis compression.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, a low-rank factor code for the posterior covariance basis itself, so posterior samples can be reconstructed without storing all basis vectors explicitly.

Prior art: SWAG (PA181) and Subspace Inference (PA182). Native truncated SVD factorization is mandatory, with full SWAG basis as reference. Mirror and native factorization use the same deterministic decomposition; any exact alias is a negative Mirror-specific result.

## Frozen protocol

Synthetic Gaussian Bayesian linear regression p=64, rank-16 covariance plus residual diagonal, 64 trajectory snapshots, 512 test examples, seed 72201. Estimate covariance from snapshots, eigendecompose, form a p×r SWAG basis, then factorize basis U≈A_q C_q with q={2,4,8,12,16}. Add diagonal residual correction so marginal variance is retained. Compare dense covariance, full SWAG rank-16 basis, truncated SVD-factor basis (native) and Mirror factor code. Measure predictive NLL, 90% interval coverage, covariance Frobenius error, actual serialized full inference bundle bytes and sample compute proxy.

### Gates

PASS requires NLL within .02 and coverage within .03 of dense, >=25% full bundle byte savings, and a better frontier than native SVD at matched q. Exact native SVD alias is FAIL for Mirror-specific value. No audit after alias/frozen development failure.

## Scope

Synthetic linear posterior only; no neural posterior or benchmark claim.

## Results — development seed 72201

**FACT:** Dense covariance is 12,846 B (NLL 1.79769, coverage .87305). Full SWAG basis plus diagonal is 5,225 B (NLL 1.79756). q=4 basis factorization is 2,792 B, NLL 1.80382, coverage .87109, and covariance Frobenius error .5642; this is 46.6% fewer bytes than the full SWAG basis, while NLL/coverage remain within frozen tolerances. q=8 is 3,995 B (23.5% below full SWAG basis), q=16 is 6,366 B. Every Mirror factor payload/hash/function exactly aliases native SVD factorization at the same q. All methods share the same synthetic evaluation inputs/targets.

**INTERPRETATION:** Basis factorization can reduce posterior state while preserving predictive metrics in this synthetic screen, but ordinary truncated SVD explains the full gain. q=4 also costs a 4x sample multiply proxy versus rank-1, but is cheaper than full dense covariance. Coordinate samples are not independent capacity.

**HYPOTHESIS:** Structured basis compression may be useful as a general storage method if its basis decoder beats native compression; this result gives no evidence for Mirror-specific benefit.

**D:** FAIL for Mirror-specific attribution; fresh/audit unopened.

**C:** The synthetic posterior has a decaying low-rank spectrum and diagonal correction; it is favorable to factorization, and one seed cannot establish calibration robustness.

**U:** Natural/neural model posterior, multi-seed calibration, ECE/OOD, training cost and serving runtime.

### H / T / D / C / U

- **H:** Mirror factors compress the posterior covariance basis while preserving calibration better than native SVD.
- **T:** p=64, rank-16 Gaussian covariance, 64 snapshots, 512 test rows, seed 72201; dense, SWAG basis, q=2/4/8/12/16 native and Mirror factorizations.
- **D:** FAIL for Mirror-specific value; q4 meets quality and saves 46.6% vs full SWAG basis, but exact native SVD alias.
- **C:** Ordinary low-rank matrix factorization fully explains the result.
- **U:** Natural/neural posterior and audit evidence.
