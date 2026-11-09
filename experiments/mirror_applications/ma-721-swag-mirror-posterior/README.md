# MA-721 — SWAG covariance subspace to Mirror posterior coordinates

Status: SCREENING

## Hypothesis

H: A structured Mirror posterior coordinate over a shared posterior mean and low-rank covariance can preserve predictive NLL and calibration at lower complete serialized bytes than full covariance, and improve on native SWAG / rank-1 Bayesian controls.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, a sampled latent coordinate over a shared mean and covariance basis, so multiple posterior predictive functions can be sampled without storing independent full parameter vectors.

Prior art PA181 SWAG, PA41 rank-1 Bayesian neural nets. Native SWAG with the same mean, basis, diagonal correction and rank is the essential control. If it matches exactly, posterior multiplicity is established as ordinary posterior sampling, not Mirror-specific value.

## Frozen protocol

Synthetic Bayesian linear regression with p=64, n_train=128, n_test=512, noise std .2, dev seed 72101. Generate a rank-8 plus diagonal Gaussian parameter posterior and draw 64 SWA-like parameter snapshots for covariance estimation. Compare exact dense covariance, SWAG low-rank-plus-diagonal approximations ranks 1/2/4/8/16, Mirror coordinates sampling the exact same posterior factors, and rank-1 BNN factorized variance control. Use 256 posterior predictive samples per test input for NLL and 90% interval coverage. Serialize full mean, covariance representation, posterior sampling state and metadata; count shared latent bases and codes. Development-only, no audit worlds.

### Gates

PASS requires NLL within .02 nat/example and coverage within .03 absolute of the dense posterior predictive reference, at least 25% fewer complete payload bytes than full covariance, and a better NLL/coverage/byte frontier than matched-rank SWAG and rank-1 BNN. Exact SWAG/Mirror payload/function alias is FAIL for Mirror-specific value. Stop at development if the alias is exact.

## Scope

This is a synthetic linear Bayesian regression screen, not neural-network or benchmark calibration evidence.

## Results — development seed 72101

**FACT:** Dense Gaussian posterior bundle is 12,865 B, NLL 1.7660, 90% interval coverage .8691. Rank-8 SWAG and Mirror bundles are exactly the same 3,293 B payload and predictions: NLL 1.7879, coverage .8516 (NLL delta .0218, coverage delta .0176). Rank-16 is 5,223 B, NLL 1.7866, coverage .8496 (NLL delta .0206). Rank-1 is 1,582 B, NLL 1.8089; the rank-1 BNN control matches that rank-1 prediction quality at 1,385 B. Maximum fixed test set is 512; all controls share it.

**INTERPRETATION:** Low-rank plus diagonal posterior storage is 74.4% below dense at rank 8, but Mirror is exactly native SWAG reparameterization. Rank 8/16 narrowly miss the frozen NLL tolerance; coverage differences are within threshold. The rank-1 Bayesian control is smaller than the corresponding Mirror state.

**HYPOTHESIS:** Structured covariance compression may be useful only if it beats native SWAG at matched calibration, NLL and stored state; this experiment does not show that.

**D:** FAIL for Mirror-specific value; no audit opened after exact native alias.

**C:** The Gaussian linear model and finite 64-snapshot covariance estimate do not represent neural posterior geometry; denser snapshots or other data may shift calibration.

**U:** Neural benchmark quality, calibration over multiple worlds, training trajectory cost, ECE/OOD, inference throughput and any method beating SWAG.

### H / T / D / C / U

- **H:** A Mirror coordinate over posterior covariance improves the NLL/coverage/bytes frontier over dense posterior and native SWAG.
- **T:** p=64, 128 synthetic train examples, 512 test examples, 64 posterior snapshots, dev seed 72101; dense, SWAG ranks 1/2/4/8/16, matched Mirror, rank-1 BNN.
- **D:** FAIL; exact Mirror/SWAG payload and function alias, rank-1 BNN smaller; rank-8/16 NLL gate narrowly missed.
- **C:** Synthetic Gaussian posterior and Monte Carlo covariance-estimation error.
- **U:** Neural/natural posterior generalization and runtime.
