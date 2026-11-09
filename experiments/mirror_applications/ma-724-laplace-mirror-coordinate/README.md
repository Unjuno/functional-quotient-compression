# MA-724 — Laplace posterior over Mirror coordinates only

Status: SCREENING

## Hypothesis

H: Fitting a Laplace posterior only over a low-dimensional Mirror View code preserves held-out predictive quality and calibration while reducing complete posterior state versus full/subnetwork Laplace, and improves over ordinary subspace/diagonal controls.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, a stochastic low-dimensional code in a frozen random-feature network's output-weight subspace, so the posterior can vary logical functions without a full output-weight covariance.

Prior art: Laplace Redux (PA183), Rank-1 BNN (PA41). Controls: full last-layer Laplace, diagonal full-weight Laplace, code-space Laplace, native PCA/subnetwork Laplace with identical subspace, and rank-1 posterior. Exact subspace alias is FAIL for Mirror-specific attribution.

## Frozen protocol

Generate nonlinear binary classification from a fixed 64-feature random Fourier network, 256 train and 512 test points, seed 72401. Fit logistic output weights by MAP Newton iterations with Gaussian prior. Estimate full Hessian Laplace; for code-space methods derive a rank-q output-weight subspace from training-only posterior covariance (q=4/8/16/32), compute Hessian in that subspace, and sample 128 posterior draws. Compare predictive NLL, accuracy, 10-bin ECE, and actual serialized posterior payload bytes including mean, precision/covariance, basis and metadata. Development-only.

### Gates

PASS requires NLL within .02 nat/example, accuracy within .01 and ECE within .03 of full Laplace; >=25% complete state savings vs full; better frontier than native subspace and rank-1 controls. Exact native subspace alias is FAIL for Mirror-specific value. No audit after development alias/failure.

## Scope

Fixed random-feature classifier mechanism only; no trained neural network, natural dataset or calibrated deployment claim.

## Results — development seed 72401

**FACT:** Full Laplace payload is 12,924 B (NLL .59327, accuracy .6680, ECE .0360). Rank-4 native subspace and Mirror code payloads are exactly identical at 1,920 B (NLL .59048, accuracy .6738, ECE .0400), meeting frozen metric tolerances and saving 85.2% vs full. Diagonal Laplace is 922 B (NLL .59348, accuracy .6719, ECE .0241). The rank-1 covariance-factor control is 931 B with NLL .59048, accuracy .6738 and ECE .0400, matching the rank-4 function metrics at less than half the bytes. Newton MAP converged in 6 steps. All metrics use 512 held-out synthetic points.

**INTERPRETATION:** Low-dimensional curvature can preserve the tested predictive metrics at much lower storage than the full covariance, but the same result is ordinary subspace Laplace. The rank-1 control is smaller while matching rank-4 metrics. No Mirror-specific storage or quality gain is established.

**HYPOTHESIS:** More expressive nonlinear networks might require multiple code directions; this fixed-feature result cannot establish that.

**D:** FAIL for Mirror-specific value; exact native subspace alias and smaller rank-1 control. Audit unopened.

**C:** The MAP model is a fixed random Fourier classifier, posterior is local Gaussian and calibration is a single synthetic split. The rank-1 control is a simple single-factor Gaussian approximation, not a complete reproduction of the PA41 Rank-1 BNN.

**U:** Trained neural network, natural data, multi-seed calibration, ECE robustness, fit-time comparison, actual GPU sampling, exact Rank-1 BNN and full/subnetwork Laplace implementations.

### H / T / D / C / U

- **H:** Code-space Laplace preserves predictive quality and calibration while reducing bytes beyond native subspace controls.
- **T:** Fixed 64-feature RFF nonlinear classifier, 256 train/512 test, one seed 72401, six Newton steps; full/diagonal Laplace, q4/8/16/32 native subspace and Mirror, simple rank-1 posterior.
- **D:** FAIL for Mirror attribution. q4 reduces full Laplace bytes by 85.2%, but exactly aliases native subspace, and rank-1 control is smaller with the same metrics.
- **C:** Synthetic fixed-feature/local-Gaussian setup and simplified Rank-1 control.
- **U:** Neural/natural generalization and deployment runtime.
