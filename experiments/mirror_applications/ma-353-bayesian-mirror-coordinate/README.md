# MA-353 — Bayesian posterior over Mirror coordinates

Status: **FAIL at development for storage and Mirror-specific gates; fresh sealed**
Dedicated branch: `research/ma-353-bayesian-mirror-coordinate-20261008`
Base: `research/mirror-application-worker-ready-20261007` (`c935a90`)
Prior art: PA41 rank-1 BNN; MA-349 two-mode sparse posterior.

## H — Hypothesis

A Gaussian posterior over one continuous Mirror coordinate can represent a three-mode logistic posterior with calibrated predictions and fewer actual bytes than a matched rank-1 multiplicative BNN posterior and independent three-model archive.

## Mirror insertion

> **Mirror insertion:** this experiment adds a posterior-distributed scalar `m` to a shared logistic classifier so three nearby weight functions can be represented as samples along one functional coordinate instead of stored as separate weight vectors.

The model is `w(m)=w0 + m*d*e_j`; prediction integrates five fixed Gauss-Hermite nodes. Native controls are deterministic mean, three independent full vectors, rank-1 multiplicative Gaussian BNN, and ordinary direct Gaussian scalar code. All posterior tensors, coordinate code, quadrature nodes/weights, indices and metadata are charged.

## T — Test

Oracle synthetic 16D Bayesian logistic posterior, 10,000 examples per world. Teacher posterior has equal modes `w0 + {-d,0,+d}e_j`; predictive targets are sampled from exact mixture probabilities. Development seeds 35301/35302; fresh seeds 35311–35313 remained sealed. No fitting updates; deterministic five-point quadrature used for continuous posterior predictions. ZIP/NPY FP32 payload was reloaded before scoring.

## D — Decision

**FAIL at development; fresh sealed.** Both seeds showed essentially identical predictions for Mirror Gaussian coordinate, direct Gaussian scalar, and rank-1 Gaussian BNN. Their NLL was approximately 0.660–0.664, ECE 0.009–0.010, calibration MSE 1.3e-5–1.5e-5, and shifted-input NLL 0.425–0.435. Mirror used 1,525B versus 1,493B for rank-1 BNN; the ordinary direct Gaussian scalar had identical bytes and hash to Mirror. The independent three-model posterior used 591–592B with exact teacher calibration and comparable NLL. The frozen 10% storage gate failed decisively.

## Fact / Interpretation / Hypothesis

**Fact:** Mirror and direct scalar payload bytes/hashes were identical; Mirror was larger than rank-1 BNN and the independent three-mode posterior. Metrics matched the rank-1 Gaussian factor control to numerical precision.

**Interpretation:** Moving a Gaussian distribution to the Mirror address did not add a useful posterior representation in this one-dimensional planted problem. Rank-1 Bayesian factors express the same uncertainty more compactly; independent vectors compress well at only three modes.

**Hypothesis:** Posterior coordinates may matter when many posterior modes share a nonlinear orbit, but the next test should first establish a byte advantage over both rank-1 BNN and direct scalar posterior under non-oracle fitting.

## C — Strongest counter-hypothesis

The exact posterior is oracle supplied and one-dimensional; the BNN control is nearly the same parameterization, while three independent vectors are unusually small. A learned high-dimensional posterior may have different amortization economics.

## U — Unconfirmed

No learned posterior, real dataset, neural hidden layer, posterior fitting, multi-dimensional orbit, near-converged capacity study, or fresh replication.

## Reproduction

```bash
python experiments/mirror_applications/ma-353-bayesian-mirror-coordinate/source/run.py --dev-only
python -m pytest -q experiments/mirror_applications/ma-353-bayesian-mirror-coordinate/tests
python experiments/mirror_applications/ma-353-bayesian-mirror-coordinate/source/verify.py
```
