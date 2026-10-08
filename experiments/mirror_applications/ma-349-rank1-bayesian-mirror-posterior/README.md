# MA-349 — Sparse stochastic Mirror posterior vs rank-1 BNN

Status: **FAIL at development for the frozen posterior storage gate; fresh seeds sealed**
Dedicated branch: `research/ma-349-rank1-bayesian-mirror-posterior-20261008`
Prior art: PA41, rank-1 Bayesian neural networks.

## H — Hypothesis

A posterior over two weight functions that differ on one sparse direction can be stored as one shared mean plus a stochastic Mirror coordinate. This should retain predictive NLL and calibration while reducing actual bytes versus both a rank-1 multiplicative BNN posterior and two independent posterior models.

## T — Test

Synthetic Bayesian logistic regression with 32 inputs. The true posterior has two equally likely weights `w0 ± δ e_j`; labels are sampled from the exact mixture predictive probability. Each world uses 12,000 test examples. Methods are the deterministic posterior mean, two independent full posterior weights, a rank-1 multiplicative BNN posterior, Mirror scalar posterior, and an ordinary direct sparse scalar posterior. The two discrete modes are enumerated exactly, so predictive probabilities are not Monte Carlo estimates. Development seeds 34901 and 34902 were run. Fresh seeds 34911–34913 remain sealed.

Actual inference posteriors were serialized as deterministic ZIP/NPY FP32 payloads and reloaded before scoring. No optimizer or posterior fitting was run; the two-mode teacher distribution and sparse direction were known.

## D — Decision

**FAIL at development for the frozen storage hypothesis.** In both development worlds, Mirror produced exact teacher probability calibration (probability MSE 0), predictive NLL about 0.658, Brier about 0.233, and ECE about 0.012. The rank-1 BNN matched these metrics within floating-point noise and used 1,195B versus Mirror's 1,196B. The independent two-model posterior was smaller at 651–652B with the same predictive quality. The deterministic mean model had worse NLL (~0.691), Brier (~0.244), ECE (~0.071), and calibration MSE (~0.011).

The Mirror and ordinary direct sparse scalar-code posterior payloads had identical bytes and hashes. Fresh seeds stayed sealed after the development storage gate failed.

## Fact / Interpretation / Hypothesis

**Fact:** Mirror, rank-1 BNN and independent two-mode posterior had identical predictions/calibration. Mirror was larger than independent full models and one byte larger than the rank-1 BNN. Ten development payload/hash/metric rows replay exactly.

**Interpretation:** The stochastic scalar coordinate can express the exact posterior modes, but rank-1 multiplicative factors already express the same distribution. For two models at this scale, storing the independent weights compresses better in the actual archive.

**Hypothesis:** A larger posterior family or multimodal task may change amortization, but a new protocol must beat rank-1 BNN at actual bytes and preserve calibration.

## C — Strongest counter-hypothesis

The synthetic posterior was chosen to be exactly rank one and sparse. The rank-1 BNN is therefore an exact strong prior-art control, while the independent two-vector archive benefits from the tiny two-mode scale.

## U — Unconfirmed

No trained posterior, variational inference, natural dataset, neural hidden layer, out-of-distribution calibration, ensemble diversity, or fresh-world replication was run.

## Reproduction

```bash
python experiments/mirror_applications/ma-349-rank1-bayesian-mirror-posterior/source/run.py --dev-only
python -m pytest -q experiments/mirror_applications/ma-349-rank1-bayesian-mirror-posterior/tests
python experiments/mirror_applications/ma-349-rank1-bayesian-mirror-posterior/source/verify.py
```
