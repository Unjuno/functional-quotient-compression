# MA-349 — Rank-1 Bayesian Mirror posterior

Status: SCREENING. Prior art: PA41 Rank-1 Bayesian neural networks.

## H

For binary logistic prediction in two dimensions, a posterior over a single functional Mirror angle may achieve predictive NLL/calibration comparable to a rank-1 Gaussian weight posterior with fewer serialized posterior parameters. A deterministic model and a matched Monte Carlo budget are controls.

## Mirror insertion

> **Mirror insertion:** this experiment puts posterior uncertainty on a scalar angle `m` that rotates a shared two-dimensional logistic weight vector, instead of placing uncertainty in rank-one weight perturbations.

Compare deterministic MAP, Mirror posterior over angle, and a Rank-1 BNN-style posterior (w=mu+a z). Both posterior methods use the same number of Monte Carlo samples at inference. Report posterior predictive NLL, Brier score, ECE, accuracy, OOD NLL, actual payload bytes, sampled operation proxy and latency.

## T

A two-feature binary logistic teacher with 64 labeled training examples, 1,024 IID test examples and 1,024 shifted/OOD examples per world. Development worlds 34921–34922; fresh worlds 34931–34933. Train variational posteriors for 1,500 updates with fixed 16 Monte Carlo draws per update. Inference uses 64 posterior samples. No test/OOD data are used in fitting or selection.

## Gates

**PROMISING:** Mirror predictive NLL within 0.02 of Rank-1 BNN on IID and OOD, ECE within 0.02, and >=20% fewer actual posterior payload bytes. **FAIL:** quality/calibration gate misses or the byte frontier is not improved. **NOT ESTABLISHED:** optimizer failure or nonfinite posterior.

## C / U

This is a tiny synthetic Bayesian classification mechanism test, not a neural network benchmark. It does not establish rank-1 BNN reproduction, natural-data calibration, or useful ensemble capacity.
