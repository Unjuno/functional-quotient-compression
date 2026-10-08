# MA-724 — Laplace posterior over Mirror code

Status: SCREENING  
Evidence lane: QUALITY / STORAGE / RUNTIME  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Random draw #18: pool size 1038, index 653, selected MA-724.

## Hypothesis

H: With one trained shared MLP and a compact Givens feature view `m`, a posterior only over `m` can approach subnetwork/full diagonal Laplace predictive quality while using fewer actual serialized posterior-state bytes and lower sampling cost.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` to the hidden-feature-to-readout interface so that posterior samples over functional views yield multiple predictive functions without posterior storage over all shared-network weights.

A shared 30→32→32 MLP and output head form physical θ; `m` is 16 Givens angles over the 32-D hidden representation before the shared output layer.

## Prior art and controls

PA183 Laplace Redux motivates curvature/posterior approximation choices; PA41 Rank-1 Bayesian neural nets is a direct uncertainty control family. Controls are MAP, full-network diagonal empirical-Fisher Laplace, full-covariance last-layer Laplace, full-covariance code-space Gauss-Newton Laplace, diagonal code-space Laplace, and a 5-model independent ensemble where compute allows. This does not claim the Laplace or rank-1 Bayesian concepts as new.

## Data and split

Use sklearn's bundled UCI Wisconsin Diagnostic Breast Cancer data. Stratified nested splits: train 60%, development 20%, fresh 20%, seeds 724/725. Standardizer is fit only on train. Dev alone chooses posterior prior precision from `(0.1, 1, 10, 100)`; fresh is evaluated once after config freeze.

## Frozen gates

PASS requires fresh predictive NLL within 0.02 nats and ECE-10 within 0.02 of the best Laplace control, while the m posterior uses at least 4× fewer actual serialized posterior-state bytes; complete shared-base plus posterior-state bytes are separately charged and reported and at least 2× lower 100-sample predictive wall time than full-network diagonal Laplace. FAIL if quality or resource gates miss.

## Fact / interpretation / hypothesis / boundary

Will be completed after dev selection, fresh evaluation and verification.

## Development result (fresh remains locked)

FACT: One shared MLP reached dev NLL 0.01487, ECE-10 0.01325, and accuracy 1.000. Train selection stopped at epoch 25 after 65 updates on 341 training examples; training time was 0.0617s.

FACT: All four Laplace variants selected prior precision λ=100 using dev only. The best dev Laplace control was subnet_full (NLL 0.01593, ECE 0.01423). Code full-cov m: NLL 0.01499, ECE 0.01336; code diagonal m: NLL 0.01499, ECE 0.01336.

FACT: The shared base payload is 11,632 B. Full-diagonal posterior state is 11,856 B, code full covariance is 2,924 B, and code diagonal is 1,964 B. Thus the full-covariance m state is 4.05× smaller than full diagonal state; complete base+state totals are 23,488 B and 14,556 B, respectively.

INTERPRETATION: Development meets the quality and marginal posterior-state byte gates provisionally. Full-covariance and diagonal m predictions are nearly identical, so current evidence supports compact uncertainty state, not a benefit from structured covariance.

HYPOTHESIS: A posterior over `m` may retain predictive calibration while reducing marginal posterior bytes and sampling time. Fresh evaluation will decide this once.

U: Fresh NLL/ECE, actual batch-1 posterior-predictive latency, and independent-seed stability are unmeasured. PA41 rank-1 BNN is not directly implemented; it remains a counter-hypothesis.
