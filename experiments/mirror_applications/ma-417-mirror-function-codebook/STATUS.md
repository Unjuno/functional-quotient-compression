# MA-417 status

Status: **FAIL** at the frozen development gate; fresh remained sealed.

## H
A shared eight-entry codebook with one function address may compress clustered latent states without losing useful function quality.

## T
A fixed shared neural decoder, 64 train and 64 held-out functions, four private-residual levels, DeepSDF full codes, VQ codebook, PCA rank-4, exact native VQ and oracle; seeds 41701/41702.

## D
FAIL. Codebook total payload was 0.778–0.780x full DeepSDF, but at σ=0 its query NRMSE was 0.0538/0.0660 versus DeepSDF 0.0427/0.0606. At σ=0.05 it degraded to 0.2337/0.2005. Interpolation NRMSE exceeded 0.10 for σ=0.05. VQ exactly matched native nearest-centroid quantization.

## C
The tested task was deliberately generated from codebook clusters, yet nearest-centroid state still discarded function-relevant detail. The method also has no functional difference from conventional VQ.

## U
Natural task utility and private-residual compositions remain untested.

FACT: all 40 result rows replay from serialized FP16 inference payloads; initial invalid interpolation results are retained. INTERPRETATION: codebook addressing saves bytes but loses query/interpolation quality and offers no Mirror-specific gain. HYPOTHESIS: private residuals may improve this frontier under a new protocol.
